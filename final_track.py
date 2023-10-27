# 本文件为bytetrack结合yolov5+deepsort添加检测目标下行上行的功能
import argparse
import os
import os.path as osp
import cv2
import numpy as np
import torch
import queue
import time
import threading


from PIL import Image, ImageDraw, ImageFont
# 设置队列，用于存储videocap读取的图片，通过多线程的方式进行再从队列中取出

q = queue.Queue()

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./videos/test1080.mp4", help="path to images or video"
        # "--path", default="rtsp://admin:sxd123!!@192.168.11.53/h264/ch1/main/av_stream", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=False,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='exps/example/mot/yolox_nano_mix_det.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='assets/bytetrack_nano_mot17.pth.tar', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


exit_event = threading.Event()


# 解决puttext不能显示中文的问题
def cv2AddChineseText(img, text, position, textColor=(0, 0, 255), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def wait_for_key():
    if cv2.waitKey(1) == 27:
        # cv2.waitKey(0)
        exit_event.set()

# 队列显示线程
def Display():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            # frame = cv2.resize(frame, (960, 540))
            cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    # start------------------画线代码-------------------- #
    width = 1920
    height = 1080
    fps = 25
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    # 用于记录轨迹信息
    pts = {}

    # 测试视频撞线坐标
    # list_pts_blue=[[379,419],[1801,425],[1808,430],[380,420]]

    # 摄像头撞线坐标
    # list_pts_blue = [[1350, 369], [1801, 475], [1808, 480], [1350, 370]]

    # 客流视频蓝色撞线坐标
    list_pts_blue = [[460, 0], [1400, 0], [1400, 250], [560, 250]]
    # list_pts_blue = [[280, 100], [650, 100], [650, 150], [280, 150]]
    # list_pts_blue = [int(x/2) for x in list_pts_blue]

    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    # 填充1
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    # 在z上增加一维
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]
    # 填充第二个撞线polygon（黄色）
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    # 测试视频撞线坐标
    # list_pts_yellow=[[400,430],[1801,440],[1808,450],[411,440]]

    # 摄像头撞线坐标
    # list_pts_yellow = [[1340, 380], [1751, 470], [1620, 500], [1261, 390]]

    # 客流视频黄色撞线坐标
    list_pts_yellow = [[560, 404], [1300, 404], [1300, 710], [560, 710]]

    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]
    # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用

    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    # cv2.INTER_NEAREST 最邻近插值 才不会出现黄色区域周围在缩小尺寸后有蓝色区域的情况
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width // 2, height // 2),
                                              interpolation=cv2.INTER_NEAREST)
    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image

    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (width // 2, height // 2))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 下行数量
    down_count = 0
    # 上行数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int((width / 2) * 0.01), int((height / 2) * 0.05))

    show_text_postion = (int((width / 2) * 0.01), int((height / 2) * 0.1))

    # end------------------画线代码-------------------- #

    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    frame_id = 0
    results = []
    ret_val, frame = cap.read()

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # 每隔6帧推理一次，因为推理这里
            if frame_id % 6 == 0:
                frame = cv2.resize(frame, (width // 2, height // 2))
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
                outputs_or, img_info_or = predictor.inference(frame, timer)
                print('outputs:',outputs)
                outputs = outputs_or
                img_info = img_info_or
                # 在检测结果上添加红蓝撞区
                output_image_frame = cv2.add(img_info['raw_img'], color_polygons_image)
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    list_bboxs = online_targets
                    if len(list_bboxs) > 0:
                        for t in list_bboxs:
                            tlwh = t.tlwh
                            tid = t.track_id
                            # start-------------撞线检测-------------#
                            x1, y1, x2, y2, track_id = tlwh[0], tlwh[1], tlwh[2], tlwh[3], tid
                            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                            # 检测点为检测框中点
                            y1_offset = int(y1 + (y2 * 0.5))
                            x1_offset = int(x1 + (x2 * 0.5))
                            # 撞线的点
                            y = y1_offset
                            x = x1_offset
                            # 然后每检测出一个预测框，就将中心点加入队列
                            center = (x, y)
                            if track_id in pts:
                                pts[track_id].append(center)
                            else:
                                pts[track_id] = []
                                pts[track_id].append(center)

                            thickness = 2
                            cv2.circle(output_image_frame, (center), 1, [255, 255, 255], thickness)
                            # cv2.imshow('qe', output_image_frame)
                            for j in range(1, len(pts[track_id])):
                                if pts[track_id][j - 1] is None or pts[track_id][j] is None:
                                    continue
                                cv2.line(output_image_frame, (pts[track_id][j - 1]), (pts[track_id][j]),
                                         [255, 255, 255],
                                         thickness)
                            try:
                                if polygon_mask_blue_and_yellow[y, x] == 1:
                                    print("撞 蓝polygon", track_id)
                                    blue_info1 = '撞 蓝polygon' + str(track_id)
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_blue_polygon:
                                        print("蓝polygon list 无此 track_id")
                                        blue_info2 = "蓝polygon list 无此 track_id"
                                        list_overlapping_blue_polygon.append(track_id)
                                    # 判断 黄polygon list里是否有此 track_id
                                    # 有此track_id，则认为是 UP (上行)方向
                                    if track_id in list_overlapping_yellow_polygon:
                                        print("上行")
                                        blue_info3 = "上行"
                                        # 上行+1
                                        up_count += 1
                                        print('up count:', up_count, ', up id:', list_overlapping_yellow_polygon)
                                        blue_info4 = 'up count:' + str(up_count) + ', up id:' + str(
                                            list_overlapping_yellow_polygon)
                                        # 删除 黄polygon list 中的此id
                                        list_overlapping_yellow_polygon.remove(track_id)

                                elif polygon_mask_blue_and_yellow[y, x] == 2:
                                    print("撞 黄polygon", track_id)
                                    yellow_info1 = '撞 黄polygon' + str(track_id)
                                    # 如果撞 黄polygon
                                    if track_id not in list_overlapping_yellow_polygon:
                                        print("黄polygon list 无此 track_id")
                                        yellow_info2 = "黄polygon list 无此 track_id"
                                        list_overlapping_yellow_polygon.append(track_id)
                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 DOWN（下行）方向
                                    if track_id in list_overlapping_blue_polygon:
                                        print("下行")
                                        yellow_info3 = "下行"
                                        # 下行+1
                                        down_count += 1
                                        print('down count:', down_count, ', down id:', list_overlapping_blue_polygon)
                                        yellow_info4 = 'down count:' + str(down_count) + ', down id:' + str(
                                            list_overlapping_blue_polygon)
                                        # 删除 蓝polygon list 中的此id
                                        list_overlapping_blue_polygon.remove(track_id)
                            except IndexError:
                                print("数组越界异常")
                            # end-------------撞线检测-------------#
                            # start---------------------清除无用id（每隔一帧）----------------------#
                            if frame_id % 300 == 0:  # 70帧还检测不到删除
                                print('清除无用id')
                                list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                                for id1 in list_overlapping_all:
                                    is_found = False
                                    for t in list_bboxs:
                                        bbox_id = t.track_id
                                        if bbox_id == id1:
                                            is_found = True
                                    if not is_found:
                                        # 如果没找到，删除id
                                        if id1 in list_overlapping_yellow_polygon:
                                            list_overlapping_yellow_polygon.remove(id1)

                                        if id1 in list_overlapping_blue_polygon:
                                            list_overlapping_blue_polygon.remove(id1)
                                list_overlapping_all.clear()
                                # 清空list
                                list_bboxs.clear()
                            # end---------------------清除无用id（每隔一帧）----------------------#
                            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)
                                results.append(
                                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )

                        timer.toc()
                        output_image_frame = plot_tracking(
                            output_image_frame, online_tlwhs, online_ids, frame_id=frame_id + 1,
                            fps=1. / timer.average_time
                        )
                        text_draw = 'DOWN: ' + str(down_count) + \
                                    ' , UP: ' + str(up_count)

                        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                                         org=draw_text_postion,
                                                         fontFace=font_draw_number,
                                                         fontScale=0.75, color=(0, 0, 255), thickness=2)
                        if len(list_overlapping_blue_polygon) == 0:
                            str_blue = ''
                        if len(list_overlapping_blue_polygon) == 1:
                            str_blue = str(list_overlapping_blue_polygon)
                        else:
                            mylist0 = []
                            for i in list_overlapping_blue_polygon:
                                mylist0.append(str(i))
                            mylist1 = [i for i in mylist0]  # 将list[list]转换成list
                            str_blue = ";".join(mylist1).strip("['").strip("']")
                        if len(list_overlapping_yellow_polygon) == 0:
                            str_yellow = ''
                        if len(list_overlapping_yellow_polygon) == 1:
                            str_blue = str(list_overlapping_yellow_polygon)
                        else:
                            mylist2 = []
                            for i in list_overlapping_yellow_polygon:
                                mylist2.append(str(i))
                            mylist3 = [''.join(i) for i in mylist2]  # 将list[list]转换成list
                            str_yellow = ";".join(mylist3).strip("['").strip("']")
                        q.put(output_image_frame)
                    else:
                        # 如果图像中没有任何的bbox，则清空list
                        list_overlapping_blue_polygon.clear()
                        list_overlapping_yellow_polygon.clear()

                    show_draw = '实时显示: '
                    # show_draw = '实时显示: ' + blue_info1 + '\n' + blue_info2 + '\n' + blue_info3 + '\n' + blue_info4 + '\n' + yellow_info1 + '\n' + \
                    #           yellow_info2 + '\n' + yellow_info2 + '\n' + yellow_info3 + '\n' + yellow_info4 \
                    # + '\n' + "蓝色list:" + str_blue + '\n' + "黄色list:" + str_yellow
                    # output_image_frame = cv2.putText(img=output_image_frame, text=show_draw,
                    #                                  org=draw_text_postion,
                    #                                  fontFace=font_draw_number,
                    #                                  fontScale=0.75, color=(0, 0, 255), thickness=2)
                    # output_image_frame = cv2AddChineseText(output_image_frame, show_draw, show_text_postion, (255, 0, 0),25)

                else:
                    timer.toc()
                    online_im = output_image_frame
                if args.save_result:
                    vid_writer.write(output_image_frame)

                if exit_event.is_set():
                    break
            else:
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    list_bboxs = online_targets
                    if len(list_bboxs) > 0:
                        for t in list_bboxs:
                            tlwh = t.tlwh
                            tid = t.track_id
                            # start-------撞线检测--------#
                            x1, y1, x2, y2, track_id = tlwh[0], tlwh[1], tlwh[2], tlwh[3], tid
                            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                            # 检测点为检测框中点
                            y1_offset = int(y1 + (y2 * 0.5))
                            x1_offset = int(x1 + (x2 * 0.5))
                            # 撞线的点
                            y = y1_offset
                            x = x1_offset
                            # 然后每检测出一个预测框，就将中心点加入队列
                            center = (x, y)
                            if track_id in pts:
                                pts[track_id].append(center)
                            else:
                                pts[track_id] = []
                                pts[track_id].append(center)

                            thickness = 2
                            cv2.circle(output_image_frame, (center), 1, [255, 255, 255], thickness)
                            # cv2.imshow('qe', output_image_frame)
                            for j in range(1, len(pts[track_id])):
                                if pts[track_id][j - 1] is None or pts[track_id][j] is None:
                                    continue
                                cv2.line(output_image_frame, (pts[track_id][j - 1]), (pts[track_id][j]),
                                         [255, 255, 255],
                                         thickness)
                            try:
                                if polygon_mask_blue_and_yellow[y, x] == 1:
                                    print("撞 蓝polygon", track_id)
                                    blue_info1 = '撞 蓝polygon' + str(track_id)
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_blue_polygon:
                                        print("蓝polygon list 无此 track_id")
                                        blue_info2 = "蓝polygon list 无此 track_id"
                                        list_overlapping_blue_polygon.append(track_id)
                                    # 判断 黄polygon list里是否有此 track_id
                                    # 有此track_id，则认为是 UP (上行)方向
                                    if track_id in list_overlapping_yellow_polygon:
                                        print("上行")
                                        blue_info3 = "上行"
                                        # 上行+1
                                        up_count += 1
                                        print('up count:', up_count, ', up id:', list_overlapping_yellow_polygon)
                                        blue_info4 = 'up count:' + str(up_count) + ', up id:' + str(
                                            list_overlapping_yellow_polygon)
                                        # 删除 黄polygon list 中的此id
                                        list_overlapping_yellow_polygon.remove(track_id)

                                elif polygon_mask_blue_and_yellow[y, x] == 2:
                                    print("撞 黄polygon", track_id)
                                    yellow_info1 = '撞 黄polygon' + str(track_id)
                                    # 如果撞 黄polygon
                                    if track_id not in list_overlapping_yellow_polygon:
                                        print("黄polygon list 无此 track_id")
                                        yellow_info2 = "黄polygon list 无此 track_id"
                                        list_overlapping_yellow_polygon.append(track_id)
                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 DOWN（下行）方向
                                    if track_id in list_overlapping_blue_polygon:
                                        print("下行")
                                        yellow_info3 = "下行"
                                        # 下行+1
                                        down_count += 1
                                        print('down count:', down_count, ', down id:', list_overlapping_blue_polygon)
                                        yellow_info4 = 'down count:' + str(down_count) + ', down id:' + str(
                                            list_overlapping_blue_polygon)
                                        # 删除 蓝polygon list 中的此id
                                        list_overlapping_blue_polygon.remove(track_id)
                            except IndexError:
                                print("数组越界异常")
                            # end ------撞线检测--------#

                            # start----------------------清除无用id（每隔一帧）----------------------
                            if frame_id % 300 == 0:  # 300帧还检测不到删除
                                print('清除无用id')
                                list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                                for id1 in list_overlapping_all:
                                    is_found = False
                                    for t in list_bboxs:
                                        bbox_id = t.track_id
                                        if bbox_id == id1:
                                            is_found = True
                                    if not is_found:
                                        # 如果没找到，删除id
                                        if id1 in list_overlapping_yellow_polygon:
                                            list_overlapping_yellow_polygon.remove(id1)

                                        if id1 in list_overlapping_blue_polygon:
                                            list_overlapping_blue_polygon.remove(id1)
                                list_overlapping_all.clear()
                                # 清空list
                                list_bboxs.clear()
                            # end----------------------清除无用id（每隔一帧）----------------------
                            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)
                                results.append(
                                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )

                        timer.toc()
                        output_image_frame = plot_tracking(
                            output_image_frame, online_tlwhs, online_ids, frame_id=frame_id + 1,
                            fps=1. / timer.average_time
                        )
                        text_draw = 'DOWN: ' + str(down_count) + \
                                    ' , UP: ' + str(up_count)
                        # 绘制DOWN UP
                        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                                         org=draw_text_postion,
                                                         fontFace=font_draw_number,
                                                         fontScale=0.75, color=(0, 0, 255), thickness=2)
                        # start----------------------将蓝色红色区域的id在界面以文字显示出来----------------------
                        # 转换成字符串
                        if len(list_overlapping_blue_polygon) == 0:
                            str_blue = ''
                        if len(list_overlapping_blue_polygon) == 1:
                            str_blue = str(list_overlapping_blue_polygon)
                        else:
                            mylist0 = []
                            for i in list_overlapping_blue_polygon:
                                mylist0.append(str(i))
                            # 将list[list]转换成list
                            mylist1 = [i for i in mylist0]
                            str_blue = ";".join(mylist1).strip("['").strip("']")
                        if len(list_overlapping_yellow_polygon) == 0:
                            str_yellow = ''
                        if len(list_overlapping_yellow_polygon) == 1:
                            str_blue = str(list_overlapping_yellow_polygon)
                        else:
                            mylist2 = []
                            for i in list_overlapping_yellow_polygon:
                                mylist2.append(str(i))
                            mylist3 = [''.join(i) for i in mylist2]  # 将list[list]转换成list
                            str_yellow = ";".join(mylist3).strip("['").strip("']")
                        # 将处理后的图像入栈
                        q.put(output_image_frame)
                    else:
                        # 如果图像中没有任何的bbox，则清空list
                        list_overlapping_blue_polygon.clear()
                        list_overlapping_yellow_polygon.clear()

                    show_draw = '实时显示: '
                    # show_draw = '实时显示: ' + blue_info1 + '\n' + blue_info2 + '\n' + blue_info3 + '\n' + blue_info4 + '\n' + yellow_info1 + '\n' + \
                    #           yellow_info2 + '\n' + yellow_info2 + '\n' + yellow_info3 + '\n' + yellow_info4 \
                    # + '\n' + "蓝色list:" + str_blue + '\n' + "黄色list:" + str_yellow
                    # output_image_frame = cv2.putText(img=output_image_frame, text=show_draw,
                    #                                  org=draw_text_postion,
                    #                                  fontFace=font_draw_number,
                    #                                  fontScale=0.75, color=(0, 0, 255), thickness=2)
                    # output_image_frame = cv2AddChineseText(output_image_frame, show_draw, show_text_postion, (255, 0, 0),25)

                else:
                    timer.toc()
                    online_im = output_image_frame
                if args.save_result:
                    vid_writer.write(output_image_frame)

                if exit_event.is_set():
                    break
        else:
            break
            capture.release()
            video.release()
            cv2.destroyAllWindows()
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
        # exp.test_size = (960, 540)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        p1 = threading.Thread(target=imageflow_demo, args=[predictor, vis_folder, current_time, args])
        p2 = threading.Thread(target=Display)
        p3 = threading.Thread(target=wait_for_key)
        p1.start()
        p2.start()
        p3.start()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
