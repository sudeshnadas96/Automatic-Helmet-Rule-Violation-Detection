import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

import cv2
import numpy as np
from tensorflow.keras.models import load_model

helmet_model = load_model('helmet_detection.h5')

def helmetDetection(img):
    img = cv2.resize(img, (224, 224))
    img = np.array(img,dtype='float32')
    img = img.reshape(1, 224, 224, 3)
    img = img/255.0
    resp = int(helmet_model.predict(img)[0][0])              
    return resp

import sys
import time
import numpy as np
import cv2
from myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

PATH_TO_CKPT = 'head_detection.pb'
tDetector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT)
    
def headDetection(img):
    im_height, im_width, im_channel = img.shape
    boxes, scores, classes, num_detections = tDetector.run(img, im_height, im_width)
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)

    head_count = 0
    coords = []
    for score, box in zip(scores, boxes):
        if score > 0.15:
            head_count += 1
            left = int(box[1]*im_width)
            top = int(box[0]*im_height)
            right = int(box[3]*im_width)
            bottom = int(box[2]*im_height)
            coords.append([left, top, right, bottom])
    return head_count, coords

from openalpr_ocr import ocr

import cv2
from myUtils import *
from tracking.tracking import Tracking
from tracking.unit_object import UnitObject

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    LABELS = ['Auto', 'Bus', 'Car', 'Rikshaw', 'Truck', 'Two-wheeler']

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    tracker = Tracking()
    prev_frame_time = time.time()
    prev_positions = {}

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            height, width, _ = im0.shape
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            bboxes = []
            coordinates = []
        
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        if names[c] in LABELS:
                            color=colors(c, True)
                            x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                            if names[c] != 'Two-wheeler':
                                cv2.rectangle(im0, (x, y), (w, h), color, 1)
                                cv2.putText(im0, names[c], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color, 1)

                                c1, c2 = (x, y), (w, h)
                                bboxes.append([c, conf, names[c], c1, c2])
                            
                            if names[c] == 'Two-wheeler':
                                helmet_count = 0
                                y = int(y - ((h - y)/2))

                                c1, c2 = (x, y), (w, h)
                                bboxes.append([c, conf, names[c], c1, c2])
                                
                                head_count, coords = headDetection(im0)
                                
                                for row in coords:
                                    x1 = x - 150
                                    y1 = y - 150
                                    w1 = w + 150
                                    h1 = h + 150

                                    rmp = int(row[0] +((row[2] - row[0]) / 2))
                                    cmp = int(row[1] +((row[3] - row[1]) / 2))
                                     
                                    if rmp > x1 and rmp < w1 and cmp > y1 and cmp < h1:
                                        img = im0[row[1]:row[3] , row[0]:row[2]]
                                        c1 = helmetDetection(img)
                                        res = ['helmet','no-helmet'][c1]
                                        if res == 'helmet':
                                            helmet_count += 1
                                        cv2.rectangle(im0, (row[0], row[1]), (row[2], row[3]), color, 1)
                                        cv2.putText(im0, res, (row[0], row[1]-5), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color, 1)

                                        if row[1] > y:
                                            y = row[1]

                                cv2.rectangle(im0, (x, y-50), (w, h), color, 1)
                                cv2.putText(im0, names[c], (x, y-55), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color, 1)

                                print("Head count: ", head_count, ", helmet count: ", helmet_count)
                                if head_count != helmet_count:
                                    cv2.imwrite('numberplate.jpg', im0[y-50:h, x:w])
                                    number = ocr('numberplate.jpg')
                                    cv2.putText(im0, f"no helmet violation by: {number}", (x+5, h-5), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color, 1)
                                    
                                if head_count > 2:
                                    cv2.imwrite('numberplate.jpg', im0[y-50:h, x:w])
                                    number = ocr('numberplate.jpg')
                                    cv2.putText(im0, f"triple riding violation by: {number}", (x+5, h-25), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color, 1)
                                    
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            for box in bboxes:
                coordinates.append(UnitObject( [box[3][0], box[3][1], box[4][0], box[4][1] ], 1))
            
            tracker.update(coordinates)

            current_frame_time = time.time()
            elapsed_time = current_frame_time - prev_frame_time

            for j in range(len(tracker.tracker_list)):
                unit_object = tracker.tracker_list[j].unit_object
                tracking_id = tracker.tracker_list[j].tracking_id

                # Check if the object was tracked in the previous frame
                if tracking_id in prev_positions:
                    # Calculate distance traveled
                    prev_x, prev_y = prev_positions[tracking_id]
                    current_x, current_y = int(unit_object.box[0]), int(unit_object.box[1])
                    distance = ((current_x - prev_x)**2 + (current_y - prev_y)**2)**0.5

                    # Calculate speed (distance per second)
                    speed1 = distance / elapsed_time ##pixcel per second
                    speed2 = speed1 *0.1
                    speed = int(speed2 * 3600 / 60)

                    cv2.putText(im0, f"{speed} kmph", (current_x+5, current_y+20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color, 1)

                # Update the previous position for the next frame
                prev_positions[tracking_id] = (int(unit_object.box[0]), int(unit_object.box[1]))

            # Update the previous frame time for the next iteration
            prev_frame_time = current_frame_time

            # Stream results
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='cars.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
