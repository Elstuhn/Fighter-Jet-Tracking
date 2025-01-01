import argparse
import os
import os.path as osp
import shutil
import cv2
import ffmpeg
from tqdm import tqdm
import time

from mmdet.apis import inference_detector, init_detector
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms
from filterpy.kalman import KalmanFilter
import numpy as np 

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint pth file')
    parser.add_argument(
        'video_path', help='video file')
    parser.add_argument(
        '--tmpdir', help='tmp dir for writing some results',
        default='./tmpdir')
    parser.add_argument(
        '--output', help='output filepath for the compiled video',
        default='./output.mp4')
    args = parser.parse_args()

    return args

def detect(
        model:init_detector, 
        img:str, 
        conf_thres:float = 0.28,
        nms_thres: float = 0.5):
    result = inference_detector(model, img).pred_instances
    bboxes = result.bboxes
    labels = result.labels
    scores = result.scores
    idx = scores > conf_thres
    bboxes = bboxes[idx]
    labels = labels[idx]
    scores = scores[idx]
    idx = nms(bboxes, scores, nms_thres)
    bboxes = bboxes.tolist()
    labels = labels.tolist()
    scores = scores.tolist()
    bboxes = [bboxes[i] for i in idx]
    labels = [labels[i] for i in idx]
    scores = [scores[i] for i in idx]
    return bboxes, labels, scores

def compute_iou(box1, box2):
    """
    Compute the Intersection-over-Union (IoU) of two bounding boxes.

    Parameters:
        box1: List or tuple with coordinates [x1, y1, x2, y2] for the first box.
        box2: List or tuple with coordinates [x1, y1, x2, y2] for the second box.

    Returns:
        iou: IoU value between the two boxes (float).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou

class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        #state vector: (cx​,cy​,w,h,cx​'​,cy​'​,w') center coords, width height, velocity
        #measurement vector: (cx,cy,w,h)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0], 
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ]) # State transition matrix (state evolution matrix basically)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ]) # Observation matrix
        self.kf.R[2:, 2:] *= 10.0  # Measurement noise 
        self.kf.P[4:, 4:] *= 1000.0  # Initial velocity uncertainty
        self.kf.P *= 10.0  # Process noise, uncertainty in system dynamics
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1 
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(-1, 1)

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        self.kf.update(np.array([cx, cy, w, h])) # x=x′+K(z−Hx′) uses prediction and measurement to update state
        # z-Hx' residual, diff between predicted measurement and actual measurement
        #K=P′HT(HP′HT+R)−1 Kalman gain, how much to trust measurement vs prediction P large, trust measurement, R large, trust prediction
        

    def predict(self):
        self.kf.predict() #x'=Fx (predicts next state using state transition matrix)
        cx, cy, w, h = self.kf.x[:4]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        return [x1, y1, x2, y2]

def main():
    args = parse_args()
    shutil.rmtree(args.tmpdir, ignore_errors=True)
    os.makedirs(args.tmpdir, exist_ok=True)
    model = init_detector(args.config, args.checkpoint, device='cuda')
    
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ffmpeg.input(args.video_path).output(
        osp.join(args.tmpdir, 'frame_%d.jpg')
    ).run()

    trackers = []
    start_time = time.perf_counter()
    for frame in tqdm(os.listdir(args.tmpdir)):
        frame_path = osp.join(args.tmpdir, frame)
        bboxes, labels, scores = detect(model, frame_path)
        img = cv2.imread(frame_path)

        new_trackers = []
        for bbox in bboxes:
            matched = False
            for tracker in trackers:
                tracker_bbox = tracker.predict()
                iou = compute_iou(bbox, tracker_bbox)  
                if iou > 0.2:  # Threshold to match tracker with detection
                    tracker.update(bbox)
                    new_trackers.append(tracker)
                    matched = True
                    break
            if not matched:
                new_trackers.append(KalmanBoxTracker(bbox))

        trackers = new_trackers

        for tracker in trackers:
            x1, y1, x2, y2 = tracker.predict()
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite(frame_path, img)

    end_time = time.perf_counter()

    input_pattern = osp.join(args.tmpdir, 'frame_%d.jpg')
    try:
        (
            ffmpeg
            .input(input_pattern, framerate=fps)
            .output(args.output, vcodec='libx264', pix_fmt='yuv420p')
            .run()
        )
    except ffmpeg.Error as e:
        print("An error occurred with ffmpeg:")
        print(e.stderr.decode())

    print("Inference Done")
    print(f"Time taken: {end_time - start_time:.3f} seconds")

#16.3 fps
if __name__ == '__main__':
    main()