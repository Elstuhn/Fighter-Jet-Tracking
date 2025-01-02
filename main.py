import argparse
import os
import os.path as osp
import shutil
import cv2
import ffmpeg
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms
import time

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
    
    start_time = time.perf_counter()
    for frame in tqdm(os.listdir(args.tmpdir)):
        frame_path = osp.join(args.tmpdir, frame)
        bboxes, labels, scores = detect(model, frame_path)
        img = cv2.imread(frame_path)
        for bbox, label, score in zip(bboxes, labels, scores):
            x1, y1, x2, y2 = bbox
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

if __name__ == '__main__':
    main()