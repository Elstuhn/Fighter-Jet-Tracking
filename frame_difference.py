import argparse
import os
import os.path as osp
import shutil
import cv2
import ffmpeg
from tqdm import tqdm
import time

import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms
import numpy as np 

def parse_args():
    parser = argparse.ArgumentParser(description='Get frame difference')
    parser.add_argument(
        'video_path', help='video file')
    parser.add_argument(
        '--tmpdir', help='tmp dir for writing some results',
        default='./tmpdir')
    parser.add_argument(
        '--output-dir', help='output directory for the compiled videos',
        default='./frame-differences')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    shutil.rmtree(args.tmpdir, ignore_errors=True)
    os.makedirs(args.tmpdir, exist_ok=True)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'frame_diffs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
    
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ffmpeg.input(args.video_path).output(
        osp.join(args.tmpdir, 'frame_%d.jpg')
    ).run()

    frames = sorted(
        os.listdir(args.tmpdir),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    img1 = cv2.imread(osp.join(args.tmpdir, frames[0]))
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = None
    kernel = np.array((9,9), dtype=np.uint8)

    start_time = time.perf_counter()
    for idx, frame in tqdm(enumerate(frames[1:])):
        img2 = cv2.imread(osp.join(args.tmpdir, frame))
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
        frame_diff = cv2.subtract(img2, img1)
        frame_diff_scaled = frame_diff * 10

        blurred_diff = cv2.medianBlur(frame_diff, 3)
    
        mask = cv2.adaptiveThreshold(blurred_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 11, 3)
        
        mask = cv2.medianBlur(mask, 3)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1) # fill blanks

        cv2.imwrite(osp.join(args.output_dir, 'frame_diffs', f'frame_{idx+1}.jpg'), frame_diff_scaled)
        cv2.imwrite(osp.join(args.output_dir, 'masks', f'frame_{idx+1}.jpg'), mask)
        img1 = img2.copy()


    end_time = time.perf_counter()

    input_pattern = osp.join(args.output_dir, 'frame_diffs', 'frame_%d.jpg')
    try:
        (
            ffmpeg
            .input(input_pattern, framerate=fps)
            .output(osp.join(args.output_dir, 'frame_diff.mp4'), vcodec='libx264', pix_fmt='yuv420p')
            .run()
        )
    except ffmpeg.Error as e:
        print("An error occurred with ffmpeg:")
        print(e.stderr.decode())

    input_pattern = osp.join(args.output_dir, 'masks', 'frame_%d.jpg')
    try:
        (
            ffmpeg
            .input(input_pattern, framerate=fps)
            .output(osp.join(args.output_dir, 'mask.mp4'), vcodec='libx264', pix_fmt='yuv420p')
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

#2.173 seconds 473 images (frame differences both mask and frame diff)
#62.710 seconds 1036 images (main.py)
# 62.276 seconds 1036 images (kalman)

