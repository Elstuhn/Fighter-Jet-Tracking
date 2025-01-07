# Visual Fighter-Jet Tracking
A small study to compare tracking methods for accurate and consistent fighter-jet tracking (and to learn how to implement kalman filters as well as cv2)

Training a robust object detector that generalizes well and implementing tracking methods such as kalman filtering makes it such that no VOD methods has to be used which may require more data, computation and hassle of setting up and training.

## Benchmark
| Mode  | Avg img process time | Performance |
| ------------- | ------------- | ------------- |
| Object Detection (RTMDet)  | 60.5ms  | NaN |
| ONNX on Python  | 60.1ms  | NaN |
| Frame Difference and Masking  | 4.59ms  | NaN |


## Methods
### Object Detection With Kalman Filter
The model used for object detection was RTMDet trained from scratch without any pretraining for around 200 epochs on fighter-jet images (still-images).

https://github.com/user-attachments/assets/f0b2c401-8985-4b6d-8b87-0ada3d9c44b7


 
### Frame Differencing
https://github.com/user-attachments/assets/52292b1d-7768-4496-9a12-4f2d3cc60070


### Masking
https://github.com/user-attachments/assets/c98b12df-7428-4674-9453-cafb18a511c9

## Improvements
1. Contour detection using masked frames
2. Template matching for noisy background
3. Class balancing for jet classification problem
