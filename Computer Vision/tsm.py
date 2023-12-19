import cv2 as cv
import numpy as np

"""
This is an easy way to implement temporal shift module using numpy
For Pytorch version, one can just use the similar functions to operate tensor
"""

def TSM(frames:np.ndarray, n_segment, n_fold):
    f, c, h, w = frames.shape
    segments, fold = f // n_segment, c // n_fold
    frames = frames[:segments*n_segment] # discard the left frames
    frames = frames.reshape(segments, n_segment, c, h, w)  # B, T, C, H, W
    out = np.zeros_like(frames)
    out[:, 1:, :fold] = frames[:, :-1, :fold] # Fold1 : 0 - t-1 -> 1 - t
    out[:, :-1, fold:2*fold] = frames[:, 1:, fold:2*fold] # Fold2 : 1 - t -> 0 - t-1
    out[:, :, 2*fold: ] = frames[:, :, 2 * fold: ] # Fold3 : no changed
    return out.reshape(-1, c, h, w)

def output_video(frames, fps = 30):
    output_file_name = './test.MP4'
    fourcc = cv.VideoWriter_fourcc('X','2','6','4')
    video_writer = cv.VideoWriter(output_file_name, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
    

video_path = "/Users/ziyuanwang/Documents/阿根廷中国行/PublishVideo.MP4"
cap = cv.VideoCapture(video_path)

frame_array = []
while True:
    ret, frame = cap.read()  
    if not ret:
        break
    frame_array.append(frame)

frame_array = np.array(frame_array).transpose(0, 3, 1, 2)
new_frame_array = TSM(frame_array, 10, 3).transpose(0, 2, 3, 1)
output_video(new_frame_array)