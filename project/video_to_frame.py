import os
import cv2
import argparse

def video_to_frame(video_path, frame_path, frame_rate=1):
    """
    video_path: path to video
    frame_path: path to save frames
    frame_rate: frame rate
    """
    # Create folder to save frames
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    # Read video
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_rate == 0:
            frame_idx = int(frame_count // frame_rate)
            cv2.imwrite(os.path.join(frame_path, 'frame%d.jpg' % frame_idx), frame)
            cv2.imwrite(os.path.join(reverse_frame_path, 'frame%d.jpg' % frame_idx), frame)
        frame_count += 1
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', type=str, default='./video/apple_cup.mp4', help='video path')
    parser.add_argument('--frame_rate', '-fr', type=int, default=1, help='frame rate')
    args = parser.parse_args()
    
    video_path = args.video_path
    frame_path = f'./frames/{args.video_path[8:-4]}/ori_frames'
    reverse_frame_path = f'./frames/{args.video_path[8:-4]}/reverse_ori_frames'
    video_to_frame(video_path, frame_path, args.frame_rate)