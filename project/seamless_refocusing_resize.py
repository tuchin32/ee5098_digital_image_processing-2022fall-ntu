import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

class Refocus:
    def __init__(self, frame_path, write_path):
        self.frame_paths = self.get_frame_paths(frame_path)
        self.write_path = write_path

         # Create folder to save frames
        if not os.path.exists(self.write_path):
            os.makedirs(self.write_path)

    def get_frame_paths(self, frame_path):
        frame_paths = os.listdir(frame_path)
        frame_paths.sort(key=lambda x: int(x[5:-4]))
        frame_paths = [os.path.join(frame_path, frame) for frame in frame_paths]
        print(f'Found {len(frame_paths)} frames')
        return frame_paths

    def process_frames(self):
        img_0 = cv2.imread(self.frame_paths[0])
        cv2.imwrite(os.path.join(self.write_path, f'frame0.jpg'), img_0)

        img_last = img_0.copy()
        ratios = []
        for idx, path in enumerate(self.frame_paths[1:]):
            print(f'Processing frame {idx + 1} / {len(self.frame_paths[1:])}')

            # 1. Capture new frame
            img_curr = cv2.imread(path)

            # 2. Resize and crop the image to get minimum error from last frame
            # img_last = img_0.copy()
            img_curr_new, ratio = self.resize_crop(img_last, img_curr)

            # 3. Save the new frame
            cv2.imwrite(os.path.join(self.write_path, f'frame{idx + 1}.jpg'), img_curr_new)

            # 4. Update the last frame
            img_last = img_curr_new.copy()
            ratios.append(ratio)

        plt.plot(ratios)
        plt.show()

    def resize_crop(self, img_last, img_curr):
        height, width = img_last.shape[:2]
        gray_last = cv2.cvtColor(img_last, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
        edges_last = cv2.Canny(gray_last, 100, 200)
        edges_curr = cv2.Canny(gray_curr, 100, 200)

        # Find the best resize ratio
        resize_ratio = np.arange(1.001, 1.05, 0.0005)    # Could be changed
        errors = []

        for ratio in resize_ratio:
            # Resize the image 
            edges_curr_new = cv2.resize(edges_curr, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
           
            # Crop the image
            center = (edges_curr_new.shape[0] // 2, edges_curr_new.shape[1] // 2)
            edges_curr_new = edges_curr_new[center[0] - height // 2: center[0] + height // 2, center[1] - width // 2: center[1] + width // 2]
            
            # Calculate the error
            error = np.linalg.norm(edges_last - edges_curr_new)
            errors.append(error)

        # Find the minimum error
        min_idx = np.argmin(errors)
        best_ratio = resize_ratio[min_idx]
        print(f'Best resize ratio: {best_ratio}')

        # Resize and crop the image
        img_curr_new = cv2.resize(img_curr, None, fx=best_ratio, fy=best_ratio, interpolation=cv2.INTER_CUBIC)
        center = (img_curr_new.shape[0] // 2, img_curr_new.shape[1] // 2)
        img_curr_new = img_curr_new[center[0] - height // 2: center[0] + height // 2, center[1] - width // 2: center[1] + width // 2]
        
        return img_curr_new, best_ratio
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path', '-fp', type=str, default='./frames/apple_cup', help='Path to the original frames')
    args = parser.parse_args()

    frame_path = f'{args.frame_path}/ori_frames'
    write_path = f'{args.frame_path}/refocus_frames'
    write_path = f'{args.frame_path}/reverse_refocus_frames'
    refocus = Refocus(frame_path, write_path)
    refocus.process_frames()
    cv2.destroyAllWindows()