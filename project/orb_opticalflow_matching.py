import os
import cv2
import argparse
import numpy as np

class Refocus:
    def __init__(self, frame_path, write_path):
        self.frame_paths = self.get_frame_paths(frame_path)
        self.write_path = write_path

    def get_frame_paths(self, frame_path):
        frame_paths = os.listdir(frame_path)
        frame_paths.sort(key=lambda x: int(x[5:-4]))
        frame_paths = [os.path.join(frame_path, frame) for frame in frame_paths]
        print(f'Found {len(frame_paths)} frames')
        return frame_paths

    def process_frames(self):
        img_0 = cv2.imread(self.frame_paths[0])
        cv2.imwrite(os.path.join(self.write_path, f'frame0.jpg'), img_0)

        img_last = img_0
        for idx, path in enumerate(self.frame_paths[1:]):
            print(f'Processing frame {idx + 1} / {len(self.frame_paths[1:])}')

            # 1. Capture new frame
            img_curr = cv2.imread(path)

            # 2. Extract and match features between the two images
            # img_last = img_0.copy()
            pts_0, pts_curr = self.orb_bfmatching(img_last, img_curr)
            # pts_0, pts_curr = self.optical_flow_matching(img_last, img_curr)

            # 3. Find homography matrix
            H, _ = cv2.findHomography(pts_curr, pts_0, cv2.RANSAC, 1, maxIters=5000, confidence=0.99)

            # 4. Warp the image
            img_curr_new = cv2.warpPerspective(img_curr, H, (img_last.shape[1], img_last.shape[0]))

            # 5. Save the new frame
            cv2.imwrite(os.path.join(self.write_path, f'frame{idx + 1}.jpg'), img_curr_new)

            # 6. Update the last frame
            img_last = img_curr_new.copy()

    def orb_bfmatching(self, img1, img2):
        '''
        img1: queryImage
        img2: trainImage
        '''
        
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # # Draw first 10 matches.
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1200, 600)
        # cv2.imshow('image', img3)
        # cv2.waitKey(1000)

        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        return pts1, pts2

    def optical_flow_matching(self, img1, img2):
        # Parameters for lucas kanade optical flow
        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        
        # calculate optical flow
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        pts1 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)
        pts2, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, pts1, None, **lk_params)

        # Select good points
        if pts2 is not None:
            good_new = pts2[st == 1]
            good_old = pts1[st == 1]
        print(good_new.shape, good_old.shape)

        # draw the tracks
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #     mask = cv2.line(np.zeros_like(img1), (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)
        #     frame = cv2.circle(img2, (int(a), int(b)), 5, (255, 0, 0), -1)
        # img3 = cv2.add(frame, mask)
        # cv2.imshow('frame', img3)
        # cv2.waitKey(1000)

        return good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path', type=str, default='./frames/airpods', help='Path to the original frames')
    args = parser.parse_args()

    frame_path = f'{args.frame_path}/ori_frames'
    write_path = f'{args.frame_path}/refocus_frames'
    refocus = Refocus(frame_path, write_path)
    refocus = Refocus(frame_path, write_path)
    refocus.process_frames()
    cv2.destroyAllWindows()