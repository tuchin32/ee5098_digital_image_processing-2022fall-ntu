import os
import cv2
import argparse
import numpy as np
import globals
import shutil

def show_xy(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN and globals.finish == True :
        globals.initialize()
        globals.dots = []
        globals.dots.append([x,y])
        #print(globals.dots)
        img_c = cv2.imread('./frames/apple_cup/refocus_frames/frame0.jpg')
        if globals.dots[0][0] > img_c.shape[0]:
            globals.dots[0][0] = x-960           
        #print(globals.dots)     
        exec(open("cutting.py").read())
        exec(open("detect_blurry.py").read()) 
        if globals.play == False :
            show_frames(frame_path)
        elif globals.play == True :
            from_last_show_frames(frame_path)

        #import cutting as cut      
        #find_mask(dots)

def show_frames(frame_path):
    """
    frame_path: path to frames
    """
    globals.finish = False
    ori_list = os.listdir(frame_path + '/ori_frames')
    ori_list.sort(key=lambda x: int(x[5:-4]))
    #print(ori_list)
    #reverse_ori_list = ori_list.reverse()
    #print(reverse_ori_list)

    print(f'Found {len(ori_list)} frames')
    refocus_list = os.listdir(frame_path + '/refocus_frames')
    refocus_list.sort(key=lambda x: int(x[5:-4]))
    print(f'Found {len(refocus_list)} frames')

    for (ori, refocus) in zip(ori_list, refocus_list):
        img_o = cv2.imread(os.path.join(frame_path, 'ori_frames', ori))
        img_r = cv2.imread(os.path.join(frame_path, 'refocus_frames', refocus))
        img = np.concatenate((img_o, img_r), axis=1)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', img.shape[1], img.shape[0])
        cv2.imshow('frame', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    globals.finish = True

def from_last_show_frames(frame_path):
    """
    frame_path: path to frames
    """
    globals.finish = False
    ori_list = os.listdir(frame_path + '/reverse_ori_frames')
    ori_list.sort(key=lambda x: int(x[5:-4]),reverse = True)
    #print(ori_list)
    #print(reverse_ori_list)
    print(f'Found {len(ori_list)} frames')

    refocus_list = os.listdir(frame_path + '/reverse_refocus_frames')
    refocus_list.sort(key=lambda x: int(x[5:-4]),reverse = True)
    print(f'Found {len(refocus_list)} frames')

    for (ori, refocus) in zip(ori_list, refocus_list):
        img_o = cv2.imread(os.path.join(frame_path, 'reverse_ori_frames', ori))
        img_r = cv2.imread(os.path.join(frame_path, 'reverse_refocus_frames', refocus))
        img = np.concatenate((img_o, img_r), axis=1)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', img.shape[1], img.shape[0])
        cv2.imshow('frame', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    globals.finish = True    

if __name__ == '__main__':
    globals.play = True
    globals.target = True
    finish = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path', '-fp', type=str, default='./frames/apple_cup', help='path to frames')
    args = parser.parse_args()  
    frame_path = args.frame_path

    # defining source and destination
    # paths
    src = './frames/apple_cup/ori_frames'
    trg = './frames/apple_cup/reverse_ori_frames'
    src1 = './frames/apple_cup/refocus_frames'
    trg1 = './frames/apple_cup/reverse_refocus_frames'
    files=os.listdir(src)
    files1=os.listdir(src1)
    # iterating over all the files in
    # the source directory
    for fname in files:
        # copying the files to the
        # destination directory
        shutil.copy2(os.path.join(src,fname), trg)

    for fname in files1:
        # copying the files to the
        # destination directory
        shutil.copy2(os.path.join(src1,fname), trg1)

    show_frames(frame_path)
    cv2.setMouseCallback('frame', show_xy)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()