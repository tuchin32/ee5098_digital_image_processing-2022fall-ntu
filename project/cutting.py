import globals
#import detect_blurry as blur
from imutils import paths
import argparse
import cv2


if __name__ == '__main__':
    exec(open("image_segmentation_with_ML.py").read())
    number = 1
    globals.initialize()
    #print('cutting')
    #print(globals.dots)
    print(globals.dots[0][0],globals.dots[0][1])
    mask = cv2.imread('./frames/apple_cup/mask/mask{}.jpg'.format(number))
    reverse_mask = cv2.imread('./frames/apple_cup/mask/reverse_colour.jpg')
    #cv2.imshow("Mask", mask)
    #key = cv2.waitKey(0)

    if ((mask[globals.dots[0][1],globals.dots[0][0]] == (0,0,0)).all()):
        chosen_mask = mask
        globals.target = True
    #elif ((reverse_mask[globals.dots[0][0],globals.dots[0][1]] == (0,0,0)).all()):
    else:
        chosen_mask = reverse_mask
        globals.target = False   
        
    #cv2.imshow("chosen_mask", chosen_mask)     
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', '-i', type=str, default='./frames/apple_cup/refocus_frames', help='Path to the original frames')
    args = vars(ap.parse_args())
    count = 0

    for imagePath in paths.list_images(args["images"]):
        count +=1
        image=cv2.imread(imagePath)
        bitwise_result = cv2.bitwise_and(chosen_mask, image)
        if globals.target == True:
            bitwise_result_save = cv2.imwrite('./frames/apple_cup/bitwise/bitwise_result{}.jpg'.format(count), bitwise_result)
        if globals.target == False:
            bitwise_result_save = cv2.imwrite('./frames/apple_cup/reverse_bitwise/bitwise_result{}.jpg'.format(count), bitwise_result)
        #cv2.imshow("bitwise_Result"+str(count) , bitwise_result)
        #key = cv2.waitKey(0)
    #cv2.imshow("bitwise_Result"+str(count) , bitwise_result)