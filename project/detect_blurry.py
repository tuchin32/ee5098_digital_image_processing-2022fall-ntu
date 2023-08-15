from imutils import paths
import argparse
import cv2
import globals

def variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == '__main__':
    target_true_array = []
    target_false_array = []
    count_true = 0
    count_false = 0
    globals.initialize() 
    ap = argparse.ArgumentParser()
    ap.add_argument('--ori_images', '-o', type=str, default='./frames/apple_cup/refocus_frames', help='Path to the original frames')
    ap.add_argument('--images', '-i', type=str, default='./frames/apple_cup/bitwise', help='Path to the original frames')
    ap.add_argument('--reverse_images', '-r', type=str, default='./frames/apple_cup/reverse_bitwise', help='Path to the original frames')
    ap.add_argument("-t", "--threshold", type=float, default=200.0,help="200")
    args = vars(ap.parse_args())

    if globals.target == True :       
        for imagePath in paths.list_images(args["images"]):            
            image=cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            text = "Not Blurry"
            #print(fm)
            target_true_array.append(fm)
            if fm < args["threshold"]:               
                text = "Blurry"
            cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 3)
            #cv2.imshow("BG blocked Image", image)
            count_true +=1
            #key = cv2.waitKey(0)

    elif globals.target == False:        
        for imagePath in paths.list_images(args["reverse_images"]):
            reverse_image=cv2.imread(imagePath)
            reverse_gray = cv2.cvtColor(reverse_image, cv2.COLOR_BGR2GRAY)
            reverse_fm = variance_of_laplacian(reverse_gray)
            text = "Not Blurry"
            #print(target_false_array)
            target_false_array.append(reverse_fm)

            if reverse_fm > args["threshold"]:                
                text = "Blurry"
            cv2.putText(reverse_image, "{}: {:.2f}".format(text, reverse_fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 3)          
            #cv2.imshow("cup blocked Image", reverse_image)
            count_false +=1
            #key = cv2.waitKey(0)
     
    #print(target_true_array) 
    if globals.area < 0 and  target_true_array[0] < target_true_array[-1] :
        target_true_array = []
        globals.play = False #From first to last
        print('Play False' )

    if globals.area > 0 and  target_false_array[0] < target_false_array[-1] :
        target_false_array = []
        globals.play = True #From last to first
        print('Play True' )

    
    