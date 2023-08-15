import cv2
import numpy as np
import globals

if __name__ == '__main__':
  globals.initialize()
  globals.area = 0
  net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb", "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
  classesFile = "coco.names"
  classNames = open(classesFile).read().strip().split('\n')
  #print(classNames)

  img = cv2.imread('test.jpg')
  height, width, _ = img.shape

  blank_mask = np.zeros((height, width, 3), np.uint8)
  blank_mask[:] = (0,0,0)
  target_mask = np.zeros((height, width, 3), np.uint8)
  target_mask[:] = (0,0,0)

  blob = cv2.dnn.blobFromImage(img, swapRB=True)

  net.setInput(blob)
  boxes, masks = net.forward(["detection_out_final", "detection_masks"])
  detection_count = boxes.shape[2]

  #print(len(detection_count))

  count=0
  for i in range(detection_count):
  
    # Extract information from detection
    box = boxes[0, 0, i]
    class_id = int(box[1])
    score = box[2]
    # print(class_id, score)
    if score < 0.6:
      continue

    # print(class_id)
    class_name = (classNames[class_id])
    # print(class_name, score)
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    roi = blank_mask[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape

    targetroi = target_mask[y: y2, x: x2]
    targetroi_height, targetroi_width, _ = targetroi.shape

    #  Get the mask
    mask = masks[i, int(class_id)] 
    targetmask = masks[i, int(class_id)] 
    mask = cv2.resize(mask, (roi_width, roi_height))
    targetmask = cv2.resize(targetmask, (targetroi_width, targetroi_height))

    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    _, targetmask = cv2.threshold(targetmask, 0.5, 255, cv2.THRESH_BINARY)
    #cv2.imshow("mask"+str(count), mask)
    #cv2.imshow("targetmask"+str(count), targetmask)
    count +=1


  # Find contours of the mask
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = np.random.randint(0, 255, 3, dtype='uint8')
    color = [int(c) for c in color]

    contours1, _ = cv2.findContours(np.array(targetmask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # fill some color in segmented area
    for cnt in contours:
      cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
  
    for cnt in contours1:
      cv2.fillPoly(targetroi, [cnt], (255, 255, 255))

    #cv2.imshow("roi image", blank_mask)
    #cv2.imshow("target image", target_mask)
    if count >= 1:
      save = cv2.imwrite('./frames/apple_cup/mask/mask{}.jpg'.format(count),target_mask)

    target_mask[:] = (0,0,0)
    #cv2.waitKey(0)
    
    # Draw bounding box
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    cv2.putText(img, class_name + " " + str(score), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

  compare_area = 0
  area_mask = (mask.shape[0]*mask.shape[1])/2
  if globals.target == True:
    for i in contours:
      compare_area += cv2.contourArea(i)
  elif globals.target == False:
    for i in contours:
      compare_area += cv2.contourArea(i)
      compare_area = area_mask*2 - compare_area

  globals.area = compare_area - area_mask
  #print('area', globals.area)

  #reverse colour
  _, reverse_colour = cv2.threshold(blank_mask, 0.5, 255, cv2.THRESH_BINARY)
  reverse_colour = cv2.bitwise_not(reverse_colour)
  reverse_colour_save = cv2.imwrite('./frames/apple_cup/mask/reverse_colour.jpg', reverse_colour)

  #cv2.imshow('reverse colour', reverse_colour)
  #cv2.imshow("Black image", blank_mask)
  #cv2.imshow("Mask image", img)
  #cv2.waitKey(0)

  # alpha is the transparency of the first picture
  alpha = 1
  # beta is the transparency of the second picture
  beta = 0.8
  mask_img = cv2.addWeighted(img, alpha, blank_mask, beta, 0)
  cv2.imshow("Final Output", mask_img)
  #key = cv2.waitKey(0)