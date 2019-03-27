# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:54:52 2019

@author: william
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.transform import resize
from PIL import Image 
 
cap = cv2.VideoCapture(0)
ret=cap.set(3,120)  #width
ret=cap.set(4,90)   #height
while(1):
    # get a frame
    ret, img = cap.read()
    # show a frame
    cv2.imshow("capture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转为灰度图片
        tempimg = cv2.resize(img,(80,60),cv2.INTER_LINEAR)  #(width,height)
        #cv2.imwrite("C:/Users/william/Python Files/photo/face2.jpeg", tempimg)
        cropimg = tempimg[10:58, 16:64]
        cv2.imwrite(r"C:\Users\william\Python Files\Facial-Expression-Recognition.Pytorch-master\images\face1.jpeg", img)
        break
cap.release()
cv2.destroyAllWindows()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


raw = io.imread('images/face1.jpeg')
plt.subplot(2, 2, 1)
plt.imshow(raw)
gray = rgb2gray(raw)
plt.subplot(2, 2, 2)
plt.imshow(gray,cmap = plt.get_cmap('gray'))
gray1 = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
plt.subplot(2, 2, 3)
plt.imshow(gray1,cmap = plt.get_cmap('gray'))
img1 = gray1[:, :, np.newaxis]
img1 = np.concatenate((img1, img1, img1), axis=2)
img1 = Image.fromarray(img1)  #由array转成图像
plt.subplot(2, 2, 4)
plt.imshow(img1)