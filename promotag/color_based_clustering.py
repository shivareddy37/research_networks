import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

# threshold = 70

# filenames = glob.glob('/home/shiva/products/*.jpg')
# count = 1
# for fn in filenames[1:10]:
#     img = cv2.imread(fn)
#     print(fn)
#     blue = img[:, :, 0]
#     # green = img[]

   
#     blue_percentage = (100.*np.sum(blue >= threshold)/blue.size)
#     print(blue_percentage)
    

  

img1 = cv2.imread('/home/shiva/products/1.jpg',1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
cv2.imshow("img1A", img1)
# cv2.imshow('img1B', img1[:,:,2])

img2 = cv2.imread('/home/shiva/products/6.jpg',1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
cv2.imshow("img2A", img2)
# cv2.imshow('img2B', img2[:,:,2])

img3 = cv2.imread('/home/shiva/products/12.jpg',1)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2LAB)
cv2.imshow("img3A", img3)
# cv2.imshow('img3B', img3[:,:,2])

img4 = cv2.imread('/home/shiva/products/1179.jpg',1)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2LAB)
cv2.imshow("img4A", img4)
# cv2.imshow('img4B', img4[:,:,2])

cv2.waitKey(0)



