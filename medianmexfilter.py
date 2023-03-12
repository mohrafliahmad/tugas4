import cv2
import numpy as np

img = cv2.imread('lena.jpg')
median = cv2.medianBlur(img, 5)

kernel = np.ones((3,3),np.uint8)
min_filter = cv2.erode(img, kernel)
max_filter = cv2.dilate(img, kernel)

cv2.imshow('Input Image', img)
cv2.imshow('Median Filter', median)
cv2.imshow('Min Filter', min_filter)
cv2.imshow('Max Filter', max_filter)
cv2.waitKey(0)
cv2.destroyAllWindows() 