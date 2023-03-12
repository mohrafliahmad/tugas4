import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('hewan.jpeg', cv2.IMREAD_GRAYSCALE)

blur_img = cv2.GaussianBlur(img, (5,5), 0)
unsharp_img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
M, N = img.shape
center = (M//2, N//2)
mask = np.zeros((M,N), dtype=np.uint8)
mask[center[0]-30:center[0]+30, center[1]-30:center[1]+30] = 1
fshift_laplacian = fshift * mask
f_ishift_laplacian = np.fft.ifftshift(fshift_laplacian)
img_laplacian = np.fft.ifft2(f_ishift_laplacian).real

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
M, N = img.shape
center = (M//2, N//2)
dist = np.zeros((M,N), dtype=np.float32)
for i in range(M):
    for j in range(N):
        dist[i,j] = np.sqrt((i-center[0])**2 + (j-center[1])**2)
mask_selective = np.zeros((M,N), dtype=np.uint8)
mask_selective[(dist > 50) & (dist < 120)] = 1
fshift_selective = fshift * mask_selective
f_ishift_selective = np.fft.ifftshift(fshift_selective)
img_selective = np.fft.ifft2(f_ishift_selective).real

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(unsharp_img, cmap='gray')
plt.title('Unsharp Masking Filtered Image')

plt.subplot(2, 2, 3)
plt.imshow(img_laplacian, cmap='gray')
plt.title('Laplacian Filtered Image in Frequency Domain')

plt.subplot(2, 2, 4)
plt.imshow(img_selective, cmap='gray')
plt.title('Selective Filtered Image in Frequency Domain')

plt.show()