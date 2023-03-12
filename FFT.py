import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load  the image
img = cv2.imread('tau.jpeg', 0)

# Perform FFT on the image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
# Define  the filter parameters
d0 = 50   # Cut-off frequency
n = 2     # Order of Butterworth filter

# Ideal Lowpass Filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), d0, 255, -1)
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * mask
img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

# Butterworth Lowpass Filter
butterworth_filter = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i-crow)**2 + (j-ccol)**2)
        butterworth_filter[i,j] = 1 / (1 + (d/d0)**(2*n))
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * butterworth_filter
img_filtered_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
# Define  the filter parameters
d0 = 50   # Cut-off frequency
n = 2     # Order of Butterworth filter

# Ideal Lowpass Filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), d0, 220, -1)
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * mask
img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

# Butterworth Lowpass Filter
butterworth_filter = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i-crow)**2 + (j-ccol)**2)
        butterworth_filter[i,j] = 1 / (1 + (d/d0)**(2*n))
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * butterworth_filter
img_filtered_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

# Butterworth  highpass filter
def butterworth_highpass_filter(img, cutoff, order):
    M, N = img.shape
    center = (M//2, N//2)
    u, v = np.meshgrid(np.arange(N)-center[1], np.arange(M)-center[0])
    D = np.sqrt(u**2 + v**2)
    H = 1 / (1 + (cutoff / D)**(2*order))
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    Gshift = H * Fshift
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G).real
    return g

# Gaussian highpass filter
def gaussian_highpass_filter(img, cutoff):
    M, N = img.shape
    center = (M//2, N//2)
    u, v = np.meshgrid(np.arange(N)-center[1], np.arange(M)-center[0])
    D = np.sqrt(u**2 + v**2)
    H = 1 - np.exp(-0.5*(D/cutoff)**2)
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    Gshift = H * Fshift
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G).real
    return g

# Apply Butterworth highpass filter
cutoff = 50
order = 4
img_butterworth = butterworth_highpass_filter(img, cutoff, order)

# Apply Gaussian highpass filter
cutoff = 50
img_gaussian = gaussian_highpass_filter(img, cutoff)

# Display the original and filtered images
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(img_butterworth, cmap='gray')
plt.title('Butterworth Highpass Filtered Image')

plt.subplot(1, 3, 3)
plt.imshow(img_gaussian, cmap='gray')
plt.title('Gaussian Highpass Filtered Image')

plt.show()
# Display the results
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_filtered, cmap='gray')
plt.title('Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_filtered_butterworth, cmap='gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()

# Display the results
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_filtered, cmap='gray')
plt.title('Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_filtered_butterworth, cmap='gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
# Perform inverse FFT to get the processed image
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Display the results
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()