import numpy as np
from matplotlib import pyplot as plt
import cv2
PandaNoise1 = cv2.imread('PandaNoise.bmp').astype(float)
PandaNoise = cv2.imread('PandaNoise.bmp',0)
PandaClean = cv2.imread('PandaOriginal.bmp',0)
imglist = []
masksize = 50
subpltnumb = 241
kernel = np.ones((5, 5), np.float32) / 25


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err




dft = cv2.dft(np.float32(PandaNoise),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.imshow(magnitude_spectrum)
plt.show()
rows , cols = PandaNoise.shape
crow, ccol = rows/2, cols/2

plt.imshow(PandaNoise, cmap="gray")
plt.show()

for i in range (4):
    # create a mask first , center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[round(crow - masksize):round(crow + masksize), round(ccol - masksize):round(ccol + masksize)] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    newmagspect = 20 * np.log(cv2.magnitude(f_ishift[:, :, 0], f_ishift[:, :, 1]))
    plt.subplot(subpltnumb), plt.imshow(newmagspect)
    subpltnumb +=1
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    imglist.append(img_back)
    plt.subplot(subpltnumb), plt.imshow(img_back, cmap='gray');
    plt.title('Masked image'), plt.xticks([]), plt.yticks([])


    subpltnumb +=1
    masksize += 10

plt.show()

plt.imshow(imglist[3], cmap="gray")
plt.show()

dst = cv2.filter2D(imglist[3],-1,kernel)
blurimg = cv2.blur(imglist[3], (5, 5))
gausblur = cv2.GaussianBlur(imglist[3], (5, 5), 0)
medblur = cv2.medianBlur(imglist[3], 5)
bifilblur = cv2.bilateralFilter(imglist[3], 9, 75, 75)
plt.subplot(231), plt.imshow(dst, cmap="gray"), plt.title('Averaging')
plt.subplot(232), plt.imshow(blurimg, cmap="gray"), plt.title('Blur')
plt.subplot(233), plt.imshow(gausblur, cmap="gray"), plt.title('Guassianblur')
plt.subplot(234), plt.imshow(medblur, cmap="gray"), plt.title('Median blur')
plt.subplot(235), plt.imshow(bifilblur, cmap="gray"), plt.title('Bilateral Filtering blur')
plt.show()

print(mse(PandaClean, medblur))
