import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy import fftpack

PandaNoise = cv2.imread('PandaNoise.bmp').astype(float)
PandaClean = cv2.imread('PandaOriginal.bmp')

im = plt.imread('PandaNoise.bmp').astype(float)
im_fft = fftpack.fft2(im)

def plot_spectrum(im_fft):

    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()


plt.figure()
plot_spectrum(im_fft)
plt.title('Fourier Transform')

keep_fraction = 0.15

# Call ff a copy of the original transform. Numpy arrays have a copy
# method for this purpose.
im_fft2 = im_fft.copy()

# Set r and c to be the number of rows and columns of the array.
r, c = im_fft2.shape

# Set to zero all rows with indices between r*keep_fraction and
# r*(1-keep_fraction):
im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

# Similarly with the columns:
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

plt.figure()
plot_spectrum(im_fft2)
plt.title('Filtered Spectrum')

im_new = fftpack.ifft2(im_fft2).real

plt.figure()
plt.imshow(im_new, plt.cm.gray)
plt.title('Reconstructed image')
plt.show()



'''


dst = cv2.fastNlMeansDenoisingColored(PandaNoise,None,24,10,7,21)

meanSquareError = ((PandaNoise - PandaClean)**2).mean(axis=None)
print("original msq = " , meanSquareError)

meanSquareError = ((dst - PandaClean)**2).mean(axis=None)
print("new msq =" , meanSquareError)
plt.subplot(131), plt.imshow(PandaNoise)
plt.subplot(132), plt.imshow(dst)
plt.subplot(133), plt.imshow(PandaClean)
plt.show()
'''

