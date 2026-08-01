import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r'c:\Users\Mega Store\Downloads\coins_hazy.png', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

low = np.percentile(img, 2)
high = np.percentile(img, 98)
stretched = (img - low) / (high - low)
stretched = np.clip(stretched, 0, 1)

T = 0.5
mask = stretched > T

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original (hazy)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(stretched, cmap='gray')
plt.title('Stretched')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

# Would thresholding work directly on the original hazy image?
# No;- the pixel values are squeezed into a narrow range, so coins and
# background are too close together in brightness. Stretching first
# spreads the values out, making it much easier to pick a clean threshold.