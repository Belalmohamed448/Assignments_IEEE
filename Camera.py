import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r'c:\Users\Mega Store\Downloads\camera_low_contrast.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0


low = np.percentile(img, 2)
high = np.percentile(img, 98)

print('Low value:', low)
print('High value:', high)

fixed = (img - low) / (high - low)
fixed = np.clip(fixed, 0, 1)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original (low contrast)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(fixed)
plt.title('Fixed')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(img.ravel(), bins=256)
plt.title('Original Histogram')

plt.subplot(2, 2, 4)
plt.hist(fixed.ravel(), bins=256)
plt.title('Fixed Histogram')

plt.tight_layout()
plt.show()