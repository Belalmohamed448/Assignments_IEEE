import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'c:\Users\Mega Store\Downloads\coffee_overexposed.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img / 255.0

current_brightness = img.mean()
print('Current brightness:', current_brightness)

target_brightness = 0.5

beta = target_brightness - current_brightness
print('Beta (amount to remove):', beta)

fixed_img = img + beta
fixed_img = np.clip(fixed_img, 0, 1)
print('New brightness:', fixed_img.mean())

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original (overexposed)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(fixed_img)
plt.title('Fixed')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(img.ravel(), bins=50)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')

plt.subplot(2, 2, 4)
plt.hist(fixed_img.ravel(), bins=50)
plt.title('Fixed Histogram')
plt.xlabel('Pixel Value')

plt.tight_layout()
plt.show()