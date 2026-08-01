import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r'c:\Users\Mega Store\Downloads\brick_gamma.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0


current_mean = img.mean()
print('Current brightness:', current_mean)


target_mean = 0.5
g_fix = np.log(target_mean) / np.log(current_mean)
print('Gamma to use:', g_fix)


fixed = img ** g_fix
fixed = np.clip(fixed, 0, 1)

print('New brightness:', fixed.mean())

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original (crushed shadows)')
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