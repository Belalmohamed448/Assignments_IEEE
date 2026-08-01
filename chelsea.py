import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r'c:\Users\Mega Store\Downloads\chelsea_negative.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0


fixed = 1 - img


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original (negative)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(fixed)
plt.title('Fixed')
plt.axis('off')

plt.tight_layout()
plt.show()