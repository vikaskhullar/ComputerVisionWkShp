import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cattle.jpg')
image
image.shape
plt.imshow(image)
plt.show()

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image
image.shape
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(image)


image = cv2.imread('cattle.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image
image.shape
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(image)


'''
Make the Kernel and apply convolution
let’s make a kernel
for blurring the image. We will use Numpy to build a 3×3 matrix of ones, and
divide it by 9, as shown in Fig.1. filter2D() function gives the convolution of the input array and kernel.
'''

abc=np.ones((3,3))
abc

kernel = np.ones((3, 3), np.float32) / 9
kernel

image = cv2.imread('cattle.jpg')
img = cv2.filter2D(image, -1, kernel)
fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].imshow(image)
ax[1].imshow(img)

img.shape
image.shape


#Following is the code for having sharpening in the image,

import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('cattle.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

img = cv2.filter2D(image, -1, kernel)
fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].imshow(image)
ax[1].imshow(img)


#For implementing image embossing, we have to follow the same procedure as explained in box blur. The only change is in its kernel.

import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('cattle.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
kernel = np.array([[-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]])
img = cv2.filter2D(image, -1, kernel)
fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].imshow(image)
ax[1].imshow(img)

