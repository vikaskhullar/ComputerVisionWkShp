import numpy as np
import matplotlib.pyplot as plt
#img1 = np.array([np.array([20, 20]), np.array([20, 20])])
img1 = np.array([np.array([200, 200]), np.array([200, 200])])
img1
plt.imshow(img1)
plt.axis('off')
plt.title('img1')
plt.show()


kernel_horizontal = np.array([np.array([2, 2]), np.array([-2, -2])])
print(kernel_horizontal, 'is a kernel for detecting horizontal edges')
def apply_kernel(img, kernel):
	return np.multiply(img, kernel)


kernel_horizontal = np.array([np.array([2, 2]), np.array([-2, -2])])
print(kernel_horizontal, 'is a kernel for detecting horizontal edges')
imgker = apply_kernel(img1,kernel_horizontal)
plt.imshow(imgker)
plt.axis('off')
plt.title('img1')
plt.show()




aa= np.array([200, 200])
aa
# let img1 be an image with no features
img1 = np.array([np.array([200, 200]), np.array([200, 200])])
img1
img1.shape
np.array([200, 200]).shape


img2 = np.array([np.array([200, 200]), np.array([0, 0])])
img2.shape
img2


img3 = np.array([np.array([200, 0]), np.array([200, 0])])
img3.shape
img3



kernel_horizontal = np.array([np.array([2, 2]), np.array([-2, -2])])
print(kernel_horizontal, 'is a kernel for detecting horizontal edges')


kernel_vertical = np.array([np.array([2, -2]), np.array([2, -2])])
print(kernel_vertical, 'is a kernel for detecting vertical edges')

# We will apply the kernels on the images by
# elementwise multiplication followed by summation
def apply_kernel(img, kernel):
	return np.multiply(img, kernel)

img1

'''
RGB
0-255
'''

# Visualizing img1
plt.imshow(img1)
plt.axis('off')
plt.title('img1')
plt.show()

img1
kernel_horizontal
imgker = apply_kernel(img1,kernel_horizontal)
plt.imshow(imgker)
plt.axis('off')
plt.title('img1')
plt.show()

imgker = apply_kernel(img1,kernel_vertical)
plt.imshow(imgker)
plt.axis('off')
plt.title('img1')
plt.show()



# Checking for horizontal and vertical features in image1
print('Horizontal edge confidence score:', apply_kernel(img1,
											kernel_horizontal))
print('Vertical edge confidence score:', apply_kernel(img1,
											kernel_vertical))




img2
plt.imshow(img2)
plt.axis('off')
plt.title('img2')
plt.show()

# Checking for horizontal and vertical features in image2
print('Horizontal edge confidence score:', apply_kernel(img2,
											kernel_horizontal))
print('Vertical edge confidence score:', apply_kernel(img2,
											kernel_vertical))

img3
# Visualizing img3
plt.imshow(img3)
plt.axis('off')
plt.title('img3')
plt.show()

# Checking for horizontal and vertical features in image3
print('Horizontal edge confidence score:', apply_kernel(img3,
											kernel_horizontal))
print('Vertical edge confidence score:', apply_kernel(img3,
											kernel_vertical))
