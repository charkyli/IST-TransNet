from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
image = Image.open('data/sirst/masks/Misc_9_pixels0.png').convert("L")
width,height = image.size
X,Y = np.meshgrid(np.arange(0,width),np.arange(0,height))
Z = np.array(image)

# plt.figure(figsize=(300,300))
# plt.subplot(2,2,1)
# # plt.xticks([])
# # plt.yticks([])
# plt.xlabel("original")
# plt.imshow(image)
#
# plt.subplot(2,2,2)
# sub_image = Z[50:70,140:160]
# plt.xlabel("sub_image")
# plt.imshow(sub_image)

plt.subplot(1,2,1)
fig = plt.figure(figsize=(300,200))
dim = Axes3D(fig)
dim.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
plt.xlabel("heatmap")
plt.show()

plt.subplot(1,2,2)
fig1 = plt.figure(figsize=(300,200))
axe = Axes3D(fig1)
axe.plot_surface(X,Y,sub_image,rstride=1,cstride= 1,cmap='rainbow')
plt.show()