from PIL import Image  #RGB 012          # PIL (Pillow) library
from PIL import ImageOps         # PIL (Pillow) library #work with greyscale
import matplotlib.pyplot as plt  # matplotlib library
import numpy as np
import os
import cv2 #BGR 012

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

my_image = "lenna.png"
cwd = os.getcwd()
image_path = os.path.join(cwd, my_image)
# print(image_path)

image = cv2.imread(my_image) #or image = cv2.imread(image_path)

# print(image.max())

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.show()

# new_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10,10))
# plt.imshow(new_image)
# plt.show()

# image = cv2.imread(image_path)
#cv2.imwrite("lenna.jpg", image) #save image in jpg
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(image_gray.shape) #The image array has only two dimensions, i.e. only one color channel: (512,512)
# plt.figure(figsize=(10, 10))
#plt.imshow(image_gray, cmap='gray') #cmp-color map because pic will turn green
# plt.show()
# cv2.imwrite('lena_gray_cv.jpg', image_gray)

#ได้ผลลัพธ์เป็น NumPy array ขนาด (H, W) (ไม่มีช่องสี)
# im_gray = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE เป็น flag ที่สั่งให้อ่านภาพแบบ Grayscale
# plt.figure(figsize=(10,10))
# plt.imshow(im_gray,cmap='gray') #ให้แสดงเป็น ภาพขาว–ดำจริง ๆ
# plt.show()

baboon=cv2.imread('baboon.png')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]
im_bgr = cv2.vconcat([blue, green, red])
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("RGB image")
plt.subplot(122)
plt.imshow(im_bgr,cmap='gray')
plt.title("Different color channels  blue (top), green (middle), red (bottom)  ")
plt.show()

rows = 256
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon[0:rows,:,:], cv2.COLOR_BGR2RGB))
plt.show()

columns = 256
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon[:,0:columns,:], cv2.COLOR_BGR2RGB))
plt.show()

A = baboon.copy()
plt.imshow(A)
plt.show()

B = A
A[:,:,:] = 0
plt.imshow(B)
plt.show()

#red baboon 
baboon_red = baboon.copy()
baboon_red[:, :, 0] = 0 #BGR 012
baboon_red[:, :, 1] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_red, cv2.COLOR_BGR2RGB)) #change color to make it show natural color
plt.show()

#blue baboon 
baboon_blue = baboon.copy()
baboon_blue[:, :, 1] = 0
baboon_blue[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB))
plt.show()

#green baboon
baboon_green = baboon.copy()
baboon_green[:, :, 0] = 0
baboon_green[:, :, 2] = 0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon_green, cv2.COLOR_BGR2RGB))
plt.show()

image=cv2.imread('baboon.png')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
image=cv2.imread('baboon.png') # replace and add you image here name 
baboon_blue=image.copy()
baboon_blue[:,:,1] = 0
baboon_blue[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB))
plt.show()

#baboon red,green take out blue
baboon_blue=cv2.imread('baboon.png')
baboon_blue=cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB)
baboon_blue[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()