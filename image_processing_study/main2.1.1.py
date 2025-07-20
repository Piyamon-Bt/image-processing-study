from PIL import Image            # PIL (Pillow) library
from PIL import ImageOps         # PIL (Pillow) library #work with greyscale
import matplotlib.pyplot as plt  # matplotlib library
import numpy as np


def get_concat_h(im1, im2):      # Python built-in (function definition), uses PIL inside
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))  # PIL (Pillow) library
    dst.paste(im1, (0, 0))       # PIL (Pillow) library
    dst.paste(im2, (im1.width, 0))  # PIL (Pillow) library
    return dst                   # Python built-in

my_image = "lenna.png"           # Python built-in (string assignment)
image = Image.open(my_image)     # PIL (Pillow) library #keep image in this variable
# image.show(title='Lena')       # PIL (Pillow) library

#image_gray = ImageOps.grayscale(image)  # PIL (Pillow) library

# for n in range(3,8):
#     plt.figure(figsize=(10,10))

#     plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) #ImageOps
#     plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
#     plt.show()

baboon = Image.open('baboon.png')
# red, green, blue = baboon.split() #pillow
# get_concat_h(baboon, red).show() #more light more red
# get_concat_h(baboon, green).show() #more light more green
# get_concat_h(baboon, blue).show() #more light more blue, darker than red and green

# array= np.asarray(image) #modify but doesn't make a copy
# print(type(array))
array = np.array(image)
# print(array.shape)
# print(array)
# print(array[0, 0])
# print(array.min())
# print(array.max())
# plt.figure(figsize=(10,10))
# plt.imshow(array)
# plt.show()

# rows = 256
# plt.figure(figsize=(10,10))
# plt.imshow(array[0:rows,:,:])
# plt.show()

# columns = 256
# plt.figure(figsize=(10,10))
# plt.imshow(array[:,0:columns,:])
# plt.show()

# A = array.copy()
# plt.imshow(A)
# plt.show()

# B = A #b point to a, same memory
# A[:,:,:] = 0
# plt.title("B")
# plt.imshow(B)
# plt.show()

baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array)
plt.show()

baboon_red=baboon_array.copy()
baboon_red[:, :, 1] = 0  # ปิดช่องสีเขียว (Green) RGB 012
baboon_red[:, :, 2] = 0  # ปิดช่องสีน้ำเงิน (Blue)
plt.figure(figsize=(10,10))
plt.imshow(baboon_red)
plt.show()

baboon_blue=baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()

blue_lenna = Image.open('lenna.png')
blue_array = np.array(blue_lenna)
blue_array[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(blue_array)
plt.show()