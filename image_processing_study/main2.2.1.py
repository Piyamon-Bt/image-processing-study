import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps #flip etc.
import numpy as np
from PIL import ImageDraw 
from PIL import ImageFont

# baboon = np.array(Image.open('baboon.png'))
# plt.figure(figsize=(5,5))
# plt.imshow(baboon )
# plt.show()
# A = baboon

# print(id(A))
# print(id(baboon))
# print(id(A) == id(baboon))

# B = baboon.copy() # ใช้กับ PIL.Image object,ต้องแปลงเป็น array ก่อน (เช่น np.array(...)) ถึงใช้ได้
# id(B)==id(baboon)
# baboon[:,:,] = 0

# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(baboon)
# plt.title("baboon")
# plt.subplot(122)
# plt.imshow(A)
# plt.title("array A")
# plt.show()

# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(baboon)
# plt.title("baboon")
# plt.subplot(122)
# plt.imshow(B)
# plt.title("array B")
# plt.show()

image = Image.open("cat.png")
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.show()

array = np.array(image)
# width, height, C = array.shape
# print('width, height, C', width, height, C)

# array_flip = np.zeros((width, height, C), dtype=np.uint8) #ใช้ dtype=np.uint8 เพื่อรองรับค่าพิกเซล 0–255 (มาตรฐานภาพสี)
# for i,row in enumerate(array):
#     array_flip[width - 1 - i, :, :] = row
# plt.imshow(array_flip)
# plt.show()

# im_flip = ImageOps.flip(image)
# plt.figure(figsize=(5,5))
# plt.imshow(im_flip)
# plt.show()

# im_mirror = ImageOps.mirror(image)
# plt.figure(figsize=(5,5))
# plt.imshow(im_mirror)
# plt.show()

# im_flip = image.transpose(1)
# plt.imshow(im_flip)
# plt.show()

flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT, #dict in python
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE, 
        "TRANSVERSE": Image.TRANSVERSE}


# image_LR = image.transpose(flip["FLIP_LEFT_RIGHT"])
# plt.imshow(image_LR)
# plt.show()

# for key, values in flip.items():
#     plt.figure(figsize=(10,10))
#     plt.subplot(1,2,1) #1 แถว 2 คอลัมน์ วาดภาพที่ ตำแหน่งที่ 1 (ซ้ายสุด)
#     plt.imshow(image)
#     plt.title("orignal")
#     plt.subplot(1,2,2) #1 แถว 2 คอลัมน์ วาดภาพที่ ตำแหน่งที่ 2 (ขวาสุด)
#     plt.imshow(image.transpose(values))
#     plt.title(key)
#     plt.show()

upper = 150
lower = 400
# crop_top = array[upper: lower,:,:] #ขนาดยาว 250px ตัดที่ array y 150 to 400, สร้าง array ใหม่ ที่มีขนาดเล็กลงเฉพาะบางแถว (แนว y) เท่านั้น
# plt.figure(figsize=(5,5))
# plt.imshow(crop_top)
# plt.show()

left = 150
right = 400
# crop_horizontal = crop_top[: ,left:right,:]
# plt.figure(figsize=(5,5))
# plt.imshow(crop_horizontal)
# plt.show()

# image = Image.open("cat.png")
crop_image = image.crop((left, upper, right, lower)) #ใช้ตัวแปรข้างบน
# plt.figure(figsize=(5,5))
# plt.imshow(crop_image)
# plt.show()

crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
# plt.imshow(crop_image)
# plt.show()

array_sq = np.copy(array) # ใช้กับ NumPy array, ใช้งาน NumPy ต่อได้
array_sq[upper:lower, left:right, 1:3] = 0 #ช่องสีที่ 1 และ 2 → คือ Green + Blue, คุณ ลบสีเขียวและน้ำเงิน โดยตั้งค่าช่อง G และ B = 0
plt.imshow(array_sq) #แก้ไขค่าพิกเซลในบางบริเวณของภาพเดิม, array_sq ยังคงมีขนาด shape = (512, 512, 3) เท่าเดิม, ไม่ได้สร้าง array ใหม่
plt.show()

plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(array)
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(array_sq)
plt.title("Altered Image")
plt.show()

image_draw = image.copy()
image_fn = ImageDraw.Draw(im=image_draw) #The draw constructor creates an object that can be used to draw in the given image. The input <code>im</code> is the image we would like to draw in.
shape = [left, upper, right, lower] 
image_fn.rectangle(xy=shape,fill="red")
plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

image_fn.text(xy=(0,0),text="box",fill=(0,0,0))
plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

image_lenna = Image.open("lenna.png")
array_lenna = np.array(image_lenna)
array_lenna[upper:lower,left:right,:]=array[upper:lower,left:right,:] #cropped on top of lenna, similar to, การสร้างรูปใหม่ทำการจองพื้นที่ขนาดเท่าแมวที่ครอปแล้วเอาแมวที่ครอปไปแปะ
plt.imshow(array_lenna)
plt.show()

image_lenna.paste(crop_image, box=(left,upper)) #ใช้แทน array_lenna[upper:lower,left:right,:]=array[upper:lower,left:right,:] ได้
plt.imshow(image_lenna)
plt.show()

image = Image.open("cat.png")
new_image=image
copy_image=image.copy()
print(id(image)==id(new_image))
print(id(image)==id(copy_image)) #the address is different

image_fn= ImageDraw.Draw(im=image)
image_fn.text(xy=(0,0),text="box",fill=(0,0,0))
image_fn.rectangle(xy=shape,fill="red")
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(new_image)
plt.subplot(122)
plt.imshow(copy_image)
plt.show()