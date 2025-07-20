import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_image(image_1, image_2,title_1="Orignal", title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()

def plot_hist(old_image, new_image,title_old="Orignal", title_new="New Image"):
    intensity_values=np.array([x for x in range(256)])
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()

def thresholding(input_img,threshold,max_value=255, min_value=0):
    N,M=input_img.shape
    image_out=np.zeros((N,M),dtype=np.uint8)
        
    for i  in range(N):
        for j in range(M):
            if input_img[i,j]> threshold:
                image_out[i,j]=max_value
            else:
                image_out[i,j]=min_value
                
    return image_out        

# toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]],dtype=np.uint8)
# plt.imshow(toy_image, cmap="gray")
# plt.show()
# print("toy_image:",toy_image)

# plt.bar([x for x in range(6)],[1,5,2,0,0,0]) #6 is range of x and y axis 0-5, 1,5,2,0,0,0 are height of each x value corresponding 0 mean no chart
# plt.show()

# plt.bar([x for x in range(6)],[0,1,0,5,0,2])
# plt.show()

# goldhill = cv2.imread("images/goldhill.bmp",cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(10,10))
# plt.imshow(goldhill,cmap="gray")
# plt.show()

# hist = cv2.calcHist([goldhill],[0], None, [256], [0,256]) #range in python will write 0-255 as 0-256 not include top range, [0] for gray-scale, เก็บค่า intensity ทั้งรูป
# intensity_values = np.array([x for x in range(hist.shape[0])]) #channel o,มีแค่ 1 channel เท่านั้น ช่องแรกเลยเริ่มที่ 0, List Comprehension สร้าง ลิสต์ของตัวเลข ตั้งแต่ 0 ถึง hist.shape[0] - 1,สร้างค่าความสว่างแนวแกน x
# plt.bar(intensity_values, hist[:,0], width = 5) #width=5 → ความกว้างของแต่ละแท่ง, (intensity_values x,hist[:,0] y) y = ค่าความสว่างทั้งหมดที่เก็บได้ใน channel 0
# plt.title("Bar histogram")
# plt.show()

# PMF = hist / (goldhill.shape[0] * goldhill.shape[1])
# plt.plot(intensity_values,hist)
# plt.title("histogram")
# plt.show()

# baboon = cv2.imread("images/baboon.png")
# plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
# plt.show()

# color = ('blue','green','red')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([baboon],[i],None,[256],[0,256])
#     plt.plot(intensity_values,histr,color = col,label=col+" channel")#label เก็บคำอธิบายในกราฟ
    
#     plt.xlim([0,256]) #กำหนดช่วงค่าแกน X ให้แสดงเฉพาะจาก a ถึง b, รวมถึง 256 เพราะ "ช่วงแสดงผลบนกราฟ" ไม่ใช่ range แบบใน range() ที่ "ไม่รวมตัวสุดท้าย"
# plt.legend() #ใช้คู่กับ Label,แสดงคำอธิบายของเส้น/แท่งในกราฟ
# plt.title("Histogram Channels")
# plt.show()


#nagative เปลี่ยนขาวเป็นดำ ดำเป็นขาว
# toy_image = cv2.imread("images/baboon.png", cv2.IMREAD_GRAYSCALE) #cv2.imread() คืนค่าเป็น NumPy อยู่แล้ว ตรงนี้อาจไม่จำเป็น
# neg_toy_image = 255 - toy_image
# print("toy image\n", neg_toy_image)
# print("image negatives\n", neg_toy_image)
# plt.figure(figsize=(10,10))
# plt.subplot(1, 2, 1) 
# plt.imshow(toy_image,cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(neg_toy_image,cmap="gray")
# plt.show()
# print("toy_image:",toy_image)

# pic = cv2.imread("images/mammogram.jpeg", cv2.IMREAD_GRAYSCALE)
# cv2.imwrite("mammogram.png", pic)
# image = cv2.imread("images/mammogram.png", cv2.IMREAD_GRAYSCALE)
# cv2.rectangle(image, pt1=(160, 212), pt2=(250, 289), color = (255), thickness=2) 
# plt.figure(figsize = (10,10))
# plt.imshow(image, cmap="gray")#use cmap="gray" evry time you use v2.IMREAD_GRAYSCALE
# plt.show()

# img_neg = 255 - image
# plt.figure(figsize=(10,10))
# plt.imshow(img_neg, cmap = "gray")
# plt.show()

# goldhill = cv2.imread("images/goldhill.bmp", cv2.IMREAD_GRAYSCALE)
# alpha = 1 # Simple contrast control
# beta = 100   # Simple brightness control   
# new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
# plot_image(goldhill, new_image, title_1 = "Orignal", title_2 = "brightness control")
# plt.figure(figsize=(10,5))
# plot_hist(goldhill, new_image, "Orignal", "brightness control")

# alpha = 2# Simple contrast control ยิ่งมากยิ่งขาว?
# beta = 0 # Simple brightness control   # Simple brightness control
# new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
# plot_image(goldhill,new_image,"Orignal","contrast control")
# plt.figure(figsize=(10,5))
# plot_hist(goldhill, new_image,"Orignal","contrast control")

# alpha = 3 # Simple contrast control เอาไว้ทำให้ดำขาวขึ้น ขาวดำขึ้น
# beta = -200  # Simple brightness control   
# new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
# plot_image(goldhill, new_image, "Orignal", "brightness & contrast control")
# plt.figure(figsize=(10,5))
# plot_hist(goldhill, new_image, "Orignal", "brightness & contrast control")

#Histogram Equalization
# zelda = cv2.imread("images/zelda.png",cv2.IMREAD_GRAYSCALE)
# new_image = cv2.equalizeHist(zelda) #flattend the histogram(make it more equal to each other,it will improve more contrast)
# plot_image(zelda,new_image,"Orignal","Histogram Equalization")
# plt.figure(figsize=(10,5))
# plot_hist(zelda, new_image,"Orignal","Histogram Equalization")

# # Thresholding and Simple Segmentation 
# pic = cv2.imread("images/cameraman.jpeg")
# cv2.imwrite("cameraman.png", pic)
# toy_image = cv2.imread("images/cameraman.png", cv2.IMREAD_GRAYSCALE)
# threshold = 1
# max_value = 2
# min_value = 0
# thresholding_toy = thresholding(toy_image, threshold=threshold, max_value=max_value, min_value=min_value)
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(toy_image, cmap="gray")
# plt.title("Original Image")
# plt.subplot(1, 2, 2)
# plt.imshow(thresholding_toy, cmap="gray")
# plt.title("Image After Thresholding")
# plt.show()

image = cv2.imread("images/cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(10, 10))
# plt.imshow(image, cmap="gray")
# plt.show()

# goldhill = cv2.imread("images/goldhill.bmp", cv2.IMREAD_GRAYSCALE)
# hist = cv2.calcHist([goldhill],[0], None, [256], [0,256]) #range in python will write 0-255 as 0-256 not include top range, [0] for gray-scale, เก็บค่า intensity ทั้งรูป
# intensity_values = np.array([x for x in range(hist.shape[0])]) 
# plt.bar(intensity_values, hist[:, 0], width=5)
# plt.title("Bar histogram")
# plt.show()

threshold = 87
max_value = 255
min_value = 0
# new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)
# plot_image(image, new_image, "Orignal", "Image After Thresholding")
# plt.figure(figsize=(10,5))
# plot_hist(image, new_image, "Orignal", "Image After Thresholding")

img = cv2.imread("images/cameraman.png", cv2.IMREAD_GRAYSCALE)

# THRESH_BINARY แบบธรรมดา
# _, binary_manual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# # THRESH_OTSU แบบอัตโนมัติ
# _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # เขียนเองแบบ NumPy และแปลงเป็น uint8
# binary_numpy = np.where(img > 127, 255, 0).astype(np.uint8)

# # แสดงผลเปรียบเทียบ
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1); plt.imshow(binary_manual, cmap='gray'); plt.title("THRESH_BINARY") #number will be 0 0r 255
# plt.subplot(1, 3, 2); plt.imshow(binary_otsu, cmap='gray'); plt.title("THRESH_OTSU") #auto select for difficult to pick threshold picture
# plt.subplot(1, 3, 3); plt.imshow(binary_numpy, cmap='gray'); plt.title("NumPy + astype(uint8)")# for manual treshold method, ex.tresholding()
# plt.show()

# #By me search 
# plt.figure(figsize=(10, 5))  # Wider layout for one row
# plot_hist(image, binary_manual, "Original", "THRESH_BINARY")
# plt.show()
# plt.figure(figsize=(10, 5))  # Wider layout for one row
# plot_hist(image, binary_otsu, "Original", "THRESH_OTSU")
# plt.show()
# plt.figure(figsize=(10, 5))  # Wider layout for one row
# plot_hist(image, binary_numpy, "Original", "NumPy + astype(uint8)")
# plt.show()

ret, new_image = cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)
plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")

ret, new_image = cv2.threshold(image,86,255,cv2.THRESH_TRUNC) #cv2.THRESH_TRUNC will not change the values if the pixels are less than the threshold value:
plot_image(image,new_image,"Orignal","Image After Thresholding (TRUNC)")
plot_hist(image, new_image,"Orignal","Image After Thresholding (TRUNC)")

ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
plot_image(image,otsu,"Orignal","Otsu")
plot_hist(image, otsu,"Orignal"," Otsu's method")