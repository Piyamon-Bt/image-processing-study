##FILTER USING OPENCV
# Used to view the images
import matplotlib.pyplot as plt
# Used to perform filtering on an image
import cv2
# Used to create kernels for filtering
import numpy as np

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

# Loads the image from the specified file
image = cv2.imread("images/lenna.png")
print(image)
# Converts the order of the color from BGR (Blue Green Red) to RGB (Red Green Blue) then renders the image from the array of data
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Get the number of rows and columns in the image
rows, cols,_= image.shape
# Creates values using a normal distribution with a mean of 0 and standard deviation of 15, the values are converted to unit8 which means the values are between 0 and 255
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# Add the noise to the image
noisy_image = image + noise
# Plots the original image and the image with noise using the function defined at the top
plot_image(image, noisy_image, title_1="Orignal",title_2="Image Plus Noise")

##FILTERING NOISE
# Create a kernel which is a 6 by 6 array where each value is 1/36
kernel = np.ones((6,6))/36
# Filters the images using the kernel
image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel) #ddepth=-1 หมายถึง: ให้ใช้ depth เดิมของภาพต้นฉบับ, ddepth=-1 หมายถึง: ให้ใช้ depth เดิมของภาพต้นฉบับ

# Plots the Filtered and Image with Noise using the function defined at the top
plot_image(image_filtered, noisy_image,title_1="Big Kernel OpenCV Filtered image",title_2="Image Plus Noise")

##Kernel
# Creates a kernel which is a 4 by 4 array where each value is 1/16
kernel = np.ones((4,4))/16
# Filters the images using the kernel
image_filtered=cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
# Plots the Filtered and Image with Noise using the function defined at the top
plot_image(image_filtered , noisy_image,title_1="Small Kernel OpenCV filtered image",title_2="Image Plus Noise")

##GAUSSIANBLUR
# Filters the images using GaussianBlur on the image with noise using a 4 by 4 kernel 
image_filtered = cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4) #sigmaX = ค่าความเบลอในแนวแกน X (แนวนอน) , sigmaY = ค่าความเบลอในแนวแกน Y (แนวตั้ง) , ถ้าไม่ใส่ sigmaX และ sigmaY หรือใส่เป็น 0 → OpenCV จะคำนวณเองจากขนาด kernel
# Plots the Filtered Image then the Unfiltered Image with Noise
plot_image(image_filtered , noisy_image,title_1="Small GaussianBlur Filtered image",title_2="Image Plus Noise")

# Filters the images using GaussianBlur on the image with noise using a 11 by 11 kernel 
image_filtered = cv2.GaussianBlur(noisy_image,(11,11),sigmaX=10,sigmaY=10)
# Plots the Filtered Image then the Unfiltered Image with Noise
plot_image(image_filtered , noisy_image,title_1="Big GaussianBlur filtered image",title_2="Image Plus Noise")

##IMAGE SHARPENING
# Common Kernel for image sharpening
kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
# Applys the sharpening filter using kernel on the original image without noise
sharpened = cv2.filter2D(image, -1, kernel)
# Plots the sharpened image and the original image without noise
plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

##EDGES
# Loads the image from the specified file
img_gray = cv2.imread('images/barbara.png', cv2.IMREAD_GRAYSCALE)
print(img_gray)
# Renders the image from the array of data, notice how it is 2 diemensional instead of 3 diemensional because it has no color
plt.imshow(img_gray ,cmap='gray')

# Filters the images using GaussianBlur on the image with noise using a 3 by 3 kernel 
img_gray = cv2.GaussianBlur(img_gray,(3,3),sigmaX=0.1,sigmaY=0.1)
# Renders the filtered image
plt.imshow(img_gray ,cmap='gray')
plt.show()

ddepth = cv2.CV_16S
# Applys the filter on the image in the X direction
grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3) #dx=1 ให้ตรวจสอบการเปลี่ยนแปลงในแนวแกน X (แนวนอน) , dy=0 ไม่ตรวจสอบแนวแกน Y ,ใช้ kernel ขนาด 3x3 สำหรับ Sobel ต้องเป็นเลข 1,3,5,7
plt.imshow(grad_x,cmap='gray')
plt.show()

# Applys the filter on the image in the X direction
grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
plt.imshow(grad_y,cmap='gray')
plt.show()

# Converts the values back to a number between 0 and 255
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Adds the derivative in the X and Y direction
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Make the figure bigger and renders the image
plt.figure(figsize=(10,10))
plt.imshow(grad,cmap='gray')
plt.show()

##MEDIAN
# Load the camera man image
image = cv2.imread("images/cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
# Make the image larger when it renders
plt.figure(figsize=(10,10))
# Renders the image
plt.imshow(image,cmap="gray")
plt.show()

# Filter the image using Median Blur with a kernel of size 5
filtered_image = cv2.medianBlur(image, 5) #เหมาะสำหรับลด Salt and Pepper noise (จุดดำจุดขาวเล็กๆ)
# Make the image larger when it renders
plt.figure(figsize=(10,10))
# Renders the image
plt.imshow(filtered_image,cmap="gray")
plt.show()

##THRESHOLD FUNCTION PARAMETER
# Returns ret which is the threshold used and outs which is the image
ret, outs = cv2.threshold(src = image, thresh = 0, maxval = 255, type = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
#ret → คือ ค่าที่ Otsu คำนวณได้ ว่าเป็น threshold ที่เหมาะสม เช่น 127.0
#result → คือภาพ ขาวดำ (binary) ที่ pixel ใด ≥ ret จะเป็น 255 และ pixel ที่น้อยกว่าจะเป็น 0

# Make the image larger when it renders
plt.figure(figsize=(10,10))

# Render the image
plt.imshow(outs, cmap='gray')