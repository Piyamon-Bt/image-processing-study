import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()



toy_image = np.zeros((6,6))
toy_image[1:5,1:5]=255 #1-4
toy_image[2:4,2:4]=0 #2-3
plt.imshow(toy_image,cmap='gray')
plt.show()

new_toy = cv2.resize(toy_image,None,fx=2, fy=1, interpolation = cv2.INTER_NEAREST )
plt.imshow(new_toy,cmap='gray')
plt.show()

image = cv2.imread("images/lenna.png")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

new_image = cv2.resize(image, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

new_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

new_image = cv2.resize(image, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

rows = 100
cols = 200
new_image = cv2.resize(image, (100, 200), interpolation=cv2.INTER_CUBIC) # = cv2.resize(image, (width, height)) but for numpy #image = np.zeros((height, width), dtype=np.uint8)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

tx = 100
ty = 0
M = np.float32([[1, 0, tx], 
                [0, 1, ty]]) #tx>0 right, ty>0 down

rows, cols, _ = image.shape #numpy array, _ แปลว่าไม่ต้องการรู้ตำแหน่งนี้
new_image = cv2.warpAffine(image, M, (cols, rows)) #cv2.warpAffine() = ฟังก์ชันแปลงภาพตามเมทริกซ์ M, (cols, rows) = ขนาดของ output image (ต้องตามลำดับ width, height)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()


#Rotation
#pattern เหมือนตอนขยับ matrix, 1.เลือกรูป 2.เอาเมทริกที่ใช้ในการขยับมา Operation MAtric? 3.กำหนดขนาดที่จะแสดง
tx = 0
ty = 50
M = np.float32([[1, 0, tx],
                [0, 1, ty]])
new_iamge = cv2.warpAffine(image, M, (cols + tx, rows + ty)) #เพิ่มขนาดภาพผลลัพธ์ เพื่อรองรับพื้นที่ที่ถูกเลื่อนออกไปจากขอบภาพเดิม เพราะถ้าไม่เพิ่มขนาดจะถูกตัดออกไป
plt.imshow(cv2.cvtColor(new_iamge, cv2.COLOR_BGR2RGB))
plt.show()

theta = 45.0
M = cv2.getRotationMatrix2D(center=(3, 3), angle=theta, scale=1) #scale=1 ขนาดภาพคงเดิม (ไม่ย่อ/ขยาย)
new_toy_image = cv2.warpAffine(toy_image, M, (6, 6)) #(6, 6) คือขนาดของ output image (width, height) ขนาด 6 = 0-5, ค่า pixel ที่หมุนออกนอกขอบภาพจะหายไป (เติมด้วยสีดำ)
plot_image(toy_image, new_toy_image, title_1="Orignal", title_2="rotated image")

cols, rows, _ = image.shape
new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#Matric Operations
#Array Operations
new_image = image + 20 #add intensity to each pixel #เพิ่มค่าทุกพิกเซล (R, G, B) ด้วยค่า 20, เพิ่มค่าทุกพิกเซล (R, G, B) ด้วยค่า 20
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = 10 * image #ภาพจะ สว่างขึ้นเร็วมาก แต่พิกเซลอาจทะลุ 255 แล้ววนกลับ (ดูผิดเพี้ยน)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#noise
Noise = np.random.normal(0, 20, (rows, cols, 3)).astype(np.uint8) #สร้างค่า สุ่มจาก Gaussian distribution (ค่ากลาง 0, ส่วนเบี่ยงเบนมาตรฐาน 20), สร้าง noise ขนาดเท่ากับภาพ (image) 3 ช่องสี (RGB หรือ BGR)
print(Noise.shape)
new_image = image + Noise #ผลคือ: ภาพที่มี "ความไม่แน่นอน" หรือ "จุดรบกวน" คล้ายกล้องมีสัญญาณรบกวน
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = image*Noise
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

im_gray = cv2.imread('images/lenna.png', cv2.IMREAD_GRAYSCALE)
im_gray.shape
plt.imshow(im_gray,cmap='gray')
plt.show()

U, s, V = np.linalg.svd(im_gray , full_matrices=True)
print(s.shape)
S = np.zeros((im_gray.shape[0], im_gray.shape[1])) #set ให้ขนาด Matrix ที่จะบรรจุ x เป็น 0#im_gray.shape[0] = 512 → จำนวน แถว (แนวตั้ง) = height, im_gray.shape[1] = 512 → จำนวน คอลัมน์ (แนวนอน) = width
S[:image.shape[0], :image.shape[0]] = np.diag(s) #เอา s ใส่แนวทะแยงใน Matrix S ที่เอามาขัดเก็บ
plot_image(U,V,title_1="Matrix U ",title_2="matrix  V")
plt.imshow(S,cmap='gray')
plt.show()

#ฟื้นฟูภาพมาดูใหม่
B = S.dot(V)
plt.imshow(B,cmap='gray')
plt.show()
A = U.dot(B)
plt.imshow(A,cmap='gray')
plt.show()

for n_component in [1,10,100,200, 500]:
    S_new = S[:, :n_component] #width #เลือกเฉพาะ คอลัมน์แรกๆ ของ S จำนวน n_component (คือเก็บเฉพาะ singular values ตัวแรก ๆ ที่สำคัญมากที่สุด)
    V_new = V[:n_component, :] #height #เลือกเฉพาะ แถวบนสุดของ V เท่ากับจำนวน components ที่ต้องการ
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap='gray')
    plt.title("Number of Components:"+str(n_component))
    plt.show()

#U: เมทริกซ์ซ้าย
#S_new: เมทริกซ์ diagonal ที่เก็บเฉพาะค่าหลัก (จาก s)
#V_new: เมทริกซ์ขวาที่ลดขนาด

# U เป็นเมทริกซ์คงที่ที่ มีขนาดเท่ากับจำนวนแถวของภาพ
# เวลาเรา “ลดมิติ” หรือ “ลดจำนวน components”
# เราเปลี่ยนแค่จำนวนคอลัมน์ที่เราจะใช้จาก U เพื่อคูณกับ S_new และ V_new

# s.shape เป็น เวกเตอร์ 1 มิติ = (512,)
# ขนาดเท่ากับ จำนวนน้อยที่สุดระหว่างแถวกับคอลัมน์ของภาพต้นฉบับ
# เก็บค่า ความสำคัญเชิงโครงสร้างของภาพ (จากมาก → น้อย)