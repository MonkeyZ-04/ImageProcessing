import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- 1. โหลดรูปภาพ ---
image_path = 'test3.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    print(f"ไม่สามารถโหลดรูปภาพได้จาก: {image_path}")
else:
    # --- 2. Preprocessing (Smooth) ---
    smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # --- 3. Segmentation (Otsu's Method) ---
    otsu_threshold, binary_image = cv2.threshold(
        smoothed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- 4. ค้นหา Contours และ Hierarchy ---
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- 5. ค้นหา "รู" (Hole) ที่ใหญ่ที่สุด ---
    max_area = 0
    largest_hole_contour = None
    largest_hole_bounding_box = None

    if hierarchy is not None:
        for i, contour in enumerate(contours):
            parent_index = hierarchy[0][i][3]
            if parent_index != -1:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_hole_contour = contour
                    largest_hole_bounding_box = cv2.boundingRect(contour)

    # --- 6. Crop และแสดงผลลัพธ์ (พร้อมขยายกรอบแบบกำหนดเอง) ---
    if largest_hole_contour is not None:
        print(f"พบ Segment สีดำที่ใหญ่ที่สุด (Hole) มีพื้นที่: {max_area} pixels")
        
        # สร้างภาพสำหรับแสดงผล
        visualization_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # วาด Contour (สีเขียว)
        cv2.drawContours(visualization_image, [largest_hole_contour], -1, 
                         (0, 255, 0), 2) 
        
        # --- (ส่วนที่อัปเดต) กำหนด Padding แยกแต่ละด้าน ---
        x, y, w, h = largest_hole_bounding_box
        
        # กำหนดค่า padding ที่ต้องการ
        padding_top = 120
        padding_bottom = 110
        padding_right = 200
        padding_left = 400

        # ดึงขนาดของภาพต้นฉบับเพื่อป้องกันขอบตก
        img_height, img_width = smoothed_image.shape[:2]

        # คำนวณกรอบใหม่ (x1, y1, x2, y2)
        x1 = max(0, x - padding_left)
        y1 = max(0, y - padding_top)
        x2 = min(img_width, x + w + padding_right)
        y2 = min(img_height, y + h + padding_bottom)
        
        # วาด Bounding Box (สีแดง) โดยใช้กรอบใหม่
        cv2.rectangle(visualization_image, (x1, y1), (x2, y2), 
                      (0, 0, 255), 2) # สีแดง, หนา 2px

        # --- ทำการ Crop ภาพโดยใช้กรอบใหม่ ---
        cropped_image = smoothed_image[y1:y2, x1:x2]

        # --- แสดงผลลัพธ์ ---
        plt.figure(figsize=(18, 7))
        
        # ภาพแสดงตำแหน่งที่พบ
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
        plt.title('Hole (Left Padding = 100px)')
        plt.axis('off')
        
        # ภาพที่ Crop แล้ว
        plt.subplot(1, 2, 2)
        plt.imshow(cropped_image, cmap='gray')
        plt.title('Cropped Image (Asymmetric Padding)')
        plt.axis('off')

        plt.show()

    else:
        print("ไม่พบ Segment สีดำที่ถูกล้อมรอบด้วยสีขาว (Hole) ในภาพนี้")
