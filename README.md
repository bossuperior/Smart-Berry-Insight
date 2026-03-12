# SmartBerry Insight 🍓

**SmartBerry Insight** เป็นระบบอัจฉริยะสำหรับจำแนกผลสตรอเบอรีที่พร้อมเก็บเกี่ยวโดยใช้เทคโนโลยีการประมวลผลภาพถ่าย (Computer Vision) ร่วมกับปัญญาประดิษฐ์ (AI) ออกแบบมาเพื่อใช้งานในโรงงานผลิตพืชด้วยแสงเทียม (Plant Factory with Artificial Light: PFAL) โดยมุ่งเน้นการตรวจสอบคุณภาพแบบไม่ทำลายผล (Non-Destructive) เพื่อลดการสูญเสียในกระบวนการเก็บเกี่ยว

## 🚀 Key Features

* **Non-Destructive Quality Assessment:** ตรวจสอบระดับความสุกแก่และความหวานได้โดยไม่ต้องสุ่มเจาะหรือทำลายผลผลิต
* **Total Quality Control:** สามารถตรวจสอบคุณภาพผลผลิตได้แบบ 100% แทนการใช้การสุ่มตรวจทางสถิติแบบเดิม 
* **Real-time Monitoring:** แสดงผลการวิเคราะห์ผ่าน Dashboard เพื่อให้ผู้ผลิตวางแผนการเก็บเกี่ยวได้ตรงจังหวะที่สุด
* **Cost-Effectiveness:** ใช้กล้อง RGB เกรดอุตสาหกรรมที่มีต้นทุนต่ำกว่าเทคโนโลยี NIR หรือ Hyperspectral Imaging

## 🛠️ Technical Architecture

### 1. Computer Vision Pipeline
ระบบประมวลผลภาพดิจิทัลเพื่อให้ได้ข้อมูลที่แม่นยำประกอบด้วย:
* **Image Segmentation:** ใช้สถาปัตยกรรม **YOLOv8-seg** เพื่อจำแนกและคัดแยกพื้นที่ส่วนของผลสตรอเบอรี (ROI) ออกจากใบ ก้าน และแสงสะท้อน
  
<img width="616" height="462" alt="image" align="center" src="https://github.com/user-attachments/assets/57bb22a4-4db2-47b1-907a-00f48ee67a39" />





* **Color Calibration:** ปรับมาตรฐานค่าสีเพื่อลดผลกระทบจากความผันผวนของแสงเทียม (Artificial Lighting) ในระบบ PFAL
* **Perspective Correction:** ปรับแก้รูปทรงและมุมมองของภาพให้เป็นมาตรฐานเดียวกัน

### 2. AI & Machine Learning Models
* **Detection Model:** ใช้การทำ Instance Segmentation เพื่อแยกวัตถุในระดับพิกเซล
* **Sweetness Prediction Model:** ใช้ Regression Analysis (Supervised Learning) เพื่อทำนายค่าความหวาน ($Brix$) จากคุณลักษณะทางทัศนีย์ เช่น ปริภูมิสี (RGB, HSV, Lab*), พื้นผิว (Texture) และสัณฐานวิทยา (Morphology)

<img width="809" height="607" alt="image" align="center" src="https://github.com/user-attachments/assets/62155a7b-d3d1-4aad-bbb5-e2c3507d2d52" />


### 3. Workflow
1.  **Capture:** กล้อง RGB ใน Imaging Box บันทึกภาพภายใต้แสงที่ควบคุมคงที่ 
2.  **Process:** Processing Unit รับภาพและส่งให้ Detection Model เพื่อหา ROI
3.  **Predict:** Sweetness Model วิเคราะห์สีและคำนวณค่าความหวาน
4.  **Display:** แสดงสถานะ **Ready** (พร้อมเก็บเกี่ยว) หรือ **Not Ready** ผ่าน Dashboard

<img width="985" height="452" alt="image" align="center" src="https://github.com/user-attachments/assets/d23b11b4-ad2a-4699-8c2e-8ef8d5c97a02" />

