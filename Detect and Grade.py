import cv2
import numpy as np
from ultralytics import YOLO
import os

# 1. Load your newly trained custom model
model = YOLO("runs/detect/fabric_defect_model/weights/best.pt")

def inspect_fabric(image_path):
    # Read image to get width and height
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image at {image_path}")
        return
    
    height, width, _ = img.shape
    total_img_area = height * width

    # Run AI inference on the image
    results = model(image_path)[0]
    
    total_defect_area = 0
    defects_found = []

    # Calculate defect surface area from bounding boxes
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_area = (x2 - x1) * (y2 - y1)
        total_defect_area += box_area
        
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        defects_found.append(cls_name)

    # Defect area ratio
    defect_percentage = (total_defect_area / total_img_area) * 100

    # Custom Industrial Quality Grading Logic
    if len(defects_found) == 0:
        grade = "GRADE A (PASS)"
        color = (0, 255, 0)  # Green
    elif defect_percentage < 1.0 and "hole" not in defects_found:
        grade = f"GRADE B (WARNING - {defect_percentage:.2f}%)"
        color = (0, 255, 255)  # Yellow
    else:
        grade = f"GRADE C (REJECTED - {defect_percentage:.2f}%)"
        color = (0, 0, 255)  # Red

    # Print Inspection Report to Terminal
    print("\n" + "="*45)
    print(f"📋 FABRIC INSPECTION REPORT: {os.path.basename(image_path)}")
    print("="*45)
    print(f"Defects Detected : {defects_found if defects_found else 'None'}")
    print(f"Defect Area      : {defect_percentage:.2f}% of total frame")
    print(f"Quality Rating   : {grade}")
    print("="*45 + "\n")

    # Draw result overlay on the image
    annotated_img = results.plot()
    cv2.putText(annotated_img, f"Status: {grade}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    output_path = "inspection_result.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"📸 Saved visual inspection result to '{output_path}'")

if __name__ == "__main__":
    # Grab the first image from your test folder automatically
    test_folder = "dataset/test/images"
    if os.path.exists(test_folder) and os.listdir(test_folder):
        first_test_img = os.path.join(test_folder, os.listdir(test_folder)[0])
        inspect_fabric(first_test_img)
    else:
        print("Please place a test image in dataset/test/images/")