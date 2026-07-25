from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    print("Starting Fresh Fabric Defect AI Model Training...")

    results = model.train(
        data="dataset/data.yaml",
        epochs=25,
        imgsz=640,
        batch=8,
        name="fabric_defect_v2"  # Changed name to trigger a fresh full run
    )

    print("Training Complete!")

if __name__ == "__main__":
    main()