from ultralytics import YOLO

# 1. Load your neural network
model = YOLO("yolov8n.pt")

print("Running AI Detection...")

# 2. Give it a test image to inspect (this uses a famous test photo of a bus and people)
results = model.predict("https://ultralytics.com/images/bus.jpg", save=True)

print(" Inspection Complete! Look inside your project folder for the 'runs' folder to see the image!")
