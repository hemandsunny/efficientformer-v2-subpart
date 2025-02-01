import cv2
import torch
import timm
import json
import numpy as np
from PIL import Image
from torchvision import transforms

# Load the pretrained EfficientFormerV2 model
model_name = "efficientformerv2_s0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(model_name, pretrained=True).to(device).eval().half()

# Load class labels from synset_Values.json
with open('synset_Values.json', 'r') as f:
    class_labels = json.load(f)

# Load synset mapping from synset_map.txt
synset_map = {}
with open('synset_map.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            synset_map[parts[0]] = parts[1]

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define GStreamer pipeline for Jetson Nano
GST_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
)

cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(image).unsqueeze(0).to(device).half()

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Get the predicted class label
    probabilities = torch.nn.functional.softmax(outputs, dim=-1)
    predicted_class_idx = probabilities.argmax().item()
    synset_value = class_labels[predicted_class_idx] if 0 <= predicted_class_idx < len(class_labels) else "Unknown"
    predicted_class = synset_map.get(synset_value, "Unknown")

    # Display the prediction on the frame
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Jetson Inference", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
