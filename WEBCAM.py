import cv2
import torch
import timm
import json
import numpy as np
from PIL import Image
from torchvision import transforms

# Load the pretrained EfficientFormerV2 model
model_name = "efficientformerv2_s0"
model = timm.create_model(model_name, pretrained=True)
model.eval()

# Load class labels from synset_Values.json
with open('synset_Values.json', 'r') as f:
    class_labels = json.load(f)  # Assuming this is a list of synset values

# Load synset mapping from synset_map.txt
synset_map = {}
with open('synset_map.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            synset_map[parts[0]] = parts[1]  # Mapping synset value to class name

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to model's expected input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet
])

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
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
    cv2.imshow("Webcam Inference", frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
