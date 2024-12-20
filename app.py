import os
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from model import FoodClassifier

app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('models/best_model.pth', map_location=device)
num_classes = len(checkpoint['class_names'])
model = FoodClassifier(num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

class_names = checkpoint['class_names']
temperature = checkpoint.get('temperature', 1.0)

# Define image transformations
transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Read and preprocess image
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)
        
        augmented = transform(image=image_np)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            # Apply temperature scaling
            outputs = outputs / temperature
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        confidences, predictions = torch.topk(probabilities[0], 3)
        
        results = []
        for pred, conf in zip(predictions, confidences):
            results.append({
                'label': class_names[pred.item()],
                'confidence': conf.item() * 100
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        })

if __name__ == '__main__':
    app.run(debug=True)

print("Server is running. Access it at http://localhost:5000")

