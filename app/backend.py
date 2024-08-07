from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as transforms
import numpy as np
from model import mnist_model

app = Flask(__name__)
device = torch.device('cpu')
model = mnist_model(784, 128, 10)
model.load_state_dict(torch.load('../models/MNIST_model_revised.pth', map_location=device))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_image(image):
    resize = transforms.Resize([28, 28])
    image = resize(image)
    image = torch.from_numpy(np.array(image))
    image = image[:, :, 3].flatten()
    image = (image/image.max()).unsqueeze(dim=0)
    return image

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = preprocess_image(image)
    with torch.no_grad():
        out = model(image)
        prob = out.max().item() * 100
        prob = f'{prob:.2f}%'
        predicted = torch.argmax(out)
    return jsonify({'digit': predicted.item(), 'prob': prob})

if __name__ == '__main__':
    app.run()
