from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import torch
from model.model import mnist_model

app = Flask(__name__)
model = mnist_model(784, 128, 10)
model.load_state_dict(torch.load('../models/MNIST_model_revised.pth'))
model.eval()

def preprocess_image(image):
    ## Implement preprocessing here (e.g., resizing, normalizing)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    print(data)
    #image_data = data['image'].split(",")[1]
    #image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    #image = preprocess_image(image)
    #image_tensor = torch.tensor(image).unsqueeze(0)
    #with torch.no_grad():
    #    output = model(image_tensor)
    #    _, predicted = torch.max(output, 1)
    return jsonify({'digit': 69})

if __name__ == '__main__':
    app.run(debug=True)