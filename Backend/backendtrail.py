from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.autograd import Variable
import segmentation_models_pytorch as smp
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app) 

def resize_if_needed(img):
    """ Resize image only if dimensions are not divisible by 32 """
    h, w, c = img.shape
    if h % 32 == 0 and w % 32 == 0:
        return img  # No resizing needed
    new_h = (h // 32 + 1) * 32
    new_w = (w // 32 + 1) * 32
    print(f"Resizing image from ({h}, {w}) to ({new_h}, {new_w})")
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Load model once to avoid reloading on every request
device = torch.device("cpu")  # Default to CPU
model = smp.Unet(in_channels=3, classes=1, activation="sigmoid")
model.load_state_dict(torch.load("unet_best_weights.pth", map_location=device))  # Ensure model file is present
model.to(device)
model.eval()

@app.route("/upload", methods=["POST"])
def segment_image():
    try:
        file = request.files['image']
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        image_np = np.array(Image.open(file).convert("RGB"))[:, :, ::-1]  # Convert to OpenCV format
        image_np = resize_if_needed(image_np)
        image_tensor = transforms.ToTensor()(image_np.copy())
        image_tensor = torch.unsqueeze(image_tensor, 0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
        
        out_img = output.cpu().data.squeeze().numpy()
        out_img = (1 - (out_img - out_img.min()) / (out_img.max() - out_img.min())) * 255
        out_img = Image.fromarray(out_img.astype(np.uint8))
        
        img_io = BytesIO()
        out_img.save(img_io, "JPEG")
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
