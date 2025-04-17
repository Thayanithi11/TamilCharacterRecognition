from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import segmentation_models_pytorch as smp
from neuralnet import FSRCNN_model  # Local import without sys.path

app = Flask(__name__)
CORS(app)

# Load models once
device = torch.device("cpu")

# UNet (Binarization)
unet_model = smp.Unet(in_channels=3, classes=1, activation="sigmoid")
unet_model.load_state_dict(torch.load("unet_best_weights.pth", map_location=device))
unet_model.to(device)
unet_model.eval()

# FSRCNN (Enhancement)
fsrcnn_model = FSRCNN_model(scale=4)
fsrcnn_model.load_state_dict(torch.load("FSRCNN-x4.pt", map_location=device))
fsrcnn_model.eval()

# Store binarized image in memory
binarized_image_bytes = None


def resize_if_needed(img):
    h, w, c = img.shape
    if h % 32 == 0 and w % 32 == 0:
        return img
    new_h = (h // 32 + 1) * 32
    new_w = (w // 32 + 1) * 32
    print(f"Resizing image from ({h}, {w}) to ({new_h}, {new_w})")
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


@app.route("/upload", methods=["POST"])
def segment_image():
    global binarized_image_bytes
    try:
        file = request.files['image']
        if not file:
            return jsonify({"error": "No file provided"}), 400

        image_np = np.array(Image.open(file).convert("RGB"))[:, :, ::-1]
        image_np = resize_if_needed(image_np)
        image_tensor = transforms.ToTensor()(image_np.copy()).unsqueeze(0).to(device)

        with torch.no_grad():
            output = unet_model(image_tensor)

        out_img = output.cpu().data.squeeze().numpy()
        out_img = (1 - (out_img - out_img.min()) / (out_img.max() - out_img.min())) * 255
        out_img = Image.fromarray(out_img.astype(np.uint8)).convert("L")

        img_io = BytesIO()
        out_img.save(img_io, "JPEG")
        img_io.seek(0)

        # Store in memory for later enhancement
        binarized_image_bytes = img_io.getvalue()

        return send_file(BytesIO(binarized_image_bytes), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/enhance", methods=["GET"])
def enhance_image():
    global binarized_image_bytes
    try:
        if binarized_image_bytes is None:
            return jsonify({"error": "No binarized image available"}), 400

        # Load binarized image from memory
        bin_image = Image.open(BytesIO(binarized_image_bytes)).convert("RGB")
        bin_np = np.array(bin_image)
        bin_tensor = transforms.ToTensor()(bin_np).unsqueeze(0)

        with torch.no_grad():
            sr_tensor = fsrcnn_model(bin_tensor).squeeze(0).clamp(0, 1)

        sr_image = transforms.ToPILImage()(sr_tensor)
        img_io = BytesIO()
        sr_image.save(img_io, "JPEG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
