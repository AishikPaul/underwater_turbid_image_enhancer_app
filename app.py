from flask import Flask, render_template, request, send_file
import os
from utils import enhance_image
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            image_bytes = uploaded_file.read()
            input_img, enhanced_rgb, enhanced_gray = enhance_image(image_bytes)

            # Save input temporarily for preview (but delete later if you want)
            cv2_input = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("static/input_preview.jpg", cv2_input)

            return render_template('index.html', result=True)
    return render_template('index.html', result=False)

@app.route('/download/<version>')
def download(version):
    if version == 'rgb':
        path = 'static/output_rgb.png'
    else:
        path = 'static/output_gray.png'
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
