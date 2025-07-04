![Tensor](https://img.shields.io/badge/TensorFlow.js-purple)
![Python](https://img.shields.io/badge/Python-3.88+-blue)
![Html](https://img.shields.io/badge/HTML-ux)
![Js](https://img.shields.io/badge/JavaScript-orange)
![Web](https://img.shields.io/badge/Browser-red)
![Windows](https://img.shields.io/badge/Windows-green)
![Android](https://img.shields.io/badge/Android-ux)
# ArtFlow
Diffussion image generator on TensorFlow.js
## Usage & Installation
1. Cloning the repository:
```bash
git clone https://ImHacker890-890/ArtFlow
cd ArtFlow
```
2. Install requirements.txt:
```bash
pip install -r requirements.txt
```
3. Load Coco:
```bash
python coco_load.py
```
4. Train the model:
```bash
python train.py
```
5. Convert to TensorFlow.js:
```bash
pip install tensorflowjs
tensorflowjs_converter --input_format keras diffusion_model.h5 tfjs_model
```
6. Launch TensorFlow.js:

Launch in browser file main.html
