from pycocotools.coco import COCO
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import requests
import tarfile


def download_coco(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    url = "http://images.cocodataset.org/zips/train2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    
    if not os.path.exists(f"{data_dir}/train2017.zip"):
        print("Downloading COCO images...")
        r = requests.get(url, stream=True)
        with open(f"{data_dir}/train2017.zip", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    
    
    if not os.path.exists(f"{data_dir}/annotations.zip"):
        print("Downloading COCO annotations...")
        r = requests.get(ann_url, stream=True)
        with open(f"{data_dir}/annotations.zip", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    
    
    if not os.path.exists(f"{data_dir}/train2017"):
        print("Extracting images...")
        with tarfile.open(f"{data_dir}/train2017.zip") as tar:
            tar.extractall(data_dir)
    
    if not os.path.exists(f"{data_dir}/annotations"):
        print("Extracting annotations...")
        with tarfile.open(f"{data_dir}/annotations.zip") as tar:
            tar.extractall(data_dir)


def load_coco_data(data_dir="data", max_samples=5000, img_size=128):
    download_coco(data_dir)
    coco = COCO(f"{data_dir}/annotations/instances_train2017.json")
    coco_caps = COCO(f"{data_dir}/annotations/captions_train2017.json")
    
    
    img_ids = coco.getImgIds()[:max_samples]
    images = []
    texts = []
    
    for img_id in img_ids:
      
        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{data_dir}/train2017/{img_info['file_name']}"
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0  # [-1, 1]
        images.append(img)
        
        
        ann_ids = coco_caps.getAnnIds(imgIds=img_id)
        anns = coco_caps.loadAnns(ann_ids)
        texts.append(np.random.choice([ann['caption'] for ann in anns]))
    
    return tf.stack(images), texts


images, texts = load_coco_data()
print(f"Download {len(images)} images")
print(f"Text Example: {texts[0]}")
plt.imshow((images[0].numpy() + 1) / 2)  # Денормализация для отображения
plt.title(texts[0])
plt.show()
