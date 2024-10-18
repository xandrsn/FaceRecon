import os
from PIL import Image

import cv2
from matplotlib import pyplot as plt


dataset_path = 'data'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

plt.figure(figsize=(12, 12))
for idx, emotion in enumerate(os.listdir(train_path)):
    emotion_folder = os.path.join(train_path, emotion)
    
    image_name = os.listdir(emotion_folder)[0]
    image_path = os.path.join(emotion_folder, image_name)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(3, 3, idx + 1)
    plt.imshow(image_rgb)
    plt.title(f"{emotion} ({len(os.listdir(emotion_folder))} images)")
    plt.axis('off')
    
    with Image.open(image_path) as img:
        print(f"Emoción: {emotion}, Tamaño Carpeta: {len(os.listdir(emotion_folder))}, Tamaño Imagen: {img.size}, capas -> {img.layers}")

plt.tight_layout()
plt.show()
