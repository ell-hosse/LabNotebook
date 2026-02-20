import matplotlib.pyplot as plt
import numpy as np
import requests
import torch

from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
   print("Loading CLIP model...")
   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
   return model, processor


def get_sample_images():
   urls = [
      "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",  # Cat
      "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Polar_Bear_-_Alaska_%28cropped%29.jpg/1200px-Polar_Bear_-_Alaska_%28cropped%29.jpg",
      # Polar Bear
      "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Hot_dog_with_mustard.png/1200px-Hot_dog_with_mustard.png",
      # Hot dog
      # ... more images ...
   ]

   images = []
   headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
   }
   for url in urls:
      try:
         response = requests.get(url, headers=headers, stream=True)
         response.raise_for_status()
         img = Image.open(BytesIO(response.content)).convert("RGB")
         images.append(img)
      except Exception as e:
         print(f"Error downloading {url}")

   return images


def get_image_embeddings(model, processor, images):
   # 1. Process images (resize, normalize)
   inputs = processor(images=images, return_tensors="pt", padding=True)

   # 2. Pass through model to get features
   with torch.no_grad():  # We don't need gradients for inference, saves memory
      image_features = model.get_image_features(**inputs)

   # 3. Normalize embeddings
   # This makes comparing them easier (cosine similarity)
   image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
   return image_features.numpy().shape




video_path = r"D:\hf\hub\datasets--turing-motors--CoVLA-Dataset\snapshots\0a6d39e41659903a26dde957744e70dbc360bb6d\videos\0000b7dc6478371b.mp4"

model, processor = load_clip_model()
images = get_sample_images()

print(get_image_embeddings(model, processor, images))