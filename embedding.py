import torch
import torchvision.models
from torchvision import transforms, models
from torchgeo.models import swin_v2_t, Swin_V2_T_Weights
from torchgeo.models import resnet50, ResNet50_Weights
from torchgeo.models import FarSeg
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = resnet50(weights=ResNet50_Weights.SENTINEL2_MI_RGB_SATLAS)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = 'rsdata'
output_csv = 'image_embeddings04271_Vit.csv'

embeddings = []
cnt = 0

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    cnt += 1
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                embedding = model(img)
                embedding = embedding.squeeze()

            embedding_array = embedding.numpy()
            embeddings.append([img_name] + embedding_array.tolist())
            print(cnt)

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

columns = ['Image Name'] + [f'Feature_{i}' for i in range(embedding_array.shape[0])]
embedding_df = pd.DataFrame(embeddings, columns=columns)
embedding_df.to_csv(output_csv, index=False)

print(f"Embeddings saved to {output_csv}")
