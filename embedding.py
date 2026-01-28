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
#model = torchvision.models.vit_b_16(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])

# 设置为评估模式
model.eval()

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像尺寸为224x224
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载单张图像

image_folder = 'rsdata'  # 替换为你的图片文件夹路径
output_csv = 'image_embeddings04271_Vit.csv'  # 输出CSV文件路径


embeddings = []

cnt = 0
# 遍历文件夹中的所有图片
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    cnt+=1
    # 仅处理图像文件
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # 加载图像并进行预处理
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)  # 转换为Tensor并增加批次维度
            # 提取特征（嵌入）
            with torch.no_grad():
                embedding = model(img)  # 通过模型提取特征
                embedding = embedding.squeeze()  # 移除批次维度，变为一维向量

            # 将图片文件名和特征向量添加到嵌入列表中
            embedding_array = embedding.numpy()
            embeddings.append([img_name] + embedding_array.tolist())  # 存储文件名和嵌入向量
            print(cnt)

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

# 提取特征（嵌入）

columns = ['Image Name'] + [f'Feature_{i}' for i in range(embedding_array.shape[0])]
embedding_df = pd.DataFrame(embeddings, columns=columns)

# 将嵌入结果保存到 CSV 文件
embedding_df.to_csv(output_csv, index=False)

print(f"Embeddings saved to {output_csv}")