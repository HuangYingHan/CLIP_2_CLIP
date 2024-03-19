from PIL import Image
import requests
import clip
import torch
import os
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch.nn.functional as F
import json
from sklearn.metrics import precision_score, recall_score
import faiss
from model.models import tokenize


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/CLIP/weight/openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/CLIP/weight/openai/clip-vit-large-patch14")

    clip_model.eval()

    json_data = "/home/yinghanhuang/Dataset/self_clip/positive_and_negative_image_search_path_data.json"


    with open(json_data, 'r') as f:
        lines = json.load(f)

    index = faiss.IndexFlatL2(clip_model.projection_dim)
    for _, (idx, img_path) in enumerate(lines.items()):

        target_image = processor(images=Image.open(img_path), return_tensors="pt").to(device)

        with torch.no_grad():
            target_image_features = clip_model.get_image_features(**target_image)
            target_image_features /= target_image_features.norm(dim=-1, keepdim=True)
            target_image_features = target_image_features.cpu().numpy()

            index.add(target_image_features)

    source_data_folder = "/home/yinghanhuang/Dataset/for_search/"
    for folder in os.listdir(source_data_folder):
        folder_path = os.path.join(source_data_folder, folder)
        if os.path.isdir(folder_path):
            count = 0
            correct_count = 0
            for image_name in os.listdir(folder_path):
                if not image_name.endswith(".jpg.cat"):
                    img_path = os.path.join(folder_path, image_name)
                    with torch.no_grad():
                        source_data = processor(images=Image.open(img_path), return_tensors="pt").to(device)
                        source_data_features = clip_model.get_image_features(**source_data)
                        source_data_features /= source_data_features.norm(dim=-1, keepdim=True)
                        source_data_features = source_data_features.cpu().numpy()

                    D, I = index.search(source_data_features, k=1)

                    filenames = [[lines[str(j)] for j in i] for i in I]

                    if folder in filenames[0][0]:
                        correct_count += 1
                    count += 1

            print(f"{folder} accuracy: {correct_count / count}")