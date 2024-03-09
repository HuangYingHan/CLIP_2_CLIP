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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    taiyi_model = CLIPModel.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/openai/clip-vit-large-patch14").to(device)
    taiyi_processor = CLIPProcessor.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/CLIP/weight/openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/CLIP/weight/openai/clip-vit-large-patch14")
    
    taiyi_model.eval()
    clip_model.eval()

    chinese_query_texts = ["一张猫的照片", "一张狮子的照片", "一张阿富汗猎犬的照片", "一张萨路基的照片", "一张壁虎的照片", "一张变色龙的照片", "一张草书的照片", "一张篆书的照片", "一张锅包肉的照片", "一张糖醋里脊的照片"]
    english_query_texts = ["a photo of cat", "a photo of lion", "a photo of Afghan hound", "a photo of Saluki", "a photo of gecko", "a photo of chameleon", "a photo of cao shu", "a photo of zhuan shu", "a photo of guobao pork", "a photo of sweet and sour pork"]

    chinese_text_tokenizer = BertTokenizer.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
    chinese_text_encoder = BertForSequenceClassification.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
    chinese_text_encoder.to(device)
    chinese_text = chinese_text_tokenizer(chinese_query_texts, return_tensors='pt', padding=True)['input_ids'].to(device)
    
    english_text_tokenizer = BertTokenizer.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/CLIP/weight/textattack/bert-base-uncased-yelp-polarity/")
    # english_text_encoder = BertForSequenceClassification.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/CLIP/weight/textattack/bert-base-uncased-yelp-polarity/", problem_type="multi_label_classification").eval()
    english_text = english_text_tokenizer(english_query_texts, return_tensors='pt', padding=True)['input_ids'].to(device)

    json_data = "/home/yinghanhuang/Dataset/self_clip/all_classes_target_data.json"

    colltected ={}
    collected_img_path = []

    with open(json_data, 'r') as f:
        lines = json.load(f)

    for _, (img_path, target) in enumerate(lines.items()):
        chinese_image = taiyi_processor(images=Image.open(img_path), return_tensors="pt").to(device)
        english_image = processor(images=Image.open(img_path), return_tensors="pt").to(device)

        with torch.no_grad():
            chinese_image_features = taiyi_model.get_image_features(**chinese_image)
            english_image_features = clip_model.get_image_features(**english_image)
            chinese_image_features /= chinese_image_features.norm(dim=-1, keepdim=True)
            english_image_features /= english_image_features.norm(dim=-1, keepdim=True)

            chinese_text_features = chinese_text_encoder(chinese_text).logits
            chinese_text_features /= chinese_text_features.norm(dim=-1, keepdim=True)
            english_text_features = clip_model.get_text_features(english_text)
            english_text_features /= english_text_features.norm(dim=-1, keepdim=True)

            chinese_similarity = (100.0 * chinese_image_features @ chinese_text_features.T)
            chinese_probs = F.softmax(chinese_similarity, dim=-1)
            chinese_prediction = chinese_probs.argmax(dim=-1)

            english_similarity = (100.0 * english_image_features @ english_text_features.T)
            english_probs = F.softmax(english_similarity, dim=-1)
            english_prediction = english_probs.argmax(dim=-1)

            if (chinese_prediction == english_prediction):
                colltected[img_path] = target

    json_output_path = "/home/yinghanhuang/Dataset/self_clip/selected_data.json"

    with open(json_output_path, "w") as f:
        json.dump(colltected, f, indent=4)

if __name__ == '__main__':
    main()