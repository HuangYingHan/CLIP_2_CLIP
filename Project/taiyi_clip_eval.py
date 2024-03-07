from PIL import Image
import requests
import clip
import torch
import os
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics import precision_score, recall_score

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/openai/clip-vit-large-patch14")
    clip_model.eval()

    query_texts = ["一张糖醋里脊的图片", "一张锅包肉的图片"] 
    text_tokenizer = BertTokenizer.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
    text_encoder = BertForSequenceClassification.from_pretrained("/home/yinghanhuang/Project/CLIP_2_CLIP/taiyi_clip/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
    text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']
    data_folder = "/home/yinghanhuang/Dataset/self_clip/"


    predict_labels = []
    true_labels = []
    image = None
    for folder in os.listdir(data_folder):
        if (folder.endswith(".json") == False):
            folder_path = os.path.join(data_folder, folder)
            for image_name in os.listdir(folder_path):
                    if (image_name.endswith(".jpg")):
                        image_path = os.path.join(folder_path, image_name)
                        animal_name = folder
                        if (animal_name == "Fried_Sweet_and_Sour_Tenderloin"):
                            image = clip_processor(images=Image.open(image_path), return_tensors="pt")
                            target = [1, 0]
                            target = np.asarray(target)
                        elif (animal_name == "Pot_bag_meat"):
                            image = clip_processor(images=Image.open(image_path), return_tensors="pt")
                            target = [0, 1]
                            target = np.asarray(target)
                        if (image is not None):
                            with torch.no_grad():
                                image_features = clip_model.get_image_features(**image)
                                text_features = text_encoder(text).logits
                                # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
                                image_features /= image_features.norm(dim=-1, keepdim=True)
                                text_features /= text_features.norm(dim=-1, keepdim=True)

                                logit_scale = clip_model.logit_scale.exp()
                                logits_per_image = logit_scale * image_features @ text_features.t()
                                # logits_per_text = logits_per_image.t()
                                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                                pred = probs.argmax()
                                tar = target.argmax()

                                # pred[probs.max(dim=-1).values < prob_threshold] = -1
                                predict_labels.append(pred)
                                true_labels.append(tar)

    predict_labels = np.asarray(predict_labels)
    true_labels = np.asarray(true_labels)

    precision = precision_score(true_labels, predict_labels, average='macro')
    recall = recall_score(true_labels, predict_labels, average='macro')

    # Print the precision and recall
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == '__main__':
    main()