import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import json
from model.clip import CLIP
from model import models
from utils.ModelData import CLipDataset
from model.models import tokenize
from sklearn.metrics import precision_score, recall_score

if __name__ == "__main__":
    datasets_path = "/home/yinghanhuang/Dataset/self_clip/"
    datasets_val_json_path = "/home/yinghanhuang/Dataset/self_clip/inclass_meat2_target_data.json"
    batch_size = 32
    num_workers = 4
    prob_threshold= 0.8

    # model = CLIP(
    #     embed_dim=512,
    #     # image
    #     image_resolution=224,
    #     vision_layers=12,s
    #     vision_patch_size=32,
    #     # text
    #     context_length=77,
    #     vocab_size= 49408,
    #     transformer_layers=12,
    #     transformer_width=768,
    #     transformer_heads=12
    # )
    with open(datasets_val_json_path, 'r') as f:
            lines = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = models.load("ViT-L/14", device=device)

    input_shape = [224, 224];
    
    val_dataset = CLipDataset(input_shape, False, lines, False)
    data_loader     = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    total_correct, step = 0, 0.
    texts = tokenize(["a photo of sweet and sour pork"]).to(device)#,"a photo of Afghan hound", "a photo of Gecko", "a photo of cao shu", "a photo of Fried Sweet and Sour Tenderloin", "a photo of others"]).to(device)
    predict_labels = []
    true_labels = []
    for iteration, (data, target) in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            if model.cuda:
                data, target  = data.cuda(), target.cuda()
            
                image_feature = model.encode_image(data)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)

                text_feature = model.encode_text(text=texts)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)

                similarity = image_feature @ text_feature.T
                # probs = F.softmax(similarity, dim=-1)

                # pred = probs.argmax(dim = -1)
                # tar = target.argmax(dim = -1)


                # pred[probs.max(dim=-1).values < prob_threshold] = -1
                # predict_labels.append(pred.cpu().numpy())
                # true_labels.append(tar.cpu().numpy())

    # predict_labels = np.concatenate(predict_labels)
    # true_labels = np.concatenate(true_labels)
                predict = torch.ones_like(similarity)
                predict[similarity[:, -1] < 0.25] = -1

                # tar = target
                predict_labels.append(predict.cpu().numpy())
                # true_labels.append(tar.cpu().numpy())
    predict_labels = np.concatenate(predict_labels)
    # true_labels = np.concatenate(true_labels)

    # # # 计算真正例的数量
    true_positive = np.sum(predict_labels[predict_labels == 1])

    # # # 计算数量
    #total_num = np.sum(true_labels[true_labels == 1])
    print(true_positive)
    # print(total_num)
    # # # 计算精确度
    recall = true_positive / 200
    # print(recall)
    # precision = precision_score(true_labels, predict_labels, average='macro')
    # recall = recall_score(true_labels, predict_labels, average='macro')

    # # Print the precision and recall
    # print("Precision:", precision)
    print("Recall:", recall)