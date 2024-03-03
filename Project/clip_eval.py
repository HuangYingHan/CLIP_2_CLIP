import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import json
from model.clip import CLIP
from model import models
from utils.ModelData import CLipDataset, dataset_collate
from utils.metrics import itm_eval

if __name__ == "__main__":
    datasets_path = "/home/yinghanhuang/Dataset/self_clip/"
    datasets_val_json_path = "/home/yinghanhuang/Dataset/self_clip/all_data.json"
    batch_size = 32
    num_workers = 4

    # model = CLIP(
    #     embed_dim=512,
    #     # image
    #     image_resolution=224,
    #     vision_layers=12,
    #     vision_width=768,
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
    model, preprocess = models.load("ViT-B/32", device=device)

    input_shape = [224, 224];
    
    val_dataset = CLipDataset(input_shape, False, lines, False)
    gen_val     = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=False, collate_fn=dataset_collate, sampler=None)
    
    i_features = []
    t_features = []

    for iteration, batch in tqdm(enumerate(gen_val)):
        images, texts = batch
        with torch.no_grad():
            if model.cuda:
                images  = images.cuda()
            
            images_feature = model.encode_image(images)
            i_features.append(images_feature)

    texts       = gen_val.dataset.text
    num_text    = len(texts)
    for i in tqdm(range(0, num_text, batch_size)):
        text = texts[i: min(num_text, i + batch_size)]
        with torch.no_grad():
            texts_feature = model.encode_text(texts=text)
            t_features.append(texts_feature)

    i_features = torch.cat(i_features, 0)
    t_features = torch.cat(t_features, 0)
    
    i_features  = i_features / i_features.norm(dim=-1, keepdim=True)
    t_features  = t_features / t_features.norm(dim=-1, keepdim=True)

    logits_per_image    = i_features @ t_features.t()
    logits_per_text     = logits_per_image.t()

    logits_per_image    = logits_per_image.cpu().numpy()
    logits_per_text     = logits_per_text.cpu().numpy()
    print(itm_eval(logits_per_image, logits_per_text, gen_val.dataset.txt2img, gen_val.dataset.img2txt))

