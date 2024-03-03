import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model.clip import CLIP


if __name__ == "__main__":
    datasets_path = "/home/yinghanhuang/Dataset/self_clip/"
    datasets_val_json_path = "/home/yinghanhuang/Dataset/self_clip/"
    batch_size = 32
    num_workers = 4

    model = CLIP(
        embed_dim=512,
        # image
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        # text
        context_length=77,
        transformer_layers=12,
        transformer_width=768,
        transformer_heads=12
    )
