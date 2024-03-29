import os
import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .util import preprocess_input


class CLipDataset(data.Dataset):
    def __init__(self, input_shape, random, lines, autoaugment_flag=True):
        self.input_shape = input_shape
        self.random = random
        self.lines = lines

        self.text = []
        self.image_path = []
        self.txt2img = {}
        self.img2txt = {}
        txt_id = 0

        for img_id, (img_path, desc) in enumerate(self.lines.items()):
            self.image_path.append(img_path)
            self.img2txt[img_id] = [txt_id]
            self.txt2img[txt_id] = img_id
            caption = desc["description"]
            self.text.append(caption)
            txt_id += 1
        # for image_path, desc in self.lines.items():
        #     self.image_path.append(image_path)
        #     caption = desc["description"]
        #     self.text.append(caption)
        #     self.img2txt[image_path] = caption
        #     self.txt2img[caption] = image_path

        self.autoaugment_flag = autoaugment_flag

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        photo_path = self.image_path[index]
        caption = self.text[index]
        
        image = Image.open(photo_path).convert('RGB')

        # if self.autoaugment_flag:
        image = self.get_random_data(image, self.input_shape, random=self.random)
        image = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))

        return image, caption

    def get_random_data(self, image, input_shape, random=True):
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

        return image_data

def dataset_collate(batch):
    images      = []
    captions    = []
    for image, caption in batch:
        images.append(image)
        captions.append(caption)
        
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    return images, captions

