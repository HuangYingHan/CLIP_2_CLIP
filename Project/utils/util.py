import math
from functools import partial

import re
import numpy as np
from PIL import Image

from .utils_aug import center_crop, resize

#---------------------------------------------------#
#   Resize input image
#---------------------------------------------------#
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

#---------------------------------------------------------#
#   Image to RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    

def preprocess_input(image):
    image /= 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
    return image