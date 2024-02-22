import numpy as np
from PIL import Image
import paddle
import paddle.vision.transforms as T

def crop(image, region):
    croopped_iamge = T,crop(image, *region)
    return croopped_iamge

class CenterCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        w, h = image.size()
        ch, cw = self.size
        crop_top = int (round(h - ch) / 2.)
        crop_left = int (round(w- cw) / 2.)
        return crop(image, (crop_top, crop_left, ch, cw))
class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        T.resize(image, self.size)
class ToTensor():
    def __init__(self):
        pass
    def __call__(self, image):
        w, h = image.size
        image = paddle.to_tensor(image)
        if image.dtype == paddle.uint8:
            image = paddle.cast(image, dtype="float32") / 255.
        image = image.transpose([2, 0, 1])
        return image

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

def main():
    img = Image.open("../VIT_Base/data/7.jpg")

    transforms = Compose([Resize([256, 256]),
                            CenterCrop([112, 112]),
                            ToTensor()])
    out = transforms(img)
    print(out)
    print(out.shape)

if __name__ == "__main__":
    main()