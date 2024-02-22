import paddle
import numpy as np
from PIL import Image

paddle.set_device('cpu')

def main():
    t = paddle.zeros([3, 3])
    # print(t)

    t = paddle.randn([5,3])
    # print(t)

    image = np.array(Image.open('./data/7.jpg'))

    t = paddle.to_tensor(image, dtype='float32')

    # print(type(t))
    # print(t.dtype)

    # print(t.transpose([1, 0]))

    t = paddle.randint(0 , 10, [5, 15])
    qkv = t.chunk(3, -1) #chunk last dim into 3
    print(type(qkv))
    q, k, v = qkv
    print(q)




if __name__ == "__main__":
    main()