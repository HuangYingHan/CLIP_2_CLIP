import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class MLP(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size,in_channels, embed_dim, dropout = 0 ):
        super().__init__()

        self.patch_embed = nn.Conv2D(in_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                                     bias_attr=False)
        
        # use patch size as kernel and stride to take the row tensor
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        # x: [n, embed_dim, h, w]
        x = x.flatten(2) # flatten in second dimension [n, embed_dim, h*w]
        x = x.transpose([0, 2, 1]) # [n, h * w, embed_dim]
        x = self.dropout(x)
        return x
    
class Encoder(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = Identity()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x= h + x

        h = x
        x = self.mlp_norm(x)
        x = self.attn(x)
        x = h + x
        return x

class ViT(nn.Layer):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)
        layer_list = [Encoder(16) for i in range(5)]
        self.encoders = nn.LayerList(layer_list)
        self.head = nn.Linear(16, 10)
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        # layer norm
        # [n, h*w, em_bed]
        x = x.transpose([0, 2, 1])
        x = self.avgpool(x) # [n, c, 1]
        x = x.flatten(1) #[n ,c]
        x = self.head(x)
        return x 

def main():
    # img = Image.open('./data/7.jpg')
    # img = np.array(img, dtype='float32')

    # print(img.size)
    
    # sample = paddle.to_tensor(img)
    # sample = sample.reshape([1, 3, 290, 290])


    # # patch embedding
    # patch_embed = PatchEmbedding(image_size=290, patch_size = 29, in_channels=3, embed_dim = 1)
    # out = patch_embed(sample)
    # print(out.shape)

    # mlp = MLP(1)
    # out = mlp(out)
    # print(out.shape)

    t = paddle.randn([4, 3, 224, 224])
    model = ViT()
    out = model(t) 
    print(out.shape)
    
if __name__ == '__main__':
    main()