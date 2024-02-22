import copy
import paddle
import paddle.nn as nn 

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
    def __init__(self, image_size=224, patch_size=16,in_channels=3, embed_dim=768, dropout = 0 ):
        super().__init__()
        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2D(in_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                                     bias_attr=False)
        
        # use patch size as kernel and stride to take the row tensor
        
        self.dropout = nn.Dropout(dropout)

        # add class token
        self.class_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.)
        )

        # add distill token
        self.distill_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02)
        )


        # add pos embedding
        self.position_embedding = paddle.create_parameter(
            shape =[1, n_patches+1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02)
        )

    def forward(self, x):
        cls_tokens = self.class_token.expand([x.shape[0], 1, self.embed_dim]) # each class token for a sample, expand to same with batch

        distill_token = self.distill_token.expand([x.shape[0], 1, self.embed_dim])

        x = self.patch_embed(x)
        # x: [n, embed_dim, h, w]
        x = x.flatten(2) # flatten in second dimension [n, embed_dim, h*w]
        x = x.transpose([0, 2, 1]) # [n, h * w, embed_dim]

        x = self.dropout(x)
        x = paddle.concat([cls_tokens, distill_token, x], axis=1)

        x= x + self.position_embedding  # broadcast
        return x

class EncoderLayer(nn.Layer):
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

class Encoder(nn.Layer):
    def __init__(self, embed_dim, depth):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim)
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # return cls and distill tokens
        return x[:, 0], x[:, 1]

class DeiT(nn.Layer):
    def __init__(self, 
                 image_size = 224,
                 patch_size = 16,
                 in_channels = 3,
                 num_classes = 1000,
                 embed_dim = 768,
                 depth=3,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout = 0.,
                 attention_dropout = 0.,
                 droppath = 0.):
        super().__init__()
        self.depth = depth
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.encoder = Encoder(embed_dim, depth)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier_distill = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x [N,C, H, W]
        N, C, H, W = x.shape
        x = self.patch_embedding(x) # [N, embed_dim, h', w']
        # x = x.flatten(2) # [N, embed_dim, h'*w'] h'*w' = num_patches
        # x = x.transpose([0, 2, 1]) # [N, num_patches, embed_dim]
        x, x_distill = self.encoder(x) # x
        x_distill = self.classifier_distill(x)

        if self.training:
            return x, x_distill
        else:
            return (x + x_distill) /2
        x = self.classifier(x[:, 0])
        return x


def main():
    model = DeiT()
    print(model)
    paddle.summary(model, (4, 3, 224, 224)) # must be tuple

if __name__ == "__main__":
    main()