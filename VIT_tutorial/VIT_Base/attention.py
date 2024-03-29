import paddle
import paddle.nn as nn

class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, qkv_bias, qk_scale=None, dropout=0., attention_dropout=False):
        # num head: number of multihead attention,here 4 as it is a 4 *4 patch, attention to a path itself
        # number of pathches: number of tokens
        # num_head * num_patch = embed_dim
        super().__init__()
        self.embed_dim = embed_dim
        # multi head but keep the channel number the same (embed_dim the same)
        self.head_dim = int (embed_dim / num_heads)
        self.num_heads = num_heads

        self.all_head_dim = self.head_dim * num_heads

        # qkv combine together
        # all_head_dim 
        # *3 q, k, v
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if qkv_bias is False else None)
        
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale

        self.softmax = nn.Softmax(-1) # for last dimension

        self.proj = nn.Linear(self.all_head_dim, embed_dim)

    def transpose_multi_head(self, x):
        # N is numebr of patchs
        # x: [B, N, all_head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # x: [B, N, num_heads, head_dim]
        x = x. transpose([0, 2, 1, 3])
        # x: [B, number_heads, N, head_dim]
        return x


    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        # qkv [B, N ,all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv) # map: give all qkv into function and return

        # q,k,v are all [B, number_heads, N, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True) # q * k ^t
        attn = self.scale * attn
        attn = self.softmax(attn)

        # dropout
        # attn: [B , num_heads, N, N]

        out = paddle.matmul(attn, v) # softmax(scale*(q*k')) *v
        out = out.transpose([0, 2, 1, 3])
        # out [B, num_patches, num_heads, num_patches],
        out = out.reshape([B, N, -1])
        out = self.proj(out)
        return out




def main():
    t = paddle.randn([8, 16, 96]) # img tokens after patch embedding, 16 is 4 *4 patch , 96 is feature, each patch calculate the attention
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qk_scale=None)
    print(model)
    out = model(t)
    print(out.shape)


if __name__ == "__main__":
    main()
