import paddle
import paddle.nn as nn

class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride = patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x) # [n, embed_dim, h', w']
        x = x.flatten(2)  #[n, embed_dim, h'*w']
        x = x.transpose([0, 2, 1])
        x = self.norm(x)
        return x


class PatchMerge(nn.Layer):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape

        x = x.reshape([b, h, w, c])

        x_0 = x[:, 0::2, 0::2, :]
        x_1 = x[:, 1::2, 0::2, :]
        x_2 = x[:, 0::2, 1::2, :]
        x_3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x_0, x_1, x_2, x_3], -1) # [B, h/2, w/2, 4c]

        x = x.reshape([b,-1,4 * c])
        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class MLP(nn.Layer):
    def __init__(self, dim, mlp_ratio = 4., dropout = 0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B , H // window_size, window_size, W // window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # [B , H//WC, W//WC, ws, ws, c]
    x = x.reshape([-1, window_size, window_size, C])
    # [B * num_patches, ws, ws, c]
    return x

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] // (H / window_size * W / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x

class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.qkv = nn.Linear(dim ,dim * 3)
        self.proj = nn.Linear(dim ,dim)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])  # [B, num_heads, num_patches]
        return x

    def forward(self, x):
        # x : [B, num_patches, embed_dim]
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)

        out = paddle.matmul(attn, v)  # [B, num_heads, num_patches, dim_head]
        out = out.transpose([0, 2, 1, 3]) # [B, num_patches, num_heads, dim_head]
        out = out.reshape([B, N, C])
        return out
    

class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        H, W = self.resolution
        B, N, C = x.shape

        h= x
        x = self.attn_norm(x)

        x = x.reshape([B, H, W, C])
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])

        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        x= window_reverse(attn_windows, self.window_size, H, W)
        # [B, H, W, C]
        x = x.reshape([B, H * W, C])
        x = self.attn(x)

        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x


def main():
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(4, 96)
    swin_block = SwinBlock(dim=96, input_resolution=(56, 56), num_heads=4, window_size=7)
    patch_merging = PatchMerge( input_resolution=(56, 56), dim = 96)

    out = patch_embedding(t) # [ 4, 56, 56, 96]
    print("patch embedding output = ", out.shape)
    out = swin_block(out)
    print("swin block output = ", out.shape)
    out = patch_merging(out)
    print("patch merging output = ", out.shape)

if __name__ == "__main__":
    main()