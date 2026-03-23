import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    """
    Configurable multi-layer perceptron (MLP) neural network
    Applicable to classification and regression tasks
    """

    def __init__(self, input_dim, num_classes, hidden_dims=None,
                 dropout_rates=None, activation='LeakyReLU',
                 use_normalization=True, use_input_layernorm=False,
                 task_type='classification'):
        super(NeuralNet, self).__init__()

        self.task_type = task_type
        self.input_layernorm = nn.LayerNorm(
            input_dim) if use_input_layernorm else None

        if len(dropout_rates) != len(hidden_dims):
            dropout_rates = dropout_rates * len(hidden_dims)
            dropout_rates = dropout_rates[:len(hidden_dims)]
        # select activation function by name (and provide reasonable default parameters)
        activation_map = {
            'ReLU': (nn.ReLU, {}),
            'LeakyReLU': (nn.LeakyReLU, {'negative_slope': 0.05}),
            'ELU': (nn.ELU, {'alpha': 1.0}),
            'GELU': (nn.GELU, {}),
            'Swish': (nn.SiLU, {}),
            'Mish': (nn.Mish, {})
        }
        act_cls, act_kwargs = activation_map.get(
            activation, (nn.LeakyReLU, {'negative_slope': 0.05}))

        # build network layers
        layers = []
        prev_dim = input_dim

        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_normalization:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(act_cls(**act_kwargs))
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        # output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # optional: apply LayerNorm to input (sample level)
        if self.input_layernorm is not None:
            x = self.input_layernorm(x)

        x = F.normalize(x, p=2, dim=1)

        return self.net(x)

    def cal_loss(self, pred, target, criterion):
        return criterion(pred, target)


# 1) use MHA to build Encoder block, return attn for each head in each layer

class MHAEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.act = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None, key_padding_mask=None,
                need_weights=False, average_attn_weights=False):
        # self-attention
        attn_out, attn_w = self.mha(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights
        )
        # attn_w: None (if need_weights=False) or (B,H,L,L) (if average_attn_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))

        ff = self.linear2(self.dropout2(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout3(ff))
        return x, attn_w  # return attn_w (None or with values)

# 2)  keep input/output and MLP, only replace Encoder


class TransformerMLPModel(nn.Module):
    def __init__(self, input_dim, num_heads, transformer_dim, num_layers,
                 seq_length, num_classes,
                 hidden_dims=None, dropout_rates=None,
                 task_type='classification', return_attn_weights=False,
                 transformer_dropout=0.1, transformer_activation='relu'):
        super().__init__()
        self.seq_length = seq_length
        self.return_attn_weights = return_attn_weights
        self.task_type = task_type
        self.num_heads = num_heads
        self.transformer_dropout = transformer_dropout

        # stack of MHA encoder blocks
        self.layers = nn.ModuleList([
            MHAEncoderBlock(
                d_model=seq_length,
                nhead=num_heads,
                dim_feedforward=transformer_dim,
                dropout=transformer_dropout,
                activation=transformer_activation
            ) for _ in range(num_layers)
        ])

        # MLP
        self.net = NeuralNet(
            input_dim=seq_length,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation='LeakyReLU',
            use_normalization=True,
            use_input_layernorm=False,
            task_type=task_type
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None, return_embedded=False, return_debug=False):
        # x: (B, seq_len) contains 0/1 bits
        # e.g. x[0] = [1, 0, 1, 0, ...]
        B, L = x.size()  # L is seq_len

        # 1. build a L×L identity matrix, each column is the one-hot encoding of the corresponding index
        eye = torch.eye(L, device=x.device, dtype=torch.float32)   # (L, L)

        # 2. adjust dimensions, so broadcasting can align
        eye = eye.unsqueeze(0)      # (1, L, L)
        bits = x.unsqueeze(-1)      # (B, L, 1)  convert (B, L) to (B, L, 1)

        # 3. multiply column by column: bit = 1 → keep the column one-hot; bit = 0 → all 0
        expanded = bits * eye       # (B, L, L) = (64, 167, 167)

        embedded = expanded

        # each layer: (B, H, L, L) (B = batch size, H = number of heads, L = sequence length)
        attn_all_layers = []
        debug_layer_outputs = [] if return_debug else None

        for li, block in enumerate(self.layers):
            embedded, attn_w = block(
                embedded,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=self.return_attn_weights,
                average_attn_weights=False  # key: get attn for each head
            )
            if self.return_attn_weights and attn_w is not None:
                # print(attn_w.shape)  # (B,H,L,L) = (64,8,167,167)
                # new PyTorch: (B,H,L,L); if old version get (B*H,L,L), can view here
                if attn_w.dim() == 3:
                    L1, L2 = attn_w.shape[-2:]
                    attn_w = attn_w.view(B, self.num_heads, L1, L2)
                attn_all_layers.append(attn_w)
            if return_debug:
                debug_layer_outputs.append({
                    'layer_index': li,
                    'post_block_mean': embedded.mean().detach().cpu().item(),
                    'post_block_std': embedded.std().detach().cpu().item()
                })

        # if self.return_attn_weights:
        #     assert len(attn_all_layers)==len(self.layers) and all(A.dim()==4 and A.shape[1]==self.num_heads and torch.allclose(A.sum(-1), torch.ones(A.shape[:3], device=A.device), atol=1e-5) for A in attn_all_layers), "attn_all_layers is incomplete (number of layers/heads/sum check failed)"

        # average pooling + MLP
        pooled = embedded.mean(dim=1)
        # print(pooled.shape) # (B, embedding_dim) = (64, 167)
        out = self.net(pooled)
        # print(out.shape) # (B, num_classes) = (64, 1)

        if self.return_attn_weights:
            # last layer head-avg, suitable for direct LxL heatmap
            last_headavg = attn_all_layers[-1].mean(dim=1)  # (B, L, L)
            if return_embedded:
                if return_debug:
                    # also return embedded and debug info
                    return out, last_headavg, attn_all_layers, embedded, {
                        'embedded_linear': expanded,
                        'debug_layers': debug_layer_outputs
                    }
                # attn_all_layers: [each layer (B,H,L,L)]
                return out, last_headavg, attn_all_layers, embedded
            if return_debug:
                return out, last_headavg, attn_all_layers, {
                    'embedded_linear': expanded,
                    'debug_layers': debug_layer_outputs
                }
            return out, last_headavg, attn_all_layers
        else:
            if return_embedded:
                if return_debug:
                    return out, embedded, {
                        'embedded_linear': expanded,
                        'debug_layers': debug_layer_outputs
                    }
                return out, embedded
            if return_debug:
                return out, {
                    'embedded_linear': expanded,
                    'debug_layers': debug_layer_outputs
                }
            return out


def init_weights(m):
    """Weight initialization function, using xavier initialization (random seed is set globally)"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # bias initialization to 0
