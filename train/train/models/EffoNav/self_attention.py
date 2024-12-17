import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class ChannelAttention(nn.Module):
	def __init__(self, in_planes=7, ratio=2):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.max_pool = nn.AdaptiveMaxPool1d(1)

		self.shareMLP = nn.Sequential(
			nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
			nn.ReLU(),
			nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# x=x.permute(0,2,1)
		avgout = self.shareMLP(self.avg_pool(x))
		maxout = self.shareMLP(self.max_pool(x))
		return self.sigmoid(avgout + maxout)
     
class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)

        self.ca = ChannelAttention(in_planes=seq_len, ratio=2)

        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])

        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]

        x=self.ca(x)*x # token attention

        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
