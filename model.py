import math
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_gat, out_channel_conv_1, out_channel_conv_2, hidden_dim_fc_3,
                 num_classes, num_heads_gat, num_heads_transformer_encoder, num_transformer_encoders, dp,
                 negative_slope, num_nodes):
        super(MyModel, self).__init__()
        self.feature_proj = nn.Linear(input_dim, hidden_dim_gat)
        self.gat = GATConv(in_channels=input_dim, out_channels=hidden_dim_gat // num_heads_gat,
                           heads=num_heads_gat, negative_slope=negative_slope)
        self.relu_1 = nn.ReLU()
        self.dropout_gat = nn.Dropout(p=dp)
        self.max_seq_len = num_nodes
        self.pos_encoding = self._generate_positional_encoding(
            seq_len=self.max_seq_len,
            d_model=hidden_dim_gat
        )
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim_gat,
            nhead=num_heads_transformer_encoder,
            dim_feedforward=hidden_dim_gat * 4,
            dropout=dp,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_transformer_encoders
        )
        self.relu_2 = nn.ReLU()

        self.conv_1d_1 = nn.Conv1d(in_channels=1, out_channels=out_channel_conv_1, kernel_size=3, stride=1, padding=1)
        self.relu_3 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_1d_2 = nn.Conv1d(in_channels=out_channel_conv_1, out_channels=out_channel_conv_2, kernel_size=5, stride=1, padding=2)
        self.relu_4 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(out_channel_conv_2 * (hidden_dim_gat // 4), hidden_dim_gat)

        self.fc_3 = nn.Linear(hidden_dim_gat * 2, hidden_dim_fc_3)
        self.relu_5 = nn.ReLU()
        self.fc_4 = nn.Linear(hidden_dim_fc_3, num_classes)

    def _generate_positional_encoding(self, seq_len, d_model):
        pos_enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        pos_enc.requires_grad = False
        return pos_enc

    def forward(self, x, edge_index, batch):
        x_gat = self.gat(x, edge_index)
        x_gat = self.relu_1(x_gat)
        x_gat = self.dropout_gat(x_gat)
        g_local = gap(x_gat, batch)

        x_proj = self.feature_proj(x)
        batch_size = batch.max().item() + 1
        x_transformer = torch.zeros(batch_size, max_nodes, x_proj.shape[-1], device=x.device)
        start_idx = 0
        for i in range(batch_size):
            end_idx = start_idx + node_per_graph[i]
            x_transformer[i, :node_per_graph[i]] = x_proj[start_idx:end_idx]
            start_idx = end_idx
        x_transformer_with_pos = x_transformer + self.pos_encoding.to(x.device)
        x_trans = self.transformer_encoder(x_transformer_with_pos)
        x_trans = self.relu_2(x_trans)
        g_global = torch.mean(x_trans, dim=1)

        g = g_local + g_global

        g_conv = g.unsqueeze(1)
        g_conv = self.relu_3(self.conv_1d_1(g_conv))
        g_conv = self.max_pool_1(g_conv)
        g_conv = self.relu_4(self.conv_1d_2(g_conv))
        g_conv = self.max_pool_2(g_conv)
        g_conv = g_conv.view(g_conv.size(0), -1)
        g_conv = self.fc_1(g_conv)

        g_fused = torch.cat([g, g_conv], dim=1)

        g_fused = self.fc_3(g_fused)
        g_fused = self.relu_5(g_fused)
        o = self.fc_4(g_fused)

        return o
