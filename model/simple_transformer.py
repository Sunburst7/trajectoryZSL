import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim:int, feature_dim:int, num_heads:int, num_layers:int, num_classes:int):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, feature_dim)  # 确保这里从128维映射到feature_dim
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(feature_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Rearrange input to [seq_length, batch_size, feature_dim]
        
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over the sequence
        x = self.dropout(x)
        x = self.fc_out(x)
        return x