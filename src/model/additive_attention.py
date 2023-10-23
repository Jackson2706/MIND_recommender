import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, query_dim: int, embed_dim: int):
        '''
            Additive attention initialization

            Args:
                query_dim: the dimension of the additive attention query vector
                embed_dim: the dimension of the "embedding"
        '''
        super().__init__()
        self.projection = nn.Linear(in_features=embed_dim, out_features=query_dim)
        self.query_vector = nn.Parameter(nn.init.xavier_uniform_(torch.empty(query_dim,1), 
                                                                 gain= nn.init.calculate_gain('tanh')).squeeze())
        
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        '''
        Forward propagation

        Args:
            embedings: tensor of shape(batch_size, seq_length, embed_dim)
            mask: tensor of shape(batch_size, seq_length). Position with True are allowed to attend
        
        Returns:
            Tensor of shape(batch_size, embed_dim)
        '''

        attnt_weight = torch.matmul(torch.tanh(self.projection(embeddings)), self.query_vector)
        attnt_weight.masked_fill_(~mask, 1e-30)
        attnt_weight = f.softmax(attnt_weight, dim = 1)
        seq_repr = torch.bmm(attnt_weight.unsqueeze(dim=1), embeddings).squeeze(dim=1)
        return seq_repr
        


