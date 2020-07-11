import numpy as np
import torch
from torch import nn
import random
import math

class ClassificationTransformer(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, word_to_ix, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        '''
        :param word_to_ix: dictionary mapping words to unique indices
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        '''        
        super(ClassificationTransformer, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.vocab_size = len(word_to_ix)
        
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.positional_embedding = nn.Embedding(self.max_length, self.word_embedding_dim)

        # Multi-Heads
        self.k = [nn.Linear(self.hidden_dim, self.dim_k) for i in range(self.num_heads)]
        self.v = [nn.Linear(self.hidden_dim, self.dim_v) for i in range(self.num_heads)]
        self.q = [nn.Linear(self.hidden_dim, self.dim_q) for i in range(self.num_heads)]
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        self.F1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.F1rel = nn.ReLU()
        self.F2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)

        self.final = nn.Linear(self.hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        '''
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups. 

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        '''
        outputs = self.final_layer(self.feedforward_layer(self.multi_head_attention(self.embed(inputs))))
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        input_mask = ( inputs != 0 ).unsqueeze(1)
        embeddings = self.word_embedding(inputs) + self.positional_embedding(torch.LongTensor([i for i in range(self.max_length)]))
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        z = []
        for head in range(self.num_heads):
            currQ = self.q[head](inputs)
            currK = self.k[head](inputs)
            currV = self.v[head](inputs)
            currZ = self.softmax(torch.stack([torch.mm(currQ[i], currK[i].permute(1,0))/math.sqrt(self.dim_k)
                            for i in range(currK.shape[0])], dim=0))
            currZ = torch.stack([torch.mm(currZ[i], currV[i]) for i in range(currV.shape[0])], dim=0)
            z.append(currZ)
        
        z = self.attention_head_projection(torch.cat(tuple(z), dim=2))
        outputs = self.norm_mh(z + inputs)
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        out = self.F2(self.F1rel(self.F1(inputs)))
        outputs = self.norm_ffn(inputs + out)
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,1)
        """
        outputs = None
        inputs = inputs[:,0,:]
        outputs = self.sig(self.final(inputs))
        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
