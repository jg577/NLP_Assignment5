#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils



class Highway(nn.Module):
    """ Highway network:
        - Two linear layers
    """
    
    def __init__(self, embed_size, dropout_rate=0.3, device=None):
        """ Init Highway model
        
        @param embed_size: Embedding size of each word(dimensionality)
        @param dropout_rate: Dropout probability
        
        
        """
        super(Highway, self).__init__()
        self.embedding_size = embed_size
        self.dropout_rate = dropout_rate
        self.T = nn.Linear(in_features=self.embedding_size,
                           out_features= self.embedding_size,
                           bias=True)
        self.C = nn.Linear(in_features=self.embedding_size,
                           out_features=self.embedding_size,
                           bias=True)
        self.device = device
        
    # quite not sure if we want a tensor as an input, probably though
    #implement batches
    def forward(self, x_conv_out:torch.Tensor) -> torch.Tensor:
        """
        Computes one forward step of the Highway model.
        
        @param x_conv_out(Tensor): Batched tensor of shape(b, embedding_size) that comes
                    as an output from the convolution model.
        
        @returns x_word_emb(Tensor): Batched words tensor of shape (b, embedding_size)
            that will be used at the embedding vector for the word eventually for 
            rest of the model

        """    
        x_proj = F.relu(self.T(x_conv_out))
        x_gate = torch.sigmoid(self.C(x_conv_out))
        units = torch.ones(self.embedding_size, device=self.device)
        x_highway = torch.sum([torch.mul(x_gate, x_proj),torch.mul((units-x_gate),x_conv_out)])
        return x_highway
        


### END YOUR CODE 

