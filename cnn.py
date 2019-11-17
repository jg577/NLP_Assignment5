#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

### YOUR CODE HERE for part 1e
class CNN(nn.Module):
    
    def __init__(self, filters=50, kernel_size=5, max_words=21):
        """
        Instantiates a one layer convolution layer.
        
        @params kernel_size: number of kernel sizes
        """
        super(CNN, self).__init__()
        self.k = kernel_size
        self.m_word = max_words
        self.f = filters
        self.conv1d = nn.Conv1d(in_channels=self.m_word,
                               out_channels=self.f,
                               kernel_size=self.k)
    
    def forward(self, x_reshaped:torch.Tensor) -> torch.Tensor:
        """ Computes the CNN to the next step
        @params x_reshaped: Reshaped tensor of shape (max_sentence_length, 
                                            batch_size, max_word_length, embed_size)
        
        @returns x_conv_out: Batched tensor of shape(b, embedding_size)
        """
        x_conv = self.conv1d(x_reshaped)
        x_conv_out = torch.max(F.relu(x_conv), 2)[0]
        return x_conv_out
        

### END YOUR CODE

