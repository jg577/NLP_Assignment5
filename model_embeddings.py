#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from highway import Highway
from cnn import CNN


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.vocab = vocab
        self.cnn_layer = CNN(filters=embed_size)
        self.highway_layer = Highway(embed_size=self.embed_size)
        self.dropout_layer = nn.Dropout(p=0.3)
        pad_token_idx = vocab.char2id['<pad>']
        self.embedding = nn.Embedding(len(vocab.char2id),self.embed_size, padding_idx=pad_token_idx)


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        sentence_length, batch_size, max_word_length = list(input.shape)
        embeddings = self.embedding(input)
        new_view = embeddings.view([sentence_length*batch_size, max_word_length, self.embed_size])
        x_conv = self.cnn_layer(new_view)
        x_highway = self.highway_layer(x_conv)
        x_word_emb = self.dropout_layer(x_highway)
        x_word_view = x_word_emb.view([sentence_length, batch_size, self.embed_size])
        
        return x_word_view
        
        
        
        
        


        ### END YOUR CODE

