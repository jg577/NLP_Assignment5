#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(hidden_size,len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size)
        self.target_vocab = target_vocab
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        X = self.decoderCharEmb(input)
        # import pdb; pdb.set_trace()
        dec_hidden, (h_t, c_t) = self.charDecoder(X,dec_hidden)
        s_t = self.char_output_projection(dec_hidden)
        return s_t, (h_t, c_t)
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        # import pdb; pdb.set_trace()
        splits = torch.split(char_sequence, split_size_or_sections=1, dim =0)
        loss = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        for i in range(len(splits)-1):
            j = i+1
            # import pdb; pdb.set_trace()
            s_t, dec_hidden = self.forward(splits[i],dec_hidden)
            s_t_sq = s_t.squeeze(0)
            # import pdb; pdb.set_trace()
            loss += loss_fn(s_t_sq, splits[j].reshape(-1))
        return loss
            

        
        
        

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        # Algorithm 1 Greedy Decoding
        # Input: Initial states h0, c0
        # Output: output word generated by the character decoder (doesn’t contain <START> or <END>)
        # 1: procedure decode greedy
        # 2: output word ← []
        # 3: current char ← <START>
        # 4: for t = 0, 1, ..., max length − 1 do
        # 5: ht+1, ct+1 ← CharDecoder(current char, ht, ct) . use last predicted character as input
        # 6: st+1 ← Wdecht+1 + bdec . compute scores
        # 7: pt+1 ← softmax(st+1) . compute probabilities
        # 8: current char ← argmaxc pt+1(c) . the most likely next character
        # 9: if current char=<END> then
        # 10: break
        # 11: output word ← output word + [current char] . append this character to output word
        # 12: return output word
        
        output_words = []
        h_t, c_t = initialStates
        batch_size = list(h_t.shape)[1]
        current_char_index = torch.tensor([[self.target_vocab.start_of_word]]*batch_size, device=device)
        current_char_index = current_char_index.permute([1,0])
        for t in range(0, max_length):
            embedding_chars = self.decoderCharEmb(current_char_index)
            output, (h_t, c_t) = self.charDecoder(embedding_chars, (h_t, c_t))
            p_val = nn.functional.softmax(output)
            current_char_index = torch.argmax(p_val, dim=-1, keepdim=False)
            output_words += [current_char_index]
        output_words = torch.cat(output_words, dim=0).t()
        edited_words = []
        for lis in output_words.tolist():
            if self.target_vocab.end_of_word in lis:
                end_point = lis.index(self.target_vocab.end_of_word)
            else:
                end_point = len(lis)
            edited_words.append(''.join([self.target_vocab.id2char[x] for x in lis[:end_point]]))
        return edited_words
        
        
            
        
        ### END YOUR CODE

