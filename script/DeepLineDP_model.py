import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


# Model structure
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        """
        vocab_size: number of words in the vocabulary of the model
        embed_dim: dimension of word embeddings
        word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        word_gru_num_layers: number of layers in word-level GRU
        sent_gru_num_layers: number of layers in sentence-level GRU
        word_att_dim: dimension of word-level attention layer
        sent_att_dim: dimension of sentence-level attention layer
        use_layer_norm: whether to use layer normalization
        dropout: dropout rate; 0 to not use dropout
        """
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        self.fc = nn.Linear(2 * sent_gru_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, code_tensor):
        
        code_lengths = []
        sent_lengths = []

        for file in code_tensor:
            code_line = []
            code_lengths.append(len(file))
            for line in file:
                code_line.append(len(line))
            sent_lengths.append(code_line)
        
        code_tensor = code_tensor.type(torch.LongTensor)
        code_lengths = torch.tensor(code_lengths).type(torch.LongTensor).cuda()
        sent_lengths = torch.tensor(sent_lengths).type(torch.LongTensor).cuda()
        
        code_embeds, word_att_weights, sent_att_weights, sents = self.sent_attention(code_tensor, code_lengths, sent_lengths)

        scores = self.fc(code_embeds)
        final_scrs = self.sig(scores)

        return final_scrs, word_att_weights, sent_att_weights, sents

class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers, word_att_dim, use_layer_norm, dropout)

        # Bidirectional sentence-level GRU
        self.gru = nn.GRU(2 * word_gru_hidden_dim, sent_gru_hidden_dim, num_layers=sent_gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, code_tensor, code_lengths, sent_lengths):

        # Sort code_tensor by decreasing order in length
        code_lengths, code_perm_idx = code_lengths.sort(dim=0, descending=True)
        code_tensor = code_tensor[code_perm_idx]
        sent_lengths = sent_lengths[code_perm_idx]

        # Make a long batch of sentences by removing pad-sentences
        # i.e. `code_tensor` was of size (num_code_tensor, padded_code_lengths, padded_sent_length)
        # -> `packed_sents.data` is now of size (num_sents, padded_sent_length)
        packed_sents = pack_padded_sequence(code_tensor, lengths=code_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_sents.batch_sizes

        # Make a long batch of sentence lengths by removing pad-sentences
        # i.e. `sent_lengths` was of size (num_code_tensor, padded_code_lengths)
        # -> `packed_sent_lengths.data` is now of size (num_sents)
        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=code_lengths.tolist(), batch_first=True)

    
    
        # Word attention module
        sents, word_att_weights = self.word_attention(packed_sents.data, packed_sent_lengths.data)

        sents = self.dropout(sents)

        # Sentence-level GRU over sentence embeddings
        packed_sents, _ = self.gru(PackedSequence(sents, valid_bsz))

        if self.use_layer_norm:
            normed_sents = self.layer_norm(packed_sents.data)
        else:
            normed_sents = packed_sents

        # Sentence attention
        att = torch.tanh(self.sent_attention(normed_sents))
        att = self.sentence_context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val)

        # Restore as documents by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as documents by repadding
        code_tensor, _ = pad_packed_sequence(packed_sents, batch_first=True)

        # Compute document vectors
        code_tensor = code_tensor * sent_att_weights.unsqueeze(2)
        code_tensor = code_tensor.sum(dim=1)

        # Restore as documents by repadding
        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)

        # Restore the original order of documents (undo the first sorting)
        _, code_tensor_unperm_idx = code_perm_idx.sort(dim=0, descending=False)
        code_tensor = code_tensor[code_tensor_unperm_idx]

        word_att_weights = word_att_weights[code_tensor_unperm_idx]
        sent_att_weights = sent_att_weights[code_tensor_unperm_idx]

        return code_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # output (batch, hidden_size)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        embeddings: embeddings to init with
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        """
        self.embeddings.weight.requires_grad = freeze

    def forward(self, sents, sent_lengths):
        """
        sents: encoded sentence-level data; LongTensor (num_sents, pad_len, embed_dim)
        return: sentence embeddings, attention weights of words
        """
        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents.cuda())

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_words.batch_sizes

        # Apply word-level GRU over word embeddings
        packed_words, _ = self.gru(packed_words)

        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words.data)
        else:
            normed_words = packed_words

        # Word Attenton
        att = torch.tanh(self.attention(normed_words.data))
        att = self.context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val) # att.size: (n_words)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        # Compute sentence vectors
        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        att_weights = att_weights[sent_unperm_idx]

        return sents, att_weights