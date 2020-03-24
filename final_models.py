import torch
import torch.nn as nn


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, projection_dim, projection, embed_dim, hidden_dim, no_of_layers, dp_ratio, bidirectional, torch_hidden_size):
        super(Encoder, self).__init__()
        input_size = projection_dim if projection else embed_dim
        dropout = 0 if no_of_layers == 1 else dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                           num_layers=no_of_layers, dropout=dropout,
                           bidirectional=bidirectional)
        self.torch_hidden_size = torch_hidden_size
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.torch_hidden_size, batch_size, self.hidden_dim
        h0 = c0 = inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.bidirectional else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class RNN_3L_notfixed_notprojected_bi_2L(nn.Module):

    def __init__(self, vocab_length, num_classes, embed_dim=100, projection=False, projection_dim=300, dp_ratio=0.2, hidden_dim=300, bidirectional=True,  fix_emb=False, no_of_layers=3):
        super(RNN_3L_notfixed_notprojected_bi_2L, self).__init__()
        torch_hidden_size = no_of_layers
        if bidirectional:
            torch_hidden_size *= 2
        self.projection = projection
        self.fix_emb = fix_emb
        self.embed = nn.Embedding(vocab_length, embed_dim)
        self.projection_layer = Linear(embed_dim, projection_dim)
        self.encoder = Encoder(projection_dim, projection, embed_dim, hidden_dim,
                               no_of_layers, dp_ratio, bidirectional, torch_hidden_size)
        self.dropout = nn.Dropout(p=dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2*hidden_dim
        if bidirectional:
            seq_in_size *= 2
        lin_config = [seq_in_size]*2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, num_classes))

    def forward(self, batch):
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        if self.fix_emb:
            prem_embed = prem_embed.detach()
            hypo_embed = hypo_embed.detach()
        if self.projection:
            prem_embed = self.relu(self.projection_layer(prem_embed))
            hypo_embed = self.relu(self.projection_layer(hypo_embed))
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores
