from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢?
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        beta, gamma = None, None

        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size_train, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_lstm":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == "cln_lstm":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == "cln_att":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.W_q = nn.Linear(hidden_size, hidden_size)
            self.W_k = nn.Linear(hidden_size, hidden_size)
        elif shaking_type == "cat_att":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
            self.W_q = nn.Linear(hidden_size, hidden_size)
            self.W_k = nn.Linear(hidden_size, hidden_size)

        if "mix" in shaking_type:
            self.lamtha = Parameter(torch.rand(hidden_size))
        if "lstm" in shaking_type:
            self.inner_context_lstm = nn.LSTM(hidden_size,
                                              hidden_size,
                                              num_layers=1,
                                              bidirectional=False,
                                              batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type):
        # seq_hiddens: (batch_size_train, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            pooling = None
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling

        inner_context = None
        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size_train, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size_train, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)

            shaking_hiddens = None
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_lstm":
                inner_context = self.enc_inner_hiddens(visible_hiddens, "lstm")
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_lstm":
                inner_context = self.enc_inner_hiddens(visible_hiddens, "lstm")
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)
            elif self.shaking_type == "cln_att":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                query = self.W_q(shaking_hiddens)
                key = torch.transpose(self.W_k(seq_hiddens), 1, 2)
                att = F.softmax(torch.matmul(query, key) / (query.size()[-1] ** 0.5), dim=-1)
                shaking_hiddens = torch.matmul(att, seq_hiddens)
            elif self.shaking_type == "cat_att":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
                query = self.W_q(shaking_hiddens)
                key = torch.transpose(self.W_k(seq_hiddens), 1, 2)
                att = F.softmax(torch.matmul(query, key) / (query.size()[-1] ** 0.5), dim=-1)
                shaking_hiddens = torch.matmul(att, seq_hiddens)
            #                 weighted_shaking_hiddens = []
            #                 for tp_ind in range(shaking_hiddens.size()[-2]):
            #                     query = shaking_hiddens[:, tp_ind: (tp_ind + 1), :].repeat(1, seq_len, 1)
            #                     keys = seq_hiddens
            # #                     set_trace()
            #                     att = self.V(torch.tanh(self.W(torch.cat([query, keys], dim = -1))))
            #                     weighted_shaking_hiddens.append(torch.sum(att * seq_hiddens, dim = -2)[:, None, :])
            #                 shaking_hiddens = torch.cat(weighted_shaking_hiddens, dim = -2)

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_Time': 2, '2_Group': 3, '2_Nation': 4,
                       '2_Indeterminate': 5, '2_Population_Center': 6, '2_Government': 7,
                       '2_Commercial': 8, '2_Non_Governmental': 9, '2_Media': 10, '2_Building_Grounds': 11,
                       '2_Numeric': 12, '2_State_or_Province': 13, '2_Region_General': 14, '2_Sports': 15,
                       '2_Crime': 16, '2_Land': 17, '2_Air': 18, '2_Water': 19, '2_Airport': 20,
                       '2_Sentence': 21, '2_Educational': 22, '2_Celestial': 23, '2_Underspecified': 24,
                       '2_Shooting': 25, '2_Special': 26, '2_Subarea_Facility': 27, '2_Path': 28,
                       '2_GPE_Cluster': 29, '2_Exploding': 30, '2_Water_Body': 31, '2_Land_Region_Natural': 32,
                       '2_Nuclear': 33, '2_Projectile': 34, '2_Region_International': 35, '2_Medical_Science': 36,
                       '2_Continent': 37, '2_Job_Title': 38, '2_County_or_District': 39, '2_Religious': 40,
                       '2_Contact_Info': 41, '2_Chemical': 42, '2_Subarea_Vehicle': 43, '2_Entertainment': 44,
                       '2_Biological': 45, '2_Boundary': 46, '2_Plant': 47, '2_Address': 48, '2_Sharp': 49,
                       '2_Blunt': 50
                       }

dep_dict = {'O': 0, 'punct': 1, 'iobj': 2, 'parataxis': 3, 'auxpass': 4, 'aux': 5,
            'conj': 6, 'advcl': 7, 'acl:relcl': 8, 'nsubjpass': 9, 'csubj': 10, 'compound': 11,
            'compound:prt': 12, 'mwe': 13, 'cop': 14, 'neg': 15, 'nmod:poss': 16, 'appos': 17,
            'cc:preconj': 18, 'nmod': 19, 'nsubj': 20, 'xcomp': 21, 'det:predet': 22,
            'nmod:npmod': 23, 'acl': 24, 'amod': 25, 'expl': 26, 'csubjpass': 27, 'case': 28,
            'ccomp': 29, 'dobj': 30, 'ROOT': 31, 'discourse': 32, 'nmod:tmod': 33, 'dep': 34,
            'nummod': 35, 'mark': 36, 'advmod': 37, 'cc': 38, 'det': 39
            }


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, dep_embed_dim, gcn_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()

        self.gcn_dim = gcn_dim
        self.dep_embed_dim = dep_embed_dim
        self.pooling = pooling

        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = Edgeupdate(gcn_dim, self.dep_embed_dim, dropout_ratio=0.5)

    def forward(self, weight_adj, node_hiddens):
        """
        :param weight_adj: [batch, seq, seq, dim_e]
        :param node_hiddens: [batch, seq, dim]
        :return:
        """
    
        batch, seq, dim = node_hiddens.shape
        weight_adj = weight_adj.permute(0, 3, 1, 2)  # [batch, dim_e, seq, seq]

        node_hiddens = node_hiddens.unsqueeze(1).expand(batch, self.dep_embed_dim, seq, dim)
        Ax = torch.matmul(weight_adj, node_hiddens)  # [batch, dim_e, seq, dim]
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        
        gcn_outputs = self.W(Ax)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        # Edge update weight_adj[batch, dim_e, seq, seq]
        weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()  # [batch, seq, seq, dim_e]
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)
        return edge_outputs, node_outputs


class Edgeupdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.5):
        super(Edgeupdate, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)

    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, seq, seq, dim_e]
        :param node: [batch, seq, seq, dim]
        :return:
        """

        node = torch.cat([node1, node2], dim=-1) # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]


class EDModel(nn.Module):

    def __init__(self, args, id_to_tag, device, pre_word_embed):
        super(EDModel, self).__init__()

        self.device = device
        self.gcn_model = EEGCN(device, pre_word_embed, args)
        self.gcn_dim = args.gcn_dim
        self.classifier = nn.Linear(self.gcn_dim, len(id_to_tag))

    def forward(self, word_sequence, x_len, entity_type_sequence, adj, dep):
        outputs, weight_adj = self.gcn_model(word_sequence, x_len, entity_type_sequence, adj, dep)
        logits = self.classifier(outputs)
        return logits, weight_adj


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0, c0


class EEGCN(nn.Module):
    def __init__(self, device, pre_word_embeds, args):
        super().__init__()

        self.device = device
        self.in_dim = args.word_embed_dim + args.bio_embed_dim
        self.maxLen = args.num_steps

        self.rnn_hidden = args.rnn_hidden
        self.rnn_dropout = args.rnn_dropout
        self.rnn_layers = args.rnn_layers

        self.gcn_dropout = args.gcn_dropout
        self.num_layers = args.num_layers
        self.gcn_dim = args.gcn_dim

        # Word Embedding Layer
        self.word_embed_dim = args.word_embed_dim
        self.wembeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pre_word_embeds), freeze=False)

        # Entity Label Embedding Layer
        self.bio_size = len(entity_subtype_dict)
        self.bio_embed_dim = args.bio_embed_dim
        if self.bio_embed_dim:
            self.bio_embeddings = nn.Embedding(num_embeddings=self.bio_size,
                                               embedding_dim=self.bio_embed_dim)

        self.dep_size = len(dep_dict)
        self.dep_embed_dim = args.dep_embed_dim
        self.edge_embeddings = nn.Embedding(num_embeddings=self.dep_size,
                                            embedding_dim=self.dep_embed_dim,
                                            padding_idx=0)

        self.rnn = nn.LSTM(self.in_dim, self.rnn_hidden, self.rnn_layers, batch_first=True,
                           dropout=self.rnn_dropout, bidirectional=True)
        self.rnn_drop = nn.Dropout(self.rnn_dropout)  # use on last layer output

        self.input_W_G = nn.Linear(self.rnn_hidden * 2, self.gcn_dim)
        self.pooling = args.pooling
        self.gcn_layers = nn.ModuleList()
        self.gcn_drop = nn.Dropout(self.gcn_dropout)
        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(self.device, self.gcn_dim, self.dep_embed_dim, args.pooling))
        self.aggregate_W = nn.Linear(self.gcn_dim + self.num_layers * self.gcn_dim, self.gcn_dim)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        # seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.rnn_hidden, self.rnn_layers)
        h0, c0 = h0.to(self.device), c0.to(self.device)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, word_sequence, x_len, entity_type_sequence, adj, edge):

        BATCH_SIZE = word_sequence.shape[0]
        BATCH_MAX_LEN = x_len[0]

        word_sequence = word_sequence[:, :BATCH_MAX_LEN].contiguous()
        adj = adj[:, :BATCH_MAX_LEN, :BATCH_MAX_LEN].contiguous()
        edge = edge[:, :BATCH_MAX_LEN, :BATCH_MAX_LEN].contiguous()
        weight_adj = self.edge_embeddings(edge)  # [batch, seq, seq, dim_e]

        word_emb = self.wembeddings(word_sequence)
        x_emb = word_emb
        if self.bio_embed_dim:
            entity_type_sequence = entity_type_sequence[:, :BATCH_MAX_LEN].contiguous()
            entity_label_emb = self.bio_embeddings(entity_type_sequence)
            x_emb = torch.cat([x_emb, entity_label_emb], dim=2)

        rnn_outputs = self.rnn_drop(self.encode_with_rnn(x_emb, x_len, BATCH_SIZE))
        gcn_inputs = self.input_W_G(rnn_outputs)
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        src_mask = (word_sequence != 0)
        src_mask = src_mask[:, :BATCH_MAX_LEN].unsqueeze(-2).contiguous()

        for _layer in range(self.num_layers):
            gcn_outputs, weight_adj = self.gcn_layers[_layer](weight_adj, gcn_outputs)  # [batch, seq, dim]
            gcn_outputs = self.gcn_drop(gcn_outputs)
            weight_adj = self.gcn_drop(weight_adj)
            layer_list.append(gcn_outputs)

        outputs = torch.cat(layer_list, dim=-1)
        aggregate_out = self.aggregate_W(outputs)
        return aggregate_out, weight_adj