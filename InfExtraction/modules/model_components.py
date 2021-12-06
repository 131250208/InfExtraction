import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from InfExtraction.modules.utils import MyMatrix


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


class MatchingLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MatchingLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

    def my_forward(self, input_tensor, matching_tensor):
        # matching_tensor: (batch_size, matching types, hidden size)
        # hidden_size of matching tensor must be the same as the in_features
        assert self.weight.size()[1] == matching_tensor.size()[-1]
        batch_size, _, _ = matching_tensor.size()
        new_weight = torch.matmul(matching_tensor.permute(0, 2, 1)[:, :, :, None], self.weight.permute(1, 0)[:, None, :])
        new_weight = new_weight.view(new_weight.size()[0], new_weight.size()[1], -1)
        new_bias = torch.matmul(torch.mean(matching_tensor, dim=-1)[:, :, None], self.bias[None, :]).view(batch_size, -1)

        # input (batch_size, seq_len, seq_len, hidden_size),
        # new_weight: (batch_size, inp_features, out_features)
        # new_bias: (batch_size, out_features)
        seq_len = input_tensor.size()[1]
        input_tensor = input_tensor.view(batch_size, seq_len * seq_len, -1)
        res = torch.matmul(input_tensor, new_weight).permute(1, 0, 2) + new_bias
        res = res.permute(1, 0, 2).view(batch_size, seq_len, seq_len, -1)
        return res


class SingleSourceHandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, only_look_after=True, distance_emb_dim=-1):
        super().__init__()
        self.shaking_types = shaking_type.split("+")
        self.only_look_after = only_look_after
        cat_length = 0

        if "cat" in self.shaking_types:
            self.cat_fc = nn.Linear(hidden_size * 2, hidden_size)
            cat_length += hidden_size

        if "cmm" in self.shaking_types:
            self.cat_fc = nn.Linear(hidden_size * 4, hidden_size)
            self.guide_fc = nn.Linear(hidden_size, hidden_size)
            self.vis_fc = nn.Linear(hidden_size, hidden_size)
            cat_length += hidden_size

        if "cln" in self.shaking_types:
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            cat_length += hidden_size

        if "lstm" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(hidden_size,
                                    hidden_size,
                                    num_layers=1,
                                    bidirectional=False,
                                    batch_first=True)
            cat_length += hidden_size

        elif "gru" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.GRU(hidden_size,
                                   hidden_size,
                                   num_layers=1,
                                   bidirectional=False,
                                   batch_first=True)
            cat_length += hidden_size

        if "bilstm" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(hidden_size,
                                    hidden_size // 2,
                                    num_layers=1,
                                    bidirectional=False,
                                    batch_first=True)
            self.lstm4span_back = nn.LSTM(hidden_size,
                                         hidden_size // 2,
                                         num_layers=1,
                                         bidirectional=False,
                                         batch_first=True)
            cat_length += hidden_size
        elif "bigru" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.GRU(hidden_size,
                                   hidden_size // 2,
                                   num_layers=1,
                                   bidirectional=False,
                                   batch_first=True)
            self.lstm4span_back = nn.GRU(hidden_size,
                                        hidden_size // 2,
                                        num_layers=1,
                                        bidirectional=False,
                                        batch_first=True)
            cat_length += hidden_size

        if "biaffine" in self.shaking_types:
            self.biaffine = nn.Bilinear(hidden_size, hidden_size, hidden_size)
            cat_length += hidden_size

        self.distance_emb_dim = distance_emb_dim
        if distance_emb_dim > 0:
            self.dist_emb = nn.Embedding(512, distance_emb_dim)
            self.dist_ids_matrix = None  # for cache
            cat_length += distance_emb_dim

        self.aggr_fc = nn.Linear(cat_length, hidden_size)

    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size_x)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        '''
        # seq_len = seq_hiddens.size()[1]
        batch_size, seq_len, vis_hidden_size = seq_hiddens.size()

        guide = seq_hiddens[:, :, None, :].repeat(1, 1, seq_len, 1)
        visible = guide.permute(0, 2, 1, 3)
        feature_pre_list = []

        if self.only_look_after:
            if len({"lstm", "bilstm", "gru", "bigru"}.intersection(self.shaking_types)) > 0:
                # batch_size, _, matrix_size, vis_hidden_size = visible.size()
                # mask lower triangle part
                upper_visible = visible.permute(0, 3, 1, 2).triu().permute(0, 2, 3, 1).contiguous()

                # visible4lstm: (batch_size * matrix_size, matrix_size, hidden_size)
                visible4lstm = upper_visible.view(batch_size * seq_len, seq_len, -1)
                span_pre, _ = self.lstm4span(visible4lstm)
                span_pre = span_pre.view(batch_size, seq_len, seq_len, -1)

                if len({"bilstm", "bigru"}.intersection(self.shaking_types)) > 0:
                    # mask upper triangle part
                    lower_visible = visible.permute(0, 3, 1, 2).tril().permute(0, 2, 3, 1).contiguous()
                    visible4lstm_back = lower_visible.view(batch_size * seq_len, seq_len, -1)

                    visible4lstm_back = torch.flip(visible4lstm_back, [1, ])
                    span_pre_back, _ = self.lstm4span_back(visible4lstm_back)
                    span_pre_back = torch.flip(span_pre_back, [1, ])
                    span_pre_back = span_pre_back.view(batch_size, seq_len, seq_len, -1)
                    span_pre_back = span_pre_back.permute(0, 2, 1, 3)
                    span_pre = torch.cat([span_pre, span_pre_back], dim=-1)

                # drop lower triangle and convert matrix to sequence
                # span_pre: (batch_size, shaking_seq_len, hidden_size)
                span_pre = MyMatrix.upper_reg2seq(span_pre)
                # shaking_pre = add_presentation(shaking_pre, span_pre)
                feature_pre_list.append(span_pre)

            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = MyMatrix.upper_reg2seq(guide)
            visible = MyMatrix.upper_reg2seq(visible)

        if "cat" in self.shaking_types:
            tp_cat_pre = torch.cat([guide, visible], dim=-1)
            tp_cat_pre = torch.relu(self.cat_fc(tp_cat_pre))
            # shaking_pre = add_presentation(shaking_pre, tp_cat_pre)
            feature_pre_list.append(tp_cat_pre)

        if "cmm" in self.shaking_types:  # cat and multiple
            tp_cat_pre = torch.cat([guide, visible,
                                    torch.abs(guide - visible),
                                    torch.mul(self.guide_fc(guide), self.vis_fc(visible))], dim=-1)
            tp_cat_pre = torch.relu(self.cat_fc(tp_cat_pre))
            # shaking_pre = add_presentation(shaking_pre, tp_cat_pre)
            feature_pre_list.append(tp_cat_pre)

        if "cln" in self.shaking_types:
            tp_cln_pre = self.tp_cln(visible, guide)
            # shaking_pre = add_presentation(shaking_pre, tp_cln_pre)
            feature_pre_list.append(tp_cln_pre)

        if "biaffine" in self.shaking_types:
            biaffine_pre = self.biaffine(guide, visible)
            biaffine_pre = torch.relu(biaffine_pre)
            # shaking_pre = add_presentation(shaking_pre, biaffine_pre)
            feature_pre_list.append(biaffine_pre)

        if self.distance_emb_dim > 0:
            if self.dist_ids_matrix is None or \
                    self.dist_ids_matrix.size()[0] != batch_size or \
                    self.dist_ids_matrix.size()[1] != seq_len:  # need to update cached distance ids
                t = torch.arange(0, seq_len).to(seq_hiddens.device)[:, None].repeat(1, seq_len)
                self.dist_ids_matrix = torch.abs(t - t.permute(1, 0)).long()[None, :, :].repeat(batch_size, 1, 1)
                if self.only_look_after:  # matrix to handshaking seq
                    self.dist_ids_matrix = MyMatrix.upper_reg2seq(self.dist_ids_matrix[:, :, :, None]).view(batch_size, -1)
            dist_embeddings = self.dist_emb(self.dist_ids_matrix)
            feature_pre_list.append(dist_embeddings)

        output_hiddens = self.aggr_fc(torch.cat(feature_pre_list, dim=-1))
        return output_hiddens


class HandshakingKernel(nn.Module):
    def __init__(self, guide_hidden_size, vis_hidden_size, shaking_type, only_look_after=True):
        '''
            guide_hidden_size = seq_hiddens_x.size()[-1]
            vis_hidden_size = seq_hiddens_y.size()[-1]
            output_size = vis_hidden_size, guide_hidden_size is not necessary equal to vis_hidden_size
        '''
        super().__init__()
        self.shaking_type = shaking_type
        self.only_look_after = only_look_after

        if "cat" in shaking_type:
            self.cat_fc = nn.Linear(guide_hidden_size + vis_hidden_size, vis_hidden_size)
        if "cln" in shaking_type:
            self.tp_cln = LayerNorm(vis_hidden_size, guide_hidden_size, conditional=True)
        if "lstm" in shaking_type:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(vis_hidden_size,
                                     vis_hidden_size,
                                     num_layers=1,
                                     bidirectional=False,
                                     batch_first=True)
        if "biaffine" in shaking_type:
            self.biaffine = nn.Bilinear(guide_hidden_size, vis_hidden_size, vis_hidden_size)

    def forward(self, seq_hiddens_x, seq_hiddens_y):
        '''
        seq_hiddens_x: (batch_size, seq_len, hidden_size_x)
        seq_hiddens_y: (batch_size, seq_len, hidden_size_y)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        '''
        # seq_len = seq_hiddens_y.size()[1]
        # assert seq_hiddens_y.size()[1] == seq_hiddens_x.size()[1]

        guide = seq_hiddens_x[:, :, None, :].repeat(1, 1, seq_hiddens_y.size()[1], 1)
        visible = seq_hiddens_y[:, None, :, :].repeat(1, seq_hiddens_x.size()[1], 1, 1)

        shaking_pre = None
        pre_num = 0

        def add_presentation(all_prst, prst):
            if all_prst is None:
                all_prst = prst
            else:
                all_prst += prst
            return all_prst

        if self.only_look_after:
            if "lstm" in self.shaking_type:
                batch_size, _, matrix_size, vis_hidden_size = visible.size()
                # mask lower triangle
                upper_visible = visible.permute(0, 3, 1, 2).triu().permute(0, 2, 3, 1).contiguous()

                # visible4lstm: (batch_size * matrix_size, matrix_size, hidden_size)
                visible4lstm = upper_visible.view(-1, matrix_size, vis_hidden_size)
                span_pre, _ = self.lstm4span(visible4lstm)
                span_pre = span_pre.view(batch_size, matrix_size, matrix_size, vis_hidden_size)

                # drop lower triangle and convert matrix to sequence
                # span_pre: (batch_size, shaking_seq_len, hidden_size)
                span_pre = MyMatrix.upper_reg2seq(span_pre)
                shaking_pre = add_presentation(shaking_pre, span_pre)
                pre_num += 1

            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = MyMatrix.upper_reg2seq(guide)
            visible = MyMatrix.upper_reg2seq(visible)

        if "cat" in self.shaking_type:
            tp_cat_pre = torch.cat([guide, visible], dim=-1)
            tp_cat_pre = torch.relu(self.cat_fc(tp_cat_pre))
            shaking_pre = add_presentation(shaking_pre, tp_cat_pre)
            pre_num += 1

        if "cln" in self.shaking_type:
            tp_cln_pre = self.tp_cln(visible, guide)
            shaking_pre = add_presentation(shaking_pre, tp_cln_pre)
            pre_num += 1

        if "biaffine" in self.shaking_type:
            biaffine_pre = self.biaffine(guide, visible)
            biaffine_pre = torch.relu(biaffine_pre)
            shaking_pre = add_presentation(shaking_pre, biaffine_pre)
            pre_num += 1

        return shaking_pre / pre_num


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

        node = torch.cat([node1, node2], dim=-1)  # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]
