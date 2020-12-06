from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from InfExtraction.modules.preprocess import Indexer


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
    def __init__(self, guide_hidden_size, vis_hidden_size, shaking_type, only_look_after=True):
        '''
            guide_hidden_size = seq_hiddens_x.size()[-1]
            vis_hidden_size = seq_hiddens_y.size()[-1]
            output_size = vis_hidden_size, guide_hidden_size is not necessary equal to vis_hidden_size
        '''
        super().__init__()
        self.shaking_type = shaking_type
        self.only_look_after = only_look_after
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(guide_hidden_size + vis_hidden_size, vis_hidden_size)
        elif shaking_type == "cat_lstm":
            self.combine_fc = nn.Linear(vis_hidden_size * 2 + guide_hidden_size, vis_hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(vis_hidden_size, guide_hidden_size, conditional=True)
        elif shaking_type == "cln_lstm":
            self.tp_cln = LayerNorm(vis_hidden_size, guide_hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(vis_hidden_size, vis_hidden_size, conditional=True)

        if "mix" in shaking_type:
            self.lamtha = Parameter(torch.rand(vis_hidden_size))
        if "lstm" in shaking_type:
            self.inner_context_lstm = nn.LSTM(vis_hidden_size,
                                              vis_hidden_size,
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

    def forward(self, seq_hiddens_x, seq_hiddens_y):
        '''
        seq_hiddens_x: (batch_size, seq_len, hidden_size)
        seq_hiddens_y: (batch_size, seq_len, hidden_size)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        '''
        seq_len = seq_hiddens_x.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            vis_start_ind = ind if self.only_look_after else 0
            guide_hiddens = seq_hiddens_x[:, ind:ind+1, :].repeat(1, seq_len - vis_start_ind, 1)
            visible_hiddens = seq_hiddens_y[:, vis_start_ind:, :]

            shaking_hiddens = None
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([guide_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_lstm":
                inner_context = self.enc_inner_hiddens(visible_hiddens, "lstm")
                shaking_hiddens = torch.cat([guide_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, guide_hiddens)
            elif self.shaking_type == "cln_lstm":
                shaking_hiddens = self.tp_cln(visible_hiddens, guide_hiddens)
                inner_context = self.enc_inner_hiddens(visible_hiddens, "lstm")
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)
        if self.only_look_after:
            fin_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        else:
            fin_shaking_hiddens = torch.stack(shaking_hiddens_list, dim=1)
        return fin_shaking_hiddens


class InteractionKernel(nn.Module):
    def __init__(self, ent_dim, rel_dim, ent_att_heads=16, rel_att_heads=16):
        super(InteractionKernel, self).__init__()
        # self.fc_rel2ent = nn.Linear(rel_dim, ent_dim)
        # self.fc_ent2rel = nn.Linear(ent_dim, rel_dim)
        # self.ent_multihead_attn = nn.MultiheadAttention(ent_dim, ent_att_heads)
        # self.rel_multihead_attn = nn.MultiheadAttention(rel_dim, rel_att_heads)
        self.ent_guide_rel_cln = LayerNorm(rel_dim, ent_dim, conditional=True)
        self.rel_guide_ent_cln = LayerNorm(ent_dim, rel_dim, conditional=True)

    def _mirror(self, shaking_seq):
        batch_size, shaking_seq_len, dim_size = shaking_seq.size()
        matrix_size = int((2 * shaking_seq_len + 0.25) ** 0.5 - 0.5)
        map_ = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        matrix_hidden_list = []
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i <= j:
                    shaking_idx = map_[i][j]
                else:
                    shaking_idx = map_[j][i]
                matrix_hidden_list.append(shaking_seq[:, shaking_idx, :])
        matrix = torch.cat(matrix_hidden_list, dim=1).view(batch_size, matrix_size, matrix_size, dim_size)
        return matrix

    def forward(self, ent_hs_hiddens, rel_hs_hiddens):
        batch_size, matrix_size, _, _ = rel_hs_hiddens.size()

        ent_hs_hiddens_mirror = self._mirror(ent_hs_hiddens)
        ent_context_list = []
        for i in range(matrix_size):
            for j in range(matrix_size):
                # ent_keys = torch.cat([ent_hs_hiddens_mirror[:, i, :, :],
                #                       ent_hs_hiddens_mirror[:, j, :, :]],
                #                      dim=1,
                #                      )
                # ent_vals = ent_keys
                # rel_query = rel_hs_hiddens[:, i, j, :].repeat(1, matrix_size * 2, 1)
                # rel_query = self.fc_rel2ent(rel_query)
                # ent_con = self.ent_multihead_attn(rel_query, ent_keys, ent_vals)[0]

                ent_con = (torch.mean(ent_hs_hiddens_mirror[:, i, :, :], dim=1) +
                           torch.mean(ent_hs_hiddens_mirror[:, j, :, :], dim=1)) / 2

                ent_context_list.append(ent_con)
        ent_context = torch.cat(ent_context_list, dim=1).view(batch_size, matrix_size, matrix_size, -1)
        assert ent_context.size()[-1] == ent_hs_hiddens.size()[-1]

        rel_hs_hiddens_guided = self.ent_guide_rel_cln(rel_hs_hiddens, ent_context)

        rel_context_list = []
        map_ = Indexer.get_shaking_idx2matrix_idx(matrix_size)
        for i in range(ent_hs_hiddens.size()[1]):
            mat_i, mat_j = map_[i]
            # rel_keys = torch.cat([rel_hs_hiddens_guided[:, mat_i, :, :],
            #                       rel_hs_hiddens_guided[:, mat_j, :, :],
            #                       rel_hs_hiddens_guided[:, :, mat_i, :],
            #                       rel_hs_hiddens_guided[:, :, mat_j, :],
            #                       ],
            #                      dim=1,
            #                      )
            # rel_vals = rel_keys
            # ent_query = ent_hs_hiddens[:, i, :].repeat(1, matrix_size * 4, 1)
            # ent_query = self.fc_ent2rel(ent_query)
            # rel_con = self.rel_multihead_attn(ent_query, rel_keys, rel_vals)[0]

            rel_con = (torch.mean(rel_hs_hiddens_guided[:, mat_i, :, :], dim=1) +
                       torch.mean(rel_hs_hiddens_guided[:, mat_j, :, :], dim=1) +
                       torch.mean(rel_hs_hiddens_guided[:, :, mat_i, :], dim=1) +
                       torch.mean(rel_hs_hiddens_guided[:, :, mat_j, :], dim=1)) / 4
            rel_context_list.append(rel_con)
        rel_context = torch.cat(rel_context_list, dim=1).view(batch_size, ent_hs_hiddens.size()[1], -1)
        assert rel_context.size()[-1] == rel_hs_hiddens.size()[-1] == rel_hs_hiddens_guided.size()[-1]

        ent_hs_hiddens_guided = self.rel_guide_ent_cln(ent_hs_hiddens, rel_context)
        assert ent_hs_hiddens_guided.size()[-1] == ent_hs_hiddens.size()[-1]

        return ent_hs_hiddens_guided, rel_hs_hiddens_guided


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

