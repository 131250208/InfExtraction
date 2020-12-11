from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from InfExtraction.modules.utils import MyMatrix
import time
import re


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


class HandshakingKernelDora(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cat4ent_tp = nn.Linear(hidden_size * 2, hidden_size)
        self.cln4rel_tp = LayerNorm(hidden_size, hidden_size, conditional=True)
        self.lstm4span = nn.LSTM(hidden_size,
                                 hidden_size,
                                 num_layers=1,
                                 bidirectional=False,
                                 batch_first=True)
        self.W_guide = nn.Linear(hidden_size, hidden_size)
        self.W_ent_vis = nn.Linear(hidden_size, hidden_size)
        self.W_rel_vis = nn.Linear(hidden_size, hidden_size)

    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size_x)
        '''
        batch_size, seq_len, hidden_size = seq_hiddens.size()
        ent_vis = torch.relu(self.W_ent_vis(seq_hiddens))
        ent_vis = ent_vis[:, None, :, :].repeat(1, seq_len, 1, 1)
        ent_guide = ent_vis.permute(0, 2, 1, 3)

        # mask lower triangle
        upper_visible = ent_vis.permute(0, 3, 1, 2).triu().permute(0, 2, 3, 1).contiguous()
        # visible4lstm: (batch_size * matrix_size, matrix_size, hidden_size)
        visible4lstm = upper_visible.view(-1, seq_len, hidden_size)
        span_pre, _ = self.lstm4span(visible4lstm)
        span_pre = span_pre.view(batch_size, seq_len, seq_len, hidden_size)

        # drop lower triangle and convert matrix to sequence
        # span_pre: (batch_size, shaking_seq_len, hidden_size)
        span_pre = MyMatrix.drop_lower_diag(span_pre)
        ent_guide_sks = MyMatrix.drop_lower_diag(ent_guide)
        ent_vis_sks = MyMatrix.drop_lower_diag(ent_vis)
        ent_pre = self.cat4ent_tp(ent_vis_sks, ent_guide_sks) + span_pre

        rel_guide = torch.relu(self.W_guide(seq_hiddens))
        # rel_guide_: (batch_size, hidden_size, seq_len, 1)
        rel_guide_ = rel_guide.permute(0, 2, 1)[:, :, :, None]

        rel_vis = torch.relu(self.W_rel_vis(seq_hiddens))
        # rel_vis_: (batch_size, hidden_size, 1, seq_len)
        rel_vis_ = rel_vis.permute(0, 2, 1)[:, :, None, :]

        rel_pre = torch.matmul(rel_guide_, rel_vis_).permute(0, 2, 3, 1) + self.cln4rel_tp(rel_vis_, rel_guide_)

        return ent_pre, rel_pre


class SingleSourceHandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, only_look_after=True):
        super().__init__()
        self.shaking_type = shaking_type
        self.only_look_after = only_look_after

        if "cat" in shaking_type:
            self.cat_fc = nn.Linear(hidden_size * 2, hidden_size)
        if "cln" in shaking_type:
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        if "lstm" in shaking_type:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(hidden_size,
                                     hidden_size,
                                     num_layers=1,
                                     bidirectional=False,
                                     batch_first=True)
        if "biaffine" in shaking_type:
            self.biaffine = nn.Bilinear(hidden_size, hidden_size, hidden_size)

    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size_x)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        '''
        seq_len = seq_hiddens.size()[1]

        guide = seq_hiddens[:, :, None, :].repeat(1, 1, seq_len, 1)
        visible = guide.permute(0, 2, 1, 3)

        shaking_pre = None
        # pre_num = 0

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
                span_pre = MyMatrix.drop_lower_diag(span_pre)
                shaking_pre = add_presentation(shaking_pre, span_pre)
                # pre_num += 1

            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = MyMatrix.drop_lower_diag(guide)
            visible = MyMatrix.drop_lower_diag(visible)

        if "cat" in self.shaking_type:
            tp_cat_pre = torch.cat([guide, visible], dim=-1)
            tp_cat_pre = torch.relu(self.cat_fc(tp_cat_pre))
            shaking_pre = add_presentation(shaking_pre, tp_cat_pre)
            # pre_num += 1

        if "cln" in self.shaking_type:
            tp_cln_pre = self.tp_cln(visible, guide)
            shaking_pre = add_presentation(shaking_pre, tp_cln_pre)
            # pre_num += 1

        if "biaffine" in self.shaking_type:
            set_trace()
            biaffine_pre = self.biaffine(guide, visible)
            biaffine_pre = torch.relu(biaffine_pre)
            shaking_pre = add_presentation(shaking_pre, biaffine_pre)
            # pre_num += 1

        return shaking_pre


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
                span_pre = MyMatrix.drop_lower_diag(span_pre)
                shaking_pre = add_presentation(shaking_pre, span_pre)
                pre_num += 1

            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = MyMatrix.drop_lower_diag(guide)
            visible = MyMatrix.drop_lower_diag(visible)
        
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
            set_trace()
            biaffine_pre = self.biaffine(guide, visible)
            biaffine_pre = torch.relu(biaffine_pre)
            shaking_pre = add_presentation(shaking_pre, biaffine_pre)
            pre_num += 1

        return shaking_pre / pre_num


class CrossLSTM(nn.Module):
    def __init__(self,
                 in_feature_dim=None,
                 out_feature_dim=None,
                 num_layers=1,
                 hv_comb_type="cat"
                 ):
        super().__init__()
        self.vertical_lstm = nn.LSTM(in_feature_dim,
                                  out_feature_dim // 2,
                                  num_layers=num_layers,
                                  bidirectional=True,
                                  batch_first=True)
        self.horizontal_lstm = nn.LSTM(in_feature_dim,
                                  out_feature_dim // 2,
                                  num_layers=num_layers,
                                  bidirectional=True,
                                  batch_first=True)

        self.hv_comb_type = hv_comb_type
        if hv_comb_type == "cat":
            self.combine_fc = nn.Linear(out_feature_dim * 2, out_feature_dim)
        elif hv_comb_type == "add":
            pass
        elif hv_comb_type == "interpolate":
            self.lamtha = Parameter(torch.rand(out_feature_dim))  # [0, 1)

    def forward(self, matrix):
        # matrix: (batch_size, matrix_ver_len, matrix_hor_len, hidden_size)
        batch_size, matrix_ver_len, matrix_hor_len, hidden_size = matrix.size()
        hor_context, _ = self.horizontal_lstm(matrix.view(-1, matrix_hor_len, hidden_size))
        hor_context = hor_context.view(batch_size, matrix_ver_len, matrix_hor_len, hidden_size)

        ver_context, _ = self.vertical_lstm(matrix.permute(0, 2, 1, 3).contiguous().view(-1, matrix_ver_len, hidden_size))
        ver_context = ver_context.view(batch_size, matrix_hor_len, matrix_ver_len, hidden_size)
        ver_context = ver_context.permute(0, 2, 1, 3)

        comb_context = None
        if self.hv_comb_type == "cat":
            comb_context = torch.relu(self.combine_fc(torch.cat([hor_context, ver_context], dim=-1)))
        elif self.hv_comb_type == "interpolate":
            comb_context = self.lamtha * hor_context + (1 - self.lamtha) * ver_context
        elif self.hv_comb_type == "add":
            comb_context = (hor_context + ver_context) / 2

        return comb_context


class CrossConv(nn.Module):
    def __init__(self,
                 channel_dim,
                 hor_dim,
                 ver_dim
                 ):
        super(CrossConv, self).__init__()
        self.alpha = Parameter(torch.randn([channel_dim, hor_dim, 1]))
        self.beta = Parameter(torch.randn([channel_dim, 1, ver_dim]))

    def forward(self, matrix_tensor):
        # matrix_tensor: (batch_size, ver_dim, hor_dim, hidden_size)
        # hor_cont: (batch_size, hidden_size (channel dim), ver_dim, 1)
        hor_cont = torch.matmul(matrix_tensor.permute(0, 3, 1, 2), self.alpha)
        # ver_cont: (batch_size, hidden_size, 1, hor_dim)
        ver_cont = torch.matmul(self.beta, matrix_tensor.permute(0, 3, 1, 2))
        # cross_context: (batch_size, ver_dim, hor_dim, hidden_size)
        cross_context = torch.matmul(hor_cont, ver_cont).permute(0, 2, 3, 1)
        return cross_context


class CrossPool(nn.Module):
    def __init__(self, hidden_size):
        super(CrossPool, self).__init__()
        self.lamtha = Parameter(torch.rand(hidden_size))

    def mix_pool(self, tensor, dim):
        return self.lamtha * torch.mean(tensor, dim=dim) + (1 - self.lamtha) * torch.max(tensor, dim=dim)[0]

    def forward(self, matrix_tensor):
        # matrix_tensor: (batch_size, ver_dim, hor_dim, hidden_size)
        # hor_cont: (batch_size, hidden_size, ver_dim, 1)
        hor_cont = self.mix_pool(matrix_tensor, dim=2)[:, :, None, :].permute(0, 3, 1, 2)

        # ver_cont: (batch_size, hidden_size, 1, hor_dim)
        ver_cont = self.mix_pool(matrix_tensor, dim=1)[:, None, :, :].permute(0, 3, 1, 2)

        # cross_context: (batch_size, ver_dim, hor_dim, hidden_size)
        cross_context = torch.matmul(hor_cont, ver_cont).permute(0, 2, 3, 1)
        return cross_context


class InteractionKernel(nn.Module):
    def __init__(self,
                 ent_dim,
                 rel_dim,
                 cross_enc_type,
                 cross_enc_config,
                 ):
        super(InteractionKernel, self).__init__()
        # self.ent_alpha = Parameter(torch.randn([ent_dim, matrix_size, 1]))
        # self.ent_beta = Parameter(torch.randn([ent_dim, 1, matrix_size]))
        # self.rel_alpha = Parameter(torch.randn([rel_dim, matrix_size, 1]))
        # self.rel_beta = Parameter(torch.randn([rel_dim, 1, matrix_size]))

        self.cross_enc_type = cross_enc_type
        if cross_enc_type == "bilstm":
            num_layers_crlstm = cross_enc_config["num_layers_crlstm"]
            hv_comb_type_crlstm = cross_enc_config["hv_comb_type_crlstm"]
            self.cross_enc4ent = CrossLSTM(ent_dim, ent_dim, num_layers_crlstm, hv_comb_type_crlstm)
            self.cross_enc4rel = CrossLSTM(rel_dim, rel_dim, num_layers_crlstm, hv_comb_type_crlstm)
        elif cross_enc_type == "conv":
            matrix_size = cross_enc_config["matrix_size"]
            self.cross_enc4ent = CrossConv(ent_dim, matrix_size, matrix_size)
            self.cross_enc4rel = CrossConv(rel_dim, matrix_size, matrix_size)
        elif cross_enc_type == "pool":
            self.cross_enc4ent = CrossPool(ent_dim)
            self.cross_enc4rel = CrossPool(rel_dim)

        self.lamtha4rel_cont = Parameter(torch.rand(rel_dim))  # [0, 1)

        # self.matrix_size = matrix_size
        # map_ = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        # mirror_select_ids = [map_[i][j] if i <= j else map_[j][i] for i in range(matrix_size) for j in range(matrix_size)]
        # self.mirror_select_vec = Parameter(torch.tensor(mirror_select_ids), requires_grad=False)
        # upper_gather_ids = [i * matrix_size + j for i in range(matrix_size) for j in range(matrix_size) if i <= j]
        # lower_gather_ids = [j * matrix_size + i for i in range(matrix_size) for j in range(matrix_size) if i <= j]
        # self.upper_gather_tensor = Parameter(torch.tensor(upper_gather_ids), requires_grad=False)
        # self.lower_gather_tensor = Parameter(torch.tensor(lower_gather_ids), requires_grad=False)
        # self.cached_mirror_gather_tensor = None
        # self.cached_upper_gather_tensor = None
        # self.cached_lower_gather_tensor = None

        self.ent_guide_rel_cln = LayerNorm(rel_dim, ent_dim, conditional=True)
        self.rel_guide_ent_cln = LayerNorm(ent_dim, rel_dim, conditional=True)

    # def _drop_lower_triangle(self, matrix_seq):
    #     batch_size, matrix_size, _, hidden_size = matrix_seq.size()
    #     shaking_seq = matrix_seq.view(batch_size, -1, hidden_size)
    #
    #     if self.cached_upper_gather_tensor is None or \
    #             self.cached_upper_gather_tensor.size()[0] != batch_size:
    #         self.cached_upper_gather_tensor = self.upper_gather_tensor[None, :, None].repeat(batch_size, 1, hidden_size)
    #
    #     if self.cached_lower_gather_tensor is None or \
    #             self.cached_lower_gather_tensor.size()[0] != batch_size:
    #         self.cached_lower_gather_tensor = self.lower_gather_tensor[None, :, None].repeat(batch_size, 1, hidden_size)
    #
    #     upper_shaking_hiddens = torch.gather(shaking_seq, 1, self.cached_upper_gather_tensor)
    #     lower_shaking_hiddens = torch.gather(shaking_seq, 1, self.cached_lower_gather_tensor)
    #
    #     return self.lamtha4rel_cont * upper_shaking_hiddens + (1 - self.lamtha4rel_cont) * lower_shaking_hiddens

    def forward(self, ent_hs_hiddens, rel_hs_hiddens):
        batch_size, matrix_size, _, _ = rel_hs_hiddens.size()

        # ent_hs_hiddens_mirror: (batch_size, matrix_size, matrix_size, ent_dim)
        ent_hs_hiddens_mirror = MyMatrix.mirror(ent_hs_hiddens)

        # # ent_row_cont: (batch_size, ent_dim, matrix_size, 1)
        # ent_row_cont = torch.matmul(ent_hs_hiddens_mirror.permute(0, 3, 1, 2), self.ent_alpha)
        # # ent_col_cont: (batch_size, ent_dim, 1, matrix_size)
        # ent_col_cont = torch.matmul(self.ent_beta, ent_hs_hiddens_mirror.permute(0, 3, 1, 2))
        # # ent_context: (batch_size, matrix_size, matrix_size, ent_dim)
        # ent_context = torch.matmul(ent_row_cont, ent_col_cont).permute(0, 2, 3, 1)
        ent_context = self.cross_enc4ent(ent_hs_hiddens_mirror)
        rel_hs_hiddens_guided = self.ent_guide_rel_cln(rel_hs_hiddens, ent_context)

        # # rel_row_cont: (batch_size, rel_dim, matrix_size, 1)
        # rel_row_cont = torch.matmul(rel_hs_hiddens_guided.permute(0, 3, 1, 2), self.rel_alpha)
        # # rel_col_cont: (batch_size, rel_dim, 1, matrix_size)
        # rel_col_cont = torch.matmul(self.rel_beta, rel_hs_hiddens_guided.permute(0, 3, 1, 2))
        # # rel_context: (batch_size, matrix_size, matrix_size, rel_dim)
        # rel_context = torch.matmul(rel_row_cont, rel_col_cont).permute(0, 2, 3, 1)
        rel_context = self.cross_enc4rel(rel_hs_hiddens_guided)

        rel_upper_context = MyMatrix.drop_lower_diag(rel_context)
        rel_lower_context = MyMatrix.drop_lower_diag(rel_context.permute(0, 2, 1, 3))
        # rel_context_flat: (batch_size, shaking_seq_len, rel_dim)
        rel_context_flat = self.lamtha4rel_cont * rel_upper_context + (1 - self.lamtha4rel_cont) * rel_lower_context

        ent_hs_hiddens_guided = self.rel_guide_ent_cln(ent_hs_hiddens, rel_context_flat)
        
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

