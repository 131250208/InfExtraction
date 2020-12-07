import torch.nn as nn
import torch
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import BertModel
from InfExtraction.modules.model_components import HandshakingKernel, GraphConvLayer, InteractionKernel
from InfExtraction.modules.preprocess import Indexer
from gensim.models import KeyedVectors
import logging
from IPython.core.debugger import set_trace
import time


class IEModel(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 tagger,
                 metrics_cal,
                 char_encoder_config=None,
                 subwd_encoder_config=None,
                 word_encoder_config=None,
                 ner_tag_emb_config=None,
                 pos_tag_emb_config=None,
                 dep_config=None
                 ):
        super().__init__()
        '''
        :parameters: see model settings in settings_default.py
        '''
        self.tagger = tagger
        self.metrics_cal = metrics_cal
        self.cat_hidden_size = 0

        # count bp steps
        self.bp_steps = 0

        # ner
        self.ner_tag_emb_config = ner_tag_emb_config
        if ner_tag_emb_config is not None:
            ner_tag_num = ner_tag_emb_config["ner_tag_num"]
            ner_tag_emb_dim = ner_tag_emb_config["emb_dim"]
            ner_tag_emb_dropout = ner_tag_emb_config["emb_dropout"]
            self.ner_tag_emb = nn.Embedding(ner_tag_num, ner_tag_emb_dim)
            self.ner_tag_emb_dropout = nn.Dropout(p=ner_tag_emb_dropout)
            self.cat_hidden_size += ner_tag_emb_dim

        # pos
        self.pos_tag_emb_config = pos_tag_emb_config
        if pos_tag_emb_config is not None:
            pos_tag_num = pos_tag_emb_config["pos_tag_num"]
            pos_tag_emb_dim = pos_tag_emb_config["emb_dim"]
            pos_tag_emb_dropout = pos_tag_emb_config["emb_dropout"]
            self.pos_tag_emb = nn.Embedding(pos_tag_num, pos_tag_emb_dim)
            self.pos_tag_emb_dropout = nn.Dropout(p=pos_tag_emb_dropout)
            self.cat_hidden_size += pos_tag_emb_dim

        # char
        self.char_encoder_config = char_encoder_config
        if char_encoder_config is not None:
            # char encoder
            char_size = char_encoder_config["char_size"]
            char_emb_dim = char_encoder_config["emb_dim"]
            char_emb_dropout = char_encoder_config["emb_dropout"]
            char_bilstm_hidden_size = char_encoder_config["bilstm_hidden_size"]
            char_bilstm_layers = char_encoder_config["bilstm_layers"]
            char_bilstm_dropout = char_encoder_config["bilstm_dropout"]
            max_char_num_in_subword = char_encoder_config["max_char_num_in_tok"]

            self.char_emb = nn.Embedding(char_size, char_emb_dim)
            self.char_emb_dropout = nn.Dropout(p=char_emb_dropout)
            self.char_lstm_l1 = nn.LSTM(char_emb_dim,
                                        char_bilstm_hidden_size[0] // 2,
                                        num_layers=char_bilstm_layers[0],
                                        dropout=char_bilstm_dropout[0],
                                        bidirectional=True,
                                        batch_first=True)
            self.char_lstm_dropout = nn.Dropout(p=char_bilstm_dropout[1])
            self.char_lstm_l2 = nn.LSTM(char_bilstm_hidden_size[0],
                                        char_bilstm_hidden_size[1] // 2,
                                        num_layers=char_bilstm_layers[1],
                                        dropout=char_bilstm_dropout[2],
                                        bidirectional=True,
                                        batch_first=True)
            self.char_cnn = nn.Conv1d(char_bilstm_hidden_size[1], char_bilstm_hidden_size[1], max_char_num_in_subword,
                                      stride=max_char_num_in_subword)
            self.cat_hidden_size += char_bilstm_hidden_size[1]

        # word encoder
        self.word_encoder_config = word_encoder_config
        if word_encoder_config is not None:
            ## config
            word2id = word_encoder_config["word2id"]
            word_emb_dropout = word_encoder_config["emb_dropout"]
            word_bilstm_hidden_size = word_encoder_config["bilstm_hidden_size"]
            word_bilstm_layers = word_encoder_config["bilstm_layers"]
            word_bilstm_dropout = word_encoder_config["bilstm_dropout"]
            freeze_word_emb = word_encoder_config["freeze_word_emb"]
            word_emb_file_path = word_encoder_config["word_emb_file_path"]

            ## init word embedding
            emb_file_suffix = word_emb_file_path.split(".")[-1]
            pretrained_emb, word_emb_dim = None, None
            logging.info("loading embedding file...")
            if emb_file_suffix == "txt":
                glove_df = pd.read_csv(word_emb_file_path,
                                       sep=" ", quoting=3, header=None, index_col=0)
                pretrained_emb = {key: val.values for key, val in glove_df.T.items()}
                word_emb_dim = len(list(pretrained_emb.values())[0])
            elif emb_file_suffix == "bin":
                pretrained_emb = KeyedVectors.load_word2vec_format(word_emb_file_path,
                                                                   binary=True)
                word_emb_dim = len(pretrained_emb.vectors[0])

            init_word_embedding_matrix = np.random.normal(-0.5, 0.5, size=(len(word2id), word_emb_dim))
            hit_count = 0
            for word, idx in tqdm(word2id.items(), desc="init word embedding matrix"):
                word_lower = word.lower()
                if word in pretrained_emb:
                    hit_count += 1
                    init_word_embedding_matrix[idx] = pretrained_emb[word]
                elif word_lower in pretrained_emb:
                    hit_count += 1
                    init_word_embedding_matrix[idx] = pretrained_emb[word_lower]
            print("pretrained word embedding hit rate: {}".format(hit_count / len(word2id)))

            init_word_embedding_matrix = torch.FloatTensor(init_word_embedding_matrix)
            word_emb_dim = init_word_embedding_matrix.size()[1]
            self.cat_hidden_size += word_emb_dim

            ## use lstm to encode word
            self.word_emb = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=freeze_word_emb)
            self.word_emb_dropout = nn.Dropout(p=word_emb_dropout)
            self.word_lstm_l1 = nn.LSTM(self.cat_hidden_size,
                                        word_bilstm_hidden_size[0] // 2,
                                        num_layers=word_bilstm_layers[0],
                                        dropout=word_bilstm_dropout[0],
                                        bidirectional=True,
                                        batch_first=True)
            self.word_lstm_dropout = nn.Dropout(p=word_bilstm_dropout[1])
            self.word_lstm_l2 = nn.LSTM(word_bilstm_hidden_size[0],
                                        word_bilstm_hidden_size[1] // 2,
                                        num_layers=word_bilstm_layers[1],
                                        dropout=word_bilstm_dropout[2],
                                        bidirectional=True,
                                        batch_first=True)
            self.cat_hidden_size += word_bilstm_hidden_size[1]

        # subword_encoder
        self.subwd_encoder_config = subwd_encoder_config
        if subwd_encoder_config is not None:
            bert_path = subwd_encoder_config["pretrained_model_path"]
            bert_finetune = subwd_encoder_config["finetune"]
            self.use_last_k_layers_bert = subwd_encoder_config["use_last_k_layers"]
            self.bert = BertModel.from_pretrained(bert_path)
            if not bert_finetune:  # if train without finetuning bert
                for param in self.bert.parameters():
                    param.requires_grad = False
            self.cat_hidden_size += self.bert.config.hidden_size

        # dependencies
        self.dep_config = dep_config
        if dep_config is not None:
            self.dep_type_num = dep_config["dep_type_num"]
            dep_type_emb_dim = dep_config["dep_type_emb_dim"]
            dep_type_emb_dropout = dep_config["emb_dropout"]
            self.dep_type_emb = nn.Embedding(self.dep_type_num, dep_type_emb_dim)
            self.dep_type_emb_dropout = nn.Dropout(p=dep_type_emb_dropout)

            # GCN
            dep_gcn_dim = dep_config["gcn_dim"]
            dep_gcn_dropout = dep_config["gcn_dropout"]
            dep_gcn_layer_num = dep_config["gcn_layer_num"]
            # aggregate fc
            self.aggr_fc4gcn = nn.Linear(self.cat_hidden_size, dep_gcn_dim)
            self.gcn_layers = nn.ModuleList()
            self.dep_gcn_dropout = nn.Dropout(dep_gcn_dropout)
            for _ in range(dep_gcn_layer_num):
                self.gcn_layers.append(GraphConvLayer(dep_type_emb_dim, dep_gcn_dim, "avg"))
                self.cat_hidden_size += dep_gcn_dim

    def _cat_features(self,
                      char_input_ids=None,
                      word_input_ids=None,
                      subword_input_ids=None,
                      attention_mask=None,
                      token_type_ids=None,
                      ner_tag_ids=None,
                      pos_tag_ids=None,
                      dep_adj_matrix=None):

        # features
        features = []

        # ner tag
        if self.ner_tag_emb_config is not None:
            ner_tag_embeddings = self.ner_tag_emb(ner_tag_ids)
            ner_tag_embeddings = self.ner_tag_emb_dropout(ner_tag_embeddings)
            features.append(ner_tag_embeddings)

        # pos tag
        if self.pos_tag_emb_config is not None:
            pos_tag_embeddings = self.pos_tag_emb(pos_tag_ids)
            pos_tag_embeddings = self.pos_tag_emb_dropout(pos_tag_embeddings)
            features.append(pos_tag_embeddings)

        # char
        if self.char_encoder_config is not None:
            # char_input_ids: (batch_size, seq_len * max_char_num_in_subword)
            # char_input_emb/char_hiddens: (batch_size, seq_len * max_char_num_in_subword, char_emb_dim)
            # char_conv_oudtut: (batch_size, seq_len, char_emb_dim)
            char_input_emb = self.char_emb(char_input_ids)
            char_input_emb = self.char_emb_dropout(char_input_emb)
            char_hiddens, _ = self.char_lstm_l1(char_input_emb)
            char_hiddens, _ = self.char_lstm_l2(self.char_lstm_dropout(char_hiddens))
            char_conv_oudtut = self.char_cnn(char_hiddens.permute(0, 2, 1)).permute(0, 2, 1)
            features.append(char_conv_oudtut)

        # word
        if self.word_encoder_config is not None:
            # word_input_ids: (batch_size, seq_len)
            # word_input_emb/word_hiddens: batch_size_train, seq_len, word_emb_dim)
            word_input_emb = self.word_emb(word_input_ids)
            word_input_emb = self.word_emb_dropout(word_input_emb)
            features.append(word_input_emb)
            word_hiddens, _ = self.word_lstm_l1(torch.cat(features, dim=-1))
            word_hiddens, _ = self.word_lstm_l2(self.word_lstm_dropout(word_hiddens))
            features.append(word_hiddens)

        # subword
        if self.subwd_encoder_config is not None:
            # subword_input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
            context_oudtuts = self.bert(subword_input_ids, attention_mask, token_type_ids)
            hidden_states = context_oudtuts[2]
            # subword_hiddens: (batch_size, seq_len, hidden_size)
            subword_hiddens = torch.mean(torch.stack(list(hidden_states)[-self.use_last_k_layers_bert:], dim=0),
                                         dim=0)
            features.append(subword_hiddens)

        # dependencies
        if self.dep_config is not None:
            # dep_adj_matrix: (batch_size, seq_len, seq_len)
            dep_adj_matrix = torch.transpose(dep_adj_matrix, 1, 2)
            dep_type_embeddings = self.dep_type_emb(dep_adj_matrix)
            # dep_type_embeddings: (batch_size, seq_len, seq_len, dep_emb_dim)
            weight_adj = self.dep_type_emb_dropout(dep_type_embeddings)
            gcn_outputs = self.aggr_fc4gcn(torch.cat(features, dim=-1))
            for gcn_l in self.gcn_layers:
                weight_adj, gcn_outputs = gcn_l(weight_adj, gcn_outputs)  # [batch, seq, dim]
                gcn_outputs = self.dep_gcn_dropout(gcn_outputs)
                weight_adj = self.dep_gcn_dropout(weight_adj)
                features.append(gcn_outputs)

        # concatenated features
        # concatenated_hiddens: (batch_size, seq_len, concatenated_size)
        cat_hiddens = torch.cat(features, dim=-1)

        return cat_hiddens

    def forward(self):
        if self.training:
            self.bp_steps += 1

    def generate_batch(self, batch_data):
        assert len(batch_data) > 0
        batch_dict = {
            "sample_list": [sample for sample in batch_data]
        }
        seq_length = len(batch_data[0]["features"]["tok2char_span"])

        if self.subwd_encoder_config is not None:
            subword_input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            for sample in batch_data:
                subword_input_ids_list.append(sample["features"]["subword_input_ids"])
                attention_mask_list.append(sample["features"]["attention_mask"])
                token_type_ids_list.append(sample["features"]["token_type_ids"])
            batch_dict["subword_input_ids"] = torch.stack(subword_input_ids_list, dim=0)
            batch_dict["attention_mask"] = torch.stack(attention_mask_list, dim=0)
            batch_dict["token_type_ids"] = torch.stack(token_type_ids_list, dim=0)

        if self.word_encoder_config is not None:
            word_input_ids_list = [sample["features"]["word_input_ids"] for sample in batch_data]
            batch_dict["word_input_ids"] = torch.stack(word_input_ids_list, dim=0)

        if self.char_encoder_config is not None:
            char_input_ids_list = [sample["features"]["char_input_ids"] for sample in batch_data]
            batch_dict["char_input_ids"] = torch.stack(char_input_ids_list, dim=0)

        if self.ner_tag_emb_config is not None:
            ner_tag_ids_list = [sample["features"]["ner_tag_ids"] for sample in batch_data]
            batch_dict["ner_tag_ids"] = torch.stack(ner_tag_ids_list, dim=0)

        if self.pos_tag_emb_config is not None:
            pos_tag_ids_list = [sample["features"]["pos_tag_ids"] for sample in batch_data]
            batch_dict["pos_tag_ids"] = torch.stack(pos_tag_ids_list, dim=0)

        if self.dep_config is not None:
            dep_matrix_points_batch = [sample["features"]["dependency_points"] for sample in batch_data]
            batch_dict["dep_adj_matrix"] = Indexer.points2matrix_batch(dep_matrix_points_batch, seq_length)

        # batch_dict["golden_tags"] need to be set by inheritors
        return batch_dict

    @abstractmethod
    def pred_output2pred_tag(self, pred_output):
        '''
        output to tag id
        :param pred_output: the output of the forward function
        :return:
        '''
        pass

    @abstractmethod
    def get_metrics(self, pred_outputs, gold_tags):
        '''
        :param pred_outputs: the outputs of the forward function
        :param gold_tags: golden tags from batch_dict["gold_tags"]
        :return:
        '''
        pass


class TPLinkerPlus(IEModel):
    def __init__(self,
                 tagger,
                 metrics_cal,
                 handshaking_kernel_config=None,
                 fin_hidden_size=None,
                 **kwargs,
                 ):
        super().__init__(tagger, metrics_cal, **kwargs)
        '''
        :parameters: see model settings in settings_default.py
        '''

        self.tag_size = tagger.get_tag_size()
        self.metrics_cal = metrics_cal

        self.aggr_fc4handshaking_kernal = nn.Linear(self.cat_hidden_size, fin_hidden_size)

        # handshaking kernel
        shaking_type = handshaking_kernel_config["shaking_type"]
        self.handshaking_kernel = HandshakingKernel(fin_hidden_size, fin_hidden_size, shaking_type)

        # decoding fc
        self.dec_fc = nn.Linear(fin_hidden_size, self.tag_size)

    def generate_batch(self, batch_data):
        seq_length = len(batch_data[0]["features"]["tok2char_span"])
        batch_dict = super(TPLinkerPlus, self).generate_batch(batch_data)
        # shaking tag
        tag_points_batch = [sample["tag_points"] for sample in batch_data]
        batch_dict["golden_tags"] = [Indexer.points2shaking_seq_batch(tag_points_batch, seq_length, self.tag_size), ]
        return batch_dict

    def forward(self, **kwargs):
        super(TPLinkerPlus, self).forward()
        cat_hiddens = self._cat_features(**kwargs)
        cat_hiddens = self.aggr_fc4handshaking_kernal(cat_hiddens)
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        # shaking_seq_len: max_seq_len * vf - sum(1, vf)
        shaking_hiddens = self.handshaking_kernel(cat_hiddens, cat_hiddens)

        # predicted_oudtuts: (batch_size, shaking_seq_len, tag_num)
        predicted_oudtuts = self.dec_fc(shaking_hiddens)

        return predicted_oudtuts

    def pred_output2pred_tag(self, pred_output):
        return (pred_output > 0.).long()

    def get_metrics(self, pred_outputs, gold_tags):
        pred_out, gold_tag = pred_outputs, gold_tags[0]
        pred_tag = self.pred_output2pred_tag(pred_out)
        return {
            "loss": self.metrics_cal.multilabel_categorical_crossentropy(pred_out, gold_tag, self.bp_steps),
            "seq_accuracy": self.metrics_cal.get_tag_seq_accuracy(pred_tag, gold_tag),
        }


class TPLinkerPP(IEModel):
    def __init__(self,
                 tagger,
                 metrics_cal,
                 handshaking_kernel_config=None,
                 ent_dim=None,
                 rel_dim=None,
                 matrix_size=None,
                 conv_config=None,
                 inter_kernel_config=None,
                 **kwargs,
                 ):
        super().__init__(tagger, metrics_cal, **kwargs)

        self.ent_tag_size, self.rel_tag_size = tagger.get_tag_size()

        self.metrics_cal = metrics_cal

        self.aggr_fc4ent_hsk = nn.Linear(self.cat_hidden_size, ent_dim)
        self.aggr_fc4rel_hsk = nn.Linear(self.cat_hidden_size, rel_dim)

        # handshaking kernel
        ent_shaking_type = handshaking_kernel_config["ent_shaking_type"]
        rel_shaking_type = handshaking_kernel_config["rel_shaking_type"]
        self.ent_handshaking_kernel = HandshakingKernel(ent_dim,
                                                        ent_dim,
                                                        ent_shaking_type,
                                                        )
        self.rel_handshaking_kernel = HandshakingKernel(rel_dim,
                                                        rel_dim,
                                                        rel_shaking_type,
                                                        only_look_after=False,
                                                        )

        # learn local info
        self.conv_config = conv_config
        if conv_config is not None:
            self.ent_convs = nn.ModuleList()
            self.rel_convs = nn.ModuleList()
            ent_conv_layers = conv_config["ent_conv_layers"]
            rel_conv_layers = conv_config["rel_conv_layers"]
            ent_conv_kernel_size = conv_config["ent_conv_kernel_size"]
            ent_conv_padding = (ent_conv_kernel_size - 1) // 2
            rel_conv_kernel_size = conv_config["rel_conv_kernel_size"]
            rel_conv_padding = (rel_conv_kernel_size - 1) // 2
            for _ in range(ent_conv_layers):
                self.ent_convs.append(nn.Conv1d(ent_dim,
                                                ent_dim,
                                                ent_conv_kernel_size,
                                                padding=ent_conv_padding))
            for _ in range(rel_conv_layers):
                self.rel_convs.append(nn.Conv2d(rel_dim,
                                                rel_dim,
                                                rel_conv_kernel_size,
                                                padding=rel_conv_padding))

        self.inter_kernel_config = inter_kernel_config
        if self.inter_kernel_config is not None:
            self.inter_kernel = InteractionKernel(ent_dim, rel_dim, matrix_size)

        # decoding fc
        self.ent_fc = nn.Linear(ent_dim, self.ent_tag_size)
        self.rel_fc = nn.Linear(rel_dim, self.rel_tag_size)

    def generate_batch(self, batch_data):
        seq_length = len(batch_data[0]["features"]["tok2char_span"])
        batch_dict = super(TPLinkerPP, self).generate_batch(batch_data)
        # tags
        batch_ent_points = [sample["ent_points"] for sample in batch_data]
        batch_rel_points = [sample["rel_points"] for sample in batch_data]
        batch_dict["golden_tags"] = [Indexer.points2shaking_seq_batch(batch_ent_points,
                                                                      seq_length,
                                                                      self.ent_tag_size,
                                                                      ),
                                     Indexer.points2multilabel_matrix_batch(batch_rel_points,
                                                                            seq_length,
                                                                            self.rel_tag_size,
                                                                            ),
                                     ]
        return batch_dict

    def forward(self, **kwargs):
        super(TPLinkerPP, self).forward()

        cat_hiddens = self._cat_features(**kwargs)

        ent_hiddens = self.aggr_fc4ent_hsk(cat_hiddens)
        rel_hiddens = self.aggr_fc4rel_hsk(cat_hiddens)

        ent_hs_hiddens = self.ent_handshaking_kernel(ent_hiddens, ent_hiddens)
        rel_hs_hiddens = self.rel_handshaking_kernel(rel_hiddens, rel_hiddens)

        if self.conv_config is not None:
            for conv in self.ent_convs:
                ent_hs_hiddens = conv(ent_hs_hiddens.permute(0, 2, 1)).permute(0, 2, 1)
            for conv in self.rel_convs:
                rel_hs_hiddens = conv(rel_hs_hiddens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if self.inter_kernel_config is not None:
            self.inter_kernel(ent_hs_hiddens, rel_hs_hiddens)

        pred_ent_output = self.ent_fc(ent_hs_hiddens)
        pred_rel_output = self.rel_fc(rel_hs_hiddens)

        return pred_ent_output, pred_rel_output

    def pred_output2pred_tag(self, pred_output):
        return (pred_output > 0.).long()

    def get_metrics(self, pred_outputs, gold_tags):
        ent_pred_out, rel_pred_out, ent_gold_tag, rel_gold_tag = pred_outputs[0], pred_outputs[1], gold_tags[0], \
                                                                 gold_tags[1]
        ent_pred_tag = self.pred_output2pred_tag(ent_pred_out)
        rel_pred_tag = self.pred_output2pred_tag(rel_pred_out)
        loss = self.metrics_cal.multilabel_categorical_crossentropy(ent_pred_out, ent_gold_tag, self.bp_steps) + \
               self.metrics_cal.multilabel_categorical_crossentropy(rel_pred_out, rel_gold_tag, self.bp_steps)

        return {
            "loss": loss,
            "ent_seq_acc": self.metrics_cal.get_tag_seq_accuracy(ent_pred_tag, ent_gold_tag),
            "rel_seq_acc": self.metrics_cal.get_tag_seq_accuracy(rel_pred_tag, rel_gold_tag),
        }


class TriggerFreeEventExtractor(IEModel):
    def __init__(self,
                 tagger,
                 metrics_cal,
                 handshaking_kernel_config=None,
                 fin_hidden_size=None,
                 **kwargs,
                 ):
        super().__init__(tagger, metrics_cal, **kwargs)
        self.tag_size = tagger.get_tag_size()
        self.metrics_cal = metrics_cal

        self.aggr_fc4handshaking_kernal = nn.Linear(self.cat_hidden_size, fin_hidden_size)

        # handshaking kernel
        shaking_type = handshaking_kernel_config["shaking_type"]
        self.handshaking_kernel = HandshakingKernel(fin_hidden_size,
                                                    fin_hidden_size,
                                                    shaking_type,
                                                    only_look_after=False,  # full handshaking
                                                    )

        # decoding fc
        self.dec_fc = nn.Linear(fin_hidden_size, self.tag_size)

    def generate_batch(self, batch_data):
        batch_dict = super(TriggerFreeEventExtractor, self).generate_batch(batch_data)
        seq_length = len(batch_data[0]["features"]["tok2char_span"])
        # shaking tag
        tag_points_batch = [sample["tag_points"] for sample in batch_data]
        batch_dict["golden_tags"] = [
            Indexer.points2multilabel_matrix_batch(tag_points_batch, seq_length, self.tag_size), ]
        return batch_dict

    def forward(self, **kwargs):
        super(TriggerFreeEventExtractor, self).forward()

        cat_hiddens = self._cat_features(**kwargs)
        cat_hiddens = self.aggr_fc4handshaking_kernal(cat_hiddens)
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        # shaking_seq_len: max_seq_len * vf - sum(1, vf)
        shaking_hiddens = self.handshaking_kernel(cat_hiddens, cat_hiddens)

        # predicted_oudtuts: (batch_size, shaking_seq_len, tag_num)
        predicted_oudtuts = self.dec_fc(shaking_hiddens)

        return predicted_oudtuts

    def pred_output2pred_tag(self, pred_output):
        return (pred_output > 0.).long()

    def get_metrics(self, pred_outputs, gold_tags):
        pred_out, gold_tag = pred_outputs, gold_tags[0]
        pred_tag = self.pred_output2pred_tag(pred_out)
        return {
            "loss": self.metrics_cal.multilabel_categorical_crossentropy(pred_out, gold_tag, self.bp_steps),
            "seq_accuracy": self.metrics_cal.get_tag_seq_accuracy(pred_tag, gold_tag),
        }
