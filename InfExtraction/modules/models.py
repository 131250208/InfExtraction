import torch.nn as nn
import torch
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import BertModel
from InfExtraction.modules.model_components import HandshakingKernel, GraphConvLayer
from InfExtraction.modules.metrics import MetricsCalculator
from gensim.models import KeyedVectors
import logging


class IEModel(metaclass=ABCMeta):
    @abstractmethod
    def generate_batch(self, batch_data):
        '''
        generate batch data for training
        :return:
        '''
        pass

    @abstractmethod
    def get_loss(self, pred_tag, gold_tag):
        pass


class TPLinkerPlus(nn.Module, IEModel):
    def __init__(self,
                 tag_size,
                 tagger,
                 char_encoder_config=None,
                 subwd_encoder_config=None,
                 word_encoder_config=None,
                 ner_tag_emb_config=None,
                 pos_tag_emb_config=None,
                 dep_config=None,
                 handshaking_kernel_config=None,
                 fin_hidden_size=None,
                 ):
        super().__init__()
        '''
        char_encoder_config = {
            "char_size": len(char2idx), 
            "emb_dim": char_emb_dim,
            "emb_dropout": char_emb_dropout,
            "bilstm_layers": char_bilstm_layers,
            "bilstm_dropout": char_bilstm_dropout,
            "max_char_num_in_tok": max_char_num_in_tok,
        }
        subwd_encoder_config = {
            "path": encoder_path,
            "fintune": bert_finetune,
            "use_last_k_layers": use_last_k_layers_hiddens,
        }
        word_encoder_config = {
            "init_word_embedding_matrix": init_word_embedding_matrix,
            "emb_dropout": word_emb_dropout,
            "bilstm_layers": word_bilstm_layers,
            "bilstm_dropout": word_bilstm_dropout,
            "freeze_word_emb": freeze_word_emb,
            "init_word_embedding_matrix": init_word_embedding_matrix,
        }
        handshaking_kernel_config = {
            "shaking_type": hyper_parameters["shaking_type"],
        }
        '''
        self.metrics_cal = MetricsCalculator()
        self.tagger = tagger
        combined_hidden_size_1 = 0

        # ner
        self.ner_tag_emb_config = ner_tag_emb_config
        if ner_tag_emb_config is not None:
            ner_tag_num = ner_tag_emb_config["ner_tag_num"]
            ner_tag_emb_dim = ner_tag_emb_config["emb_dim"]
            ner_tag_emb_dropout = ner_tag_emb_config["emb_dropout"]
            self.ner_tag_emb = nn.Embedding(ner_tag_num, ner_tag_emb_dim)
            self.ner_tag_emb_dropout = nn.Dropout(p=ner_tag_emb_dropout)
            combined_hidden_size_1 += ner_tag_emb_dim

        # pos
        self.pos_tag_emb_config = pos_tag_emb_config
        if pos_tag_emb_config is not None:
            pos_tag_num = pos_tag_emb_config["pos_tag_num"]
            pos_tag_emb_dim = pos_tag_emb_config["emb_dim"]
            pos_tag_emb_dropout = pos_tag_emb_config["emb_dropout"]
            self.pos_tag_emb = nn.Embedding(pos_tag_num, pos_tag_emb_dim)
            self.pos_tag_emb_dropout = nn.Dropout(p=pos_tag_emb_dropout)
            combined_hidden_size_1 += pos_tag_emb_dim

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
            combined_hidden_size_1 += char_bilstm_hidden_size[1]

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
                if word in pretrained_emb:
                    hit_count += 1
                    init_word_embedding_matrix[idx] = pretrained_emb[word]
            print("pretrained word embedding hit rate: {}".format(hit_count / len(word2id)))

            init_word_embedding_matrix = torch.FloatTensor(init_word_embedding_matrix)
            word_emb_dim = init_word_embedding_matrix.size()[1]

            ## word encoder model
            self.word_emb = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=freeze_word_emb)
            self.word_emb_dropout = nn.Dropout(p=word_emb_dropout)
            self.word_lstm_l1 = nn.LSTM(word_emb_dim,
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
            combined_hidden_size_1 += word_bilstm_hidden_size[1]

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
            combined_hidden_size_1 += self.bert.config.hidden_size

        # aggregate fc 1
        self.aggr_fc_1 = nn.Linear(combined_hidden_size_1, fin_hidden_size)

        # dependencies
        self.dep_config = dep_config
        if dep_config is not None:
            self.dep_type_num = dep_config["dep_type_num"]
            dep_type_emb_dim = dep_config["dep_type_emb_dim"]
            dep_type_emb_dropout = dep_config["emb_dropout"]
            dep_gcn_dim = dep_config["gcn_dim"]
            dep_gcn_dropout = dep_config["gcn_dropout"]
            dep_gcn_layer_num = dep_config["gcn_layer_num"]
            self.dep_type_emb = nn.Embedding(self.dep_type_num * 2, dep_type_emb_dim)
            self.dep_type_emb_dropout = nn.Dropout(p=dep_type_emb_dropout)
            # GCN
            self.gcn_layers = nn.ModuleList()
            self.dep_gcn_dropout = nn.Dropout(dep_gcn_dropout)
            combined_hidden_size_2 = combined_hidden_size_1
            for _ in range(dep_gcn_layer_num):
                self.gcn_layers.append(GraphConvLayer(dep_type_emb_dim, dep_gcn_dim, "avg"))
                combined_hidden_size_2 += dep_gcn_dim
            # aggregate fc 2
            self.aggr_fc_2 = nn.Linear(combined_hidden_size_2, fin_hidden_size)

        # handshaking kernel
        shaking_type = handshaking_kernel_config["shaking_type"]
        self.handshaking_kernel = HandshakingKernel(fin_hidden_size, shaking_type)

        # decoding fc
        self.dec_fc = nn.Linear(fin_hidden_size, tag_size)

    def generate_batch(self, batch_data):
        batch_dict = {}
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

        seq_length = batch_dict["subword_input_ids"].size()[1]

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
            batch_dict["dep_adj_matrix"] = self.tagger.points2matrix_batch(dep_matrix_points_batch, seq_length)

        # must
        sample_list = []
        tag_points_batch = []
        for sample in batch_data:
            sample_list.append(sample)
            tag_points_batch.append(sample["tag_points"])
        batch_dict["sample_list"] = sample_list
        # shaking tag
        batch_dict["shaking_tag"] = self.tagger.points2tag_batch(tag_points_batch, seq_length)
        return batch_dict

    def forward(self,
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
            # word_input_emb/word_hiddens: batch_size, seq_len, word_emb_dim)
            word_input_emb = self.word_emb(word_input_ids)
            word_input_emb = self.word_emb_dropout(word_input_emb)
            word_hiddens, _ = self.word_lstm_l1(word_input_emb)
            word_hiddens, _ = self.word_lstm_l2(self.word_lstm_dropout(word_hiddens))
            features.append(word_hiddens)

        # subword
        if self.subwd_encoder_config is not None:
            # subword_input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
            context_oudtuts = self.bert(subword_input_ids, attention_mask, token_type_ids)
            # last_hidden_state: (batch_size, seq_len, hidden_size)
            hidden_states = context_oudtuts[2]
            subword_hiddens = torch.mean(torch.stack(list(hidden_states)[-self.use_last_k_layers_bert:], dim=0), dim=0)
            features.append(subword_hiddens)

        # combine features
        # combined_hiddens: (batch_size, seq_len, combined_size)
        combined_hiddens = self.aggr_fc_1(torch.cat(features, dim=-1))

        # dependencies
        if self.dep_config is not None:
            # dep_adj_matrix: (batch_size, seq_len, seq_len)
            dep_adj_matrix = (dep_adj_matrix + self.dep_type_num) + torch.transpose(dep_adj_matrix, 1, 2)
            dep_type_embeddings = self.dep_type_emb(dep_adj_matrix)
            # dep_type_embeddings: (batch_size, seq_len, seq_len, dep_emb_dim)
            weight_adj = self.dep_type_emb_dropout(dep_type_embeddings)
            gcn_outputs = combined_hiddens
            for gcn_l in self.gcn_layers:
                gcn_outputs, weight_adj = gcn_l(weight_adj, gcn_outputs)  # [batch, seq, dim]
                gcn_outputs = self.gcn_drop(gcn_outputs)
                weight_adj = self.gcn_drop(weight_adj)
                features.append(gcn_outputs)
            combined_hiddens = self.aggr_fc_2(torch.cat(features, dim=-1))

        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        # shaking_seq_len: max_seq_len * vf - sum(1, vf)
        shaking_hiddens = self.handshaking_kernel(combined_hiddens)

        # predicted_oudtuts: (batch_size, shaking_seq_len, tag_num)
        predicted_oudtuts = self.dec_fc(shaking_hiddens)

        return predicted_oudtuts

    def get_loss(self, pred_tag, gold_tag, ghm=False):
        return self.metrics_cal.multilabel_categorical_crossentropy(pred_tag, gold_tag, ghm)


class TriggerFreeEventExtraction(nn.Module, IEModel):
    def get_loss(self, pred_tag, gold_tag):
        pass

    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def generate_batch(self, batch_data):
        pass