import re
from tqdm import tqdm
import copy
from transformers import BertTokenizerFast
import stanza
from InfExtraction.modules.utils import MyMatrix, save_as_json_lines, load_data
from InfExtraction.modules import utils
import torch
from pprint import pprint
import os
import Levenshtein
import hashlib
import time
import json

class Indexer:
    def __init__(self, tag2id, max_seq_len, spe_tag_dict):
        self.tag2id = tag2id
        self.max_seq_len = max_seq_len
        self.spe_tag_dict = spe_tag_dict

    def index_tag_list_w_matrix_pos(self, tags):
        '''
        :param tags: [[pos_i, pos_j, tag1], [pos_i, pos_j, tag2], ...]
        :return:
        '''
        for t in tags:
            if t[2] in self.tag2id:
                t[2] = self.tag2id[t[2]]
            else:
                t[2] = self.spe_tag_dict["[UNK]"]
        return tags

    @staticmethod
    def pad2length(tags, padding_tag, length):
        if len(tags) < length:
            tags.extend([padding_tag] * (length - len(tags)))
        return tags[:length]

    def index_tag_list(self, tags):
        '''
        tags: [t1, t2, t3, ...]
        '''
        tag_ids = []
        for t in tags:
            if t not in self.tag2id:
                tag_ids.append(self.spe_tag_dict["[UNK]"])
            else:
                tag_ids.append(self.tag2id[t])

        if len(tag_ids) < self.max_seq_len:
            tag_ids.extend([self.spe_tag_dict["[PAD]"]] * (self.max_seq_len - len(tag_ids)))

        return tag_ids[:self.max_seq_len]

    @staticmethod
    def get_shaking_idx2matrix_idx(matrix_size):
        return MyMatrix.get_shaking_idx2matrix_idx(matrix_size)

    @staticmethod
    def get_matrix_idx2shaking_idx(matrix_size):
        return MyMatrix.get_matrix_idx2shaking_idx(matrix_size)

    @staticmethod
    def points2multilabel_shaking_seq(points, matrix_size, tag_size):
        '''
        Convert points to a shaking sequence tensor

        points: [(start_ind, end_ind, tag_id), ]
        return:
            shaking_seq: (shaking_seq_len, tag_size)
        '''
        matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        shaking_seq_len = matrix_size * (matrix_size + 1) // 2
        shaking_seq = torch.zeros(shaking_seq_len, tag_size).long()
        for sp in points:
            shaking_idx = matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_seq[shaking_idx][sp[2]] = 1
        return shaking_seq

    @staticmethod
    def points2multilabel_shaking_seq_batch(batch_points, matrix_size, tag_size):
        '''
        Convert points to a shaking sequence tensor in batch (for training tags)

        batch_points: a batch of points, [points1, points2, ...]
            points: [(start_ind, end_ind, tag_id), ]
        return:
            batch_shaking_seq: (batch_size_train, shaking_seq_len, tag_size)
        '''
        matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        shaking_seq_len = matrix_size * (matrix_size + 1) // 2
        batch_shaking_seq = torch.zeros(len(batch_points), shaking_seq_len, tag_size).long()
        for batch_id, points in enumerate(batch_points):
            for sp in points:
                shaking_idx = matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_seq[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_seq

    @staticmethod
    def points2shaking_seq_batch(batch_points, matrix_size):
        '''
        Convert points to a shaking sequence tensor

        batch_points: a batch of points, [points1, points2, ...]
            points: [(start_ind, end_ind, tag_id), ]
        return:
            batch_shaking_seq: (batch_size_train, shaking_seq_len)
        '''
        matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        shaking_seq_len = matrix_size * (matrix_size + 1) // 2
        batch_shaking_seq = torch.zeros(len(batch_points), shaking_seq_len).long()
        for batch_id, points in enumerate(batch_points):
            for sp in points:
                try:
                    shaking_idx = matrix_idx2shaking_idx[sp[0]][sp[1]]
                except Exception:
                    print("debug sk")
                batch_shaking_seq[batch_id][shaking_idx] = sp[2]
        return batch_shaking_seq

    @staticmethod
    def points2matrix_batch(batch_points, matrix_size):
        '''
        Convert points to a matrix tensor

        batch_points: a batch of points, [points1, points2, ...]
            points: [(start_ind, end_ind, tag_id), ]
        return:
            batch_matrix: (batch_size_train, matrix_size, matrix_size)
        '''
        batch_matrix = torch.zeros(len(batch_points), matrix_size, matrix_size).long()
        for batch_id, points in enumerate(batch_points):
            for pt in points:
                batch_matrix[batch_id][pt[0]][pt[1]] = pt[2]
        return batch_matrix

    @staticmethod
    def points2multilabel_matrix_batch(batch_points, matrix_size, tag_size):
        '''
        Convert points to a matrix tensor for multi-label tasks

        batch_points: a batch of points, [points1, points2, ...]
            points: [(i, j, tag_id), ]
        return:
            batch_matrix: shape: (batch_size_train, matrix_size, matrix_size, tag_size) # element 0 or 1
        '''
        batch_matrix = torch.zeros(len(batch_points), matrix_size, matrix_size, tag_size).long()
        for batch_id, points in enumerate(batch_points):
            for pt in points:
                batch_matrix[batch_id][pt[0]][pt[1]][pt[2]] = 1
        return batch_matrix

    @staticmethod
    def shaking_seq2points(shaking_tag):
        '''
        shaking_tag -> points
        shaking_tag: shape: (shaking_seq_len, tag_size)
        points: [(start_ind, end_ind, tag_id), ]
        '''
        points = []
        shaking_seq_len = shaking_tag.size()[0]
        matrix_size = int((2 * shaking_seq_len + 0.25) ** 0.5 - 0.5)
        shaking_idx2matrix_idx = Indexer.get_shaking_idx2matrix_idx(matrix_size)
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = shaking_idx2matrix_idx[shaking_idx]
            point = (pos1, pos2, tag_idx)
            points.append(point)
        return points

    @staticmethod
    def matrix2points(matrix_tag):
        '''
        matrix_tag -> points
        matrix_tag: shape: (matrix_size, matrix_size, tag_size)
        points: [(i, j, tag_id), ]
        '''
        points = []
        nonzero_points = torch.nonzero(matrix_tag, as_tuple=False)
        for point in nonzero_points:
            i, j, tag_idx = point[0].item(), point[1].item(), point[2].item()
            point = (i, j, tag_idx)
            points.append(point)
        return points


class StanzaWordTokenizer:
    '''
    word level tokenizer,
    for word level encoders (LSTM, GRU, etc.)
    '''

    def __init__(self, stanza_nlp):
        # self.word2idx = word2idx
        self.stanza_nlp = stanza_nlp
        # self.word_indexer = Indexer(word2idx, max_seq_len, {"[UNK]": word2idx["[UNK]"], "[PAD]": word2idx["[PAD]"]})
        # self.pos_indexer = Indexer(pos2id, max_seq_len, {"[UNK]": pos2id["[UNK]"], "[PAD]": pos2id["[PAD]"]})
        # self.ner_tag_indexer = Indexer(ner_tag2id, max_seq_len, {"[UNK]": ner_tag2id["O"], "[PAD]": ner_tag2id["O"]})
        # self.deprel_indexer = Indexer(deprel2id, max_seq_len, {"[UNK]": deprel2id["[UNK]"], "[PAD]": deprel2id["[PAD]"]})

    # def get_stanza(self):
    #     if self.stanza_nlp is None:
    #         self.stanza_nlp = stanza.Pipeline('en')
    #     return self.stanza_nlp

    def tokenize(self, text):
        return [word.text for sent in self.stanza_nlp(text).sentences for word in sent.words]

    # def text2word_indices(self, text):
    #     # if not self.word2idx:
    #     #     raise ValueError(
    #     #         "if you invoke function text2word_indices, self.word2idx should be set when initialize StanzaWordTokenizer")
    #     words = self.tokenize(text)
    #     return self.word_indexer.get_indices(words)

    def tokenize_plus(self, text):
        word_list = []
        tok2char_span = []
        ner_tag_list = []
        pos_tag_list = []
        dependency_list = []
        for sent in self.stanza_nlp(text).sentences:
            for token in sent.tokens:
                net_tag = token.ner
                for word in token.words:
                    word_list.append(word.text)
                    start_char, end_char = word.misc.split("|")
                    start_char, end_char = int(start_char.split("=")[1]), int(end_char.split("=")[1])
                    tok2char_span.append([start_char, end_char])
                    ner_tag_list.append(net_tag)
                    pos_tag_list.append(word.xpos)
                    dependency_list.append([word.head - word.id if word.head != 0 else 0, word.deprel])

        res = {
            "word_list": word_list,
            "word2char_span": tok2char_span,
            "ner_tag_list": ner_tag_list,
            "pos_tag_list": pos_tag_list,
            "dependency_list": dependency_list,
        }

        # if self.ner_tag_indexer is not None:
        #     res["ner_tag_ids"] = self.ner_tag_indexer.get_indices(ner_tag_list)
        # if self.pos_indexer is not None:
        #     res["pos_tag_ids"] = self.pos_indexer.get_indices(pos_tag_list)
        # if self.deprel_indexer is not None:
        #     deprel_ids = self.deprel_indexer.get_indices([dep[1] for dep in dependency_list])
        #     deprel_heads = [dep[0] for dep in dependency_list]
        #     deprel_heads.extend([0] * (len(deprel_ids) - len(deprel_heads))) # padding with 0
        #
        #     res["dependency_list"] = [[head, deprel_ids[idx]] for idx, head in enumerate(deprel_heads)]

        return res


class WhiteWordTokenizer:
    '''
    word level tokenizer,
    for word level encoders (LSTM, GRU, etc.)
    '''

    @staticmethod
    def tokenize(text, ent_list=None, span_list=None):
        '''
        :param text:
        :param ent_list: ground truth entity list, to ensure the text would be split from the correct boundaries
        :param span_list: ground truth span list, to ensure the text would be split from the correct boundaries
        :return:
        '''
        boundary_ids = set()
        if ent_list is not None and len(ent_list) > 0:
            for ent in ent_list:
                for m in re.finditer(re.escape(ent), text):
                    boundary_ids.add(m.span()[0])
                    boundary_ids.add(m.span()[1])

        if span_list is not None and len(span_list) > 0:
            for sp in span_list:
                boundary_ids = boundary_ids.union(set(sp))

        if len(boundary_ids) > 0:
            split_ids = [0] + sorted(list(boundary_ids)) + [len(text)]
            segs = []
            for idx, split_id in enumerate(split_ids):
                if idx == len(split_ids) - 1:
                    break
                segs.append(text[split_id:split_ids[idx + 1]])
        else:
            segs = [text]

        word_list = []
        for seg in segs:
            word_list.extend(seg.split(" "))
        return word_list

    @staticmethod
    def get_tok2char_span_map(tokens, text=None):
        '''
        :param tokens:
        :param text: must be given if tokens are not all separated by whitespaces, e.g. text = "hello.", tokens = ["hello", "."]
        :return:
        '''
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append([char_num, char_num + len(tok)])
            char_num += len(tok)
            if text is not None and (char_num > len(text) - 1 or text[char_num] != " "):
                pass
            else:
                char_num += 1
        return tok2char_span

    @staticmethod
    def tokenize_plus(text, ent_list=None, span_list=None):
        word_list = WhiteWordTokenizer.tokenize(text, ent_list, span_list)
        res = {
            "word_list": word_list,
            "word2char_span": WhiteWordTokenizer.get_tok2char_span_map(word_list, text),
        }
        for wid, char_sp in enumerate(res["word2char_span"]):
            extr_word = text[char_sp[0]: char_sp[1]]
            assert word_list[wid] == extr_word
        return res


class ChineseWordTokenizer:
    @staticmethod
    def tokenize(text, ent_list=None, span_list=None, rm_blanks=False):
        '''
        :param text:
        :param ent_list: tokenize by entities first
        :return:
        '''
        boundary_ids = set()
        if ent_list is not None and len(ent_list) > 0:
            for ent in ent_list:
                for m in re.finditer(re.escape(ent), text):
                    boundary_ids.add(m.span()[0])
                    boundary_ids.add(m.span()[1])

        if span_list is not None and len(span_list) > 0:
            for sp in span_list:
                boundary_ids = boundary_ids.union(set(sp))

        if len(boundary_ids) > 0:
            split_ids = [0] + sorted(list(boundary_ids)) + [len(text)]
            segs = []
            for idx, split_id in enumerate(split_ids):
                if idx == len(split_ids) - 1:
                    break
                segs.append(text[split_id:split_ids[idx + 1]])
        else:
            segs = [text]

        word_pattern = "[0-9]+|\[[A-Z]+\]|[a-zA-Z]+|[^0-9a-zA-Z]"
        word_list = []
        for seg in segs:
            word_list.extend(re.findall(word_pattern, seg))

        if rm_blanks:
            word_list = [w for w in word_list if re.sub("\s+", "", w) != ""]
        return word_list

    @staticmethod
    def get_tok2char_span_map(word_list):
        return utils.get_tok2char_span_map4ch(word_list)

    @staticmethod
    def tokenize_plus(text, ent_list=None, span_list=None):
        word_list = ChineseWordTokenizer.tokenize(text, ent_list, span_list)
        res = {
            "word_list": word_list,
            "word2char_span": ChineseWordTokenizer.get_tok2char_span_map(word_list),
        }
        return res


class BertTokenizerAlignedWithStanza(BertTokenizerFast):
    '''
    why need this class?
       text: Its favored cities include Boston , Washington , Los Angeles , Seattle , San Francisco and Oakland .
       stanza tokenizer: ['It', 's', 'favored', 'cities', 'include', 'Boston', ',', 'Washington', ',', 'Los', 'Angeles', ',', 'Seattle', ',', 'San', 'Francisco', 'and', 'Oakland', '.']
       bert tokenizer: ['Its', 'favored', 'cities', 'include', 'Boston', ',', 'Washington', ',', 'Los', 'Angeles', ',', 'Seattle', ',', 'San', 'Francisco', 'and', 'Oakland', '.']

       so we need to align bert tokenizer with stanza tokenizer
   '''

    def __init__(self, *args, **kwargs):
        super(BertTokenizerAlignedWithStanza, self).__init__(*args, **kwargs)
        self.stanza_language = kwargs["stanza_language"]
        self.stanza_nlp = None

    def get_stanza_nlp(self):
        if self.stanza_nlp is None:
            self.stanza_nlp = stanza.Pipeline(self.stanza_language)
        return self.stanza_nlp

    def tokenize_fr_words(self, words, max_length=None, *args, **kwargs):
        text = " ".join(words)
        tokens = super(BertTokenizerAlignedWithStanza, self).tokenize(text, *args, **kwargs)

        if max_length is not None:
            if max_length > len(tokens):
                tokens.extend(["[PAD]"] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
        return tokens

    def tokenize(self, text, max_length=None, *args, **kwargs):
        words_by_stanza = [word.text for sent in self.get_stanza_nlp()(text).sentences for word in sent.words]
        return self.tokenize_fr_words(words_by_stanza, max_length=max_length, *args, **kwargs)

    # def encode_plus(self, text, *args, **kwargs):
    #     words_by_stanza = []
    #     word2char_span = []
    #     for sent in self.get_stanza_nlp()(text).sentences:
    #         for word in sent.words:
    #             words_by_stanza.append(word.text)
    #             start_char, end_char = word.misc.split("|")
    #             start_char, end_char = int(start_char.split("=")[1]), int(end_char.split("=")[1])
    #             word2char_span.append([start_char, end_char])
    #
    #     return self.encode_plus_fr_words(words_by_stanza, word2char_span, *args, **kwargs)

    def encode_plus_fr_words(self, words, word2char_span, *args, **kwargs):
        text = " ".join(words)

        new_char_ids2ori_char_ids = []
        for char_sp in word2char_span:
            for char_id in range(char_sp[0], char_sp[1]):
                new_char_ids2ori_char_ids.append(char_id)
            new_char_ids2ori_char_ids.append(-1)  # whitespace = -1

        features = super(BertTokenizerAlignedWithStanza, self).encode_plus(text, *args, **kwargs)

        if "offset_mapping" in features:
            new_offset_mapping = []
            for char_span in features["offset_mapping"]:
                if char_span[1] == 0:
                    new_offset_mapping.append([0, 0])
                    continue
                char_ids = new_char_ids2ori_char_ids[char_span[0]:char_span[1]]
                new_offset_mapping.append([char_ids[0], char_ids[-1] + 1])
            features["offset_mapping"] = new_offset_mapping

        max_length = kwargs["max_length"] if "max_length" in kwargs else None

        features["subword_list"] = self.tokenize_fr_words(words, max_length=max_length)

        return features


class Preprocessor:
    def __init__(self, language, pretrained_model_path, do_lower_case):
        self.subword_tokenizer = None
        self.word_tokenizer = None
        self.language = language
        self.pretrained_model_path = pretrained_model_path
        self.do_lower_case = do_lower_case

    @staticmethod
    def unique_list(inp_list):
        return utils.unique_list(inp_list)

    @staticmethod
    def list_equal(list1, list2):
        if len(list1) != len(list2):
            return False

        memory = {str(it) for it in list1}
        for item in list2:
            if str(item) not in memory:
                return False
        return True

    def get_word_tokenizer(self, type="white"):
        '''
        :param type: whitespace or stanza
        :return:
        '''
        if self.word_tokenizer is None:
            if type == "stanza":
                stanza_nlp = stanza.Pipeline(self.language)
                self.word_tokenizer = StanzaWordTokenizer(stanza_nlp)
            elif type == "white":
                self.word_tokenizer = WhiteWordTokenizer()
            elif type == "normal_chinese":
                self.word_tokenizer = ChineseWordTokenizer()
        return self.word_tokenizer

    def get_subword_tokenizer(self):
        if self.subword_tokenizer is None:
            self.subword_tokenizer = BertTokenizerAlignedWithStanza.from_pretrained(self.pretrained_model_path,
                                                                                    add_special_tokens=False,
                                                                                    do_lower_case=self.do_lower_case,
                                                                                    stanza_language=self.language)
            print("tokenizer loaded: {}".format(self.pretrained_model_path))
        return self.subword_tokenizer

    def _get_char2tok_span(self, tok2char_span):
        return utils.get_char2tok_span(tok2char_span)

    def _get_ent2char_spans(self, text, entities, ignore_subword_match=True):
        '''
        a dict mapping an entity to all possible character level spans
        it is used for adding character level spans for all entities
        e.g. {"entity1": [[0, 1], [18, 19]]}
        if ignore_subword, look for entities with whitespace around, e.g. "entity" -> " entity "
        '''
        entities = sorted(set(entities), key=lambda x: len(x), reverse=True)
        text_cp = " {} ".format(text) if ignore_subword_match else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword_match else ent
            # if not in, try lower
            if target_ent not in text_cp:
                target_ent = target_ent.lower()
                text_cp = text_cp.lower()
            for m in re.finditer(re.escape(target_ent), text_cp):
                # if consider subword, avoid matching an inner number, e.g. "76567" -> "65",
                # or it would introduce too many errors.
                # if not ignore_subword_match and \
                #         (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1]) and re.match("^\d+", target_ent)) \
                #         or (m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]]) and re.match("\d+$", target_ent)):
                #         continue
                if not ignore_subword_match and utils.is_invalid_extr_ent(target_ent, m.span(), text_cp):
                    continue
                # if ignore_subword_match, we use " {original entity} " to match. So, we need to mv span
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else [*m.span()]
                spans.append(span)
            ent2char_spans[ent] = spans
        return ent2char_spans

    @staticmethod
    def trans_duee(sample):
        text = sample["text"]

        # event list
        if "event_list" in sample:  # train or valid data
            normal_event_list = []
            for event in sample["event_list"]:
                normal_event = copy.deepcopy(event)

                # rm whitespaces
                clean_tri = normal_event["trigger"].lstrip()
                normal_event["trigger_start_index"] += len(normal_event["trigger"]) - len(clean_tri)
                normal_event["trigger"] = clean_tri.rstrip()

                normal_event["trigger_char_span"] = [normal_event["trigger_start_index"],
                                                     normal_event["trigger_start_index"] + len(
                                                         normal_event["trigger"])]
                char_span = normal_event["trigger_char_span"]
                assert text[char_span[0]:char_span[1]] == normal_event["trigger"]
                del normal_event["trigger_start_index"]

                normal_arg_list = []
                for arg in normal_event["arguments"]:
                    # clean whitespaces
                    clean_arg = arg["argument"].lstrip()
                    arg["argument_start_index"] += len(arg["argument"]) - len(clean_arg)
                    arg["argument"] = clean_arg.rstrip()

                    char_span = [arg["argument_start_index"],
                                 arg["argument_start_index"] + len(arg["argument"])]
                    assert text[char_span[0]:char_span[1]] == arg["argument"]
                    normal_arg_list.append({
                        "text": arg["argument"],
                        "type": arg["role"],
                        "char_span": char_span,
                    })
                normal_event["argument_list"] = normal_arg_list
                del normal_event["arguments"]
                normal_event_list.append(normal_event)
            sample["event_list"] = normal_event_list

    @staticmethod
    def trans_duee_fin(sample):
        def clean_txt(txt):
            txt = re.sub("\s+", " ", txt)
            return txt

        arg_fix_map = {
            'Wondery 表情': 'Wondery',
            "InterVes": "InterVest",
        }

        ents = set()
        sample["text"] = clean_txt(sample["title"] + "[SEP]" + sample["text"])
        text = sample["text"]
        # if "乐声电子回购14万股涉资约17.22万港元" in text or \
        #         "厦门金牌厨柜股份有限公司关于控股股东部分股票质押及解除质押的公告" in text or \
        #         "【重要公告】中公教育：上半年净利预增100%-135%；中环环保：联合预中标3.16亿元项目" in text:
        #     print("debug!")

        # fix cases
        if sample["id"] == 'cba17a27928c657bf6781c3ecdfd8f37':
            for event in sample["event_list"]:
                for arg in event["arguments"]:
                    if arg["argument"] == '019年8月22日':
                        arg["argument"] = '2019年8月22日'

        if sample["id"] == 'c2e95b239cafba85ca348503a8742440':
            for event in sample["event_list"]:
                for arg in event["arguments"]:
                    if arg["argument"] == '5个交易日后的6个月内':
                        arg["argument"] = '15个交易日后的6个月内'

        if sample["id"] == '3d2fee76e7442e68be78b4be5fee5804':
            for event in sample["event_list"]:
                if event["event_type"] == "质押":
                    arg_list = []
                    for arg in event["arguments"]:
                        if arg["argument"] == '3.04%' and arg["role"] == '质押物占总股比':
                            continue
                        arg_list.append(arg)
                    event["arguments"] = arg_list

        if "event_list" not in sample:
            sample["event_list"] = []
        else:
            for event in sample["event_list"]:
                ents.add(event["trigger"])
                for arg in event["arguments"]:
                    if event["event_type"] == "公司上市" and arg["role"] == "环节":
                        continue
                    arg["argument"] = clean_txt(arg["argument"])
                    if arg["argument"] in arg_fix_map:
                        arg["argument"] = arg_fix_map[arg["argument"]]
                    ents.add(arg["argument"])

            new_event_list = []
            for event in sample["event_list"]:
                # trigger_ch_sp = [[*m.span()] for m in re.finditer(re.escape(event["trigger"]), text)]
                # assert len(trigger_ch_sp) > 0
                new_event = {
                    "event_type": event["event_type"],
                    "trigger": event["trigger"],
                    # "trigger_char_span": trigger_ch_sp,
                    "argument_list": []
                }

                for arg in event["arguments"]:
                    if arg["role"] == "环节" and event["event_type"] == "公司上市":
                        new_event["event_type"] = arg["argument"]
                    else:
                        # arg_spans = [[*m.span()] for m in re.finditer(re.escape(arg["argument"]), text)]
                        # assert len(arg_spans) > 0
                        new_event["argument_list"].append({"text": arg["argument"],
                                                           "type": arg["role"],
                                                           # "char_span": arg_spans,
                                                           })
                new_event_list.append(new_event)
            sample["event_list"] = new_event_list

    @staticmethod
    def trans_duie_1(sample):
        if "spo_list" in sample:
            rel_list, ent_list = [], []
            for spo in sample["spo_list"]:
                rel_list.append({
                    "subject": spo["subject"],
                    "object": spo["object"],
                    "predicate": spo["predicate"],
                })
                ent_list.append({
                    "text": spo["subject"],
                    "type": spo["subject_type"],
                })
                ent_list.append({
                    "text": spo["object"],
                    "type": spo["object_type"],
                })

            sample["entity_list"] = Preprocessor.unique_list(ent_list)
            sample["relation_list"] = Preprocessor.unique_list(rel_list)

    @staticmethod
    def trans_duie_2(sample):
        if "spo_list" in sample:
            rel_list, ent_list = [], []
            for spo in sample["spo_list"]:
                rel_list.append({
                    "subject": spo["subject"],
                    "object": spo["object"]["@value"],
                    "predicate": spo["predicate"],
                })
                ent_list.append({
                    "text": spo["subject"],
                    "type": spo["subject_type"],
                })
                ent_list.append({
                    "text": spo["object"]["@value"],
                    "type": spo["object_type"]["@value"],
                })

                for k, item in spo["object"].items():
                    if k == "@value":
                        continue
                    rel_list.append({
                        "subject": spo["subject"],
                        "object": item,
                        "predicate": k,
                    })
                    rel_list.append({
                        "subject": spo["object"]["@value"],
                        "object": item,
                        "predicate": k,
                    })
                    ent_list.append({
                        "text": item,
                        "type": spo["object_type"][k],
                    })
            sample["entity_list"] = Preprocessor.unique_list(ent_list)
            sample["relation_list"] = Preprocessor.unique_list(rel_list)

    @staticmethod
    def transform_data(data, ori_format, dataset_type, add_id=True):
        '''
        This function is for transforming data published by previous works.
        data: original data
        ori_format: "casrel", "etl_span", "raw_nyt", "tplinker", etc.
        dataset_type: "train", "valid", "test"; only for generate id for the data
        '''
        for ind, sample in enumerate(data):
            normal_sample = {}
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            else:
                assert "id" in sample, "miss id in data!"
                normal_sample["id"] = sample["id"]

            if ori_format == "duie_1":
                Preprocessor.trans_duie_1(sample)

            if ori_format == "duie_2":
                Preprocessor.trans_duie_2(sample)

            if ori_format == "duee_1":
                Preprocessor.trans_duee(sample)
            if ori_format == "duee_fin":
                Preprocessor.trans_duee_fin(sample)

            if ori_format in {"normal", "duee_fin", "duee_1", "duie_2"}:
                yield {**normal_sample, **sample}
                continue

            text, rel_list, subj_key, pred_key, obj_key = None, None, None, None, None
            if ori_format == "casrel":
                text = sample["text"]
                rel_list = sample["triple_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "etl_span":
                text = " ".join(sample["tokens"])
                rel_list = sample["spo_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "raw_nyt":
                text = sample["sentText"]
                rel_list = sample["relationMentions"]
                subj_key, pred_key, obj_key = "em1Text", "label", "em2Text"
            normal_sample["text"] = text

            normal_rel_list = []
            normal_ent_list = []
            if ori_format != "raw_nyt":
                for rel in rel_list:
                    normal_rel_list.append({
                        "subject": rel[subj_key],
                        "predicate": rel[pred_key],
                        "object": rel[obj_key],
                    })
                    normal_ent_list.append({
                        "text": rel[subj_key],
                        "type": "DEFAULT",
                    })
                    normal_ent_list.append({
                        "text": rel[obj_key],
                        "type": "DEFAULT",
                    })
            else:
                for ent in sample["entityMentions"]:
                    normal_ent_list.append({
                        "text": ent["text"],
                        "type": ent["label"],
                    })

                for rel in rel_list:
                    normal_rel_list.append({
                        "subject": rel[subj_key],
                        "predicate": rel[pred_key],
                        "object": rel[obj_key],
                    })
            normal_sample["relation_list"] = normal_rel_list
            normal_sample["entity_list"] = normal_ent_list

            # clean
            if ori_format in {"casrel", "etl_span", "raw_nyt"}:
                def clean_text(text):
                    text = re.sub("�", "", text)
                    text = re.sub("([,;.?!]+)", r" \1 ", text)
                    text = re.sub("\s+", " ", text).strip()
                    return text

                normal_sample["text"] = clean_text(sample["text"])
                for ent in sample["entity_list"]:
                    ent["text"] = clean_text(ent["text"])

                for rel in normal_sample["relation_list"]:
                    rel["subject"] = clean_text(rel["subject"])
                    rel["object"] = clean_text(rel["object"])

            yield normal_sample

    @staticmethod
    def pre_check_data_annotation(data, language):
        def check_ent_span(entity_list, text):
            for ent in entity_list:
                ent_ext_fr_span = Preprocessor.extract_ent_fr_txt_by_char_sp(ent["char_span"], text, language)
                if ent["text"] != ent_ext_fr_span:
                    raise Exception("char span error: ent_text: {} != ent_ext_fr_span: {}".format(ent["text"],
                                                                                                  ent_ext_fr_span))

        for sample in data:
            text = sample["text"]

            if "entity_list" in sample:
                check_ent_span(sample["entity_list"], text)

            if "relation_list" in sample:
                entities_fr_rel = []
                for rel in sample["relation_list"]:
                    entities_fr_rel.append({
                        "text": rel["subject"],
                        "char_span": [*rel["subj_char_span"]]
                    })

                    entities_fr_rel.append({
                        "text": rel["object"],
                        "char_span": [*rel["obj_char_span"]]
                    })
                entities_fr_rel = Preprocessor.unique_list(entities_fr_rel)
                check_ent_span(entities_fr_rel, text)

                entities_mem = {str({"text": ent["text"], "char_span": ent["char_span"]})
                                for ent in sample["entity_list"]}
                for ent in entities_fr_rel:
                    if str(ent) not in entities_mem:
                        # print("entity list misses some entities in relation list")
                        raise Exception("entity list misses some entities in relation list")

            if "event_list" in sample:
                entities_fr_event = []
                for event in sample["event_list"]:
                    if "trigger" in event:
                        if type(event["trigger_char_span"][0]) is list:
                            for ch_sp in event["trigger_char_span"]:
                                entities_fr_event.append({
                                    "text": event["trigger"],
                                    "char_span": [*ch_sp],
                                })
                        else:
                            entities_fr_event.append({
                                "text": event["trigger"],
                                "char_span": [*event["trigger_char_span"]]
                            })
                    for arg in event["argument_list"]:
                        if type(arg["char_span"][0]) is list:
                            for ch_sp in arg["char_span"]:
                                entities_fr_event.append({
                                    "text": arg["text"],
                                    "char_span": [*ch_sp],
                                })
                        else:
                            entities_fr_event.append({
                                "text": arg["text"],
                                "char_span": [*arg["char_span"]]
                            })

                entities_fr_event = Preprocessor.unique_list(entities_fr_event)
                check_ent_span(entities_fr_event, text)

            if "open_spo_list" in sample:
                for spo in sample["open_spo_list"]:
                    check_ent_span(spo, text)
            yield sample

    def add_char_span(self, dataset, ignore_subword_match=True):
        '''
        if the dataset has not annotated character level spans, add them automatically
        :param dataset:
        :param ignore_subword_match: if a word is a subword of another word, ignore its span.
        :return:
        '''
        for sample in dataset:
            entities = []
            if "relation_list" in sample:
                entities.extend([rel["subject"] for rel in sample["relation_list"]])
                entities.extend([rel["object"] for rel in sample["relation_list"]])
            if "entity_list" in sample:
                entities.extend([ent["text"] for ent in sample["entity_list"]])
            if "event_list" in sample:
                for event in sample["event_list"]:
                    if "trigger" in event:
                        entities.append(event["trigger"])
                    entities.extend([arg["text"] for arg in event["argument_list"]])

            entities = Preprocessor.unique_list(entities)
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities,
                                                      ignore_subword_match=ignore_subword_match)
            for ent, sps in ent2char_spans.items():
                if len(sps) == 0:
                    print("\n>>>>>>>>>>>>>entity: {} not found!>>>>>>>>>>>>>>>>>".format(ent))
                    pprint(sample)

            if "relation_list" in sample:
                relation_list = []
                for rel in sample["relation_list"]:
                    subj_char_spans = ent2char_spans[rel["subject"]]
                    obj_char_spans = ent2char_spans[rel["object"]]
                    for subj_sp in subj_char_spans:
                        for obj_sp in obj_char_spans:
                            rel_cp = copy.deepcopy(rel)
                            rel_cp["subj_char_span"] = subj_sp
                            rel_cp["obj_char_span"] = obj_sp
                            relation_list.append(rel_cp)
                    # try:
                    #     # assert len(sample["relation_list"]) <= len(relation_list)
                    #     assert len(subj_char_spans) > 0 and len(obj_char_spans) > 0
                    # except Exception as e:
                    #     print("miss relations")
                    #     print(ent2char_spans)
                    #     print(sample["text"])
                    #     pprint(sample["relation_list"])
                    #     print("==========================")
                sample["relation_list"] = relation_list

            if "entity_list" in sample:
                new_ent_list = []
                for ent in sample["entity_list"]:
                    for char_sp in ent2char_spans[ent["text"]]:
                        new_ent_list.append({
                            "text": ent["text"],
                            "type": ent["type"],
                            "char_span": char_sp,
                        })
                # try:
                #     assert len(new_ent_list) >= len(sample["entity_list"])
                # except Exception as e:
                #     print("miss entities")
                sample["entity_list"] = new_ent_list

            if "event_list" in sample:
                for event in sample["event_list"]:
                    if "trigger" in event:
                        char_spans = ent2char_spans[event["trigger"]]
                        event["trigger_char_span"] = char_spans if len(char_spans) > 0 else [[]]
                    for arg in event["argument_list"]:
                        char_spans = ent2char_spans[arg["text"]]
                        arg["char_span"] = char_spans if len(char_spans) > 0 else [[]]
            yield sample

    def add_tok_span(self, data):
        '''
        add token level span according to the character spans, character level spans are required
        '''

        def char_span2tok_span(char_span, char2tok_span):
            tok_span = []
            for idx in range(0, len(char_span), 2):  # len(char_span) > 2 if discontinuous entity
                if char_span[-1] == 0:
                    return char_span
                ch_sp = [char_span[idx], char_span[idx + 1]]
                tok_span_list = char2tok_span[ch_sp[0]:ch_sp[1]]
                try:
                    tok_span.extend([tok_span_list[0][0], tok_span_list[-1][1]])
                except Exception:
                    print("error in char_span2tok_span!")
            return tok_span

        for sample in data:
            char2word_span = self._get_char2tok_span(sample["features"]["word2char_span"])
            char2subwd_span = self._get_char2tok_span(sample["features"]["subword2char_span"])

            # anns = {"relation_list", "entity_list", "event_list", "open_spo_list",
            #         "pre_relation_list", "pre_entity_list", "pos_tag_list_csp", "dependency_list_csp"}

            def add_sps(a_list):
                for item in a_list:
                    if type(item) is list:
                        add_sps(item)
                    elif type(item) is dict:
                        add_dict = {}
                        for k, v in item.items():
                            if "char_span" in k:
                                add_wd_sp_k = re.sub("char_span", "wd_span", k)
                                add_subwd_sp_k = re.sub("char_span", "subwd_span", k)

                                if type(v[0]) is list:
                                    add_dict[add_wd_sp_k] = [char_span2tok_span(ch_sp, char2word_span) for ch_sp in v]
                                    add_dict[add_subwd_sp_k] = [char_span2tok_span(ch_sp, char2subwd_span) for ch_sp in
                                                                v]
                                else:
                                    assert type(v[0]) is int
                                    add_dict[add_wd_sp_k] = char_span2tok_span(v, char2word_span)
                                    add_dict[add_subwd_sp_k] = char_span2tok_span(v, char2subwd_span)
                            elif type(v) is list:
                                add_sps(v)
                        item.update(add_dict)

            for k, v in sample.items():
                if type(v) is list:
                    add_sps(v)
                elif type(v) is dict:
                    for val in v.values():
                        if type(val) is list:
                            add_sps(val)
            yield sample
            # if "relation_list" in sample:
            #     for rel in sample["relation_list"]:
            #         subj_char_span = rel["subj_char_span"]
            #         obj_char_span = rel["obj_char_span"]
            #         rel["subj_wd_span"] = char_span2tok_span(subj_char_span, char2word_span)
            #         rel["obj_wd_span"] = char_span2tok_span(obj_char_span, char2word_span)
            #         rel["subj_subwd_span"] = char_span2tok_span(subj_char_span, char2subwd_span)
            #         rel["obj_subwd_span"] = char_span2tok_span(obj_char_span, char2subwd_span)
            #
            # if "entity_list" in sample:
            #     for ent in sample["entity_list"]:
            #         char_span = ent["char_span"]
            #         ent["wd_span"] = char_span2tok_span(char_span, char2word_span)
            #         ent["subwd_span"] = char_span2tok_span(char_span, char2subwd_span)
            #
            # if "event_list" in sample:
            #     for event in sample["event_list"]:
            #         if "trigger" in event:
            #             if type(event["trigger_char_span"][0]) is list:
            #                 event["trigger_wd_span"] = [char_span2tok_span(ch_sp, char2word_span)
            #                                             for ch_sp in event["trigger_char_span"]]
            #                 event["trigger_subwd_span"] = [char_span2tok_span(ch_sp, char2subwd_span)
            #                                             for ch_sp in event["trigger_char_span"]]
            #             else:
            #                 event["trigger_wd_span"] = char_span2tok_span(event["trigger_char_span"], char2word_span)
            #                 event["trigger_subwd_span"] = char_span2tok_span(event["trigger_char_span"], char2subwd_span)
            #         for arg in event["argument_list"]:
            #             if type(arg["char_span"][0]) is list:
            #                 arg["wd_span"] = [char_span2tok_span(ch_sp, char2word_span) for ch_sp in arg["char_span"]]
            #                 arg["subwd_span"] = [char_span2tok_span(ch_sp, char2subwd_span) for ch_sp in arg["char_span"]]
            #             else:
            #                 arg["wd_span"] = char_span2tok_span(arg["char_span"], char2word_span)
            #                 arg["subwd_span"] = char_span2tok_span(arg["char_span"], char2subwd_span)
            #
            # if "open_spo_list" in sample:
            #     for spo in sample["open_spo_list"]:
            #         for arg in spo:
            #             if "char_span" in arg:
            #                 arg["wd_span"] = char_span2tok_span(arg["char_span"], char2word_span)
            #                 arg["subwd_span"] = char_span2tok_span(arg["char_span"], char2subwd_span)

    @staticmethod
    def search_char_spans_fr_txt(target_seg, text, language, merge_sps=True):
        if target_seg == "" or target_seg is None:
            return [[0, 0]], ""

        add_text = re.sub("\S", "_", target_seg)

        # if continuous
        if language == "ch" and target_seg in text:
            # span = [*re.search(re.escape(target_seg), text).span()] # 0407
            candidate_spans = [[*m.span()] for m in re.finditer(re.escape(target_seg), text)]
            return candidate_spans, add_text

        if language == "en" and " {} ".format(target_seg) in " {} ".format(text):
            # span = [*re.search(re.escape(" {} ".format(target_seg)), " {} ".format(text)).span()]
            candidate_spans = [[m.span()[0], m.span()[1] - 2]
                               for m in re.finditer(re.escape(" {} ".format(target_seg)), " {} ".format(text))]
            # return [span[0], span[1] - 2], add_text
            return candidate_spans, add_text

        # # discontinuous but in the same order
        # if language == "ch":
        #     words = ChineseWordTokenizer.tokenize(target_seg)
        # elif language == "en":
        #     words = target_seg.split(" ")
        #
        # words = [re.escape(w) for w in words]
        # pattern = "(" + ").*?(".join(words) + ")"
        #
        # match_list = None
        # try:
        #     match_list = list(re.finditer(pattern, text))
        # except Exception:
        #     print("search error!")
        #     print(target_seg)
        #     print(text)
        #     print("================")
        #
        # if len(match_list) > 0:  # discontinuous but in the same order
        #     candidate_spans = []
        #     for m in match_list:
        #         spans = []
        #         for sp in list(m.regs)[1:]:
        #             spans.extend([*sp])
        #         candidate_spans.append(spans)
        #
        # else:  # reversed order, or some words are not in the original text
        seg_list = utils.search_segs(target_seg, text, {"[", "]", "|"})
        # if language == "ch":
        #     seg_list = ChineseWordTokenizer.tokenize(target_seg)
        # elif language == "en":
        #     seg_list = target_seg.split(" ")

        seg2spans = {}
        m_list = []
        text_cp = text[:]
        for sbwd_idx, sbwd in enumerate(sorted(seg_list, key=lambda s: len(s), reverse=True)):
            finditer = re.finditer(re.escape(" {} ".format(sbwd)), " {} ".format(text_cp)) \
                if language == "en" else re.finditer(re.escape(sbwd),
                                                     text_cp)  # a bug to fix: if language == "en", span should be [m[0], m[1] - 2]
            for m in finditer:
                m_list.append(m)

                # word_idx2spans
                if m.group() not in seg2spans:
                    seg2spans[m.group()] = []
                seg2spans[m.group()].append(m)

                # mask
                sp = m.span()
                text_ch_list = list(text_cp)
                text_ch_list[sp[0]:sp[1]] = ["_"] * (sp[1] - sp[0])
                text_cp = "".join(text_ch_list)

        seg_list = [s for s in seg_list if s in seg2spans]

        word2surround_sps = {}
        for sbwd_idx, sbwd in enumerate(seg_list):
            pre_spans = seg2spans[seg_list[sbwd_idx - 1]] if sbwd_idx != 0 else []
            # try:
            post_spans = seg2spans[seg_list[sbwd_idx + 1]] if sbwd_idx != len(seg_list) - 1 else []
            # except Exception:
            #     print("TTTTT")
            if sbwd not in word2surround_sps:
                word2surround_sps[sbwd] = {}
            word2surround_sps[sbwd]["pre"] = pre_spans
            word2surround_sps[sbwd]["post"] = post_spans

        dist_map = [0] * len(m_list)
        for mid_i, mi in enumerate(m_list):
            sur_sps = word2surround_sps[mi.group()]
            # try:
            pre_dists = [min(abs(mi.span()[0] - mj.span()[1]), abs(mi.span()[1] - mj.span()[0])) for mj in
                         sur_sps["pre"]]
            dist_map[mid_i] += min(pre_dists) if len(pre_dists) > 0 else 0
            post_dists = [min(abs(mi.span()[0] - mj.span()[1]), abs(mi.span()[1] - mj.span()[0])) for mj in
                          sur_sps["post"]]
            dist_map[mid_i] += min(post_dists) if len(post_dists) > 0 else 0

            # except Exception:
            #     print("!!!!!!")
            # for mid_j, mj in enumerate(m_list):
            #     if mid_i == mid_j:
            #         continue
            #     dist_map[mid_i] += min(abs(mi.span()[0] - mj.span()[1]), abs(mi.span()[1] - mj.span()[0]))

        m_list_ = [{"score": dist_map[mid], "mention": m} for mid, m in enumerate(m_list)]
        word2cand_sps = {}

        m_list_ = sorted(m_list_, key=lambda m: m["score"], reverse=True)
        for m in m_list_:
            if m["mention"].group() not in word2cand_sps:
                word2cand_sps[m["mention"].group()] = []
            word2cand_sps[m["mention"].group()].append(m["mention"])

        # choose the most cohesive spans as candidates
        cand_spans = []
        add_list = []
        for wd in seg_list:
            if wd in word2cand_sps:
                # last word first
                last_w = word2cand_sps[wd].pop() if len(word2cand_sps[wd]) > 1 else word2cand_sps[wd][-1]
                cand_spans.append(last_w.span())
                add_list.append("_")
            else:
                add_list.append(wd)
        add_text = "".join(add_list)

        spans = []
        for idx, sp in enumerate(cand_spans):
            spans.extend(sp)

        candidate_spans = [spans]

        # merge continuous spans
        new_candidate_spans = []
        for spans in candidate_spans:
            new_spans = utils.merge_spans(spans) if merge_sps else spans
            new_candidate_spans.append(new_spans)

        return new_candidate_spans, add_text

    @staticmethod
    def extract_ent_fr_txt_by_char_sp(char_span, text, language=None):
        return utils.extract_ent_fr_txt_by_char_sp(char_span, text, language)

    @staticmethod
    def tok_span2char_span(tok_span, tok2char_span):
        char_span = []
        if len(tok_span) == 0:
            return []

        for idx in range(0, len(tok_span), 2):
            tk_sp = [tok_span[idx], tok_span[idx + 1]]
            if tk_sp[-1] == 0:
                return tok_span
            try:
                char_span_list = tok2char_span[tk_sp[0]:tk_sp[1]]
                char_span.extend([char_span_list[0][0], char_span_list[-1][1]])
            except Exception:
                print("tok_span2char_span!")
        return char_span

    @staticmethod
    def extract_ent_fr_txt_by_tok_sp(tok_span, tok2char_span, text, language):
        char_span = Preprocessor.tok_span2char_span(tok_span, tok2char_span)
        return Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text, language)

    # @staticmethod
    # def extract_ent_fr_toks(tok_span, toks, language):
    #     segs = []
    #     for idx in range(0, len(tok_span), 2):
    #         segs.extend(toks[tok_span[idx]:tok_span[idx + 1]])
    #     return utils.join_segs(segs)

    @staticmethod
    def check_tok_span(data, language):
        '''
        check if text is equal to the one extracted by the annotated token level spans
        :param data: 
        :return: 
        '''

        for sample in data:
            text = sample["text"]
            word2char_span = sample["features"]["word2char_span"]
            subword2char_span = sample["features"]["subword2char_span"]

            if "entity_list" in sample:
                bad_entities = []
                for ent in sample["entity_list"]:
                    word_span = ent["wd_span"]
                    subword_span = ent["subwd_span"]
                    ent_wd = Preprocessor.extract_ent_fr_txt_by_tok_sp(word_span, word2char_span, text, language)
                    ent_subwd = Preprocessor.extract_ent_fr_txt_by_tok_sp(subword_span, subword2char_span, text,
                                                                          language)

                    if not (ent_wd == ent_subwd == ent["text"]):
                        bad_ent = copy.deepcopy(ent)
                        bad_ent["extr_ent_wd"] = ent_wd
                        bad_ent["extr_ent_subwd"] = ent_subwd
                        bad_entities.append(bad_ent)
                if len(bad_entities) > 0:
                    print(text)
                    print(bad_entities)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            if "relation_list" in sample:
                bad_rels = []
                for rel in sample["relation_list"]:
                    subj_wd_span = rel["subj_wd_span"]
                    obj_wd_span = rel["obj_wd_span"]
                    subj_subwd_span = rel["subj_subwd_span"]
                    obj_subwd_span = rel["obj_subwd_span"]

                    subj_wd = Preprocessor.extract_ent_fr_txt_by_tok_sp(subj_wd_span, word2char_span, text, language)
                    obj_wd = Preprocessor.extract_ent_fr_txt_by_tok_sp(obj_wd_span, word2char_span, text, language)
                    subj_subwd = Preprocessor.extract_ent_fr_txt_by_tok_sp(subj_subwd_span, subword2char_span, text,
                                                                           language)
                    obj_subwd = Preprocessor.extract_ent_fr_txt_by_tok_sp(obj_subwd_span, subword2char_span, text,
                                                                          language)

                    if not (subj_wd == rel["subject"] == subj_subwd and obj_wd == rel["object"] == obj_subwd):
                        bad_rel = copy.deepcopy(rel)
                        bad_rel["extr_subj_wd"] = subj_wd
                        bad_rel["extr_subj_subwd"] = subj_subwd
                        bad_rel["extr_obj_wd"] = obj_wd
                        bad_rel["extr_obj_subwd"] = obj_subwd
                        bad_rels.append(bad_rel)
                if len(bad_rels) > 0:
                    print(text)
                    print(bad_rels)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            if "event_list" in sample:
                bad_events = []
                for event in sample["event_list"]:
                    bad_event = copy.deepcopy(event)
                    bad = False

                    if "trigger" in event:
                        trigger_wd_span = event["trigger_wd_span"]
                        trigger_subwd_span = event["trigger_subwd_span"]

                        if type(trigger_wd_span[0]) is list or type(trigger_subwd_span[0]) is list:
                            pass
                        else:
                            trigger_wd_span = [trigger_wd_span, ]
                            trigger_subwd_span = [trigger_subwd_span, ]

                        for sp_idx, wd_sp in enumerate(trigger_wd_span):
                            # try:
                            subwd_sp = trigger_subwd_span[sp_idx]
                            # except Exception:
                            #     print("??????")
                            extr_trigger_wd = Preprocessor.extract_ent_fr_txt_by_tok_sp(wd_sp, word2char_span, text,
                                                                                        language)
                            extr_trigger_subwd = Preprocessor.extract_ent_fr_txt_by_tok_sp(subwd_sp, subword2char_span,
                                                                                           text, language)

                            if not (extr_trigger_wd == extr_trigger_subwd == event["trigger"]):
                                bad = True
                                bad_event.setdefault("extr_trigger_wd", []).append(extr_trigger_wd)
                                bad_event.setdefault("extr_trigger_subwd", []).append(extr_trigger_subwd)

                    for arg in bad_event["argument_list"]:
                        arg_wd_span = arg["wd_span"]
                        arg_subwd_span = arg["subwd_span"]
                        if type(arg_wd_span[0]) is list or type(arg_subwd_span[0]) is list:
                            pass
                        else:
                            arg_wd_span = [arg_wd_span, ]
                            arg_subwd_span = [arg_subwd_span, ]

                        for sp_idx, wd_sp in enumerate(arg_wd_span):
                            subwd_sp = arg_subwd_span[sp_idx]
                            extr_arg_wd = Preprocessor.extract_ent_fr_txt_by_tok_sp(wd_sp, word2char_span, text,
                                                                                    language)
                            extr_arg_subwd = Preprocessor.extract_ent_fr_txt_by_tok_sp(subwd_sp, subword2char_span,
                                                                                       text, language)

                            if not (extr_arg_wd == extr_arg_subwd == arg["text"]):
                                bad = True
                                bad_event.setdefault("extr_arg_wd", []).append(extr_arg_wd)
                                bad_event.setdefault("extr_arg_subwd", []).append(extr_arg_subwd)

                    if bad:
                        bad_events.append(bad_event)
                if len(bad_events) > 0:
                    print(text)
                    print(bad_events)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            yield sample

    @staticmethod
    def get_all_possible_char_spans(sample):
        sp_list = []
        if "entity_list" in sample:
            # ent_list.extend([ent["text"] for ent in sample["entity_list"]])
            sp_list.extend([ent["char_span"] for ent in sample["entity_list"]])

        if "relation_list" in sample:
            # ent_list.extend([spo["subject"] for spo in sample["relation_list"]])
            # ent_list.extend([spo["object"] for spo in sample["relation_list"]])
            sp_list.extend([spo["subj_char_span"] for spo in sample["relation_list"]])
            sp_list.extend([spo["obj_char_span"] for spo in sample["relation_list"]])

        if "event_list" in sample:
            for event in sample["event_list"]:
                if "trigger" in event:
                    if type(event["trigger_char_span"][0]) is list:
                        sp_list.extend(event["trigger_char_span"])
                    else:
                        sp_list.append(event["trigger_char_span"])
                for arg in event["argument_list"]:
                    if type(arg["char_span"][0]) is list:
                        sp_list.extend(arg["char_span"])
                    else:
                        sp_list.append(arg["char_span"])

        if "open_spo_list" in sample:
            for spo in sample["open_spo_list"]:
                for arg in spo:
                    # ent_list.append(arg["text"])
                    if "char_span" not in arg or len(arg["char_span"]) == 0:
                        continue
                    sp_list.append(arg["char_span"])

        return sp_list

    @staticmethod
    def get_all_possible_entities(sample):
        ent_list = []
        if "entity_list" in sample:
            ent_list.extend([ent["text"] for ent in sample["entity_list"]])

        if "relation_list" in sample:
            ent_list.extend([spo["subject"] for spo in sample["relation_list"]])
            ent_list.extend([spo["object"] for spo in sample["relation_list"]])

        if "event_list" in sample:
            for event in sample["event_list"]:
                if "trigger" in event:
                    ent_list.append(event["trigger"])
                for arg in event["argument_list"]:
                    ent_list.append(arg["text"])

        if "open_spo_list" in sample:
            for spo in sample["open_spo_list"]:
                for arg in spo:
                    if len(arg["char_span"]) == 0:
                        continue
                    ent_list.append(arg["text"])

        return Preprocessor.unique_list(ent_list)

    @staticmethod
    def exist_nested_entities(sp_list):
        return utils.exist_nested_entities(sp_list)

    def create_features(self, data, word_tokenizer_type="white",
                        parse_format=None,
                        parse_results=None,
                        ent_spo_extractor=None
                        ):
        '''
        :param data:
        :param word_tokenizer_type: stanza, white, normal_chinese;
        :return:
        '''
        # create features
        for sample_idx, sample in enumerate(data):
            text = sample["text"]

            # word level
            word_features = {}
            if "word_list" not in sample or "word2char_span" not in sample:
                # generate word_list, word2char_span
                word_tokenizer = self.get_word_tokenizer(word_tokenizer_type)
                if word_tokenizer_type in {"white", "normal_chinese"}:
                    all_sps = Preprocessor.get_all_possible_char_spans(sample)
                    wd_tok_res = word_tokenizer.tokenize_plus(text, span_list=all_sps)
                else:
                    wd_tok_res = word_tokenizer.tokenize_plus(text)
                sample["word_list"] = wd_tok_res["word_list"]
                sample["word2char_span"] = wd_tok_res["word2char_span"]

            # subword level features
            # sample["word_list"] = ["".join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')
            #                        for w in sample["word_list"]]  # rm accents
            codes = self.get_subword_tokenizer().encode_plus_fr_words(sample["word_list"],
                                                                      sample["word2char_span"],
                                                                      return_offsets_mapping=True,
                                                                      add_special_tokens=False,
                                                                      )
            subword_features = {
                "subword_list": codes["subword_list"],
                "subword2char_span": codes["offset_mapping"],
            }

            if parse_results is not None:  # (ner_tag_list, pos_tag_list, dependency_list)
                if parse_format == "ddp":
                    ctok_word2char_span = sample["word2char_span"]
                    cchar2tok_span = utils.get_char2tok_span(ctok_word2char_span)

                    ddp_res = parse_results[sample_idx]
                    ddp_tok2char_span = utils.get_tok2char_span_map4ch(ddp_res["word"])
                    ddp_char2tok_span = utils.get_char2tok_span(ddp_tok2char_span)

                    pos_tag_list_csp, deprel_list_csp = [], []

                    def invalid_tok4bert(csp):
                        return csp[-1] > subword_features["subword2char_span"][-1][-1]
                        # return tok.strip().strip("[\s|\u200b|\xad|�|\uf447]+").strip() == ""

                    for ddp_tok_id, pos_tag in enumerate(ddp_res["postag"]):
                        ch_sp = ddp_tok2char_span[ddp_tok_id]
                        wd = ddp_res["word"][ddp_tok_id]
                        assert wd == text[ch_sp[0]:ch_sp[1]]
                        if not invalid_tok4bert(ch_sp):
                            pos_tag_list_csp.append({
                                # "word": ddp_res["word"][ddp_tok_id],
                                "type": pos_tag,
                                "char_span": ch_sp,
                            })
                    for ddp_tok_id, deprel in enumerate(ddp_res["deprel"]):
                        head_id = ddp_res["head"][ddp_tok_id] - 1
                        tail = ddp_res["word"][ddp_tok_id]
                        head = ddp_res["word"][head_id]
                        tail_char_span = ddp_tok2char_span[ddp_tok_id]
                        head_char_span = ddp_tok2char_span[head_id]
                        assert tail == text[tail_char_span[0]:tail_char_span[1]]
                        assert head == text[head_char_span[0]:head_char_span[1]]
                        if not invalid_tok4bert(tail_char_span) and not invalid_tok4bert(head_char_span):
                            deprel_list_csp.append({
                                # "tail": tail,
                                "subj_char_span": tail_char_span,
                                # "head": head,
                                "obj_char_span": head_char_span,
                                "predicate": deprel
                            })
                    sample["dependency_list_csp"] = deprel_list_csp
                    sample["pos_tag_list_csp"] = pos_tag_list_csp

                    # align to word list
                    pos_tag_list, deprel_list = [], []
                    for wid, word in enumerate(sample["word_list"]):
                        ch_sp = ctok_word2char_span[wid]  # word to char span
                        tok_sps = ddp_char2tok_span[ch_sp[0]:ch_sp[1]]  # char span to ddp tok span
                        # try:
                        #     assert tok_sps[0][0] == tok_sps[-1][1] - 1
                        # except:
                        #     print("pre duie")
                        if tok_sps[0][0] == tok_sps[-1][
                            1] - 1:  # if this word corresponds to a single token in ddp result
                            ddp_tok_id = tok_sps[0][0]
                        else:
                            # if this word corresponds to multiple tokens in ddp result,
                            # find the most similar token by edit distance
                            # ignore "[" and "]"
                            ddp_tok_id = None
                            min_dis = 9999
                            for tk_idx in range(tok_sps[0][0], tok_sps[-1][1]):
                                ddp_wd = ddp_res["word"][tk_idx]
                                l_dis = Levenshtein.distance(re.sub("[\[\]]", "", ddp_wd), re.sub("[\[\]]", "", word))
                                if l_dis < min_dis:
                                    min_dis = l_dis
                                    ddp_tok_id = tk_idx
                        pos_tag = ddp_res["postag"][ddp_tok_id]
                        pos_tag_list.append(pos_tag)

                        ddp_head_tok_id = ddp_res["head"][ddp_tok_id] - 1
                        deprel = ddp_res["deprel"][ddp_tok_id]

                        ddp_head_ch_sp = ddp_tok2char_span[ddp_head_tok_id]
                        ddp_head_ctok_spans = cchar2tok_span[ddp_head_ch_sp[0]:ddp_head_ch_sp[1]]
                        for head_ctok_idx in range(ddp_head_ctok_spans[0][0], ddp_head_ctok_spans[-1][1]):
                            deprel_list.append([wid, head_ctok_idx, deprel])

                    ent_tag = {"TIME", "PER", "ORG", "LOC"}
                    ent_list = []

                    start_idx = None
                    for tok_idx, tag in enumerate(pos_tag_list):
                        if tag not in ent_tag:
                            start_idx = None

                        if tag in ent_tag and (tok_idx - 1 < 0 or pos_tag_list[tok_idx - 1] != tag):
                            start_idx = tok_idx
                        if tag in ent_tag and (tok_idx + 1 >= len(pos_tag_list) or pos_tag_list[tok_idx + 1] != tag):
                            end_idx = tok_idx + 1
                            # try:
                            assert start_idx is not None
                            # except Exception:
                            #     print("de")
                            char_spans = ctok_word2char_span[start_idx:end_idx]
                            char_span = [char_spans[0][0], char_spans[-1][1]]
                            ent_list.append({
                                "char_span": char_span,
                                "text": text[char_span[0]:char_span[1]],
                                "type": tag,
                            })
                    sample["pre_entity_list"] = ent_list
                    sample["pos_tag_list"] = pos_tag_list
                    sample["dependency_list"] = deprel_list

            # extracted possible entities and relations by dicts
            if ent_spo_extractor is not None:
                ent_list, spo_list = ent_spo_extractor.extract_items(text,
                                                                     add_entities=sample.get("pre_entity_list", []))
                ent_list += sample.get("pre_entity_list", [])

                # filter ground truth
                golden_ent_list = sample.get("entity_list", [])
                golden_rel_list = sample.get("relation_list", [])
                # if len(golden_ent_list) != 0 and len(golden_rel_list) != 0:
                #     print("!debug")
                golden_ent_marks = {",".join([ent["text"], ent["type"]])
                                    for ent in golden_ent_list}
                golden_rel_marks = {",".join([rel["subject"], rel["predicate"], rel["object"]])
                                    for rel in golden_rel_list}
                sample["pre_entity_list"] = [ent for ent in ent_list
                                             if ",".join([ent["text"], ent["type"]]) not in golden_ent_marks]
                sample["pre_relation_list"] = [spo for spo in spo_list
                                               if ",".join([spo["subject"], spo["predicate"], spo["object"]])
                                               not in golden_rel_marks]

            for key in {"ner_tag_list", "word_list", "pos_tag_list", "dependency_list", "word2char_span",
                        "dependency_list_csp", "pos_tag_list_csp"}:
                if key in sample:
                    word_features[key] = sample[key]
                    del sample[key]

            sample["features"] = word_features

            ## generate subword2word_id
            char2word_span = self._get_char2tok_span(sample["features"]["word2char_span"])

            subword2word_id = []
            for subw_id, char_sp in enumerate(subword_features["subword2char_span"]):
                wd_sps = char2word_span[char_sp[0]:char_sp[1]]
                assert wd_sps[0][0] == wd_sps[-1][1] - 1  # the same word idx
                subword2word_id.append(wd_sps[0][0])

            ## generate word2subword_span
            word2subword_span = [[-1, -1] for _ in range(len(sample["features"]["word_list"]))]
            for subw_id, wid in enumerate(subword2word_id):
                if word2subword_span[wid][0] == -1:
                    word2subword_span[wid][0] = subw_id
                word2subword_span[wid][1] = subw_id + 1

            # align
            if "dependency_list" in word_features:
                ## transform word dependencies to matrix point
                word_dependency_list = word_features["dependency_list"]
                if len(word_dependency_list[0]) == 2:
                    new_word_dep_list = [[wid, dep[0] + wid, dep[1]] for wid, dep in enumerate(word_dependency_list)]
                else:
                    assert len(word_dependency_list[0]) == 3
                    new_word_dep_list = word_dependency_list
                word_features["word_dependency_list"] = new_word_dep_list
                del word_features["dependency_list"]

                ## generate subword level dependency list
                subword_dep_list = []
                for dep in sample["features"]["word_dependency_list"]:
                    for subw_id1 in range(*word2subword_span[dep[0]]):  # debug
                        try:
                            for subw_id2 in range(*word2subword_span[dep[1]]):
                                subword_dep_list.append([subw_id1, subw_id2, dep[2]])
                        except IndexError as e:
                            print("index error")
                subword_features["subword_dependency_list"] = subword_dep_list

            # add subword level features into the feature list
            sample["features"] = {
                **sample["features"],
                **subword_features,
                "subword2word_id": subword2word_id,
                "word2subword_span": word2subword_span,
            }

            # check
            feats = sample["features"]
            num_words = len(word2subword_span)
            for k in {"ner_tag_list", "pos_tag_list", "word2char_span", "word_list"}:
                if k in feats:
                    # try:
                    assert len(feats[k]) == num_words
                    # except Exception:
                    #     print("!")
            assert len(feats["subword_list"]) == len(feats["subword2char_span"]) == len(subword2word_id)
            for subw_id, wid in enumerate(subword2word_id):
                subw = sample["features"]["subword_list"][subw_id]
                word = sample["features"]["word_list"][wid]

                subw_ = re.sub("##", "", subw)

                if re.match("^[\uAC00-\uD7FFh]+$", word) is not None:  # skip korean
                    continue
                word = utils.rm_accents(word)
                subw_ = utils.rm_accents(subw_)

                try:
                    if self.do_lower_case:
                        assert subw_.lower() in word.lower() or subw_ == "[UNK]"
                    else:
                        assert subw_ in word or subw_ == "[UNK]"
                except Exception:
                    print("subw({}) not in word({})".format(subw_, word))

            for subw_id, char_sp in enumerate(feats["subword2char_span"]):
                subw = sample["features"]["subword_list"][subw_id]
                subw = re.sub("##", "", subw)
                subw_extr = sample["text"][char_sp[0]:char_sp[1]]

                if re.match("^[\uAC00-\uD7FFh]+$", subw_extr) is not None:
                    continue
                subw_extr = utils.rm_accents(subw_extr)
                subw = utils.rm_accents(subw)
                try:
                    if self.do_lower_case:
                        assert subw_extr.lower() == subw.lower() or subw == "[UNK]"
                    else:
                        assert subw_extr == subw or subw == "[UNK]"
                except Exception:
                    print("subw_extr({}) != subw({})".format(subw_extr, subw))

            yield sample

    def generate_supporting_data(self, data_paths, max_word_dict_size, min_word_freq):
        pos_tag_list = list()
        ner_tag_list = list()
        deprel_type_list = list()
        word2num = dict()
        word_set = set()
        char_set = set()

        rel_type_set = set()

        ent_type_set = set()

        event_type_set = set()
        argument_type_set = set()

        # # oie
        oie_arg_type_set = set()

        max_word_seq_length, max_subword_seq_length = 0, 0
        ent_exist, rel_exist, event_exist, oie_exist = False, False, False, False

        max_ann_subwd_span = 0
        max_ann_wd_span = 0

        pos_tag, ner_tag, wd_dep = False, False, False

        for data_path in data_paths:
            with open(data_path, "r", encoding="utf-8") as file_in:
                for line in tqdm(file_in, desc="gen sup data 4 {}".format(data_path)):
                    sample = json.loads(line)
                    min_subwd_span_start = 99999
                    max_subwd_span_end = 0
                    min_wd_span_start = 99999
                    max_wd_span_end = 0

                    t1 = time.time()
                    # POS tag
                    if "pos_tag_list" in sample["features"]:
                        pos_tag_list.extend(sample["features"]["pos_tag_list"])
                    # NER tag
                    if "ner_tag_list" in sample["features"]:
                        ner_tag_list.extend(sample["features"]["ner_tag_list"])
                    # dependency relations
                    if "word_dependency_list" in sample["features"]:
                        deprel_type_list.extend([deprel[-1] for deprel in sample["features"]["word_dependency_list"]])

                    t2 = time.time()

                    subwd_span_list = []
                    wd_span_list = []

                    # entity
                    if "entity_list" in sample:
                        ent_exist = True
                        for ent in sample["entity_list"]:
                            ent_type_set.add(ent["type"])
                            subwd_span_list.append(ent["subwd_span"])
                            wd_span_list.append(ent["wd_span"])

                    # relation
                    if "relation_list" in sample:
                        rel_exist = True
                        for rel in sample["relation_list"]:
                            rel_type_set.add(rel["predicate"])
                            subwd_span_list.append(rel["subj_subwd_span"])
                            wd_span_list.append(rel["subj_wd_span"])
                            subwd_span_list.append(rel["obj_subwd_span"])
                            wd_span_list.append(rel["obj_wd_span"])

                    t3 = time.time()

                    # event
                    if "event_list" in sample:
                        event_exist = True
                        for event in sample["event_list"]:
                            event_type_set.add(event["event_type"])

                            def add_sps(sp_list, span):
                                if type(span[0]) is list:
                                    sp_list.extend(span)
                                else:
                                    sp_list.append(span)

                            if "trigger" in event:
                                add_sps(subwd_span_list, event["trigger_subwd_span"])
                                add_sps(wd_span_list, event["trigger_wd_span"])

                            for arg in event["argument_list"]:
                                argument_type_set.add(arg["type"])
                                add_sps(subwd_span_list, arg["subwd_span"])
                                add_sps(wd_span_list, arg["wd_span"])

                    if "open_spo_list" in sample:
                        oie_exist = True
                        for spo in sample["open_spo_list"]:
                            for arg in spo:
                                oie_arg_type_set.add(arg["type"])
                                if "char_span" not in arg or len(arg["char_span"]) == 0:
                                    continue
                                subwd_span_list.append(arg["subwd_span"])
                                wd_span_list.append(arg["wd_span"])
                    t4 = time.time()

                    # span
                    if "entity_list" in sample and "relation_list" not in sample and \
                            "event_list" not in sample and "open_spo_list" not in sample:
                        for sp in subwd_span_list:
                            max_ann_subwd_span = max(sp[1] - sp[0], max_ann_subwd_span)
                        for sp in wd_span_list:
                            max_ann_wd_span = max(sp[1] - sp[0], max_ann_wd_span)
                    else:
                        for sp in subwd_span_list:
                            min_subwd_span_start = min(min_subwd_span_start, sp[0])
                            max_subwd_span_end = max(max_subwd_span_end, sp[1])
                        for sp in wd_span_list:
                            min_wd_span_start = min(min_wd_span_start, sp[0])
                            max_wd_span_end = max(max_wd_span_end, sp[1])
                        max_ann_subwd_span = max(max_subwd_span_end - min_subwd_span_start, max_ann_subwd_span)
                        max_ann_wd_span = max(max_wd_span_end - min_wd_span_start, max_ann_wd_span)
                    t5 = time.time()
                    # character
                    char_set |= set(sample["text"])
                    # word
                    for word in sample["features"]["word_list"]:
                        word2num[word] = word2num.get(word, 0) + 1
                    max_word_seq_length = max(max_word_seq_length, len(sample["features"]["word_list"]))
                    max_subword_seq_length = max(max_subword_seq_length, len(sample["features"]["subword_list"]))

            # print("{:.5}, {:.5}, {:.5}, {:.5}, {:.5}".format(t2 - t1, t3 - t2, t4 - t3, t5 - t4, time.time() - t5))

        def get_dict(tag_set):
            tag2id = {tag: ind + 2 for ind, tag in enumerate(sorted(tag_set))}
            tag2id["[PAD]"] = 0
            tag2id["[UNK]"] = 1
            return tag2id

        char2id = get_dict(char_set)

        word2num = dict(sorted(word2num.items(), key=lambda x: x[1], reverse=True))
        for tok, num in word2num.items():
            if num < min_word_freq:  # filter words with a frequency of less than <min_freq>
                continue
            word_set.add(tok)
            if len(word_set) == max_word_dict_size:
                break
        word2id = get_dict(word_set)

        data_statistics = {
            "word_num": len(word2id),
            "char_num": len(char2id),
            "max_word_seq_length": max_word_seq_length,
            "max_subword_seq_length": max_subword_seq_length,
            "max_ann_subword_span": max_ann_subwd_span,
            "max_ann_word_span": max_ann_wd_span,
        }
        if ent_exist:
            data_statistics["ent_type_num"] = len(ent_type_set)
        if rel_exist:
            data_statistics["rel_type_num"] = len(rel_type_set)
        if event_exist:
            data_statistics["event_type_num"] = len(event_type_set)
            data_statistics["arg_type_num"] = len(argument_type_set)
        if oie_exist:
            data_statistics["oie_arg_type_num"] = len(oie_arg_type_set)

        if ent_exist:
            data_statistics["ent_types"] = list(ent_type_set)
        if rel_exist:
            data_statistics["rel_types"] = list(rel_type_set)
        if event_exist:
            data_statistics["event_types"] = list(event_type_set)
            data_statistics["arg_types"] = list(argument_type_set)
        if oie_exist:
            data_statistics["oie_arg_types"] = list(oie_arg_type_set)

        dicts = {
            "char2id": char2id,
            "word2id": word2id,
        }
        if pos_tag:
            pos_tag2id = get_dict(set(pos_tag_list))
            data_statistics["pos_tag_num"] = len(pos_tag2id)
            dicts["pos_tag2id"] = pos_tag2id

        if ner_tag:
            ner_tag_set = set(ner_tag_list)
            ner_tag_set.remove("O")
            ner_tag2id = {ner_tag: ind + 1 for ind, ner_tag in enumerate(sorted(ner_tag_set))}
            ner_tag2id["O"] = 0
            data_statistics["ner_tag_num"] = len(ner_tag2id)
            dicts["ner_tag2id"] = ner_tag2id

        if wd_dep:
            deprel_type2id = get_dict(set(deprel_type_list))
            data_statistics["deprel_type_num"] = len(deprel_type2id)
            dicts["deprel_type2id"] = deprel_type2id

        return dicts, data_statistics

    @staticmethod
    def choose_features_by_token_level4sample(sample, token_level, do_lower_case=False):
        features = sample["features"]
        if token_level == "subword":
            subword2word_id = features["subword2word_id"]
            new_features = {
                "subword_list": features["subword_list"],
                "tok2char_span": features["subword2char_span"],
                "word_list": [features["word_list"][wid] for wid in subword2word_id],
            }
            if "ner_tag_list" in features:
                new_features["ner_tag_list"] = [features["ner_tag_list"][wid] for wid in subword2word_id]
            if "pos_tag_list" in features:
                new_features["pos_tag_list"] = [features["pos_tag_list"][wid] for wid in subword2word_id]
            if "subword_dependency_list" in features:
                new_features["dependency_list"] = features["subword_dependency_list"]
            if "pos_tag_list_csp" in features:
                new_features["pos_tag_list_csp"] = features["pos_tag_list_csp"]
            if "dependency_list_csp" in features:
                new_features["dependency_list_csp"] = features["dependency_list_csp"]
            sample["features"] = new_features
        else:
            subwd_list = [w.lower() for w in features["word_list"]] if do_lower_case else features["word_list"]
            new_features = {
                "word_list": features["word_list"],
                "subword_list": subwd_list,
                "tok2char_span": features["word2char_span"],
            }
            if "ner_tag_list" in features:
                new_features["ner_tag_list"] = features["ner_tag_list"]
            if "pos_tag_list" in features:
                new_features["pos_tag_list"] = features["pos_tag_list"]
            if "word_dependency_list" in features:
                new_features["dependency_list"] = features["word_dependency_list"]
            if "pos_tag_list_csp" in features:
                new_features["pos_tag_list_csp"] = features["pos_tag_list_csp"]
            if "dependency_list_csp" in features:
                new_features["dependency_list_csp"] = features["dependency_list_csp"]
            sample["features"] = new_features

    @staticmethod
    def choose_features_by_token_level(data, token_level, do_lower_case=False):
        for sample in tqdm(data, desc="choose features"):
            Preprocessor.choose_features_by_token_level4sample(sample, token_level, do_lower_case)
        return data

    @staticmethod
    def choose_features_by_token_level_gen(data, token_level, do_lower_case=False):
        for sample in data:
            Preprocessor.choose_features_by_token_level4sample(sample, token_level, do_lower_case)
            yield sample

    @staticmethod
    def choose_spans_by_token_level4sample(sample, token_level):
        tok_key = "subwd_span" if token_level == "subword" else "wd_span"

        def choose_sps(a_list):
            for item in a_list:
                if type(item) is list:
                    choose_sps(item)
                elif type(item) is dict:
                    add_dict = {}
                    for k, v in item.items():
                        if tok_key in k:
                            add_tok_sp_k = re.sub(tok_key, "tok_span", k)
                            add_dict[add_tok_sp_k] = v
                        elif type(v) is list:
                            choose_sps(v)
                    item.update(add_dict)

        for k, v in sample.items():
            if type(v) is list:
                choose_sps(v)
            elif type(v) is dict:
                for val in v.values():
                    if type(val) is list:
                        choose_sps(val)

    @staticmethod
    def choose_spans_by_token_level_gen(data, token_level):
        '''
        :param data:
        :param token_level: "subword" or "word"
        :return:
        '''
        for sample in data:
            Preprocessor.choose_spans_by_token_level4sample(sample, token_level)
            yield sample

    @staticmethod
    def choose_spans_by_token_level(data, token_level):
        '''
        :param data:
        :param token_level: "subword" or "word"
        :return:
        '''
        for sample in tqdm(data, desc="choose span level"):
            Preprocessor.choose_spans_by_token_level4sample(sample, token_level)

        # for sample in tqdm(data, desc="choose span level"):
        #     if "entity_list" in sample:
        #         for ent in sample["entity_list"]:
        #             ent["tok_span"] = ent["subwd_span"] if token_level == "subword" else ent["wd_span"]
        #     if "relation_list" in sample:
        #         for rel in sample["relation_list"]:
        #             rel["subj_tok_span"] = rel["subj_subwd_span"] if token_level == "subword" else rel["subj_wd_span"]
        #             rel["obj_tok_span"] = rel["obj_subwd_span"] if token_level == "subword" else rel["obj_wd_span"]
        #     if "event_list" in sample:
        #         for event in sample["event_list"]:
        #             if "trigger" in event:
        #                 event["trigger_tok_span"] = event["trigger_subwd_span"] \
        #                     if token_level == "subword" else event["trigger_wd_span"]
        #             for arg in event["argument_list"]:
        #                 arg["tok_span"] = arg["subwd_span"] if token_level == "subword" else arg["wd_span"]
        #     if "open_spo_list" in sample:
        #         tok_key = "subwd_span" if token_level == "subword" else "wd_span"
        #         for spo in sample["open_spo_list"]:
        #             for arg in spo:
        #                 if tok_key in arg:
        #                     arg["tok_span"] = arg[tok_key]
        return data

    @staticmethod
    def filter_spans(inp_list, start_ind, end_ind):
        limited_span = [start_ind, end_ind]
        filter_res = []
        for item in inp_list:
            if any("tok_span" in k and not utils.span_contains(limited_span, v) for k, v in item.items()):
                pass
            else:
                filter_res.append(item)

        return filter_res

    @staticmethod
    def filter_annotations(sample, start_ind, end_ind):
        '''
        filter annotations in [start_ind, end_ind]
        :param sample:
        :param start_ind:
        :param end_ind:
        :return:
        '''
        filter_res = {}
        limited_span = [start_ind, end_ind]

        if "relation_list" in sample:
            # sub_rel_list = []
            # for rel in sample["relation_list"]:
            #     subj_tok_span = rel["subj_tok_span"]
            #     obj_tok_span = rel["obj_tok_span"]
            #     # if subject and object are both in this subtext, add this spo to new sample
            #     # if subj_tok_span[0] >= start_ind and subj_tok_span[-1] <= end_ind \
            #     #         and obj_tok_span[0] >= start_ind and obj_tok_span[-1] <= end_ind:
            #     if utils.span_contains(limited_span, subj_tok_span) \
            #             and utils.span_contains(limited_span, obj_tok_span):
            #         rel_cp = copy.deepcopy(rel)
            #         sub_rel_list.append(rel_cp)
            filter_res["relation_list"] = Preprocessor.filter_spans(sample["relation_list"], start_ind, end_ind)

        if "entity_list" in sample:
            # sub_ent_list = []
            # for ent in sample["entity_list"]:
            #     tok_span = ent["tok_span"]
            #     # if entity in this subtext, add the entity to new sample
            #     # if tok_span[0] >= start_ind and tok_span[-1] <= end_ind:
            #     if utils.span_contains(limited_span, tok_span):
            #         ent_cp = copy.deepcopy(ent)
            #         sub_ent_list.append(ent_cp)
            # filter_res["entity_list"] = sub_ent_list
            filter_res["entity_list"] = Preprocessor.filter_spans(sample["entity_list"], start_ind, end_ind)

        if "event_list" in sample:
            sub_event_list = []
            for event in sample["event_list"]:
                event_cp = copy.deepcopy(event)
                if "trigger" in event_cp:
                    trigger_tok_span = event["trigger_tok_span"]
                    trigger_ch_span = event["trigger_char_span"]
                    if type(trigger_tok_span[0]) is list:
                        new_tok_span = []
                        new_ch_span = []
                        for sp_idx, tok_sp in enumerate(trigger_tok_span):
                            if utils.span_contains(limited_span, tok_sp):
                                new_tok_span.append(tok_sp)
                                new_ch_span.append(trigger_ch_span[sp_idx])
                        event_cp["trigger_tok_span"] = new_tok_span
                        event_cp["trigger_char_span"] = new_ch_span

                        if len(event_cp["trigger_tok_span"]) == 0:
                            del event_cp["trigger"]
                            del event_cp["trigger_tok_span"]
                            del event_cp["trigger_char_span"]
                    else:
                        if not utils.span_contains(limited_span, trigger_tok_span):
                            del event_cp["trigger"]
                            del event_cp["trigger_tok_span"]
                            del event_cp["trigger_char_span"]

                new_arg_list = []
                for arg in event_cp["argument_list"]:
                    tok_span = arg["tok_span"]
                    ch_span = arg["char_span"]
                    if type(tok_span[0]) is list:
                        new_tok_span = []
                        new_ch_span = []
                        for sp_idx, tok_sp in enumerate(tok_span):
                            if utils.span_contains(limited_span, tok_sp):
                                new_tok_span.append(tok_sp)
                                new_ch_span.append(ch_span[sp_idx])
                        arg["tok_span"] = new_tok_span
                        arg["char_span"] = new_ch_span

                        if len(arg["tok_span"]) > 0:
                            new_arg_list.append(arg)
                    else:
                        if utils.span_contains(limited_span, tok_span):
                            new_arg_list.append(arg)

                new_trigger_list = []
                for trigger in event_cp.get("trigger_list", []):
                    tok_span = trigger["tok_span"]
                    ch_span = trigger["char_span"]
                    if type(tok_span[0]) is list:
                        new_tok_span = []
                        new_ch_span = []
                        for sp_idx, tok_sp in enumerate(tok_span):
                            if utils.span_contains(limited_span, tok_sp):
                                new_tok_span.append(tok_sp)
                                new_ch_span.append(ch_span[sp_idx])
                        trigger["tok_span"] = new_tok_span
                        trigger["char_span"] = new_ch_span

                        if len(trigger["tok_span"]) > 0:
                            new_trigger_list.append(trigger)
                    else:
                        if utils.span_contains(limited_span, tok_span):
                            new_trigger_list.append(trigger)

                if len(new_arg_list) > 0 or "trigger" in event_cp:
                    event_cp["argument_list"] = new_arg_list
                    if len(new_trigger_list) > 0:
                        event_cp["trigger_list"] = new_trigger_list
                    sub_event_list.append(event_cp)
            filter_res["event_list"] = sub_event_list

        if "open_spo_list" in sample:
            sub_open_spo_list = []
            for spo in sample["open_spo_list"]:
                new_spo = []
                bad_spo = False
                for arg in spo:
                    if utils.span_contains(limited_span, arg["tok_span"]):
                        new_spo.append(arg)
                    elif not utils.span_contains(limited_span, arg["tok_span"]) \
                            and arg["type"] in {"predicate", "object", "subject"}:
                        bad_spo = True
                        break
                if not bad_spo:
                    sub_open_spo_list.append(new_spo)

                # if any(not utils.span_contains(limited_span, arg["tok_span"]) for arg in spo):
                #     continue
                # sub_open_spo_list.append(spo)
            filter_res["open_spo_list"] = sub_open_spo_list
        return filter_res

    # @staticmethod
    # def split_into_short_samples(data, max_seq_len, sliding_len, data_type,
    #                              token_level, task_type, wordpieces_prefix="##", early_stop=True,
    #                              drop_neg_samples=False):
    #     '''
    #     split long samples into short samples
    #     :param data: original data
    #     :param max_seq_len: the max sequence length of a subtext
    #     :param sliding_len: the size of the sliding window
    #     :param data_type: train, valid, test
    #     :return:
    #     '''
    #
    #     split_sample_list = []
    #
    #     for sample in tqdm(data, desc="splitting"):
    #         id = sample["id"]
    #         text = sample["text"]
    #         features = sample["features"]
    #         tokens = features["subword_list"] if token_level == "subword" else features["word_list"]
    #         tok2char_span = features["tok2char_span"]
    #
    #         # split by sliding window
    #         for start_ind in range(0, len(tokens), sliding_len):
    #             if token_level == "subword":
    #                 while wordpieces_prefix in tokens[start_ind]:
    #                     start_ind -= 1
    #             end_ind = start_ind + max_seq_len
    #
    #             # split text
    #             char_span_list = tok2char_span[start_ind:end_ind]
    #             char_span = (char_span_list[0][0], char_span_list[-1][1])
    #             sub_text = text[char_span[0]:char_span[1]]
    #
    #             # offsets
    #             tok_level_offset, char_level_offset = start_ind, char_span[0]
    #
    #             # split features
    #             short_word_list = features["word_list"][start_ind:end_ind]
    #             short_subword_list = features["subword_list"][start_ind:end_ind]
    #             split_features = {"word_list": short_word_list,
    #                               "subword_list": short_subword_list,
    #                               "tok2char_span": [[char_sp[0] - char_level_offset, char_sp[1] - char_level_offset]
    #                                                 for char_sp in features["tok2char_span"][start_ind:end_ind]],
    #                               }
    #             if "pos_tag_list" in features:
    #                 split_features["pos_tag_list"] = features["pos_tag_list"][start_ind:end_ind]
    #             if "ner_tag_list" in features:
    #                 split_features["ner_tag_list"] = features["ner_tag_list"][start_ind:end_ind]
    #             if "dependency_list" in features:
    #                 split_features["dependency_list"] = []
    #                 for dep in features["dependency_list"]:
    #                     if start_ind <= dep[0] < end_ind and start_ind <= dep[1] < end_ind:
    #                         new_dep = [dep[0] - tok_level_offset, dep[1] - tok_level_offset, dep[2]]
    #                         split_features["dependency_list"].append(new_dep)
    #             if "pos_tag_list_csp" in features:
    #                 pos_tag_list_csp = Preprocessor.filter_spans(features["pos_tag_list_csp"],
    #                                                              start_ind, end_ind)
    #                 split_features["pos_tag_list_csp"] = Preprocessor.list_offset(pos_tag_list_csp,
    #                                                                               - tok_level_offset,
    #                                                                               - char_level_offset)
    #             if "dependency_list_csp" in features:
    #                 dependency_list_csp = Preprocessor.filter_spans(features["dependency_list_csp"],
    #                                                                 start_ind, end_ind)
    #                 split_features["dependency_list_csp"] = Preprocessor.list_offset(dependency_list_csp,
    #                                                                                  - tok_level_offset,
    #                                                                                  - char_level_offset)
    #             new_sample = {
    #                 "id": id,
    #                 "text": sub_text,
    #                 "features": split_features,
    #                 "tok_level_offset": tok_level_offset,
    #                 "char_level_offset": char_level_offset,
    #             }
    #             if "entity_list" in sample:
    #                 new_sample["entity_list"] = []
    #             if "relation_list" in sample:
    #                 new_sample["relation_list"] = []
    #             if "event_list" in sample:
    #                 new_sample["event_list"] = []
    #             if "open_spo_list" in sample:
    #                 new_sample["open_spo_list"] = []
    #
    #             # # if "components" exists -> it is a combined sample
    #             # if "components" in sample:
    #             #     sub_comps_ = []
    #             #     for comp in sample["components"]:
    #             #         if comp["offset_in_this_comb"][-1] <= start_ind \
    #             #                 or comp["offset_in_this_comb"][0] >= end_ind:
    #             #             continue
    #             #         sub_comps_.append(comp)
    #             #
    #             #     sub_comps = copy.deepcopy(sub_comps_)
    #             #     subcomb_start_tok_id = sub_comps[0]["offset_in_this_comb"][0]
    #             #     sub_comps[0]["offset_in_ori_txt"]["tok_level_offset"] = start_ind - subcomb_start_tok_id
    #             #     start_char_id = tok2char_span[start_ind][0]
    #             #     subcomb_start_char_id = tok2char_span[subcomb_start_tok_id][0]
    #             #     sub_comps[0]["offset_in_ori_txt"]["char_level_offset"] = start_char_id - subcomb_start_char_id
    #             #
    #             #     sub_comps[0]["offset_in_this_comb"][0] = start_ind
    #             #     if end_ind < sub_comps[-1]["offset_in_this_comb"][-1]:
    #             #         sub_comps[-1]["offset_in_this_comb"][-1] = end_ind
    #             #
    #             #     for cp in sub_comps:
    #             #         cp["offset_in_this_comb"][0] -= start_ind
    #             #         cp["offset_in_this_comb"][1] -= start_ind
    #             #
    #             #     new_sample["components"] = sub_comps
    #
    #             # if train or debug, filter annotations
    #             if data_type not in {"train", "debug"}:
    #                 if len(sub_text) > 0:
    #                     split_sample_list.append(new_sample)
    #                 if end_ind > len(tokens):
    #                     break
    #             else:
    #                 # if train data, need to filter annotations in the subtext
    #                 filtered_res = Preprocessor.filter_annotations(sample, start_ind, end_ind)
    #                 if "entity_list" in filtered_res:
    #                     new_sample["entity_list"] = filtered_res["entity_list"]
    #                 if "relation_list" in filtered_res:
    #                     new_sample["relation_list"] = filtered_res["relation_list"]
    #                 if "event_list" in filtered_res:
    #                     new_sample["event_list"] = filtered_res["event_list"]
    #                 if "open_spo_list" in filtered_res:
    #                     new_sample["open_spo_list"] = filtered_res["open_spo_list"]
    #
    #                 # do not introduce excessive negative samples
    #                 if drop_neg_samples and data_type == "train":
    #                     if ("entity_list" not in new_sample or len(new_sample["entity_list"]) == 0) \
    #                             and ("relation_list" not in new_sample or len(new_sample["relation_list"]) == 0) \
    #                             and ("event_list" not in new_sample or len(new_sample["event_list"]) == 0) \
    #                             and ("open_spo_list" not in new_sample or len(new_sample["open_spo_list"]) == 0):
    #                         continue
    #
    #                 # offset
    #                 anns = Preprocessor.span_offset(new_sample, - tok_level_offset, - char_level_offset)
    #                 new_sample = {**new_sample, **anns}
    #                 split_sample_list.append(new_sample)
    #                 if early_stop and end_ind > len(tokens):
    #                     break
    #
    #     return split_sample_list

    @staticmethod
    def split_into_short_samples(data, max_seq_len, sliding_len, data_type,
                                 token_level, task_type, wordpieces_prefix="##", early_stop=True,
                                 drop_neg_samples=False):
        '''
        split long samples into short samples
        :param data: original data
        :param max_seq_len: the max sequence length of a subtext
        :param sliding_len: the size of the sliding window
        :param data_type: train, valid, test
        :return:
        '''

        # split_sample_list = []
        for sample in data:
            id = sample["id"]
            text = sample["text"]
            features = sample["features"]
            tokens = features["subword_list"] if token_level == "subword" else features["word_list"]
            # try:
            tok2char_span = features["tok2char_span"]
            # except Exception:
            #     print("debug")
            # split by sliding window
            for start_ind in range(0, len(tokens), sliding_len):
                if token_level == "subword":
                    while wordpieces_prefix in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                # split text
                char_span_list = tok2char_span[start_ind:end_ind]
                char_span = (char_span_list[0][0], char_span_list[-1][1])
                sub_text = text[char_span[0]:char_span[1]]

                # offsets
                tok_level_offset, char_level_offset = start_ind, char_span[0]

                # split features
                short_word_list = features["word_list"][start_ind:end_ind]
                short_subword_list = features["subword_list"][start_ind:end_ind]
                split_features = {"word_list": short_word_list,
                                  "subword_list": short_subword_list,
                                  "tok2char_span": [[char_sp[0] - char_level_offset, char_sp[1] - char_level_offset]
                                                    for char_sp in features["tok2char_span"][start_ind:end_ind]],
                                  }
                if "pos_tag_list" in features:
                    split_features["pos_tag_list"] = features["pos_tag_list"][start_ind:end_ind]
                if "ner_tag_list" in features:
                    split_features["ner_tag_list"] = features["ner_tag_list"][start_ind:end_ind]
                if "dependency_list" in features:
                    split_features["dependency_list"] = []
                    for dep in features["dependency_list"]:
                        if start_ind <= dep[0] < end_ind and start_ind <= dep[1] < end_ind:
                            new_dep = [dep[0] - tok_level_offset, dep[1] - tok_level_offset, dep[2]]
                            split_features["dependency_list"].append(new_dep)
                if "pos_tag_list_csp" in features:
                    pos_tag_list_csp = Preprocessor.filter_spans(features["pos_tag_list_csp"],
                                                                 start_ind, end_ind)
                    split_features["pos_tag_list_csp"] = Preprocessor.list_offset(pos_tag_list_csp,
                                                                                  - tok_level_offset,
                                                                                  - char_level_offset)
                if "dependency_list_csp" in features:
                    dependency_list_csp = Preprocessor.filter_spans(features["dependency_list_csp"],
                                                                    start_ind, end_ind)
                    split_features["dependency_list_csp"] = Preprocessor.list_offset(dependency_list_csp,
                                                                                     - tok_level_offset,
                                                                                     - char_level_offset)
                new_sample = {
                    "id": id,
                    "text": sub_text,
                    "features": split_features,
                    "tok_level_offset": tok_level_offset,
                    "char_level_offset": char_level_offset,
                }
                if "entity_list" in sample:
                    new_sample["entity_list"] = []
                if "relation_list" in sample:
                    new_sample["relation_list"] = []
                if "event_list" in sample:
                    new_sample["event_list"] = []
                if "open_spo_list" in sample:
                    new_sample["open_spo_list"] = []

                # if train or debug, filter annotations
                if data_type not in {"train", "debug"}:
                    if len(sub_text) > 0:
                        yield new_sample
                    if end_ind > len(tokens):
                        break
                else:
                    # if train data, need to filter annotations in the subtext
                    filtered_res = Preprocessor.filter_annotations(sample, start_ind, end_ind)
                    if "entity_list" in filtered_res:
                        new_sample["entity_list"] = filtered_res["entity_list"]
                    if "relation_list" in filtered_res:
                        new_sample["relation_list"] = filtered_res["relation_list"]
                    if "event_list" in filtered_res:
                        new_sample["event_list"] = filtered_res["event_list"]
                    if "open_spo_list" in filtered_res:
                        new_sample["open_spo_list"] = filtered_res["open_spo_list"]

                    # do not introduce excessive negative samples
                    if drop_neg_samples and data_type == "train":
                        if ("entity_list" not in new_sample or len(new_sample["entity_list"]) == 0) \
                                and ("relation_list" not in new_sample or len(new_sample["relation_list"]) == 0) \
                                and ("event_list" not in new_sample or len(new_sample["event_list"]) == 0) \
                                and ("open_spo_list" not in new_sample or len(new_sample["open_spo_list"]) == 0):
                            continue

                    # offset
                    anns = Preprocessor.span_offset(new_sample, - tok_level_offset, - char_level_offset)
                    new_sample = {**new_sample, **anns}
                    yield new_sample
                    if early_stop and end_ind > len(tokens):
                        break

    # @staticmethod
    # def combine(data, comb_len):
    #     assert len(data) > 0
    #
    #     def get_new_com_sample():
    #         new_combined_sample = {
    #             "id": "combined_{}".format(len(new_data)),
    #             "text": "",
    #             "features": {
    #                 "word_list": [],
    #                 "subword_list": [],
    #                 "tok2char_span": [],
    #                 "pos_tag_list": [],
    #                 "ner_tag_list": [],
    #                 "dependency_list": []
    #             },
    #             "components": [],
    #         }
    #
    #         if "entity_list" in data[0]:
    #             new_combined_sample["entity_list"] = []
    #         if "relation_list" in data[0]:
    #             new_combined_sample["relation_list"] = []
    #         if "event_list" in data[0]:
    #             new_combined_sample["event_list"] = []
    #         if "open_spo_list" in data[0]:
    #             new_combined_sample["open_spo_list"] = []
    #         return new_combined_sample
    #
    #     new_data = []
    #     combined_sample = get_new_com_sample()
    #
    #     for sample in tqdm(data, desc="combining"):
    #         if len(combined_sample["features"]["tok2char_span"] + sample["features"]["tok2char_span"]) > comb_len:
    #             new_data.append(combined_sample)
    #             combined_sample = get_new_com_sample()
    #
    #         # combine features
    #         if len(combined_sample["text"]) > 0:
    #             combined_sample["text"] += " "  # use white space as a separator
    #
    #         combined_sample["text"] += sample["text"]
    #
    #         combined_sample["features"]["word_list"].extend(sample["features"]["word_list"])
    #         combined_sample["features"]["subword_list"].extend(sample["features"]["subword_list"])
    #         if "pos_tag_list" in sample["features"]:
    #             combined_sample["features"]["pos_tag_list"].extend(sample["features"]["pos_tag_list"])
    #         if "ner_tag_list" in sample["features"]:
    #             combined_sample["features"]["ner_tag_list"].extend(sample["features"]["ner_tag_list"])
    #         token_offset = len(combined_sample["features"]["tok2char_span"])
    #         char_offset = 0
    #         if token_offset > 0:
    #             char_offset = combined_sample["features"]["tok2char_span"][-1][1] + 1  # +1: white space
    #
    #         new_tok2char_span = [[char_sp[0] + char_offset, char_sp[1] + char_offset] for char_sp in
    #                              sample["features"]["tok2char_span"]]
    #         combined_sample["features"]["tok2char_span"].extend(new_tok2char_span)
    #
    #         if "dependency_list" in sample["features"]:
    #             new_dependency_list = [[dep[0] + token_offset, dep[1] + token_offset, dep[2]] for dep in
    #                                    sample["features"]["dependency_list"]]
    #             combined_sample["features"]["dependency_list"].extend(new_dependency_list)
    #
    #         # offsets for recovering
    #         combined_sample["components"].append({
    #             "id": sample["id"],
    #             "offset_in_this_comb": [token_offset, token_offset + len(sample["features"]["tok2char_span"])],
    #             "offset_in_ori_txt": {
    #                 "tok_level_offset": 0,
    #                 "char_level_offset": 0,
    #             }
    #         })
    #
    #         # combine annotations
    #         anns = Preprocessor.span_offset(sample, token_offset, char_offset)
    #         if "entity_list" in anns:
    #             combined_sample["entity_list"].extend(anns["entity_list"])
    #         if "relation_list" in anns:
    #             combined_sample["relation_list"].extend(anns["relation_list"])
    #         if "event_list" in anns:
    #             combined_sample["event_list"].extend(anns["event_list"])
    #         if "open_spo_list" in anns:
    #             combined_sample["open_spo_list"].extend(anns["open_spo_list"])
    #
    #         # if len(combined_sample["features"]["tok2char_span"]) >= comb_len:
    #         #     new_data.append(combined_sample)
    #         #     combined_sample = get_new_com_sample()
    #
    #     # do not forget the last one
    #     if combined_sample["text"] != "":
    #         new_data.append(combined_sample)
    #     return new_data

    @staticmethod
    def combine(data, comb_len):
        # 上面的是yield之前的
        def get_new_com_sample(com_idx):
            new_combined_sample = {
                "id": "combined_{}".format(com_idx),
                "text": "",
                "features": {
                    "word_list": [],
                    "subword_list": [],
                    "tok2char_span": [],
                    "pos_tag_list": [],
                    "ner_tag_list": [],
                    "dependency_list": []
                },
                "components": [],
            }
            return new_combined_sample

        # new_data = []
        comb_count = 0
        combined_sample = get_new_com_sample(comb_count)

        for sample in data:
            if len(combined_sample["features"]["tok2char_span"] + sample["features"]["tok2char_span"]) > comb_len:
                yield combined_sample
                comb_count += 1
                combined_sample = get_new_com_sample(comb_count)

            # combine features
            if len(combined_sample["text"]) > 0:
                combined_sample["text"] += " "  # use white space as a separator

            combined_sample["text"] += sample["text"]

            combined_sample["features"]["word_list"].extend(sample["features"]["word_list"])
            combined_sample["features"]["subword_list"].extend(sample["features"]["subword_list"])
            if "pos_tag_list" in sample["features"]:
                combined_sample["features"]["pos_tag_list"].extend(sample["features"]["pos_tag_list"])
            if "ner_tag_list" in sample["features"]:
                combined_sample["features"]["ner_tag_list"].extend(sample["features"]["ner_tag_list"])
            token_offset = len(combined_sample["features"]["tok2char_span"])
            char_offset = 0
            if token_offset > 0:
                char_offset = combined_sample["features"]["tok2char_span"][-1][1] + 1  # +1: white space

            new_tok2char_span = [[char_sp[0] + char_offset, char_sp[1] + char_offset] for char_sp in
                                 sample["features"]["tok2char_span"]]
            combined_sample["features"]["tok2char_span"].extend(new_tok2char_span)

            if "dependency_list" in sample["features"]:
                new_dependency_list = [[dep[0] + token_offset, dep[1] + token_offset, dep[2]] for dep in
                                       sample["features"]["dependency_list"]]
                combined_sample["features"]["dependency_list"].extend(new_dependency_list)

            # offsets for recovering
            combined_sample["components"].append({
                "id": sample["id"],
                "offset_in_this_comb": [token_offset, token_offset + len(sample["features"]["tok2char_span"])],
                "offset_in_ori_txt": {
                    "tok_level_offset": 0,
                    "char_level_offset": 0,
                }
            })

            # combine annotations
            anns = Preprocessor.span_offset(sample, token_offset, char_offset)
            if "entity_list" in anns:
                combined_sample.setdefault("entity_list", []).extend(anns["entity_list"])
            if "relation_list" in anns:
                combined_sample.setdefault("relation_list", []).extend(anns["relation_list"])
            if "event_list" in anns:
                combined_sample.setdefault("event_list", []).extend(anns["event_list"])
            if "open_spo_list" in anns:
                combined_sample.setdefault("open_spo_list", []).extend(anns["open_spo_list"])

            # if len(combined_sample["features"]["tok2char_span"]) >= comb_len:
            #     new_data.append(combined_sample)
            #     combined_sample = get_new_com_sample()

        # do not forget the last one
        if combined_sample["text"] != "":
            yield combined_sample

    # @staticmethod
    # def split_into_short_samples(data, max_seq_len, sliding_len, data_type,
    #                              token_level, task_type, wordpieces_prefix="##", early_stop=True,
    #                              drop_neg_samples=False):
    #     '''
    #     split samples with long text into samples with short subtexts
    #     :param data: original data
    #     :param max_seq_len: the max sequence length of a subtext
    #     :param sliding_len: the size of the sliding window
    #     :param data_type: train, valid, test
    #     :return:
    #     '''
    #     split_sample_list = []
    #     for sample in tqdm(data, desc="splitting"):
    #         id = sample["id"]
    #         text = sample["text"]
    #         features = sample["features"]
    #         tokens = features["subword_list"] if token_level == "subword" else features["word_list"]
    #         tok2char_span = features["tok2char_span"]
    #
    #         # split by sliding window
    #         for start_ind in range(0, len(tokens), sliding_len):
    #             if token_level == "subword":
    #                 while wordpieces_prefix in tokens[start_ind]:
    #                     start_ind -= 1
    #             end_ind = start_ind + max_seq_len
    #
    #             # split text
    #             char_span_list = tok2char_span[start_ind:end_ind]
    #             char_span = (char_span_list[0][0], char_span_list[-1][1])
    #             sub_text = text[char_span[0]:char_span[1]]
    #
    #             # offsets
    #             tok_level_offset, char_level_offset = start_ind, char_span[0]
    #
    #             # split features
    #             short_word_list = features["word_list"][start_ind:end_ind]
    #             short_subword_list = features["subword_list"][start_ind:end_ind]
    #             split_features = {"word_list": short_word_list,
    #                               "subword_list": short_subword_list,
    #                               "tok2char_span": [[char_sp[0] - char_level_offset, char_sp[1] - char_level_offset]
    #                                                 for char_sp in features["tok2char_span"][start_ind:end_ind]],
    #                               }
    #             if "pos_tag_list" in features:
    #                 split_features["pos_tag_list"] = features["pos_tag_list"][start_ind:end_ind]
    #             if "ner_tag_list" in features:
    #                 split_features["ner_tag_list"] = features["ner_tag_list"][start_ind:end_ind]
    #             if "dependency_list" in features:
    #                 split_features["dependency_list"] = []
    #                 for dep in features["dependency_list"]:
    #                     if start_ind <= dep[0] < end_ind and start_ind <= dep[1] < end_ind:
    #                         new_dep = [dep[0] - tok_level_offset, dep[1] - tok_level_offset, dep[2]]
    #                         split_features["dependency_list"].append(new_dep)
    #
    #             new_sample = {
    #                 "id": id,
    #                 "text": sub_text,
    #                 "features": split_features,
    #                 "tok_level_offset": tok_level_offset,
    #                 "char_level_offset": char_level_offset,
    #             }
    #             if "entity_list" in sample:
    #                 new_sample["entity_list"] = []
    #             if "relation_list" in sample:
    #                 new_sample["relation_list"] = []
    #             if "event_list" in sample:
    #                 new_sample["event_list"] = []
    #             if "open_spo_list" in sample:
    #                 new_sample["open_spo_list"] = []
    #
    #             if data_type not in {"train", "debug"}:
    #                 if len(sub_text) > 0:
    #                     split_sample_list.append(new_sample)
    #                 if end_ind > len(tokens):
    #                     break
    #             else:
    #                 # if train data, need to filter annotations in the subtext
    #                 filtered_sample = Preprocessor.filter_annotations(sample, start_ind, end_ind)
    #                 if "entity_list" in filtered_sample:
    #                     new_sample["entity_list"] = filtered_sample["entity_list"]
    #                 if "relation_list" in filtered_sample:
    #                     new_sample["relation_list"] = filtered_sample["relation_list"]
    #                 if "event_list" in filtered_sample:
    #                     new_sample["event_list"] = filtered_sample["event_list"]
    #                 if "open_spo_list" in filtered_sample:
    #                     new_sample["open_spo_list"] = filtered_sample["open_spo_list"]
    #
    #                 # do not introduce excessive negative samples
    #                 if drop_neg_samples and data_type == "train":
    #                     if ("entity_list" not in new_sample or len(new_sample["entity_list"]) == 0) \
    #                             and ("relation_list" not in new_sample or len(new_sample["relation_list"]) == 0) \
    #                             and ("event_list" not in new_sample or len(new_sample["event_list"]) == 0) \
    #                             and ("open_spo_list" not in new_sample or len(new_sample["open_spo_list"]) == 0):
    #                         continue
    #
    #                 # offset
    #                 new_sample = Preprocessor.span_offset(new_sample, - tok_level_offset, - char_level_offset)
    #                 split_sample_list.append(new_sample)
    #                 if early_stop and end_ind > len(tokens):
    #                     break
    #     return split_sample_list
    #
    # @staticmethod
    # def combine(data, max_seq_len):
    #     assert len(data) > 0
    #
    #     def get_new_com_sample():
    #         new_combined_sample = {
    #             "id": "combined_{}".format(len(new_data)),
    #             "text": "",
    #             "features": {
    #                 "word_list": [],
    #                 "subword_list": [],
    #                 "tok2char_span": [],
    #                 "pos_tag_list": [],
    #                 "ner_tag_list": [],
    #                 "dependency_list": []
    #             },
    #             "splits": [],
    #         }
    #         if "entity_list" in data[0]:
    #             new_combined_sample["entity_list"] = []
    #         if "relation_list" in data[0]:
    #             new_combined_sample["relation_list"] = []
    #         if "event_list" in data[0]:
    #             new_combined_sample["event_list"] = []
    #         if "open_spo_list" in data[0]:
    #             new_combined_sample["open_spo_list"] = []
    #         return new_combined_sample
    #
    #     new_data = []
    #     combined_sample = get_new_com_sample()
    #
    #     for sample in tqdm(data, desc="combining splits"):
    #         if len(combined_sample["features"]["tok2char_span"] + sample["features"]["tok2char_span"]) > max_seq_len:
    #             new_data.append(combined_sample)
    #             combined_sample = get_new_com_sample()
    #
    #         # combine features
    #         if len(combined_sample["text"]) > 0:
    #             combined_sample["text"] += " "
    #         combined_sample["text"] += sample["text"]
    #
    #         combined_sample["features"]["word_list"].extend(sample["features"]["word_list"])
    #         combined_sample["features"]["subword_list"].extend(sample["features"]["subword_list"])
    #         if "pos_tag_list" in sample["features"]:
    #             combined_sample["features"]["pos_tag_list"].extend(sample["features"]["pos_tag_list"])
    #         if "ner_tag_list" in sample["features"]:
    #             combined_sample["features"]["ner_tag_list"].extend(sample["features"]["ner_tag_list"])
    #         token_offset = len(combined_sample["features"]["tok2char_span"])
    #         char_offset = 0
    #         if token_offset > 0:
    #             char_offset = combined_sample["features"]["tok2char_span"][-1][1] + 1  # +1: whitespace
    #         new_tok2char_span = [[char_sp[0] + char_offset, char_sp[1] + char_offset] for char_sp in
    #                              sample["features"]["tok2char_span"]]
    #         combined_sample["features"]["tok2char_span"].extend(new_tok2char_span)
    #
    #         if "dependency_list" in sample["features"]:
    #             new_dependency_list = [[dep[0] + token_offset, dep[1] + token_offset, dep[2]] for dep in
    #                                    sample["features"]["dependency_list"]]
    #             combined_sample["features"]["dependency_list"].extend(new_dependency_list)
    #
    #         # offsets for recovering
    #         combined_sample["splits"].append({
    #             "id": sample["id"],
    #             "offset_in_this_comb": [token_offset, token_offset + len(sample["features"]["tok2char_span"])],
    #             "offset_in_ori_txt": {
    #                 "tok_level_offset": sample["tok_level_offset"],
    #                 "char_level_offset": sample["char_level_offset"],
    #             }
    #         })
    #
    #         # combine annotations
    #         # sample_cp = copy.deepcopy(sample)
    #         sample_cp = Preprocessor.span_offset(sample, token_offset, char_offset)
    #         if "entity_list" in sample_cp:
    #             combined_sample["entity_list"].extend(sample_cp["entity_list"])
    #         if "relation_list" in sample_cp:
    #             combined_sample["relation_list"].extend(sample_cp["relation_list"])
    #         if "event_list" in sample_cp:
    #             combined_sample["event_list"].extend(sample_cp["event_list"])
    #         if "open_spo_list" in sample_cp:
    #             combined_sample["open_spo_list"].extend(sample_cp["open_spo_list"])
    #
    #     # do not forget the last one
    #     if combined_sample["text"] != "":
    #         new_data.append(combined_sample)
    #     return new_data

    @staticmethod
    def decompose2splits(data):
        '''
        decompose combined samples to splits by the list "splits"
        :param data:
        :return:
        '''
        new_data = []
        for sample in data:
            if "components" in sample:
                text = sample["text"]
                tok2char_span = sample["features"]["tok2char_span"]
                # decompose
                for spl in sample["components"]:
                    split_sample = {
                        "id": spl["id"],
                        "tok_level_offset": spl["offset_in_ori_txt"]["tok_level_offset"],
                        "char_level_offset": spl["offset_in_ori_txt"]["char_level_offset"],
                    }
                    text_tok_span = spl["offset_in_this_comb"]
                    char_sp_list = tok2char_span[text_tok_span[0]:text_tok_span[1]]

                    text_char_span = [char_sp_list[0][0], char_sp_list[-1][1]]
                    assert text_char_span[0] < text_char_span[1]

                    # text
                    split_sample["text"] = text[text_char_span[0]:text_char_span[1]]
                    # filter annotations
                    filtered_sample = Preprocessor.filter_annotations(sample, text_tok_span[0], text_tok_span[1])
                    if "entity_list" in filtered_sample:
                        split_sample["entity_list"] = filtered_sample["entity_list"]
                    if "relation_list" in filtered_sample:
                        split_sample["relation_list"] = filtered_sample["relation_list"]
                    if "event_list" in filtered_sample:
                        split_sample["event_list"] = filtered_sample["event_list"]
                    if "open_spo_list" in filtered_sample:
                        split_sample["open_spo_list"] = filtered_sample["open_spo_list"]
                    # recover spans
                    anns = Preprocessor.span_offset(split_sample, -text_tok_span[0], -text_char_span[0])
                    split_sample = {**split_sample, **anns}
                    new_data.append(split_sample)
            else:
                new_data.append(sample)
        return new_data

    @staticmethod
    def list_offset(inp_list, tok_level_offset, char_level_offset):
        def list_add(ori_list, add_num):
            if len(ori_list) > 0 and type(ori_list[0]) is list:
                return [[e + add_num for e in sub_list] for sub_list in ori_list]
            else:
                return [e + add_num for e in ori_list]

        res_list = copy.deepcopy(inp_list)
        for item in res_list:
            for k, v in item.items():
                if "tok_span" in k:
                    item[k] = list_add(v, tok_level_offset)
                elif "char_span" in k:
                    item[k] = list_add(v, char_level_offset)
        return res_list

    @staticmethod
    def span_offset(sample_spans, tok_level_offset, char_level_offset):
        '''
        add offset
        :param sample_spans:
        :param tok_level_offset:
        :param char_level_offset:
        :return:
        '''

        def list_add(ori_list, add_num):
            if len(ori_list) > 0 and type(ori_list[0]) is list:
                return [[e + add_num for e in sub_list] for sub_list in ori_list]
            else:
                return [e + add_num for e in ori_list]

        annotations = {}
        if "relation_list" in sample_spans:
            annotations["relation_list"] = copy.deepcopy(sample_spans["relation_list"])
            for rel in annotations["relation_list"]:
                rel["subj_tok_span"] = list_add(rel["subj_tok_span"], tok_level_offset)
                rel["obj_tok_span"] = list_add(rel["obj_tok_span"], tok_level_offset)
                rel["subj_char_span"] = list_add(rel["subj_char_span"], char_level_offset)
                rel["obj_char_span"] = list_add(rel["obj_char_span"], char_level_offset)

        if "entity_list" in sample_spans:
            annotations["entity_list"] = copy.deepcopy(sample_spans["entity_list"])
            for ent in annotations["entity_list"]:
                ent["tok_span"] = list_add(ent["tok_span"], tok_level_offset)
                ent["char_span"] = list_add(ent["char_span"], char_level_offset)

        if "event_list" in sample_spans:
            annotations["event_list"] = copy.deepcopy(sample_spans["event_list"])
            for event in annotations["event_list"]:
                if "trigger" in event:
                    event["trigger_tok_span"] = list_add(event["trigger_tok_span"], tok_level_offset)
                    event["trigger_char_span"] = list_add(event["trigger_char_span"], char_level_offset)
                for arg in event["argument_list"]:
                    arg["tok_span"] = list_add(arg["tok_span"], tok_level_offset)
                    arg["char_span"] = list_add(arg["char_span"], char_level_offset)
        if "open_spo_list" in sample_spans:
            annotations["open_spo_list"] = copy.deepcopy(sample_spans["open_spo_list"])
            for spo in annotations["open_spo_list"]:
                for arg in spo:
                    arg["tok_span"] = list_add(arg["tok_span"], tok_level_offset)
                    arg["char_span"] = list_add(arg["char_span"], char_level_offset)
        return annotations

    @staticmethod
    def check_spans(data, language):
        sample_id2mismatched_ents = {}
        for sample in tqdm(data, desc="checking splits"):
            text = sample["text"]
            tok2char_span = sample["features"]["tok2char_span"]

            bad_entities = []
            bad_rels = []
            bad_events = []
            bad_open_spos = []

            if "entity_list" in sample:
                for ent in sample["entity_list"]:
                    extr_ent_t = Preprocessor.extract_ent_fr_txt_by_tok_sp(ent["tok_span"], tok2char_span, text,
                                                                           language)
                    extr_ent_c = Preprocessor.extract_ent_fr_txt_by_char_sp(ent["char_span"], text, language)

                    if not (extr_ent_t == ent["text"] == extr_ent_c):
                        bad_ent = copy.deepcopy(ent)
                        bad_ent["extr_ent_t"] = extr_ent_t
                        bad_ent["extr_ent_c"] = extr_ent_c
                        bad_entities.append(bad_ent)
                sample_id2mismatched_ents[sample["id"]] = {
                    "bad_entites": bad_entities,
                }

            if "relation_list" in sample:
                for rel in sample["relation_list"]:
                    extr_subj_t = Preprocessor.extract_ent_fr_txt_by_tok_sp(rel["subj_tok_span"], tok2char_span, text,
                                                                            language)
                    extr_obj_t = Preprocessor.extract_ent_fr_txt_by_tok_sp(rel["obj_tok_span"], tok2char_span, text,
                                                                           language)
                    extr_subj_c = Preprocessor.extract_ent_fr_txt_by_char_sp(rel["subj_char_span"], text, language)
                    extr_obj_c = Preprocessor.extract_ent_fr_txt_by_char_sp(rel["obj_char_span"], text, language)

                    if not (extr_subj_t == rel["subject"] == extr_subj_c and extr_obj_t == rel["object"] == extr_obj_c):
                        bad_rel = copy.deepcopy(rel)
                        bad_rel["extr_subj_t"] = extr_subj_t
                        bad_rel["extr_obj_t"] = extr_obj_t
                        bad_rel["extr_subj_c"] = extr_subj_c
                        bad_rel["extr_obj_c"] = extr_obj_c
                        bad_rels.append(bad_rel)
                sample_id2mismatched_ents[sample["id"]]["bad_relations"] = bad_rels

            if "event_list" in sample:
                for event in sample["event_list"]:
                    bad_event = copy.deepcopy(event)
                    bad = False
                    if "trigger" in event:
                        trigger_tok_span = event["trigger_tok_span"]
                        trigger_ch_span = event["trigger_char_span"]
                        if type(trigger_ch_span[0]) is list or type(trigger_tok_span[0]) is list:
                            tok_sp_list = trigger_tok_span
                            ch_sp_list = trigger_ch_span
                        else:
                            tok_sp_list = [trigger_tok_span, ]
                            ch_sp_list = [trigger_ch_span, ]

                        for sp_idx, tok_sp in enumerate(tok_sp_list):
                            ch_sp = ch_sp_list[sp_idx]
                            extr_trigger_t = Preprocessor.extract_ent_fr_txt_by_tok_sp(tok_sp, tok2char_span,
                                                                                       text, language)
                            extr_trigger_c = Preprocessor.extract_ent_fr_txt_by_char_sp(ch_sp, text, language)
                            if not (extr_trigger_t == event["trigger"] == extr_trigger_c):
                                bad = True
                                bad_event.setdefault("extr_trigger_t", []).append(extr_trigger_t)
                                bad_event.setdefault("extr_trigger_c", []).append(extr_trigger_c)

                    for arg in event["argument_list"]:
                        arg_tok_span = arg["tok_span"]
                        arg_ch_span = arg["char_span"]
                        if type(arg_ch_span[0]) is list or type(arg_tok_span[0]) is list:
                            tok_sp_list = arg_tok_span
                            ch_sp_list = arg_ch_span
                        else:
                            tok_sp_list = [arg_tok_span, ]
                            ch_sp_list = [arg_ch_span, ]

                        for sp_idx, tok_sp in enumerate(tok_sp_list):
                            ch_sp = ch_sp_list[sp_idx]
                            extr_arg_t = Preprocessor.extract_ent_fr_txt_by_tok_sp(tok_sp, tok2char_span,
                                                                                   text, language)
                            extr_arg_c = Preprocessor.extract_ent_fr_txt_by_char_sp(ch_sp, text, language)
                            if not (extr_arg_t == arg["text"] == extr_arg_c):
                                bad = True
                                bad_event.setdefault("extr_arg_t", []).append(extr_arg_t)
                                bad_event.setdefault("extr_arg_c", []).append(extr_arg_c)

                    if bad:
                        bad_events.append(bad_event)
            if "open_spo_list" in sample:
                for spo in sample["open_spo_list"]:
                    for arg in spo:
                        arg_tok_span = arg["tok_span"]
                        arg_ch_span = arg["char_span"]
                        if len(arg_ch_span) == 0:
                            continue
                        extr_arg_t = Preprocessor.extract_ent_fr_txt_by_tok_sp(arg_tok_span, tok2char_span,
                                                                               text, language)
                        extr_arg_c = Preprocessor.extract_ent_fr_txt_by_char_sp(arg_ch_span, text, language)

                        ori_arg = arg["text"] if arg["type"] != "predicate" \
                            else re.sub("\[OBJ\]", "", arg["text"]).strip(" ")
                        if not (extr_arg_t == ori_arg == extr_arg_c):
                            bad_open_spos.append({
                                "extr_arg_t": extr_arg_t,
                                "extr_arg_c": extr_arg_c,
                                "ori_txt": ori_arg,
                                "type": arg["type"],
                            })

            sample_id2mismatched_ents[sample["id"]] = {}
            if len(bad_entities) > 0:
                sample_id2mismatched_ents[sample["id"]]["bad_entities"] = bad_entities
                sample_id2mismatched_ents[sample["id"]]["text"] = text
            if len(bad_rels) > 0:
                sample_id2mismatched_ents[sample["id"]]["bad_relations"] = bad_rels
                sample_id2mismatched_ents[sample["id"]]["text"] = text
            if len(bad_events) > 0:
                sample_id2mismatched_ents[sample["id"]]["bad_events"] = bad_events
                sample_id2mismatched_ents[sample["id"]]["text"] = text
            if len(bad_open_spos) > 0:
                sample_id2mismatched_ents[sample["id"]]["bad_open_spos"] = bad_open_spos
                sample_id2mismatched_ents[sample["id"]]["text"] = text

            if len(sample_id2mismatched_ents[sample["id"]]) == 0:
                del sample_id2mismatched_ents[sample["id"]]
        return sample_id2mismatched_ents

    # @staticmethod
    # def index_features(data, language, key2dict, max_seq_len, max_char_num_in_tok=None):
    #     '''
    #     :param language:
    #     :param data:
    #     :param key2dict: feature key to dict for indexing
    #     :param max_seq_len:
    #     :param max_char_num_in_tok: max character number in a token, truncate or pad to this length
    #     :param pretrained_model_padding: for subword ids padding
    #     :return:
    #     '''
    #
    #     # map for replacing key names
    #     key_map = {
    #         "char_list": "char_input_ids",
    #         "word_list": "word_input_ids",
    #         "subword_list": "subword_input_ids",
    #         "ner_tag_list": "ner_tag_ids",
    #         "pos_tag_list": "pos_tag_ids",
    #         "dependency_list": "dependency_points",
    #     }
    #
    #     for sample in tqdm(data, desc="indexing"):
    #         features = sample["features"]
    #         features["token_type_ids"] = [0] * len(features["tok2char_span"])
    #         features["attention_mask"] = [1] * len(features["tok2char_span"])
    #         features["char_list"] = list(sample["text"])
    #
    #         sep = " " if language == "en" else ""
    #         features["word_list"] += ["[PAD]"] * (max_seq_len - len(features["word_list"]))
    #         fin_features = {
    #             "padded_text": sep.join(features["word_list"]),
    #             "word_list": features["word_list"],
    #         }
    #         for f_key, tags in features.items():
    #             # features need indexing and padding
    #             if f_key in key2dict.keys():
    #                 tag2id = key2dict[f_key]
    #
    #                 if f_key == "ner_tag_list":
    #                     spe_tag_dict = {"[UNK]": tag2id["O"], "[PAD]": tag2id["O"]}
    #                 else:
    #                     spe_tag_dict = {"[UNK]": tag2id["[UNK]"], "[PAD]": tag2id["[PAD]"]}
    #
    #                 indexer = Indexer(tag2id, max_seq_len, spe_tag_dict)
    #                 if f_key == "dependency_list":
    #                     fin_features[key_map[f_key]] = indexer.index_tag_list_w_matrix_pos(tags)
    #                 elif f_key == "char_list" and max_char_num_in_tok is not None:
    #                     char_input_ids = indexer.index_tag_list(tags)
    #                     # padding character ids
    #                     char_input_ids_padded = []
    #                     for span in features["tok2char_span"]:
    #                         char_ids = char_input_ids[span[0]:span[1]]
    #
    #                         if len(char_ids) < max_char_num_in_tok:
    #                             char_ids.extend([0] * (max_char_num_in_tok - len(char_ids)))
    #                         else:
    #                             char_ids = char_ids[:max_char_num_in_tok]
    #                         char_input_ids_padded.extend(char_ids)
    #                     fin_features[key_map[f_key]] = torch.LongTensor(char_input_ids_padded)
    #                 else:
    #                     fin_features[key_map[f_key]] = torch.LongTensor(indexer.index_tag_list(tags))
    #
    #             # features only need padding
    #             elif f_key in {"token_type_ids", "attention_mask"}:
    #                 fin_features[f_key] = torch.LongTensor(Indexer.pad2length(tags, 0, max_seq_len))
    #             elif f_key == "tok2char_span":
    #                 fin_features[f_key] = Indexer.pad2length(tags, [0, 0], max_seq_len)
    #             elif f_key == "pos_tag_list_csp":
    #                 tag2id = key2dict["pos_tag_list"]
    #                 fin_features["pos_tag_points"] = list(
    #                     {(pos["tok_span"][0], pos["tok_span"][1] - 1, tag2id[pos["type"]])
    #                      for pos in tags})
    #             elif f_key == "dependency_list_csp":
    #                 tag2id = key2dict["dependency_list"]
    #                 t2t_points = list(
    #                     {(deprel["subj_tok_span"][1] - 1, deprel["obj_tok_span"][1] - 1,
    #                       tag2id[deprel["predicate"]] + len(tag2id)) for deprel in tags}
    #                 )
    #                 h2h_points = list(
    #                     {(deprel["subj_tok_span"][0], deprel["obj_tok_span"][0],
    #                       tag2id[deprel["predicate"]]) for deprel in tags}
    #                 )
    #                 fin_features["deprel_points_hnt"] = h2h_points + t2t_points
    #
    #         sample["features"] = fin_features
    #     return data

    @staticmethod
    def index_features(data, language, key2dict, max_seq_len, max_char_num_in_tok=None):
        '''
        :param language:
        :param data:
        :param key2dict: feature key to dict for indexing
        :param max_seq_len:
        :param max_char_num_in_tok: max character number in a token, truncate or pad to this length
        :param pretrained_model_padding: for subword ids padding
        :return:
        '''

        # map for replacing key names
        key_map = {
            "char_list": "char_input_ids",
            "word_list": "word_input_ids",
            "subword_list": "subword_input_ids",
            "ner_tag_list": "ner_tag_ids",
            "pos_tag_list": "pos_tag_ids",
            "dependency_list": "dependency_points",
        }

        for sample in data:
            features = sample["features"]
            features["token_type_ids"] = [0] * len(features["tok2char_span"])
            features["attention_mask"] = [1] * len(features["tok2char_span"])
            features["char_list"] = list(sample["text"])

            sep = " " if language == "en" else ""
            features["word_list"] += ["[PAD]"] * (max_seq_len - len(features["word_list"]))
            fin_features = {
                "padded_text": sep.join(features["word_list"]),
                "word_list": features["word_list"],
            }
            for f_key, tags in features.items():
                # features need indexing and padding
                if f_key in key2dict.keys():
                    tag2id = key2dict[f_key]

                    if f_key == "ner_tag_list":
                        spe_tag_dict = {"[UNK]": tag2id["O"], "[PAD]": tag2id["O"]}
                    else:
                        spe_tag_dict = {"[UNK]": tag2id["[UNK]"], "[PAD]": tag2id["[PAD]"]}

                    indexer = Indexer(tag2id, max_seq_len, spe_tag_dict)
                    if f_key == "dependency_list":
                        fin_features[key_map[f_key]] = indexer.index_tag_list_w_matrix_pos(tags)
                    elif f_key == "char_list" and max_char_num_in_tok is not None:
                        char_input_ids = indexer.index_tag_list(tags)
                        # padding character ids
                        char_input_ids_padded = []
                        for span in features["tok2char_span"]:
                            char_ids = char_input_ids[span[0]:span[1]]

                            if len(char_ids) < max_char_num_in_tok:
                                char_ids.extend([0] * (max_char_num_in_tok - len(char_ids)))
                            else:
                                char_ids = char_ids[:max_char_num_in_tok]
                            char_input_ids_padded.extend(char_ids)
                        fin_features[key_map[f_key]] = char_input_ids_padded  # torch.LongTensor(char_input_ids_padded)
                    else:
                        fin_features[key_map[f_key]] = indexer.index_tag_list(
                            tags)  # torch.LongTensor(indexer.index_tag_list(tags))

                # features only need padding
                elif f_key in {"token_type_ids", "attention_mask"}:
                    fin_features[f_key] = Indexer.pad2length(tags, 0,
                                                             max_seq_len)  # torch.LongTensor(Indexer.pad2length(tags, 0, max_seq_len))
                elif f_key == "tok2char_span":
                    fin_features[f_key] = Indexer.pad2length(tags, [0, 0], max_seq_len)
                elif f_key == "pos_tag_list_csp":
                    tag2id = key2dict["pos_tag_list"]
                    fin_features["pos_tag_points"] = list(
                        {(pos["tok_span"][0], pos["tok_span"][1] - 1, tag2id[pos["type"]])
                         for pos in tags})
                elif f_key == "dependency_list_csp":
                    tag2id = key2dict["dependency_list"]
                    t2t_points = list(
                        {(deprel["subj_tok_span"][1] - 1, deprel["obj_tok_span"][1] - 1,
                          tag2id[deprel["predicate"]] + len(tag2id)) for deprel in tags}
                    )
                    h2h_points = list(
                        {(deprel["subj_tok_span"][0], deprel["obj_tok_span"][0],
                          tag2id[deprel["predicate"]]) for deprel in tags}
                    )
                    fin_features["deprel_points_hnt"] = h2h_points + t2t_points

            sample["features"] = fin_features
            yield sample


if __name__ == "__main__":
    pass
