import re
from tqdm import tqdm
import copy
from transformers import BertTokenizerFast
import stanza
import logging
import time
from IPython.core.debugger import set_trace
import torch


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
        '''
        :param matrix_size:
        :return: a list mapping shaking sequence points to matrix points
        '''
        shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(matrix_size) for end_ind in
                                  list(range(matrix_size))[ind:]]
        return shaking_idx2matrix_idx

    @staticmethod
    def get_matrix_idx2shaking_idx(matrix_size):
        '''
        :param matrix_size:
        :return: a matrix mapping matrix points to shaking sequence points
        '''
        matrix_idx2shaking_idx = [[0 for i in range(matrix_size)] for j in range(matrix_size)]
        shaking_idx2matrix_idx = Indexer.get_shaking_idx2matrix_idx(matrix_size)
        for shaking_ind, matrix_ind in enumerate(shaking_idx2matrix_idx):
            matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind
        return matrix_idx2shaking_idx

    @staticmethod
    def points2shaking_seq(points, matrix_size, tag_size):
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
    def points2shaking_seq_batch(batch_points, matrix_size, tag_size):
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


class WordTokenizer:
    '''
    word level tokenizer,
    for word level encoders (LSTM, GRU, etc.)
    '''
    def __init__(self, stanza_nlp):
        '''
        :param word2idx: a dictionary, mapping words to ids
        '''
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
    #     #         "if you invoke function text2word_indices, self.word2idx should be set when initialize WordTokenizer")
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
    
    def encode_plus(self, text, *args, **kwargs):
        words_by_stanza = []
        word2char_span = []
        for sent in self.get_stanza_nlp()(text).sentences:
            for word in sent.words:
                words_by_stanza.append(word.text)
                start_char, end_char = word.misc.split("|")
                start_char, end_char = int(start_char.split("=")[1]), int(end_char.split("=")[1])
                word2char_span.append([start_char, end_char])
                
        return self.encode_plus_fr_words(words_by_stanza, word2char_span, *args, **kwargs)
    
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
    def __init__(self, language, pretrained_model_path):
        self.subword_tokenizer = None
        self.word_tokenizer = None
        self.language = language
        self.pretrained_model_path = pretrained_model_path

    @staticmethod
    def get_tok2char_span_map(tokens):
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span

    @staticmethod
    def unique_list(inp_list):
        out_list = []
        memory = set()
        for item in inp_list:
            if str(item) not in memory:
                out_list.append(item)
                memory.add(str(item))
        return out_list

    def get_word_tokenizer(self):
        if self.word_tokenizer is None:
            stanza_nlp = stanza.Pipeline(self.language)
            self.word_tokenizer = WordTokenizer(stanza_nlp)
        return self.word_tokenizer

    def get_subword_tokenizer(self):
        if self.subword_tokenizer is None:
            self.subword_tokenizer = BertTokenizerAlignedWithStanza.from_pretrained(self.pretrained_model_path,
                                                                                   add_special_tokens=False,
                                                                                   do_lower_case=False,
                                                                                   stanza_language=self.language)
            print("tokenizer loaded: {}".format(self.pretrained_model_path))
        return self.subword_tokenizer

    def _get_char2tok_span(self, tok2char_span):
        '''

        get a map from character level index to token level span
        e.g. "She is singing" -> [
                                 [0, 1], [0, 1], [0, 1], # She
                                 [-1, -1] # whitespace
                                 [1, 2], [1, 2], # is
                                 [-1, -1] # whitespace
                                 [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] # singing
                                 ]

         tok2char_span： a map from token index to character level span
        '''

        # get the number of characters
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break

        # build a map: char index to token level span
        char2tok_span = [[-1, -1] for _ in range(char_num)]  # 除了空格，其他字符均有对应token
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为在bert中，char to tok 也可能出现1对多的情况，比如韩文。
                # 所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1: # 第一次赋值以后不再修改
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1 # 一直修改
        return char2tok_span

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
            for m in re.finditer(re.escape(target_ent), text_cp):
                # if consider subword, avoid matching an incomplete number, "76567" -> "65", or it will introduce too many errors.
                if not ignore_subword_match and re.match("\d+", target_ent):
                    if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or (
                            m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                        continue
                # if ignore_subword_match, we use " {original entity} " to match. So, we need to recover the correct span
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
                spans.append(span)
            ent2char_spans[ent] = spans
        return ent2char_spans

    @staticmethod
    def transform_data(data, ori_format, dataset_type, add_id=True):
        '''
        This function is for transforming data published by previous work on [joint extraction].
        If you want to feed new dataset to the model, just define your own function to transform data.
        data: original data
        ori_format: "casrel", "etl_span", "raw_nyt", "tplinker"
        dataset_type: "train", "valid", "test"; only for generate id for the data
        '''
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc="transforming data format"):
            normal_sample = {}
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)

            if ori_format == "tplinker":
                normal_sample_list.append({**normal_sample, **sample})
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
            for rel in rel_list:
                normal_rel = {
                    "subject": rel[subj_key],
                    "predicate": rel[pred_key],
                    "object": rel[obj_key],
                }
                normal_rel_list.append(normal_rel)
            normal_sample["relation_list"] = normal_rel_list
            normal_sample_list.append(normal_sample)

        def clean_text(text):
            text = re.sub("�", "", text)
            #             text = re.sub("([A-Za-z]+)", r" \1 ", text)
            #             text = re.sub("(\d+)", r" \1 ", text)
            #             text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(normal_sample_list, desc="Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])

        return normal_sample_list

    def add_char_span(self, dataset, ignore_subword_match=True):
        '''
        if the dataset has not annotated character level spans, add them automatically
        :param dataset:
        :param ignore_subword_match: if a word is a subword of another word, ignore its span.
        :return:
        '''
        for sample in tqdm(dataset, desc="adding char level spans"):
            entities = []
            if "relation_list" in sample:
                entities.extend([rel["subject"] for rel in sample["relation_list"]])
                entities.extend([rel["object"] for rel in sample["relation_list"]])
            if "entity_list" in sample:
                entities.extend([ent["text"] for ent in sample["entity_list"]])
            # if "event_list" in sample:
            #     for event in sample["event_list"]:
            #         entities.append(event["trigger"])
            #         entities.extend([arg["text"] for arg in event["argument_list"]])

            ent2char_spans = self._get_ent2char_spans(sample["text"], entities,
                                                      ignore_subword_match=ignore_subword_match)

            if "relation_list" in sample:
                relation_list = []
                for rel in sample["relation_list"]:
                    subj_char_spans = ent2char_spans[rel["subject"]]
                    obj_char_spans = ent2char_spans[rel["object"]]
                    for subj_sp in subj_char_spans:
                        for obj_sp in obj_char_spans:
                            relation_list.append({
                                "subject": rel["subject"],
                                "object": rel["object"],
                                "subj_char_span": subj_sp,
                                "obj_char_span": obj_sp,
                                "predicate": rel["predicate"],
                            })

                assert len(sample["relation_list"]) <= len(relation_list)
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
                assert len(new_ent_list) >= sample["entity_list"]
                sample["entity_list"] = new_ent_list

            if "event_list" in sample:
                miss_span = False
                for event in sample["event_list"]:
                    if "trigger_char_span" not in event:
                        miss_span = True
                    for arg in event["argument_list"]:
                        if "char_span" not in arg:
                            miss_span = True

                error_info = "it is not a good idea to automatically add character level spans for events. Because it will introduce too many errors"
                assert miss_span is False, error_info
        return dataset

    def add_tok_span(self, data):
        '''
        add token level span according to the character spans, character level spans are required
        '''
        def char_span2tok_span(char_span, char2tok_span):
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span

        for sample in tqdm(data, desc="adding word level and subword level spans"):
            char2word_span = self._get_char2tok_span(sample["features"]["word2char_span"])
            char2subwd_span = self._get_char2tok_span(sample["features"]["subword2char_span"])
            
            if "relation_list" in sample:
                for rel in sample["relation_list"]:
                    subj_char_span = rel["subj_char_span"]
                    obj_char_span = rel["obj_char_span"]
                    rel["subj_wd_span"] = char_span2tok_span(subj_char_span, char2word_span)
                    rel["obj_wd_span"] = char_span2tok_span(obj_char_span, char2word_span)
                    rel["subj_subwd_span"] = char_span2tok_span(subj_char_span, char2subwd_span)
                    rel["obj_subwd_span"] = char_span2tok_span(obj_char_span, char2subwd_span)
                    
            if "entity_list" in sample:
                for ent in sample["entity_list"]:
                    char_span = ent["char_span"]
                    ent["wd_span"] = char_span2tok_span(char_span, char2word_span)
                    ent["subwd_span"] = char_span2tok_span(char_span, char2subwd_span)
                    
            if "event_list" in sample:
                for event in sample["event_list"]:
                    event["trigger_wd_span"] = char_span2tok_span(event["trigger_char_span"], char2word_span)
                    event["trigger_subwd_span"] = char_span2tok_span(event["trigger_char_span"], char2subwd_span)
                    for arg in event["argument_list"]:
                        arg["wd_span"] = char_span2tok_span(arg["char_span"], char2word_span)
                        arg["subwd_span"] = char_span2tok_span(arg["char_span"], char2subwd_span)
        return data

    @staticmethod
    def _extract_ent(tok_span, tok2char_span, text):
        char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
        char_span = [char_span_list[0][0], char_span_list[-1][1]]
        text_extr = text[char_span[0]:char_span[1]]
        return text_extr

    @staticmethod
    def check_tok_span(data):
        '''
        check if text is equal to the one extracted by the annotated token level spans
        :param data: 
        :return: 
        '''
        sample_id2mismatched_ents = {}

        for sample in tqdm(data, desc="checking word level and subword level spans"):
            text = sample["text"]
            word2char_span = sample["features"]["word2char_span"]
            subword2char_span = sample["features"]["subword2char_span"]

            sample_id2mismatched_ents[sample["id"]] = {}
            if "entity_list" in sample:
                bad_entities = []
                for ent in sample["entity_list"]:
                    word_span = ent["wd_span"]
                    subword_span = ent["subwd_span"]
                    ent_wd = Preprocessor._extract_ent(word_span, word2char_span, text)
                    ent_subwd = Preprocessor._extract_ent(subword_span, subword2char_span, text)

                    if not (ent_wd == ent_subwd == ent["text"]):
                        bad_ent = copy.deepcopy(ent)
                        bad_ent["extr_ent_wd"] = ent_wd
                        bad_ent["extr_ent_subwd"] = ent_subwd
                        bad_entities.append(bad_ent)
                if len(bad_entities) > 0:
                    sample_id2mismatched_ents[sample["id"]]["bad_entites"] = bad_entities

            if "relation_list" in sample:
                bad_rels = []
                for rel in sample["relation_list"]:
                    subj_wd_span = rel["subj_wd_span"]
                    obj_wd_span = rel["obj_wd_span"]
                    subj_subwd_span = rel["subj_subwd_span"]
                    obj_subwd_span = rel["obj_subwd_span"]

                    subj_wd = Preprocessor._extract_ent(subj_wd_span, word2char_span, text)
                    obj_wd = Preprocessor._extract_ent(obj_wd_span, word2char_span, text)
                    subj_subwd = Preprocessor._extract_ent(subj_subwd_span, subword2char_span, text)
                    obj_subwd = Preprocessor._extract_ent(obj_subwd_span, subword2char_span, text)

                    if not (subj_wd == rel["subject"] == subj_subwd and obj_wd == rel["object"] == obj_subwd):
                        bad_rel = copy.deepcopy(rel)
                        bad_rel["extr_subj_wd"] = subj_wd
                        bad_rel["extr_subj_subwd"] = subj_subwd
                        bad_rel["extr_obj_wd"] = obj_wd
                        bad_rel["extr_obj_subwd"] = obj_subwd
                        bad_rels.append(bad_rel)
                if len(bad_rels) > 0:
                    sample_id2mismatched_ents[sample["id"]]["bad_relations"] = bad_rels

            if "event_list" in sample:
                bad_events = []
                for event in sample["event_list"]:
                    event_cp = copy.deepcopy(event)
                    bad = False
                    trigger_wd_span = event["trigger_wd_span"]
                    trigger_subwd_span = event["trigger_subwd_span"]
                    trigger_wd = Preprocessor._extract_ent(trigger_wd_span, word2char_span, text)
                    trigger_subwd = Preprocessor._extract_ent(trigger_subwd_span, subword2char_span, text)

                    if not (trigger_wd == trigger_subwd == event["trigger"]):
                        bad = True
                        event_cp["extr_trigger_wd"] = trigger_wd
                        event_cp["extr_trigger_subwd"] = trigger_subwd

                    for arg in event_cp["argument_list"]:
                        arg_wd_span = arg["wd_span"]
                        arg_subwd_span = arg["subwd_span"]
                        arg_wd = Preprocessor._extract_ent(arg_wd_span, word2char_span, text)
                        arg_subwd = Preprocessor._extract_ent(arg_subwd_span, subword2char_span, text)
                        if not (arg_wd == arg_subwd == arg["text"]):
                            bad = True
                            arg["extr_arg_wd"] = arg_wd
                            arg["extr_arg_subwd"] = arg_subwd
                    if bad:
                        bad_events.append(event)
                if len(bad_events) > 0:
                    sample_id2mismatched_ents[sample["id"]]["bad_events"] = bad_events

            if len(sample_id2mismatched_ents[sample["id"]]) == 0:
                del sample_id2mismatched_ents[sample["id"]]
        return sample_id2mismatched_ents

    def create_features(self, data, generate=False):
        # create features
        for sample in tqdm(data, desc="create features"):
            text = sample["text"]
            # word level
            word_level_feature_keys = {"ner_tag_list", "word_list", "pos_tag_list", "dependency_list", "word2char_span"}
            # ## to guarantee the lengths are equal, if one key does not exist, generate features from scratch
            # generate = False
            # for key in word_level_feature_keys:
            #     if key not in sample:
            #         generate = True

            word_features = {}
            if "word_list" not in sample or generate:
                word_features = self.get_word_tokenizer().tokenize_plus(text)  # generate word level features
            else:
                for key in word_level_feature_keys:
                    if key in sample:
                        word_features[key] = sample[key]
                        del sample[key]
                if "word_list" in word_features and "word2char_span" not in word_features:
                    word_features["word2char_span"] = Preprocessor.get_tok2char_span_map(word_features["word_list"])
                # word_features = {
                #     "word_list": sample["word_list"],
                #     "word2char_span": sample["word2char_span"],
                #     "pos_tag_list": sample["pos_tag_list"],
                #     "ner_tag_list": sample["ner_tag_list"],
                #     "dependency_list": sample["dependency_list"],
                # }
                # del sample["word_list"]
                # del sample["word2char_span"]
                # del sample["dependency_list"]
                # del sample["pos_tag_list"]
                # del sample["ner_tag_list"]

            sample["features"] = word_features

            # subword level features
            codes = self.get_subword_tokenizer().encode_plus_fr_words(sample["features"]["word_list"],
                                                                                 sample["features"]["word2char_span"],
                                                                                 return_offsets_mapping=True,
                                                                                 add_special_tokens=False,
                                                                                 )
            subword_features = {
                "subword_list": codes["subword_list"],
                "subword2char_span": codes["offset_mapping"],
            }

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

            if "dependency_list" in word_features:
                ## transform word dependencies to matrix point
                word_dependency_list = word_features["dependency_list"]
                new_word_dep_list = [[wid, dep[0] + wid, dep[1]] for wid, dep in enumerate(word_dependency_list)]
                word_features["word_dependency_list"] = new_word_dep_list
                del word_features["dependency_list"]

                ## generate subword level dependency list
                subword_dep_list = []
                for dep in sample["features"]["word_dependency_list"]:
                    for subw_id1 in range(*word2subword_span[dep[0]]): # debug
                        for subw_id2 in range(*word2subword_span[dep[1]]):
                            subword_dep_list.append([subw_id1, subw_id2, dep[2]])
                subword_features["subword_dependency_list"] = subword_dep_list

            # add subword level features into the feature list
            sample["features"] = {
                **sample["features"],
                **subword_features,
                "subword2word_id": subword2word_id,
                "word2subword_span": word2subword_span,
            }

            # check features
            feats = sample["features"]
            num_words = len(word2subword_span)
            for k in {"ner_tag_list", "pos_tag_list", "word2char_span", "word_list"}:
                if k in feats:
                    assert len(feats[k]) == num_words
            assert len(feats["subword_list"]) == len(feats["subword2char_span"]) == len(subword2word_id)
            for subw_id, wid in enumerate(subword2word_id):
                subw = sample["features"]["subword_list"][subw_id]
                word = sample["features"]["word_list"][wid]
                assert re.sub("##", "", subw) in word or subw == "[UNK]"
            for subw_id, char_sp in enumerate(feats["subword2char_span"]):
                subw = sample["features"]["subword_list"][subw_id]
                subw = re.sub("##", "", subw)
                subw_extr = sample["text"][char_sp[0]:char_sp[1]]
                assert subw_extr == subw or subw == "[UNK]"
        return data

    def _process_main_data(self, data, task, add_char_span, ignore_subword_match):
        # add char spans
        if add_char_span:
            self.add_char_span(data, ignore_subword_match=ignore_subword_match)

        # build features
        separator = "_"  # for event extraction task
        for sample in tqdm(data, desc="processing main data"):
            text = sample["text"]
            # features
            ## word level
            word_feature_keys = {"ner_tag_list", "word_list", "pos_tag_list", "dependency_list", "tok2char_span"}
            # if a key not exists, generate features from scratch
            # (to guarantee the lengths are equal)
            generate = False
            for key in word_feature_keys:
                if key not in sample:
                    generate = True
            if generate:
                sample["word_level_features"] = self.get_word_tokenizer().tokenize_plus(text)
            else:
                word_level_features = {
                    "word_list": sample["word_list"],
                    "tok2char_span": sample["tok2char_span"],
                    "pos_tag_list": sample["pos_tag_list"],
                    "ner_tag_list": sample["ner_tag_list"],
                    "dependency_list": sample["dependency_list"],
                }
                sample["word_level_features"] = word_level_features
                del sample["word_list"]
                del sample["tok2char_span"]
                del sample["dependency_list"]
                del sample["pos_tag_list"]
                del sample["ner_tag_list"]

            ### transform dependencies to matrix point
            new_dep_list = []
            for wid, dep in enumerate(sample["word_level_features"]["dependency_list"]):
                new_dep_list.append([wid, dep[0] + wid, dep[1]])
            sample["word_level_features"]["dependency_list"] = new_dep_list

            ## subword level
            subword_features = self.get_subword_tokenizer().encode_plus_fr_words(sample["word_level_features"]["word_list"],
                                                                           sample["word_level_features"]["tok2char_span"],
                                                                           return_offsets_mapping=True,
                                                                           add_special_tokens=False,
                                                                           )
            # generate subword2word_idx dict
            subword2char_span = subword_features["offset_mapping"]

            char2word_span = self._get_char2tok_span(sample["word_level_features"]["tok2char_span"])
            subword2word_idx = []
            for subw_id, char_sp in enumerate(subword2char_span):
                wd_sps = char2word_span[char_sp[0]:char_sp[1]]
                assert wd_sps[0][0] == wd_sps[-1][1] - 1 # the same word idx
                subword2word_idx.append(wd_sps[0][0])

            subword_features["tok2char_span"] = subword_features["offset_mapping"]
            del subword_features["offset_mapping"]
            sample["subword_level_features"] = {
                **subword_features,
                "word_list": [sample["word_level_features"]["word_list"][wid] for wid in subword2word_idx],
                "ner_tag_list": [sample["word_level_features"]["ner_tag_list"][wid] for wid in subword2word_idx],
                "pos_tag_list": [sample["word_level_features"]["pos_tag_list"][wid] for wid in subword2word_idx],
            }
            subw_feats = sample["subword_level_features"]

            ### generate subword level dependency list
            word2subword_ids = [[] for _ in range(len(sample["word_level_features"]["word_list"]))]
            for sbw_idx, wid in enumerate(subword2word_idx):
                word2subword_ids[wid].append(sbw_idx)
            subword_dep_list = []
            for dep in sample["word_level_features"]["dependency_list"]:
                for subw_id1 in word2subword_ids[dep[0]]:
                    for subw_id2 in word2subword_ids[dep[1]]:
                        subword_dep_list.append([subw_id1, subw_id2, dep[2]])
            sample["subword_level_features"]["dependency_list"] = subword_dep_list

            # check features
            assert len(subw_feats["subword_list"]) == len(subw_feats["tok2char_span"]) == len(
                subw_feats["word_list"]) == len(subw_feats["ner_tag_list"]) == len(subw_feats["pos_tag_list"])
            for subw_id, wid in enumerate(subword2word_idx):
                subw = sample["subword_level_features"]["subword_list"][subw_id]
                word = sample["word_level_features"]["word_list"][wid]
                assert re.sub("##", "", subw) in word or subw == "[UNK]"
            for subw_id, char_sp in enumerate(subword2char_span):
                subw = sample["subword_level_features"]["subword_list"][subw_id]
                subw = re.sub("##", "", subw)
                subw_extr = sample["text"][char_sp[0]:char_sp[1]]
                assert subw_extr == subw or subw == "[UNK]"

            # entities, relations
            fin_ent_list, fin_rel_list = [], []
            if "entity_list" in sample:
                fin_ent_list.extend(sample["entity_list"])
                default_ent_list = copy.deepcopy(sample["entity_list"])
                # add default tag to entities
                for ent in default_ent_list:
                    ent["type"] = "EXT:DEFAULT"
                fin_ent_list.extend(default_ent_list)

            if "ee" in task:
                for event in sample["event_list"]:
                    fin_ent_list.append({
                        "text": event["trigger"],
                        "type": "EE:{}{}{}".format("Trigger", separator, event["trigger_type"]),  # EE: event extraction
                        "char_span": event["trigger_char_span"],
                    })
                    # add default tag to entities
                    fin_ent_list.append({
                        "text": event["trigger"],
                        "type": "EXT:DEFAULT",
                        "char_span": event["trigger_char_span"],
                    })
                    for arg in event["argument_list"]:
                        fin_ent_list.append({
                            "text": arg["text"],
                            "type": "EE:{}{}{}".format("Argument", separator, arg["type"]),
                            "char_span": arg["char_span"],
                        })
                        # add default tag to entities
                        fin_ent_list.append({
                            "text": arg["text"],
                            "type": "EXT:DEFAULT",
                            "char_span": arg["char_span"],
                        })

                        fin_rel_list.append({
                            "subject": arg["text"],
                            "subj_char_span": arg["char_span"],
                            "object": event["trigger"],
                            "obj_char_span": event["trigger_char_span"],
                            "predicate": "EE:{}{}{}".format(arg["type"], separator, event["trigger_type"]),
                        })

            if "re" in task:
                # add default tag to entities
                for rel in sample["relation_list"]:
                    fin_ent_list.append({
                        "text": rel["subject"],
                        "type": "EXT:DEFAULT",
                        "char_span": rel["subj_char_span"],
                    })
                    fin_ent_list.append({
                        "text": rel["object"],
                        "type": "EXT:DEFAULT",
                        "char_span": rel["obj_char_span"],
                    })
                fin_rel_list.extend(sample["relation_list"])

            sample["entity_list"] = fin_ent_list
            sample["relation_list"] = fin_rel_list

        # if default if the only entity type, count it in the final scoring
        # entity type with EXT (extra) will be ignored at the scoring stage.
        ent_set = set()
        for sample in data:
            ent_set |= {ent["type"] for ent in sample["entity_list"]}
        if len(ent_set) == 1 and ent_set.pop() == "EXT:DEFAULT":
            for sample in data:
                for ent in sample["entity_list"]:
                    ent["type"] = "DEFAULT"

        # add token spans (word level and subword level)
        self.add_tok_span(data)

        return data

    def generate_supporting_data(self, data, max_word_dict_size, min_word_freq):
        pos_tag_set = set()
        ner_tag_set = set()
        deprel_type_set = set()
        word2num = dict()
        word_set = set()
        char_set = set()
        max_word_seq_length, max_subword_seq_length = 0, 0

        for sample in tqdm(data, desc="generating supporting data"):
            # POS tag
            if "pos_tag_list" in sample["features"]:
                pos_tag_set |= {pos_tag for pos_tag in sample["features"]["pos_tag_list"]}
            # NER tag
            if "ner_tag_list" in sample["features"]:
                ner_tag_set |= {ner_tag for ner_tag in sample["features"]["ner_tag_list"]}
            # dependency relations
            if "word_dependency_list" in sample["features"]:
                deprel_type_set |= {deprel[-1] for deprel in sample["features"]["word_dependency_list"]}

            # character
            char_set |= set(sample["text"])
            # word
            for word in sample["features"]["word_list"]:
                word2num[word] = word2num.get(word, 0) + 1
            max_word_seq_length = max(max_word_seq_length, len(sample["features"]["word_list"]))
            max_subword_seq_length = max(max_subword_seq_length, len(sample["features"]["subword_list"]))

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
        }

        dicts = {
            "char2id": char2id,
            "word2id": word2id,
        }
        if "pos_tag_list" in data[0]["features"]:
            pos_tag2id = get_dict(pos_tag_set)
            data_statistics["pos_tag_num"] = len(pos_tag2id)
            dicts["pos_tag2id"] = pos_tag2id

        if "ner_tag_list" in data[0]["features"]:
            ner_tag_set.remove("O")
            ner_tag2id = {ner_tag: ind + 1 for ind, ner_tag in enumerate(sorted(ner_tag_set))}
            ner_tag2id["O"] = 0
            data_statistics["ner_tag_num"] = len(ner_tag2id)
            dicts["ner_tag2id"] = ner_tag2id

        if "word_dependency_list" in data[0]["features"]:
            deprel_type2id = get_dict(deprel_type_set)
            data_statistics["deprel_type_num"] = len(deprel_type2id)
            dicts["deprel_type2id"] = deprel_type2id

        return dicts, data_statistics

    @staticmethod
    def choose_features_by_token_level(data, token_level):
        for sample in data:
            features = sample["features"]
            if token_level == "subword":
                subword2word_id = features["subword2word_id"]
                new_features = {
                    "subword_list": features["subword_list"],
                    "tok2char_span": features["subword2char_span"],
                    "word_list": [features["word_list"][wid] for wid in subword2word_id],
                    # "ner_tag_list": [features["ner_tag_list"][wid] for wid in subword2word_id],
                    # "pos_tag_list": [features["pos_tag_list"][wid] for wid in subword2word_id],
                    # "dependency_list": features["subword_dependency_list"],
                }
                if "ner_tag_list" in features:
                    new_features["ner_tag_list"] = [features["ner_tag_list"][wid] for wid in subword2word_id]
                if "pos_tag_list" in features:
                    new_features["pos_tag_list"] = [features["pos_tag_list"][wid] for wid in subword2word_id]
                if "subword_dependency_list" in features:
                    new_features["dependency_list"] = features["subword_dependency_list"]
                sample["features"] = new_features
            else:
                new_features = {
                    "word_list": features["word_list"],
                    "subword_list": features["word_list"],
                    "tok2char_span": features["word2char_span"],
                    # "ner_tag_list": features["ner_tag_list"],
                    # "pos_tag_list": features["pos_tag_list"],
                    # "dependency_list": features["word_dependency_list"],
                }
                if "ner_tag_list" in features:
                    new_features["ner_tag_list"] = features["ner_tag_list"]
                if "pos_tag_list" in features:
                    new_features["pos_tag_list"] = features["pos_tag_list"]
                if "word_dependency_list" in features:
                    new_features["dependency_list"] = features["word_dependency_list"]
                sample["features"] = new_features
        return data

    @staticmethod
    def choose_spans_by_token_level(data, token_level):
        '''
        :param data:
        :param token_level: "subword" or "word"
        :return:
        '''
        for sample in data:
            if "entity_list" in sample:
                for ent in sample["entity_list"]:
                    ent["tok_span"] = ent["subwd_span"] if token_level == "subword" else ent["wd_span"]
                    del ent["subwd_span"]
                    del ent["wd_span"]
            if "relation_list" in sample:
                for rel in sample["relation_list"]:
                    rel["subj_tok_span"] = rel["subj_subwd_span"] if token_level == "subword" else rel["subj_wd_span"]
                    rel["obj_tok_span"] = rel["obj_subwd_span"] if token_level == "subword" else rel["obj_wd_span"]
                    del rel["subj_wd_span"]
                    del rel["obj_wd_span"]
                    del rel["subj_subwd_span"]
                    del rel["obj_subwd_span"]
            if "event_list" in sample:
                for event in sample["event_list"]:
                    event["trigger_tok_span"] = event["trigger_subwd_span"] if token_level == "subword" else event["trigger_wd_span"]
                    del event["trigger_subwd_span"]
                    del event["trigger_wd_span"]
                    for arg in event["argument_list"]:
                        arg["tok_span"] = arg["subwd_span"] if token_level == "subword" else arg["wd_span"]
                        del arg["subwd_span"]
                        del arg["wd_span"]
        return data

    @staticmethod
    def filter_annotations(sample, start_ind, end_ind):
        '''
        filter annotations in [start_ind, end_ind]
        :param sample:
        :param start_ind:
        :param end_ind:
        :return:
        '''
        new_sample = copy.deepcopy(sample)
        if "relation_list" in sample:
            sub_rel_list = []
            for rel in sample["relation_list"]:
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                # if subject and object are both in this subtext, add this spo to new sample
                if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                        and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind:
                    rel_cp = copy.deepcopy(rel)
                    sub_rel_list.append(rel_cp)
            new_sample["relation_list"] = sub_rel_list

            # entity
        if "entity_list" in sample:
            sub_ent_list = []
            for ent in sample["entity_list"]:
                tok_span = ent["tok_span"]
                # if entity in this subtext, add the entity to new sample
                if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                    ent_cp = copy.deepcopy(ent)
                    sub_ent_list.append(ent_cp)
            new_sample["entity_list"] = sub_ent_list

            # event
        if "event_list" in sample:
            sub_event_list = []
            for event in sample["event_list"]:
                if "trigger" in event:
                    trigger_tok_span = event["trigger_tok_span"]
                    if trigger_tok_span[1] > end_ind or trigger_tok_span[0] < start_ind:
                        continue
                event_cp = copy.deepcopy(event)
                new_arg_list = []
                for arg in event_cp["argument_list"]:
                    tok_span = arg["tok_span"]
                    if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                        arg_cp = copy.deepcopy(arg)
                        new_arg_list.append(arg_cp)
                event_cp["argument_list"] = new_arg_list
                sub_event_list.append(event_cp)
            new_sample["event_list"] = sub_event_list
        return new_sample

    @staticmethod
    def split_into_short_samples(data, max_seq_len, sliding_len, data_type,
                                 token_level, wordpieces_prefix="##"):
        '''
        split samples with long text into samples with short subtexts
        :param data: original data
        :param max_seq_len: the max sequence length of a subtext
        :param sliding_len: the size of the sliding window
        :param data_type: train, valid, test
        :return:
        '''
        split_sample_list = []

        for sample in tqdm(data, desc="splitting"):
            id = sample["id"]
            text = sample["text"]
            features = sample["features"]
            tokens = features["subword_list"] if token_level == "subword" else features["word_list"]
            tok2char_span = features["tok2char_span"]

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
                split_features = {"word_list": features["word_list"][start_ind:end_ind],
                                  "subword_list": features["subword_list"][start_ind:end_ind],
                                  "tok2char_span": [[char_sp[0] - char_level_offset, char_sp[1] - char_level_offset]
                                                    for char_sp in features["tok2char_span"][start_ind:end_ind]],
                                  # "pos_tag_list": features["pos_tag_list"][start_ind:end_ind],
                                  # "ner_tag_list": features["ner_tag_list"][start_ind:end_ind],
                                  # "dependency_list": [],
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

                new_sample = {
                    "id": id,
                    "text": sub_text,
                    "features": split_features,
                    "tok_level_offset": tok_level_offset,
                    "char_level_offset": char_level_offset,
                    "entity_list": [],
                    "relation_list": [],
                    "event_list": [],
                }
                if data_type == "test":
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else:
                    # if not test data, need to filter entities, relations, and events in the subtext
                    # relation
                    if "relation_list" in sample:
                        sub_rel_list = []
                        for rel in sample["relation_list"]:
                            subj_tok_span = rel["subj_tok_span"]
                            obj_tok_span = rel["obj_tok_span"]
                            # if subject and object are both in this subtext, add this spo to new sample
                            if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                                    and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind:
                                rel_cp = copy.deepcopy(rel)
                                sub_rel_list.append(rel_cp)
                        new_sample["relation_list"] = sub_rel_list

                    # entity
                    if "entity_list" in sample:
                        sub_ent_list = []
                        for ent in sample["entity_list"]:
                            tok_span = ent["tok_span"]
                            # if entity in this subtext, add the entity to new sample
                            if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                                ent_cp = copy.deepcopy(ent)
                                sub_ent_list.append(ent_cp)
                        new_sample["entity_list"] = sub_ent_list

                    # event
                    if "event_list" in sample:
                        sub_event_list = []
                        for event in sample["event_list"]:
                            trigger_tok_span = event["trigger_tok_span"]
                            if trigger_tok_span[1] > end_ind or trigger_tok_span[0] < start_ind:
                                continue
                            event_cp = copy.deepcopy(event)
                            new_arg_list = []
                            for arg in event_cp["argument_list"]:
                                tok_span = arg["tok_span"]
                                if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                                    arg_cp = copy.deepcopy(arg)
                                    new_arg_list.append(arg_cp)
                            event_cp["argument_list"] = new_arg_list
                            sub_event_list.append(event_cp)
                        new_sample["event_list"] = sub_event_list
                    # offset
                    new_sample = Preprocessor.span_offset(new_sample, - tok_level_offset, - char_level_offset)
                    split_sample_list.append(new_sample)

                if end_ind > len(tokens):
                    break
        return split_sample_list

    @staticmethod
    def combine(data, max_seq_len):

        def get_new_com_sample():
            return {
                "id": "combined_{}".format(len(new_data)),
                "text": "",
                "features": {
                    "word_list": [],
                    "subword_list": [],
                    "tok2char_span": [],
                    "pos_tag_list": [],
                    "ner_tag_list": [],
                    "dependency_list": []
                },
                "splits": [],
                "entity_list": [],
                "relation_list": [],
                "event_list": [],
            }

        new_data = []
        combined_sample = get_new_com_sample()
        for sample in tqdm(data, desc="combining splits"):
            if len(combined_sample["features"]["tok2char_span"] + sample["features"]["tok2char_span"]) > max_seq_len:
                new_data.append(combined_sample)
                combined_sample = get_new_com_sample()
            # combine features
            if len(combined_sample["text"]) > 0:
                combined_sample["text"] += " "
            combined_sample["text"] += sample["text"]
            combined_sample["features"]["word_list"].extend(sample["features"]["word_list"])
            combined_sample["features"]["subword_list"].extend(sample["features"]["subword_list"])
            combined_sample["features"]["pos_tag_list"].extend(sample["features"]["pos_tag_list"])
            combined_sample["features"]["ner_tag_list"].extend(sample["features"]["ner_tag_list"])
            token_offset = len(combined_sample["features"]["tok2char_span"])
            char_offset = 0
            if token_offset > 0:
                char_offset = combined_sample["features"]["tok2char_span"][-1][1] + 1 # +1: whitespace
            new_tok2char_span = [[char_sp[0] + char_offset, char_sp[1] + char_offset] for char_sp in sample["features"]["tok2char_span"]]
            combined_sample["features"]["tok2char_span"].extend(new_tok2char_span)

            new_dependency_list = [[dep[0] + token_offset, dep[1] + token_offset, dep[2]] for dep in sample["features"]["dependency_list"]]
            combined_sample["features"]["dependency_list"].extend(new_dependency_list)
            # record split offsets
            combined_sample["splits"].append({
                "id": sample["id"],
                "offset_in_this_seg": [token_offset, token_offset + len(sample["features"]["tok2char_span"])],
                "ori_offset": {
                    "tok_level_offset": sample["tok_level_offset"],
                    "char_level_offset": sample["char_level_offset"],
                }
            })

            # combine annotations
            sample_cp = copy.deepcopy(sample)
            Preprocessor.span_offset(sample_cp, token_offset, char_offset)
            if "entity_list" in sample_cp:
                combined_sample["entity_list"].extend(sample_cp["entity_list"])
            if "relation_list" in sample_cp:
                combined_sample["relation_list"].extend(sample_cp["relation_list"])
            if "event_list" in sample_cp:
                combined_sample["event_list"].extend(sample_cp["event_list"])

        # do not forget the last one
        if combined_sample["text"] != "":
            new_data.append(combined_sample)
        return new_data

    @staticmethod
    def decompose2splits(data):
        '''
        decompose combined samples to splits by "splits"
        :param data:
        :return:
        '''
        new_data = []
        for sample in data:
            if "splits" in sample:
                text = sample["text"]
                tok2char_span = sample["features"]["tok2char_span"]
                # decompose
                for spl in sample["splits"]:
                    split_sample = {
                        "id": spl["id"],
                        "tok_level_offset": spl["ori_offset"]["tok_level_offset"],
                        "char_level_offset": spl["ori_offset"]["char_level_offset"],
                    }
                    text_tok_span = spl["offset_in_this_seg"]
                    char_sp_list = tok2char_span[text_tok_span[0]:text_tok_span[1]]
                    text_char_span = [char_sp_list[0][0], char_sp_list[-1][1]]
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
                    # recover spans
                    split_sample = Preprocessor.span_offset(split_sample, -text_tok_span[0], -text_char_span[0])
                    new_data.append(split_sample)
            else:
                new_data.append(sample)
        return new_data

    @staticmethod
    def span_offset(sample, tok_level_offset, char_level_offset):
        if "relation_list" in sample:
            for rel in sample["relation_list"]:
                rel["subj_tok_span"][0] += tok_level_offset
                rel["subj_tok_span"][1] += tok_level_offset
                rel["obj_tok_span"][0] += tok_level_offset
                rel["obj_tok_span"][1] += tok_level_offset
                rel["subj_char_span"][0] += char_level_offset
                rel["subj_char_span"][1] += char_level_offset
                rel["obj_char_span"][0] += char_level_offset
                rel["obj_char_span"][1] += char_level_offset
        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                ent["tok_span"][0] += tok_level_offset
                ent["tok_span"][1] += tok_level_offset
                ent["char_span"][0] += char_level_offset
                ent["char_span"][1] += char_level_offset
        if "event_list" in sample:
            for event in sample["event_list"]:
                if "trigger" in event:
                    event["trigger_tok_span"][0] += tok_level_offset
                    event["trigger_tok_span"][1] += tok_level_offset
                    event["trigger_char_span"][0] += char_level_offset
                    event["trigger_char_span"][1] += char_level_offset
                for arg in event["argument_list"]:
                    arg["tok_span"][0] += tok_level_offset
                    arg["tok_span"][1] += tok_level_offset
                    arg["char_span"][0] += char_level_offset
                    arg["char_span"][1] += char_level_offset
        return sample

    @staticmethod
    def check_spans(data):
        sample_id2mismatched_ents = {}
        for sample in tqdm(data, desc="checking splits"):
            text = sample["text"]
            tok2char_span = sample["features"]["tok2char_span"]

            bad_entities = []
            bad_rels = []
            bad_events = []
            if "entity_list" in sample:
                for ent in sample["entity_list"]:
                    tok_span = ent["tok_span"]
                    extr_ent = Preprocessor._extract_ent(tok_span, tok2char_span, text)
                    extr_ent_c = text[ent["char_span"][0]:ent["char_span"][1]]

                    if not (extr_ent == ent["text"] == extr_ent_c):
                        bad_ent = copy.deepcopy(ent)
                        bad_ent["extr_ent"] = extr_ent
                        bad_ent["extr_ent_c"] = extr_ent_c
                        bad_entities.append(bad_ent)
                sample_id2mismatched_ents[sample["id"]] = {
                    "bad_entites": bad_entities,
                }

            if "relation_list" in sample:
                for rel in sample["relation_list"]:
                    subj_tok_span = rel["subj_tok_span"]
                    obj_tok_span = rel["obj_tok_span"]

                    extr_subj = Preprocessor._extract_ent(subj_tok_span, tok2char_span, text)
                    extr_obj = Preprocessor._extract_ent(obj_tok_span, tok2char_span, text)
                    extr_subj_c = text[rel["subj_char_span"][0]:rel["subj_char_span"][1]]
                    extr_obj_c = text[rel["obj_char_span"][0]:rel["obj_char_span"][1]]

                    if not (extr_subj == rel["subject"] == extr_subj_c and extr_obj == rel["object"] == extr_obj_c):
                        bad_rel = copy.deepcopy(rel)
                        bad_rel["extr_subj"] = extr_subj
                        bad_rel["extr_obj"] = extr_obj
                        bad_rel["extr_subj_c"] = extr_subj_c
                        bad_rel["extr_obj_c"] = extr_obj_c
                        bad_rels.append(bad_rel)
                sample_id2mismatched_ents[sample["id"]]["bad_relations"] = bad_rels

            if "event_list" in sample:
                for event in sample["event_list"]:
                    bad_event = copy.deepcopy(event)
                    bad = False
                    trigger_tok_span = event["trigger_tok_span"]
                    extr_trigger = Preprocessor._extract_ent(trigger_tok_span, tok2char_span, text)
                    extr_trigger_c = text[event["trigger_char_span"][0]:event["trigger_char_span"][1]]
                    if not (extr_trigger == event["trigger"] == extr_trigger_c):
                        bad = True
                        bad_event["extr_trigger"] = extr_trigger
                        bad_event["extr_trigger_c"] = extr_trigger_c

                    for arg in event["argument_list"]:
                        arg_span = arg["tok_span"]
                        extr_arg = Preprocessor._extract_ent(arg_span, tok2char_span, text)
                        extr_arg_c = text[arg["char_span"][0]:arg["char_span"][1]]
                        if not (extr_arg == arg["text"] == extr_arg_c):
                            bad = True
                            bad_event["extr_arg"] = extr_arg
                            bad_event["extr_arg_c"] = extr_arg_c
                    if bad:
                        bad_events.append(event)

            sample_id2mismatched_ents[sample["id"]] = {}
            if len(bad_entities) > 0:
                sample_id2mismatched_ents[sample["id"]]["bad_entities"] = bad_entities
            if len(bad_rels) > 0:
                sample_id2mismatched_ents[sample["id"]]["bad_relations"] = bad_rels
            if len(bad_events) > 0:
                sample_id2mismatched_ents[sample["id"]]["bad_events"] = bad_events
            if len(sample_id2mismatched_ents[sample["id"]]) == 0:
                del sample_id2mismatched_ents[sample["id"]]
        return sample_id2mismatched_ents

    @staticmethod
    def index_features(data, key2dict, max_seq_len, max_char_num_in_tok=None):
        '''
        :param data:
        :param key2dict: feature key to dict for indexing
        :param max_seq_len:
        :param max_char_num_in_tok: max character number in a token, truncate or pad char list to this length
        :param pretrained_model_padding: for subword ids padding
        :return:
        '''

        # map for replacing key names
        key_map = {
            "char_list": "char_input_ids",
            "word_list": "word_input_ids",
            "subword_list": "subword_input_ids",
            "pos_tag_list": "pos_tag_ids",
            "ner_tag_list": "ner_tag_ids",
            "dependency_list": "dependency_points"
        }

        for sample in tqdm(data, desc="indexing"):
            features = sample["features"]
            features["token_type_ids"] = [0] * len(features["tok2char_span"])
            features["attention_mask"] = [1] * len(features["tok2char_span"])
            features["char_list"] = list(sample["text"])

            indexed_features = {}
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
                        indexed_features[key_map[f_key]] = indexer.index_tag_list_w_matrix_pos(tags)
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
                        indexed_features[key_map[f_key]] = torch.LongTensor(char_input_ids_padded)
                    else:
                        indexed_features[key_map[f_key]] = torch.LongTensor(indexer.index_tag_list(tags))

                # features only need padding
                elif f_key in {"token_type_ids", "attention_mask"}:
                    indexed_features[f_key] = torch.LongTensor(Indexer.pad2length(tags, 0, max_seq_len))
                elif f_key == "tok2char_span":
                    indexed_features[f_key] = Indexer.pad2length(tags, [0, 0], max_seq_len)
            sample["features"] = indexed_features
        return data


if __name__ == "__main__":
    bert = BertTokenizerFast.from_pretrained("../../data/pretrained_models/bert-base-uncased")
    text = "type1; type2; type3[SEP]FSAN jkfsn"
    codes = bert.encode_plus(text, return_offsets_mapping=True)
    print(bert.tokenize(text))
    print(codes["offset_mapping"])
    pass
