import re
from tqdm import tqdm
import copy
from transformers import BertTokenizerFast
import stanza
import logging
import time
from IPython.core.debugger import set_trace

class Indexer:
    def __init__(self, tag2id, max_seq_len, spe_tag_dict):
        self.tag2id = tag2id
        self.max_seq_len = max_seq_len
        self.spe_tag_dict = spe_tag_dict

    def get_indices(self, tags):
        '''
        tags: a list of tag
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
            "tok2char_span": tok2char_span,
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
        self.stanza_nlp = kwargs["stanza_nlp"]
    
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
        words_by_stanza = [word.text for sent in self.stanza_nlp(text).sentences for word in sent.words]
        return self.tokenize_fr_words(words_by_stanza, max_length=max_length, *args, **kwargs)
    
    def encode_plus(self, text, *args, **kwargs):
        words_by_stanza = []
        word2char_span = []
        for sent in self.stanza_nlp(text).sentences:
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
    def __init__(self, word_tokenizer, subword_tokenizer):
        self.subword_tokenizer = subword_tokenizer
        self.word_tokenizer = word_tokenizer

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

    def _clean_sp_char(self, dataset):
        def clean_text(text):
            text = re.sub("�", "", text)
            #             text = re.sub("([A-Za-z]+)", r" \1 ", text)
            #             text = re.sub("(\d+)", r" \1 ", text)
            #             text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(dataset, desc="Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return dataset

    def transform_data(self, data, ori_format, dataset_type, add_id=True):
        '''
        This function is for transforming data published by previous work on [joint extraction].
        If you want to feed new dataset to the model, just define your own function to transform data.
        data: original data
        ori_format: "casrel", "etl_span", "raw_nyt"
        dataset_type: "train", "valid", "test"; only for generate id for the data
        '''
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc="Transforming data format"):
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

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
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

        return self._clean_sp_char(normal_sample_list)

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
                new_relation_list = []
                for rel in sample["relation_list"]:
                    subj_char_spans = ent2char_spans[rel["subject"]]
                    obj_char_spans = ent2char_spans[rel["object"]]
                    for subj_sp in subj_char_spans:
                        for obj_sp in obj_char_spans:
                            new_relation_list.append({
                                "subject": rel["subject"],
                                "object": rel["object"],
                                "subj_char_span": subj_sp,
                                "obj_char_span": obj_sp,
                                "predicate": rel["predicate"],
                            })

                assert len(sample["relation_list"]) <= len(new_relation_list)
                sample["relation_list"] = new_relation_list

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
            char2word_span = self._get_char2tok_span(sample["word_level_features"]["tok2char_span"])
            char2subwd_span = self._get_char2tok_span(sample["subword_level_features"]["tok2char_span"])
            
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

    def check_tok_span(self, data):
        '''
        check if text is equal to the one extracted by the annotated token level spans
        :param data: 
        :return: 
        '''
        sample_id2mismatched_ents = {}

        def extract_ent(tok_span, tok2char_span, text):
            char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
            char_span = [char_span_list[0][0], char_span_list[-1][1]]
            text_extr = text[char_span[0]:char_span[1]]
            return text_extr

        for sample in tqdm(data, desc="checking word level and subword level"):
            text = sample["text"]
            word2char_span = sample["word_level_features"]["tok2char_span"]
            subword2char_span = sample["subword_level_features"]["tok2char_span"]

            bad_entities = []
            bad_rels = []
            bad_events = []
            if "entity_list" in sample:

                for ent in sample["entity_list"]:
                    word_span = ent["wd_span"]
                    subword_span = ent["subwd_span"]
                    ent_wd = extract_ent(word_span, word2char_span, text)
                    ent_subwd = extract_ent(subword_span, subword2char_span, text)

                    if not (ent_wd == ent_subwd == ent["text"]):
                        bad_ent = copy.deepcopy(ent)
                        bad_ent["extr_ent_wd"] = ent_wd
                        bad_ent["extr_ent_subwd"] = ent_subwd
                        bad_entities.append(bad_ent)
                sample_id2mismatched_ents[sample["id"]] = {
                    "bad_entites": bad_entities,
                }

            if "relation_list" in sample:

                for rel in sample["relation_list"]:
                    subj_wd_span = rel["subj_wd_span"]
                    obj_wd_span = rel["obj_wd_span"]
                    subj_subwd_span = rel["subj_subwd_span"]
                    obj_subwd_span = rel["obj_subwd_span"]

                    subj_wd = extract_ent(subj_wd_span, word2char_span, text)
                    obj_wd = extract_ent(obj_wd_span, word2char_span, text)
                    subj_subwd = extract_ent(subj_subwd_span, subword2char_span, text)
                    obj_subwd = extract_ent(obj_subwd_span, subword2char_span, text)

                    if not (subj_wd == rel["subject"] == subj_subwd and obj_wd == rel["object"] == obj_subwd):
                        bad_rel = copy.deepcopy(rel)
                        bad_rel["extr_subj_wd"] = subj_wd
                        bad_rel["extr_subj_subwd"] = subj_subwd
                        bad_rel["extr_obj_wd"] = obj_wd
                        bad_rel["extr_obj_subwd"] = obj_subwd
                        bad_rels.append(bad_rel)
                sample_id2mismatched_ents[sample["id"]]["bad_relations"] = bad_rels

            if "event_list" in sample:

                for event in sample["event_list"]:
                    bad = False
                    trigger_wd_span = event["trigger_wd_span"]
                    trigger_subwd_span = event["trigger_subwd_span"]
                    trigger_wd = extract_ent(trigger_wd_span, word2char_span, text)
                    trigger_subwd = extract_ent(trigger_subwd_span, subword2char_span, text)
                    if not (trigger_wd == trigger_subwd == event["trigger"]):
                        bad = True

                    for arg in event["argument_list"]:
                        arg_wd_span = arg["wd_span"]
                        arg_subwd_span = arg["subwd_span"]
                        arg_wd = extract_ent(arg_wd_span, word2char_span, text)
                        arg_subwd = extract_ent(arg_subwd_span, subword2char_span, text)
                        if not (arg_wd == arg_subwd == arg["text"]):
                            bad = True
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

    def process_main_data(self, data, task, add_char_span, ignore_subword_match):
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
            # (to guarantee the lengths match each other)
            generate = False
            for key in word_feature_keys:
                if key not in sample:
                    generate = True
            if generate:
                sample["word_level_features"] = self.word_tokenizer.tokenize_plus(text)
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
            subword_features = self.subword_tokenizer.encode_plus_fr_words(sample["word_level_features"]["word_list"],
                                                                           sample["word_level_features"]["tok2char_span"],
                                                                           return_offsets_mapping=True,
                                                                           add_special_tokens=False,
                                                                           )
            subword2char_span = subword_features["offset_mapping"]
            char2word_span = self._get_char2tok_span(sample["word_level_features"]["tok2char_span"])
            subword2word_idx = []
            for subw_id, char_sp in enumerate(subword2char_span):
                wd_sps = char2word_span[char_sp[0]:char_sp[1]]
                assert wd_sps[0][0] == wd_sps[-1][1] - 1 # the same word idx
                subword2word_idx.append(wd_sps[0][0])

            sample["subword_level_features"] = {
                "subword_list": subword_features["subword_list"],
                "tok2char_span": subword2char_span,
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

    def generate_supporting_data(self, data, max_word_num, min_word_freq):
        rel_type_set = set()
        ent_type_set = set()
        pos_tag_set = set()
        ner_tag_set = set()
        deprel_type_set = set()
        word2num = dict()
        word_set = set()
        char_set = set()

        for sample in tqdm(data, desc="generating supporting data"):
            # entity type
            ent_type_set |= {ent["type"] for ent in sample["entity_list"]}
            # relation type
            rel_type_set |= {rel["predicate"] for rel in sample["relation_list"]}
            # POS tag
            pos_tag_set |= {pos_tag for pos_tag in sample["word_level_features"]["pos_tag_list"]}
            # NER tag
            ner_tag_set |= {ner_tag for ner_tag in sample["word_level_features"]["ner_tag_list"]}
            # dependency relations
            deprel_type_set |= {deprel[-1] for deprel in sample["word_level_features"]["dependency_list"]}
            # character
            char_set |= set(sample["text"])
            # word
            for word in sample["word_level_features"]["word_list"]:
                word2num[word] = word2num.get(word, 0) + 1

        rel_type_set = sorted(rel_type_set)
        rel2id = {rel: ind for ind, rel in enumerate(rel_type_set)}

        ent_type_set = sorted(ent_type_set)
        ent2id = {ent: ind for ind, ent in enumerate(ent_type_set)}

        def get_dict(tag_set):
            tag2id = {tag: ind + 2 for ind, tag in enumerate(sorted(tag_set))}
            tag2id["[PAD]"] = 0
            tag2id["[UNK]"] = 1
            return tag2id

        deprel_type2id = get_dict(deprel_type_set)
        pos_tag2id = get_dict(pos_tag_set)

        char2id = get_dict(char_set)
        ner_tag_set.remove("O")
        ner_tag2id = {ner_tag: ind + 1 for ind, ner_tag in enumerate(sorted(ner_tag_set))}
        ner_tag2id["O"] = 0

        word2num = dict(sorted(word2num.items(), key=lambda x: x[1], reverse=True))
        for tok, num in word2num.items():
            if num < min_word_freq:  # filter words with a frequency of less than <min_freq>
                continue
            word_set.add(tok)
            if len(word_set) == max_word_num:
                break
        word2id = get_dict(word_set)

        data_statistics = {
            "relation_type_num": len(rel2id),
            "entity_type_num": len(ent2id),
            "pos_tag_num": len(pos_tag2id),
            "ner_tag_num": len(ner_tag2id),
            "deprel_type_num": len(deprel_type2id),
            "word_num": len(word2id),
            "char_num": len(char2id),
        }

        dicts = {
            "rel_type2id": rel2id,
            "ent_type2id": ent2id,
            "pos_tag2id": pos_tag2id,
            "ner_tag2id": ner_tag2id,
            "deprel_type2id": deprel_type2id,
            "char2id": char2id,
            "word2id": word2id,
        }

        return dicts, data_statistics

    def split_into_short_samples(self,
                                 data,
                                 max_seq_len,
                                 sliding_len=50,
                                 data_type="train"):
        '''
        split samples with long text into samples with short subtexts
        :param data: original data
        :param max_seq_len: the max sequence length of a subtext
        :param sliding_len: the size of the sliding window
        :param data_type: train, valid, test
        :return:
        '''
        new_data = []
        for sample in tqdm(data, desc="Splitting"):
            medline_id = sample["id"]
            text = sample["text"]
            tokens = self.tokenize(text)
            tok2char_span = self.get_tok2char_span_map(text)

            # sliding on token level
            for start_ind in range(0, len(tokens), sliding_len):
                if self.use_bert:  # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                tok_spans = tok2char_span[start_ind:end_ind]
                char_span = (tok_spans[0][0], tok_spans[-1][-1])
                sub_text = text[char_span[0]:char_span[1]]

                if data_type == "test":
                    if len(sub_text) > 0:
                        new_sample = {
                            "id": medline_id,
                            "text": sub_text,
                            "tok_offset": start_ind,
                            "char_offset": char_span[0],
                        }
                        new_data.append(new_sample)
                else:
                    sub_entity_list = []
                    for term in sample["entity_list"]:
                        tok_span = term["tok_span"]
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                            new_term = copy.deepcopy(term)
                            new_term["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]
                            new_term["char_span"][0] -= char_span[0]
                            new_term["char_span"][1] -= char_span[0]
                            sub_entity_list.append(new_term)

                    #                     if len(sub_entity_list) > 0:
                    new_sample = {
                        "id": medline_id,
                        "text": sub_text,
                        "entity_list": sub_entity_list,
                    }
                    new_data.append(new_sample)

                if end_ind > len(tokens):
                    break
        return new_data


if __name__ == "__main__":
    pass
