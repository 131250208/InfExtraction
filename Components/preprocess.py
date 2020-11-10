import re
from tqdm import tqdm
import copy
from transformers import BertTokenizerFast
import torch

class WordTokenizer:
    '''
    word level tokenizer,
    for word level encoders (LSTM, GRU) instead of subword level encoders (BERT)
    '''
    def __init__(self, word2idx=None):
        '''
        :param word2idx: a dictionary, mapping word (token) to word id (token id)
        '''
        self.word2idx = word2idx

    def tokenize(self, text):
        return text.split(" ")

    def text2word_indices(self, text, max_length=-1):
        if not self.word2idx:
            raise ValueError(
                "if you invoke function text2word_indices, self.word2idx should be set when initialize WordTokenizer")
        word_ids = []
        words = text.split(" ")
        for w in words:
            if w not in self.word2idx:
                word_ids.append(self.word2idx['<UNK>'])
            else:
                word_ids.append(self.word2idx[w])

        if len(word_ids) < max_length:
            word_ids.extend([self.word2idx['<PAD>']] * (max_length - len(word_ids)))
        if max_length != -1:
            word_ids = torch.tensor(word_ids[:max_length]).long()
        return word_ids

    def get_word2char_span_map(self, text, max_length=-1):
        '''
        get a map from word position to character level span
        :param text: 
        :param max_length: 
        :return: 
        '''
        words = self.tokenize(text)
        word2char_span = []
        char_num = 0
        for wd in words:
            word2char_span.append([char_num, char_num + len(wd)])
            char_num += len(wd) + 1  # +1: whitespace
        if len(word2char_span) < max_length:
            word2char_span.extend([[0, 0]] * (max_length - len(word2char_span)))
        if max_length != -1:
            word2char_span = word2char_span[:max_length]
        return word2char_span

    def encode_plus(self, text, max_length=-1):
        return {
            "input_ids": self.text2word_indices(text, max_length),
            "offset_mapping": self.get_word2char_span_map(text, max_length)
        }


class Preprocessor:
    def __init__(self, tokenizer, use_bert):
        '''
        if token_type == "subword", tokenizer must be set to bert encoder
        "word", word tokenizer
        '''
        self.use_bert = use_bert
        # get_tok2char_span_map is a map from token index to character level span
        # e.g. "I am sing #ing" ("I am singing") -> [[0, 1], [2, 4], [5, 9], [9, 12]]
        if use_bert:
            self.tokenize = tokenizer.tokenize
            self.get_tok2char_span_map = lambda text: tokenizer.encode_plus(text,
                                                                            return_offsets_mapping=True,
                                                                            add_special_tokens=False)["offset_mapping"]
        else:
            self.tokenize = tokenizer.tokenize
            self.get_tok2char_span_map = lambda text: tokenizer.get_word2char_span_map(text)

    def _get_char2tok_span(self, text):
        '''
        map character level index to token level span
        e.g. "She is singing" -> [
                                 [0, 1], [0, 1], [0, 1], # She
                                 [-1, -1] # whitespace
                                 [1, 2], [1, 2], # is
                                 [-1, -1] # whitespace
                                 [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] # singing
                                 ]
        '''
        tok2char_span = self.get_tok2char_span_map(text)

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

    def _get_ent2char_spans(self, text, entities, ignore_subword=True):
        '''
        a dict mapping an entity to all possible character level spans
        it is used for adding character level spans for all entities
        e.g. {"entity1": [[0, 1], [18, 19]]}
        if ignore_subword, look for entities with whitespace around, e.g. "entity" -> " entity "
        '''
        entities = sorted(entities, key=lambda x: len(x), reverse=True)
        text_cp = " {} ".format(text) if ignore_subword else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword else ent
            for m in re.finditer(re.escape(target_ent), text_cp):
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword else m.span()
                spans.append(span)
            #             if len(spans) == 0:
            #                 set_trace()
            ent2char_spans[ent] = spans
        return ent2char_spans

    def add_char_span(self, dataset, ignore_subword=True):
        '''
        if the dataset has not annotated character level spans, add them automatically
        this function need to adapt according to the final format of the dataset
        :param dataset:
        :param ignore_subword:
        :return:
        '''
        samples_w_wrong_entity = []  # samples with entities that do not exist in the text, please check if any
        for sample in tqdm(dataset, desc="Adding char level spans"):
            entities = [ent["text"] for ent in sample["entity_list"]]
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities, ignore_subword=ignore_subword)

            # filter duplicates
            ent_memory_set = set()
            uni_entity_list = []
            for ent in sample["entity_list"]:
                ent_memory = "{}-{}".format(ent["text"], ent["type"])
                if ent_memory not in ent_memory_set:
                    uni_entity_list.append(ent)
                    ent_memory_set.add(ent_memory)

            new_ent_list = []
            for ent in uni_entity_list:
                ent_spans = ent2char_spans[ent["text"]]
                for sp in ent_spans:
                    new_ent_list.append({
                        "text": ent["text"],
                        "type": ent["type"],
                        "char_span": sp,
                    })

            if len(sample["entity_list"]) > len(new_ent_list):
                samples_w_wrong_entity.append(sample)
            sample["entity_list"] = new_ent_list
        return dataset, samples_w_wrong_entity

    def add_tok_span(self, data):
        '''
        add token level span according to the character spans, character level spans are required
        '''
        for sample in tqdm(data, desc="Adding token level span"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                tok_span_list = char2tok_span[char_span[0]:char_span[1]]
                tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
                ent["tok_span"] = tok_span
        return data

    def check_tok_span(self, data):
        '''
        check if text is equal to the one extracted by the annotated token level spans
        :param data: 
        :return: 
        '''
        entities_to_fix = []
        for sample in tqdm(data, desc="check tok spans"):
            text = sample["text"]
            tok2char_span = self.get_tok2char_span_map(text)
            for ent in sample["entity_list"]:
                tok_span = ent["tok_span"]
                char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
                char_span = [char_span_list[0][0], char_span_list[-1][1]]
                text_extr = text[char_span[0]:char_span[1]]
                gold_char_span = ent["char_span"]
                if not (char_span[0] == gold_char_span[0] and char_span[1] == gold_char_span[1] and text_extr == ent[
                    "text"]):
                    bad_ent = copy.deepcopy(ent)
                    bad_ent["extr_text"] = text_extr
                    entities_to_fix.append(bad_ent)

        span_error_memory = set()
        for ent in entities_to_fix:
            err_mem = "gold: {} --- extr: {}".format(ent["text"], ent["extr_text"])
            span_error_memory.add(err_mem)
        return span_error_memory

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