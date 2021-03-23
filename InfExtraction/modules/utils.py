import torch
import os
import json
from torch.utils.data import Dataset
import torch.nn.functional as F
from pprint import pprint
from tqdm import tqdm
import nltk
import nltk.data
import re


def unique_list(inp_list):
    out_list = []
    memory = set()
    for item in inp_list:
        mem = str(item)
        if type(item) is dict:
            mem = str(dict(sorted(item.items())))
        if mem not in memory:
            out_list.append(item)
            memory.add(mem)
    return out_list


def exist_nested_entities(sp_list):
    sp_list = unique_list(sp_list)
    sp_list = sorted(sp_list, key=lambda x: (x[0], x[1]))
    for idx, sp in enumerate(sp_list):
        if idx != 0 and sp[0] < sp_list[idx - 1][1]:
            return True
    return False


def strip_entity(entity):
    '''
    strip abundant white spaces around entities
    :param entity:
    :return:
    '''
    assert "text" in entity and "char_span" in entity
    ent_ori_txt = entity["text"]
    strip_left_len = len(ent_ori_txt) - len(ent_ori_txt.lstrip())
    strip_right_len = len(ent_ori_txt) - len(ent_ori_txt.rstrip())
    entity["char_span"][0] += strip_left_len
    entity["char_span"][-1] -= strip_right_len
    entity["text"] = ent_ori_txt.strip()
    return entity


def strip_entities(ent_list):
    for ent in ent_list:
        strip_entity(ent)


def split_para2sents_ch(para):
    '''
    split Chinese paragraph to sentences
    :param para:
    :return:
    '''
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def split_para2sents_en(paragraph):
    '''
    split English paragraphs to sentences
    :param paragraph:
    :return:
    '''
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def span_contains(span1, span2):
    return span1[0] <= span2[0] < span2[-1] <= span1[-1]


def ids2span(ids):
    '''
    parse ids to spans, e.g. [1, 2, 3, 4, 7, 8, 9] -> [1, 5, 7, 10]
    :param ids: id list
    :return:
    '''
    spans = []
    pre = -10
    for pos in ids:
        if pos - 1 != pre:
            spans.append(pre + 1)
            spans.append(pos)
        pre = pos
    spans.append(pre + 1)
    spans = spans[1:]
    return spans


def spans2ids(spans):
    '''
    parse spans to ids, e.g. [1, 5, 7, 10] -> [1, 2, 3, 4, 7, 8, 9]
    :param spans:
    :return:
    '''
    ids = []
    for i in range(0, len(spans), 2):
        ids.extend(list(range(spans[i], spans[i + 1])))
    return ids


def merge_spans(spans):
    '''
    merge continuous spans
    :param spans: [1, 2, 2, 3]
    :return: [1, 3]
    '''
    new_spans = []
    for pid, pos in enumerate(spans):
        p = pos
        if pid == 0 or pid % 2 != 0 or pid % 2 == 0 and p != new_spans[-1]:
            new_spans.append(pos)
        elif pid % 2 == 0 and p == new_spans[-1]:
            new_spans.pop()
    return new_spans


def load_data(path, total_lines=None):
    filename = path.split("/")[-1]
    try:
        print("loading data: {}".format(filename))
        data = json.load(open(path, "r", encoding="utf-8"))
        if total_lines is not None:
            print("total number is set: {}".format(total_lines))
            data = data[:total_lines]
        sample_num = len(data) if type(data) == list else 1
        print("done! {} samples are loaded!".format(sample_num))
    except json.decoder.JSONDecodeError:
        with open(path, "r", encoding="utf-8") as file_in:
            if total_lines is not None:
                print("total number is set: {}".format(total_lines))
            data = []
            for line in tqdm(file_in, desc="loading data {}".format(filename), total=total_lines):
                data.append(json.loads(line))
                if total_lines is not None and len(data) == total_lines:
                    break
    return data


def save_as_json_lines(data, path):
    with open(path, "w", encoding="utf-8") as out_file:
        filename = path.split("/")[-1]
        for sample in tqdm(data, desc="saving data {}".format(filename)):
            line = json.dumps(sample, ensure_ascii=False)
            out_file.write("{}\n".format(line))


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DefaultLogger:
    def __init__(self, log_path, project, run_name, run_id, config2log):
        self.log_path = log_path
        log_dir = "/".join(self.log_path.split("/")[:-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.run_id = run_id
        self.line = "============================================================================"
        self.log("project: {}, run_name: {}, run_id: {}\n".format(project, run_name, run_id))
        self.log({
            "config": config2log,
        })

    def log(self, content):
        log_dict = {
            "run_id": self.run_id,
            "log_text": content,
        }
        open(self.log_path, "a", encoding="utf-8").write("{}\n{}".format(self.line, json.dumps(log_dict, indent=4)))


class MyMaths:
    @staticmethod
    def handshaking_len2matrix_size(hsk_len):
        matrix_size = int((2 * hsk_len + 0.25) ** 0.5 - 0.5)
        return matrix_size


class MyMatrix:
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
        shaking_idx2matrix_idx = MyMatrix.get_shaking_idx2matrix_idx(matrix_size)
        for shaking_ind, matrix_ind in enumerate(shaking_idx2matrix_idx):
            matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind
        return matrix_idx2shaking_idx

    @staticmethod
    def mirror(shaking_seq):
        '''
        copy upper region to lower region
        :param shaking_seq:
        :return:
        '''
        batch_size, handshaking_seq_len, hidden_size = shaking_seq.size()

        # if self.cached_mirror_gather_tensor is None or \
        #         self.cached_mirror_gather_tensor.size()[0] != batch_size:
        #     self.cached_mirror_gather_tensor = self.mirror_gather_tensor[None, :, None].repeat(batch_size, 1, hidden_size)

        matrix_size = MyMaths.handshaking_len2matrix_size(handshaking_seq_len)
        map_ = MyMatrix.get_matrix_idx2shaking_idx(matrix_size)
        mirror_select_ids = [map_[i][j] if i <= j else map_[j][i] for i in range(matrix_size) for j in range(matrix_size)]
        mirror_select_vec = torch.tensor(mirror_select_ids).to(shaking_seq.device)

        # shaking_hiddens = torch.gather(shaking_seq, 1, self.cached_mirror_gather_tensor)

        # matrix = shaking_hiddens.view(batch_size, self.matrix_size, self.matrix_size, hidden_size)

        matrix = torch.index_select(shaking_seq, dim=1, index=mirror_select_vec)
        matrix = matrix.view(batch_size, matrix_size, matrix_size, hidden_size)
        return matrix

    @staticmethod
    def upper_reg2seq(ori_tensor):
        '''
        drop lower triangular part and flat upper triangular part to sequence
        :param ori_tensor: (batch_size, matrix_size, matrix_size, hidden_size)
        :return: (batch_size, matrix_size + ... + 1, hidden_size)
        '''
        tensor = ori_tensor.permute(0, 3, 1, 2).contiguous()
        uppder_ones = torch.ones([tensor.size()[-2], tensor.size()[-1]]).long().triu().to(ori_tensor.device)
        upper_diag_ids = torch.nonzero(uppder_ones.view(-1), as_tuple=False).view(-1)
        # flat_tensor: (batch_size, matrix_size * matrix_size, hidden_size)
        flat_tensor = tensor.view(tensor.size()[0], tensor.size()[1], -1).permute(0, 2, 1)
        tensor_upper = torch.index_select(flat_tensor, dim=1, index=upper_diag_ids)
        return tensor_upper

    @staticmethod
    def lower_reg2seq(ori_tensor):
        '''
        drop upper triangular part and flat lower triangular part to sequence
        :param ori_tensor: (batch_size, matrix_size, matrix_size, hidden_size)
        :return: (batch_size, matrix_size + ... + 1, hidden_size)
        '''
        tensor = ori_tensor.permute(0, 3, 1, 2).contiguous()
        lower_ones = torch.ones([tensor.size()[-2], tensor.size()[-1]]).long().tril().to(ori_tensor.device)
        lower_diag_ids = torch.nonzero(lower_ones.view(-1), as_tuple=False).view(-1)
        # flat_tensor: (batch_size, matrix_size * matrix_size, hidden_size)
        flat_tensor = tensor.view(tensor.size()[0], tensor.size()[1], -1).permute(0, 2, 1)
        tensor_lower = torch.index_select(flat_tensor, dim=1, index=lower_diag_ids)
        return tensor_lower

    @staticmethod
    def shaking_seq2matrix(sequence):
        '''
        map sequence tensor to matrix tensor; only upper region has values, pad 0 to the lower region
        :param sequence:
        :return:
        '''
        # sequence: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = sequence.size()
        matrix_size = MyMaths.handshaking_len2matrix_size(seq_len)
        map_ = MyMatrix.get_matrix_idx2shaking_idx(matrix_size)
        index_ids = [map_[i][j] if i <= j else seq_len for i in range(matrix_size) for j in range(matrix_size)]
        sequence_w_ze = F.pad(sequence, (0, 0, 0, 1), "constant", 0)
        index_tensor = torch.LongTensor(index_ids).to(sequence.device)
        long_seq = torch.index_select(sequence_w_ze, dim=1, index=index_tensor)
        return long_seq.view(batch_size, matrix_size, matrix_size, hidden_size)
