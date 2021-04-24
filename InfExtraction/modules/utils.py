import torch
import os
import json
from torch.utils.data import Dataset
import torch.nn.functional as F
from pprint import pprint
from tqdm import tqdm
# import nltk
# import nltk.data
import re
import unicodedata
import functools


def patch_pattern():
    from pattern import text
    original_read = text._read

    @functools.wraps(original_read)
    def patched_read(*args, **kwargs):
        try:
            for r in original_read(*args, **kwargs):
                yield r
        except RuntimeError:
            pass
    text._read = patched_read


ch_jp_kr_pattern = "^[\u4e00-\u9faf|\uff00-\uffef|\u30a0-\u30ff|\u3040-\u309f|\u3000-\u303f]$"


def joint_segs(segs):
    text = segs[0]
    for seg in segs[1:]:
        if text == "" or seg == "" or \
                re.match(ch_jp_kr_pattern, text[-1]) is not None or \
                re.match(ch_jp_kr_pattern, seg[0]) is not None:
            pass
        else:
            text += " "
        text += seg
    return text


def extract_ent_fr_txt_by_char_sp(char_span, text):
    segs = [text[char_span[idx]:char_span[idx + 1]] for idx in range(0, len(char_span), 2)]
    return joint_segs(segs)


def search_best_span4ents(entities, text):
    ent2spans = {}
    ent2best_sp = {}
    for ent in entities:
        for m in re.finditer(re.escape(ent), text):
            ent2spans.setdefault(ent, []).append([*m.span()])

    for ent_i, sps_i in ent2spans.items():
        assert len(sps_i) > 0
        if len(sps_i) > 1:
            fin_ch_sp = None
            fin_dis_score = 9999
            for ch_sp_i in sps_i:
                dis_score = 0
                for ent_j, sps_j in ent2spans.items():
                    if ent_i == ent_j:
                        continue
                    dis_score += min(min(abs(ch_sp_i[0] - ch_sp_j[1]), abs(ch_sp_j[0] - ch_sp_i[1]))
                                     for ch_sp_j in sps_j if len(ch_sp_j) != 0)
                if dis_score < fin_dis_score:
                    fin_dis_score = dis_score
                    fin_ch_sp = ch_sp_i
            ent2best_sp[ent_i] = fin_ch_sp
        else:
            ent2best_sp[ent_i] = sps_i[0]
    return ent2best_sp


def rm_accents(str):
    return "".join(c for c in unicodedata.normalize('NFD', str) if unicodedata.category(c) != 'Mn')


def search_segs(search_str, text):
    '''
    "split" search_str into segments according to the text,
    e.g.
    :param search_str: '培养一个县委书记地委书记'
    :param text: '徐特立曾说：“培养一个县委书记、地委书记容易，培养一个速记员难”。'
    :return: ['培养一个县委书记', '地委书记']
    '''
    s_idx = 0
    seg_list = []
    mask_chars = {"[", "]", "|"}
    while s_idx != len(search_str):
        start_char = search_str[s_idx]
        if start_char in mask_chars:
            s_idx += 1
            continue
        start_ids = [m.span()[0] for m in re.finditer(re.escape(start_char), text)]
        if len(start_ids) == 0:
            s_idx += 1
            continue

        e_idx = 0
        while e_idx != len(search_str):
            new_start_ids = []
            for idx in start_ids:
                if idx + e_idx == len(text) or s_idx + e_idx == len(search_str):
                    continue
                search_char = search_str[s_idx + e_idx]
                if text[idx + e_idx] == search_str[s_idx + e_idx] and search_char not in mask_chars:
                    new_start_ids.append(idx)
            if len(new_start_ids) == 0:
                break
            start_ids = new_start_ids
            e_idx += 1

        seg_list.append(search_str[s_idx: s_idx + e_idx])
        s_idx += e_idx

    return seg_list


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


# def split_para2sents_en(paragraph):
#     '''
#     split English paragraphs to sentences
#     :param paragraph:
#     :return:
#     '''
#     tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#     sentences = tokenizer.tokenize(paragraph)
#     return sentences


# def span_contains(span1, span2):
#     if len(span2) == 0:
#         return True
#     return span1[0] <= span2[0] < span2[-1] <= span1[-1]

def span_contains(sp1, sp2):
    if len(sp2) == 0:
        return True
    span1 = sorted(sp1) if len(sp1) > 2 else sp1
    span2 = sorted(sp2) if len(sp2) > 2 else sp2
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


def merge_spans(spans, text=None):
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

    new_spans_ = []
    if text is not None:  # merge spans if only blanks between them
        for pid, pos in enumerate(new_spans):
            if pid != 0 and pid % 2 == 0 and re.match("^\s+$", text[new_spans[pid - 1]:pos]) is not None:
                new_spans_.pop()
            else:
                new_spans_.append(pos)
        new_spans = new_spans_

    return new_spans


def load_data(path, lines=None):
    filename = path.split("/")[-1]
    try:
        print("loading data: {}".format(filename))
        data = json.load(open(path, "r", encoding="utf-8"))
        if lines is not None:
            print("total number is set: {}".format(lines))
            data = data[:lines]
        sample_num = len(data) if type(data) == list else 1
        print("done! {} samples are loaded!".format(sample_num))
    except json.decoder.JSONDecodeError:
        data = []
        with open(path, "r", encoding="utf-8") as file_in:
            if lines is not None:
                print("total number is set: {}".format(lines))
            for line in tqdm(file_in, desc="loading data {}".format(filename), total=lines):
                data.append(json.loads(line))
                if lines is not None and len(data) == lines:
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
        mirror_select_ids = [map_[i][j] if i <= j else map_[j][i] for i in range(matrix_size) for j in
                             range(matrix_size)]
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
