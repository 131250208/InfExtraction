import torch
import os
import json
from torch.utils.data import Dataset
import torch.nn.functional as F
from pprint import pprint
from tqdm import tqdm
# import nltk
# import nltk.data
import random
import re
import unicodedata
import functools
import logging

from functools import wraps
import time


def clean_text(text):
    text = re.sub("�", "", text)
    # text = re.sub("([,;.?!]+)", r" \1 ", text)
    text = re.sub("\ufeff", "", text)
    text = re.sub("\s+", " ", text).strip()
    return text


class CodePieceTimer(object):
    '''
    用上下文管理器计时
    '''
    def __init__(self, desc="default desc"):
        self.desc = desc

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        spent = time.time() - self.t0
        if spent > 1:
            print("debug")
        print('[{}, spent time: {time:.6f}s]'.format(self.desc, time =spent))


def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__, time = t1 - t0))
        return result
    return function_timer


def is_invalid_extr_ent(ent, char_span, text):
    def check_invalid(pat):
        return (char_span[0] - 1 >= 0 and re.match(pat, text[char_span[0] - 1]) is not None and re.match("^{}+".format(pat), ent) is not None) \
        or (char_span[1] < len(text) and re.match(pat, text[char_span[1]]) is not None and re.match("{}+$".format(pat), ent) is not None)
    return check_invalid("\d") or check_invalid("[A-Za-z]")


def get_tok2char_span_map4ch(word_list):
    text_fr_word_list = ""
    word2char_span = []
    for word in word_list:
        char_span = [len(text_fr_word_list), len(text_fr_word_list) + len(word)]
        text_fr_word_list += word
        word2char_span.append(char_span)
    return word2char_span


def get_char2tok_span(tok2char_span):
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
            if tok_sp[0] == -1:  # 第一次赋值以后不再修改
                tok_sp[0] = tok_ind
            tok_sp[1] = tok_ind + 1  # 每一次都更新
    return char2tok_span


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


def join_segs(segs, sep=None):
    if len(segs) == 0:
        return ""
    if sep is not None:
        return " ".join(segs)

    text = segs[0]
    for seg in segs[1:]:
        # if text == "" or seg == "" or \
        #         re.match(ch_jp_kr_pattern, text[-1]) is not None or \
        #         re.match(ch_jp_kr_pattern, seg[0]) is not None:
        #     pass
        # else:
        #     text += " "
        if text != "" and seg != "" and \
                re.match("^[a-zA-Z]$", text[-1]) is not None and \
                re.match("^[a-zA-Z]$", seg[0]) is not None:
            text += " "
        else:
            pass
        text += seg
    return text


def extract_ent_fr_txt_by_char_sp(char_span, text, language):
    segs = [text[char_span[idx]:char_span[idx + 1]] for idx in range(0, len(char_span), 2)]

    if language == "en":
        return join_segs(segs, " ")
    else:
        return join_segs(segs)


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


def search_segs(search_str, text, mask_toks=None):
    '''
    "split" search_str into segments according to the text,
    e.g.
    :param mask_toks: tokens to skip, {"[", "]", "|"} for saoke
    :param search_str: '培养一个县委书记地委书记'
    :param text: '徐特立曾说：“培养一个县委书记、地委书记容易，培养一个速记员难”。'
    :return: ['培养一个县委书记', '地委书记']
    '''
    if mask_toks is None:
        mask_toks = {}
    s_idx = 0
    seg_list = []

    word_pattern = "[0-9\.]+|[a-zA-Z]+|[^0-9\.a-zA-Z]"
    txt_tokens = re.findall(word_pattern, text)
    se_tokens = re.findall(word_pattern, search_str)

    while s_idx != len(se_tokens):
        start_tok = se_tokens[s_idx]

        if start_tok in mask_toks:  # skip masked chars
            s_idx += 1
            continue

        start_ids = [tok_idx for tok_idx, tok in enumerate(txt_tokens) if tok == start_tok]
        if len(start_ids) == 0:  # if not in text, skip
            s_idx += 1
            continue

        e_idx = 0
        while e_idx != len(se_tokens):
            new_start_ids = []
            for idx in start_ids:
                if idx + e_idx == len(txt_tokens) or s_idx + e_idx == len(se_tokens):
                    continue
                search_char = se_tokens[s_idx + e_idx]
                if txt_tokens[idx + e_idx] == se_tokens[s_idx + e_idx] and search_char not in mask_toks:
                    new_start_ids.append(idx)
            if len(new_start_ids) == 0:
                break
            start_ids = new_start_ids
            e_idx += 1

        seg_list.append("".join(se_tokens[s_idx: s_idx + e_idx]))
        s_idx += e_idx

    return seg_list


# 》》》》》》》》》》》》》》》》》
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
    seg_list = search_segs(target_seg, text, {"[", "]", "|"})
    # if language == "ch":
    #     seg_list = ChineseWordTokenizer.tokenize(target_seg)
    # elif language == "en":
    #     seg_list = target_seg.split(" ")

    word2spans = {}
    m_list = []
    text_cp = text[:]
    for sbwd_idx, sbwd in enumerate(sorted(seg_list, key=lambda s: len(s), reverse=True)):
        finditer = re.finditer(re.escape(" {} ".format(sbwd)), " {} ".format(text_cp)) \
            if language == "en" else re.finditer(re.escape(sbwd),
                                                 text_cp)  # a bug to fix: if language == "en", span should be [m[0], m[1] - 2]
        for m in finditer:
            m_list.append(m)

            # word_idx2spans
            if m.group() not in word2spans:
                word2spans[m.group()] = []
            word2spans[m.group()].append(m)

            # mask
            sp = m.span()
            text_ch_list = list(text_cp)
            text_ch_list[sp[0]:sp[1]] = ["_"] * (sp[1] - sp[0])
            text_cp = "".join(text_ch_list)

    word2surround_sps = {}
    for sbwd_idx, sbwd in enumerate(seg_list):
        pre_spans = word2spans[seg_list[sbwd_idx - 1]] if sbwd_idx != 0 else []
        # try:
        post_spans = word2spans[seg_list[sbwd_idx + 1]] if sbwd_idx != len(seg_list) - 1 else []
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
        pre_dists = [min(abs(mi.span()[0] - mj.span()[1]), abs(mi.span()[1] - mj.span()[0])) for mj in sur_sps["pre"]]
        dist_map[mid_i] += min(pre_dists) if len(pre_dists) > 0 else 0
        post_dists = [min(abs(mi.span()[0] - mj.span()[1]), abs(mi.span()[1] - mj.span()[0])) for mj in sur_sps["post"]]
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
        new_spans = merge_spans(spans) if merge_sps else spans
        new_candidate_spans.append(new_spans)

    return new_candidate_spans, add_text


def parse_spe_txt2list(spe_txt, jt=""):
    sep = "\u2E82"
    star = spe_txt.find("[")
    end = -1
    if star != -1:
        stack = []
        for idx in range(star, len(spe_txt)):
            c = spe_txt[idx]
            if c == "[":
                stack.append(c)
            elif c == "]":
                stack.pop()
                if len(stack) == 0:
                    end = idx
                    break

    res = []
    if star != -1 and end != -1:
        pre = spe_txt[:star]
        mid = spe_txt[star + 1:end]
        post = spe_txt[end + 1:]

        mid_sub = mid[:]
        stack = []
        for idx, c in enumerate(mid):
            if c == "[":
                stack.append(c)
            elif c == "]":
                stack.pop()
            elif c == "|" and len(stack) == 0:
                mid_sub = mid_sub[:idx] + sep + mid_sub[idx + 1:]

        mid_segs = mid_sub.split(sep)
        tmp = [jt.join([pre, seg, post]) for seg in mid_segs]
        for txt in tmp:
            res.extend(parse_spe_txt2list(txt))
    else:
        res.append(spe_txt)
    return res


def get_spe_txt_spans(spe_txt, text, is_pred=False):
    # target_str = re.sub("[\]\[\|]", "", spe_txt)
    # if is_pred:
    #     target_str = re.sub("([^a-zA-Z]|^)[XYZU]([^a-zA-Z]|$)", r"\1\2", target_str)

    # if is_pred:
    #     segs = re.split("[XYZU]", spe_txt)
    # else:
    #     segs = re.split("[\]\[\|]", spe_txt)
    #
    # segs = [s.strip() for s in segs if s.strip() != ""]
    search_str = spe_txt  # "".join(segs)

    candidate_spans, _ = search_char_spans_fr_txt(search_str, text, "ch")
    spans = candidate_spans[0]
    spans = [(spans[i], spans[i + 1]) for i in range(0, len(spans), 2)]

    preid2c = {}
    pat = "[\]\[\|XYZU]+" if is_pred else "[\]\[\|]+"
    for m in re.finditer(pat, spe_txt):
        if is_pred:
            if spe_txt[m.span()[0]] in set("XYZU") and m.span()[0] - 1 >= 0 and (
                    0 <= ord(spe_txt[m.span()[0] - 1]) - ord("A") <= 25 or 0 <= ord(spe_txt[m.span()[0] - 1]) - ord(
                "a") <= 25) or \
                    spe_txt[m.span()[1] - 1] in set("XYZU") and m.span()[1] < len(spe_txt) and (
                    0 <= ord(spe_txt[m.span()[1]]) - ord("A") <= 25 or 0 <= ord(spe_txt[m.span()[1]]) - ord(
                "a") <= 25):
                continue
        preid2c[m.span()[0] - 1] = m.group()

    start = re.match("[\]\[\|XYZU]+", spe_txt) if is_pred else re.match("[\]\[\|]+", spe_txt)
    spans_str = start.group() if start is not None else ""
    offset = len(spans_str)

    for sp in spans:
        for sp_idx in range(*sp):
            spans_str += "({}, {})".format(sp_idx, sp_idx + 1)
            offset += 1
            if offset - 1 in preid2c:
                spans_str += preid2c[offset - 1]
                offset += len(preid2c[offset - 1])

    spans_str_list = []
    for sps_str in parse_spe_txt2list(spans_str):
        sps = [int(s) for s in re.findall("\d+", sps_str)]
        sps = merge_spans(sps)
        spans_str_list.append(sps)
    return spans_str_list


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


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


# def load_data(path, lines=None):
#     filename = path.split("/")[-1]
#     try:
#         print("loading data: {}".format(filename))
#         data = json.load(open(path, "r", encoding="utf-8"))
#         if lines is not None:
#             print("total number is set: {}".format(lines))
#             data = data[:lines]
#         sample_num = len(data) if type(data) == list else 1
#         print("done! {} samples are loaded!".format(sample_num))
#     except json.decoder.JSONDecodeError:
#         data = []
#         with open(path, "r", encoding="utf-8") as file_in:
#             if lines is not None:
#                 print("total number is set: {}".format(lines))
#             for line in tqdm(file_in, desc="loading data {}".format(filename), total=lines):
#                 data.append(json.loads(line))
#                 if lines is not None and len(data) == lines:
#                     break
#     return data


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_file_line_offsets(file_path, max_lines=None):
    line_offset = []
    tqdm_desc = "init line offsets: {}".format(file_path)

    with open(file_path, "r", encoding="utf-8") as file_in, tqdm(desc=tqdm_desc) as pbar:
        while True:
            offset = file_in.tell()
            line_offset.append(offset)  # returns the location of the next line
            line = file_in.readline()
            if line.strip() == "":
                break
            pbar.update()

            if max_lines is not None and len(line_offset) >= max_lines:
                break

    return line_offset[:-1]


class MyLargeFileReader:
    def __init__(self, file_path, shuffle=False, max_lines=None):
        self.line_offsets = get_file_line_offsets(file_path, max_lines)
        if shuffle:
            self.shuffle_line_offsets()
        self.path = file_path

    def __len__(self):
        return len(self.line_offsets)

    def shuffle_line_offsets(self):
        random.shuffle(self.line_offsets)

    def get_line(self, idx):
        with open(self.path, "r", encoding="utf-8") as file_in:
            file_in.seek(self.line_offsets[idx])
            line = file_in.readline()
        return line

    def get_lines_generator(self, star_idx=0, end_idx=None, shuffle=False):
        if end_idx is None:
            end_idx = len(self.line_offsets)

        selected_line_offsets = self.line_offsets[star_idx:end_idx]
        if shuffle:
            random.shuffle(selected_line_offsets)

        with open(self.path, "r", encoding="utf-8") as file_in:
            for offset in selected_line_offsets:
                file_in.seek(offset)
                yield file_in.readline()


class MyLargeJsonlinesFileReader:
    def __init__(self, filereader):
        self.filereader = filereader
        self.path = filereader.path

    def __len__(self):
        return len(self.filereader)

    def get_json(self, idx):
        job = {}
        try:
            job = json.load(self.filereader.get_line(idx))
        except Exception:
            # logging.warning("Not json line!")
            pass
        return job

    def get_jsonlines_generator(self, star_idx=0, end_idx=None, shuffle=False):
        # if end_idx is None:
        #     end_idx = len(self.filereader.line_offsets)
        # selected_line_offsets = self.filereader.line_offsets[star_idx:end_idx]
        # if shuffle:
        #     random.shuffle(selected_line_offsets)
        # for offset in selected_line_offsets:
        #     self.filereader.file.seek(offset)
        #     try:
        #         job = json.loads(self.filereader.file.readline())
        #         yield job
        #     except Exception:
        #         # logging.warning("Not json line!")
        #         continue
        # self.filereader.file.seek(0)

        freader_gen = self.filereader.get_lines_generator(star_idx=star_idx, end_idx=end_idx, shuffle=shuffle)
        for line in freader_gen:
            try:
                job = json.loads(line)
                yield job
            except Exception:
                # logging.warning("Not json line!")
                continue


def merge_gen(*gens):
    for gen in gens:
        for item in gen:
            yield item


def load_data(path, lines=None, mute=False):
    filename = path.split("/")[-1]
    try:
        data = []
        with open(path, "r", encoding="utf-8") as file_in, tqdm(desc="loading data {}".format(filename), total=lines) as bar:
            if lines is not None:
                print("max number is set: {}".format(lines))

            for line in file_in:
                data.append(json.loads(line))
                if not mute:
                    bar.update()
                if lines is not None and len(data) == lines:
                    break
        if len(data) == 1:
            data = data[0]

    except json.decoder.JSONDecodeError:
        print("loading data: {}".format(filename))
        data = json.load(open(path, "r", encoding="utf-8"))
        if lines is not None:
            print("total number is set: {}".format(lines))
            data = data[:lines]
        sample_num = len(data) if type(data) == list else 1
        print("done! {} samples are loaded!".format(sample_num))
    return data


def save_as_json_lines(data, path):
    count = 0
    filename = path.split("/")[-1]
    dir = "/".join(path.split("/")[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path, "w", encoding="utf-8") as out_file:
        for sample in tqdm(data, desc="saving data {}".format(filename)):
            line = json.dumps(sample, ensure_ascii=False)
            out_file.write("{}\n".format(line))
            count += 1
    return count


class MyMappingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, json_reader, shuffle=False):
        super(MyIterableDataset).__init__()
        self.json_reader = json_reader
        self.start = 0
        self.end = len(json_reader)
        self.shuffle = shuffle

    def __iter__(self):
        return iter(self.json_reader.get_jsonlines_generator(star_idx=self.start, end_idx=self.end, shuffle=self.shuffle))

    def __len__(self):
        return len(self.json_reader)


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


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

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


import ahocorasick
from tqdm import tqdm


# 定义
class AC_Unicode:
    """稍微封装一下，弄个支持unicode的AC自动机
    """

    def __init__(self):
        self.ac = ahocorasick.Automaton()

    def add_word(self, k, v):
        return self.ac.add_word(k, v)

    def make_automaton(self):
        return self.ac.make_automaton()

    def iter(self, s):
        return self.ac.iter(s)


class SpoSearcher(object):
    def __init__(self, spo_list, ent_list, ent_type_map=None, ent_type_mask=None, min_ent_len=1):
        if ent_type_map is None:
            ent_type_map = dict()
        self.ent_type_mask = ent_type_mask if ent_type_mask is not None else set()

        self.ent_ac = AC_Unicode()
        self.subj_obj2preds = {}

        for spo in tqdm(spo_list, desc="build so2pred"):
            subj, pred, obj = spo["subject"], spo["predicate"], spo["object"],
            if subj == '' or obj == '' or len(obj) < min_ent_len or len(subj) < min_ent_len:
                continue
            self.subj_obj2preds.setdefault((subj, obj), set()).add(pred)

        ent2types = {}
        for ent in tqdm(ent_list, desc="build ent2types"):
            if len(ent["text"]) < min_ent_len:
                continue
            ent_type = ent_type_map.get(ent["type"], ent["type"])
            ent2types.setdefault(ent["text"], set()).add(ent_type)

        for ent_text, ent_types in tqdm(ent2types.items(), desc="add word 2 ent AC"):
            self.ent_ac.add_word(ent_text, {"text": ent_text, "types": list(ent_types)})

        print("init entity AC automaton")
        self.ent_ac.make_automaton()
        print("entity AC automaton done!")
        # self.min_ent_len = min_ent_len

    def extract_items(self, text_in, add_entities=None):
        extracted_spos = []
        extracted_ents = [{"text": ent["text"],
                           "type": tp,
                           "char_span": [end_idx - len(ent["text"]) + 1, end_idx + 1]}
                          for end_idx, ent in self.ent_ac.iter(text_in) for tp in ent["types"]]
        if add_entities is not None:
            extracted_ents.extend(add_entities)
        # filter invalid ents
        extracted_ents = [ent for ent in extracted_ents if not is_invalid_extr_ent(ent["text"], ent["char_span"], text_in)]
        for ent in extracted_ents:
            assert text_in[ent["char_span"][0]:ent["char_span"][1]] == ent["text"]

        ents_in_rel = set()
        uniq_ent_list = []
        ent_mem = set()
        for ent in extracted_ents:
            new_ent = {"text": ent["text"], "char_span": ent["char_span"]}
            if str(new_ent) not in ent_mem:
                uniq_ent_list.append(new_ent)
                ent_mem.add(str(new_ent))

        for subj in uniq_ent_list:
            for obj in uniq_ent_list:
                so = (subj["text"], obj["text"])
                if so in self.subj_obj2preds:
                    for pred in self.subj_obj2preds[so]:
                        extracted_spos.append({
                            "subject": subj["text"],
                            "subj_char_span": subj["char_span"],
                            "object": obj["text"],
                            "obj_char_span": obj["char_span"],
                            "predicate": pred,
                        })
                    ents_in_rel.add(subj["text"])
                    ents_in_rel.add(obj["text"])

        extracted_ents = [ent for ent in extracted_ents
                          if ent["text"] in ents_in_rel or ent["type"] not in self.ent_type_mask
                          and re.match("^\d+$", ent["text"]) is None
                          ]
        return extracted_ents, list(extracted_spos)


# 处理后数据
spo_list = [
    {"subject": "李宽", "predicate": "父亲_@value", "object": '李世民'},
    {"subject": "唐人街探案", "predicate": "票房_@value", "object": '54亿'},
    {"subject": "唐人街探案", "predicate": "票房_inArea", "object": '国内'},
]
ent_list = [
    {"text": "李宽", "type": "历史人物"},
    {"text": "李世民", "type": "历史人物"},
    {"text": "李世民", "type": "人物"},
    {"text": "唐人街探案", "type": "影视作品"},
    {"text": "54亿", "type": "Number"},
    {"text": "54", "type": "Number"},
    {"text": "54", "type": "影视作品"},
    {"text": "154亿", "type": "Number"},
    {"text": "国内", "type": "地区"},
]

# # 调用
# spoer = SpoSearcher(spo_list, ent_list, ent_type_map={"历史人物": "人物"}, ent_type_mask={"Number", })
# text = "李宽是唐太宗李世民的第二子，生母不详，史书记载为后宫生宽。 唐人街探案票房当日破54亿人民币, 黑客帝国突破154亿， sdf54亿"
# ent_list, spo_list = spoer.extract_items(text)
# print(spo_list)
# print(ent_list)
