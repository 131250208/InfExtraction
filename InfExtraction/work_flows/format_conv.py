import json
import os
from InfExtraction.modules.preprocess import Preprocessor, WhiteWordTokenizer, ChineseWordTokenizer
from InfExtraction.modules.utils import load_data, save_as_json_lines, merge_spans
from InfExtraction.modules import utils
from tqdm import tqdm
import random
from tqdm import tqdm
from pprint import pprint
import copy
import re
import jieba
import string
from pattern.en import lexeme, lemma
import itertools
import matplotlib.pyplot as plt
import time


def trans_genia():
    data_in_dir = "../../data/ori_data/genia_bk"
    data_out_dir = "../../data/ori_data/genia"
    in_file2out_file = {
        "train_dev.genia.jsonlines": "train_data.json",
        "test.genia.jsonlines": "test_data.json",
    }

    # load data
    filename2data = {}
    for in_filename, out_filename in in_file2out_file.items():
        int_path = os.path.join(data_in_dir, in_filename)
        out_path = os.path.join(data_out_dir, out_filename)
        with open(int_path, "r", encoding="utf-8") as file_in:
            data = [json.loads(line) for line in file_in]
            out_data = []
            for batch in tqdm(data, desc="transforming"):
                ners = batch["ners"]
                sentences = batch["sentences"]
                for idx, words in enumerate(sentences):
                    text = " ".join(words)
                    tok2char_span = WhiteWordTokenizer.get_tok2char_span_map(words)
                    ent_list = []
                    for ent in ners[idx]:
                        ent_text = " ".join(words[ent[0]:ent[1] + 1])
                        char_span_list = tok2char_span[ent[0]:ent[1] + 1]
                        char_span = [char_span_list[0][0], char_span_list[-1][1]]
                        norm_ent = {"text": ent_text,
                                    "type": ent[2],
                                    "char_span": char_span}
                        assert ent_text == text[char_span[0]:char_span[1]]
                        ent_list.append(norm_ent)
                    sample = {
                        "text": text,
                        "word_list": words,
                        "entity_list": ent_list,
                    }
                    out_data.append(sample)

        json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


def clean_entity(ent):
    ent = re.sub("�", "", ent)
    ent = ent.strip("\xad")
    return ent.strip()


def trans_daixiang_data(path, data_type=None):
    with open(path, "r", encoding="utf-8") as file_in:
        lines = [line.strip("\n") for line in file_in]
        data = []
        for i in range(0, len(lines), 3):
            sample = lines[i: i + 3]
            text = sample[0]
            word_list = text.split(" ")
            annstr = sample[1]
            ent_list = []
            word2char_span = WhiteWordTokenizer.get_tok2char_span_map(word_list)

            # entities
            for ann in annstr.split("|"):
                if ann == "":
                    continue
                offsets, ent_type = ann.split(" ")
                offsets = [int(idx) for idx in offsets.split(",")]
                assert len(offsets) % 2 == 0
                for idx, pos in enumerate(offsets):
                    if idx % 2 != 0:
                        offsets[idx] += 1

                extr_segs = []
                char_span = []
                tok_span = []
                for idx in range(0, len(offsets), 2):
                    wd_sp = [offsets[idx], offsets[idx + 1]]
                    ch_sp_list = word2char_span[wd_sp[0]:wd_sp[1]]
                    ch_sp = [ch_sp_list[0][0], ch_sp_list[-1][1]]

                    seg_wd = " ".join(word_list[wd_sp[0]: wd_sp[1]])
                    seg_ch = text[ch_sp[0]:ch_sp[1]]
                    assert seg_ch == seg_wd

                    char_span.extend(ch_sp)
                    tok_span.extend(wd_sp)
                    extr_segs.append(seg_ch)
                ent_txt_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text, "en")
                ent_txt = " ".join(extr_segs)

                assert ent_txt == ent_txt_extr
                ent = {
                    "text": ent_txt,
                    "type": ent_type,
                    "char_span": char_span,
                    "tok_span": tok_span,
                }
                ent_list.append(ent)

            # merge continuous spans
            for ent in ent_list:
                ori_char_span = ent["char_span"]
                merged_span = merge_spans(ori_char_span)
                ent_ori_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(ori_char_span, text, "en")
                ent_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(merged_span, text, "en")
                ent["char_span"] = merged_span
                assert ent_ori_extr == ent_extr == ent["text"]

            new_sample = {
                "text": sample[0],
                "word_list": word_list,
                "word2char_span": word2char_span,
                "entity_list": ent_list,
            }
            if data_type is not None:
                new_sample["id"] = "{}_{}".format(data_type, len(data))

            data.append(new_sample)

    return data


def postprocess_duee():
    res_data_path = "../../data/res_data/test_data.json"
    out_path = "../../data/res_data/duee.json"
    test_data = load_data(res_data_path)
    test_data2submit = []
    for sample in test_data:
        sample2submit = {
            "id": sample["id"],
            "text": sample["text"],
            "event_list": []
        }
        for event in sample["event_list"]:
            event2submit = {
                "event_type": event["trigger_type"],
                "trigger": event["trigger"],
                "trigger_start_index": event["trigger_char_span"][0],
                "arguments": [],
            }
            for arg in event["argument_list"]:
                event2submit["arguments"].append({
                    "argument": arg["text"],
                    "role": arg["type"],
                    "argument_start_index": arg["char_span"][0],
                })
            sample2submit["event_list"].append(event2submit)
        test_data2submit.append(sample2submit)
    save_as_json_lines(test_data2submit, out_path)


def preprocess_oie4():
    data_in_dir = "../../data/ori_data/oie4_bk"
    data_out_dir = "../../data/ori_data/oie4"
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    train_filename = "openie4_labels"
    valid_filename = "dev.tsv"
    test_filename = "test.tsv"
    train_path = os.path.join(data_in_dir, train_filename)
    valid_path = os.path.join(data_in_dir, valid_filename)
    test_path = os.path.join(data_in_dir, test_filename)

    train_save_path = os.path.join(data_out_dir, "train_data.json")
    valid_save_path = os.path.join(data_out_dir, "valid_data.json")
    test_save_path = os.path.join(data_out_dir, "test_data.json")

    train_data = []
    with open(train_path, "r", encoding="utf-8") as file_in:
        text = None
        words = None
        tag_lines = []
        for line in tqdm(file_in, desc="loading data"):
            line = line.strip("\n")
            if re.search("ARG|REL|NONE", line) is not None:
                tag_lines.append(line.split(" "))
            else:
                if text is not None:
                    train_data.append({
                        "text": text,
                        "word_list": words,
                        "tag_lines": tag_lines
                    })
                    tag_lines = []
                text = line
                words = line.split(" ")
        if text is not None:
            train_data.append({
                "text": text,
                "word_list": words,
                "tag_lines": tag_lines
            })

    for sample in tqdm(train_data, desc="transforming data"):
        open_spo_list = []
        word_list = sample["word_list"]
        tok2char_span = WhiteWordTokenizer.get_tok2char_span_map(word_list)
        text = sample["text"]

        for tags in sample["tag_lines"]:
            for tag_id in range(-3, 0):
                if tags[tag_id] != "NONE":
                    assert tags[tag_id] == "REL"
                    tags[tag_id] = "ADD"
            type2indices = {}
            for idx, tag in enumerate(tags):
                if tag == "NONE":
                    continue
                if tag not in type2indices:
                    type2indices[tag] = []
                type2indices[tag].append(idx)

            spo = {"predicate": {"text": "",
                                 "complete": "",
                                 "predefined": False,
                                 "prefix": "",
                                 "suffix": "",
                                 "char_span": [0, 0],
                                 },
                   "subject": {"text": "", "char_span": [0, 0]},
                   "object": {"text": "", "char_span": [0, 0]},
                   "other_args": []}
            add_text = None
            other_args = []
            for type_, ids in type2indices.items():
                wd_spans = []
                pre = -10
                for pos in ids:
                    if pos - 1 != pre:
                        wd_spans.append(pre + 1)
                        wd_spans.append(pos)
                    pre = pos
                wd_spans.append(pre + 1)
                wd_spans = wd_spans[1:]

                ch_spans = Preprocessor.tok_span2char_span(wd_spans, tok2char_span)
                arg_text = Preprocessor.extract_ent_fr_toks(wd_spans, word_list, "en")
                arg_text_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(ch_spans, text, "en")
                assert arg_text_extr == arg_text

                type_map = {
                    "REL": "predicate",
                    "ARG1": "subject",
                    "ARG2": "object",
                    "ADD": "add",
                    "TIME": "time",
                    "LOC": "location",
                }
                if type_ in {"REL", "ARG1", "ARG2"}:
                    spo[type_map[type_]] = {
                        "text": arg_text,
                        "char_span": ch_spans,
                        "word_span": wd_spans,
                    }
                elif type_ in {"TIME", "LOC"}:
                    other_args.append({
                        "type": type_map[type_],
                        "text": arg_text,
                        "char_span": ch_spans,
                    })
                else:
                    add_info_map = {
                        "[unused1]": "be-none",
                        "[unused2]": "be-of",
                        "[unused3]": "be-from",
                    }
                    add_text = add_info_map[arg_text]

            if "predicate" not in spo:
                if add_text == "be-none":
                    spo["predicate"] = {
                        "predefined": True,
                        "text": "be",
                        "complete": "be",
                        "prefix": "",
                        "suffix": "",
                        "char_span": [0, 0]
                    }
                else:
                    spo["predicate"] = {
                        "predefined": True,
                        "text": "",
                        "complete": "DEFAULT",
                        "prefix": "",
                        "suffix": "",
                        "char_span": [0, 0]
                    }
                    raise Exception
            else:
                spo["predicate"]["prefix"] = ""
                spo["predicate"]["suffix"] = ""
                spo["predicate"]["predefined"] = False
                if add_text is not None:
                    spo["predicate"]["prefix"] = "be"
                    if add_text == "be-of":
                        spo["predicate"]["suffix"] = "of"
                    if add_text == "be-from":
                        spo["predicate"]["suffix"] = "from"
                spo["predicate"]["complete"] = " ".join([spo["predicate"]["prefix"],
                                                         spo["predicate"]["text"],
                                                         spo["predicate"]["suffix"]]).strip()
                spo["other_args"] = other_args
                open_spo_list.append(spo)

        word_list = word_list[:-3]
        text = " ".join(word_list)
        sample["word2char_span"] = tok2char_span[:-3]
        sample["text"] = text
        sample["word_list"] = word_list
        sample["open_spo_list"] = open_spo_list
        for spo in open_spo_list:
            for key, val in spo.items():
                if key == "other_args":
                    for arg in spo[key]:
                        arg_text_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(arg["char_span"], text, "en")
                        assert arg_text_extr == arg["text"]
                else:
                    arg_text_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(val["char_span"], text, "en")
                    assert arg_text_extr == val["text"]

    # valid and test
    def get_val_test_data(path):
        fix_map = {
            "the unique limestone limestone of the mountains": "the unique limestone of the mountains",
            "do n't": "don't",
            "did n't": "didn't",
        }

        with open(path, "r", encoding="utf-8") as file_in:
            lines = []
            for line in file_in:
                for key, val in fix_map.items():
                    line = re.sub(key, val, line)
                splits = line.strip("\n").split("\t")
                # puncs = re.escape(re.sub("\.", "", string.punctuation))
                new_splits = []
                # 符号与单词用空格隔开
                for sp in splits:
                    sp = re.sub("([A-Za-z]+|[0-9\.]+)", r" \1 ", sp)
                    sp = re.sub("\s+", " ", sp)
                    sp = re.sub("n ' t", "n 't", sp)
                    sp = re.sub("' s", "'s", sp)
                    sp = sp.strip()
                    new_splits.append(sp)
                lines.append(new_splits)

            text2anns = {}
            for line in lines:
                text = line[0]
                if text not in text2anns:
                    text2anns[text] = []
                spo = {"predicate": {"text": line[1],
                                     "complete": line[1],
                                     "predefined": False,
                                     "prefix": "",
                                     "suffix": "",
                                     "char_span": [0, 0],
                                     },
                       "subject": {"text": line[2], "char_span": [0, 0]},
                       "object": {"text": "", "char_span": [0, 0]},
                       "other_args": []}

                if len(line) >= 4:
                    if "C :" not in line[3]:
                        spo["object"] = {"text": line[3], }
                if len(line) >= 5:
                    if "C :" not in line[4]:
                        arg = re.sub("T : |L : ", "", line[4])
                        spo["other_args"].append({"text": arg, "type": "time/loc_1"})
                if len(line) == 6:
                    if "C :" not in line[5]:
                        arg = re.sub("T : |L : ", "", line[5])
                        spo["other_args"].append({"text": arg, "type": "time/loc_2"})
                text2anns[text].append(spo)

        data = []
        for text, anns in text2anns.items():
            data.append({
                "text": text,
                "open_spo_list": anns,
            })

        # spans
        predefined_p_set = {"belong to",
                            "come from",
                            "have a", "have",
                            "will be",
                            "exist",
                            "be", "be a", "be in", "be on", "be at", "be of", "be from", "be for", "be with"}
        prefix_set = {
            "be", "will", "will be", "have", "have no", "must", "do not", "that",
        }
        suffix_set = {"in", "by", "of", "to", "from", "at"}
        samples_w_tl = []

        def my_lexeme(ori_word):
            lexeme_ws = lexeme(ori_word)
            lexeme_ws += [ori_word[0].upper() + ori_word[1:]]
            lexeme_ws += [ori_word.lower()]
            lexeme_ws += [ori_word.upper()]

            if ori_word[-2:] == "ly":
                lexeme_ws += [ori_word[:-2]]
            if re.match("[A-Z]", ori_word[0]) is not None:
                lexeme_ws += [ori_word + "'s"]
            if ori_word == "pursued":
                lexeme_ws += ["Pursuit"]
            if ori_word == "approve":
                lexeme_ws += ["approval"]
            if ori_word == "goes":
                lexeme_ws += ["exit onto"]
            return lexeme_ws

        def try_best2get_spans(target_str, text):
            candidate_spans, add_text = Preprocessor.search_char_spans_fr_txt(target_str, text, "en")
            spans = candidate_spans[0]
            fin_spans = None
            if add_text.strip("_ ") == "" and len(spans) != 0:  # if exact match
                fin_spans = spans
            else:
                pre_add_text = add_text
                # find words need to alter
                words2lexeme = re.findall("[^_\s]+", add_text)
                # lexeme all words
                words_list = [my_lexeme(w) for w in words2lexeme]
                # enumerate all possible alternative words
                alt_words_list = [[w] for w in words_list[0]] if len(words_list) == 1 else itertools.product(
                    *words_list)

                match_num2spans = {}
                max_match_num = 0
                for alt_words in alt_words_list:
                    chs = list(target_str)
                    add_text_cp = pre_add_text[:]
                    for wid, alt_w in enumerate(alt_words):
                        # search the span of the word need to alter
                        m4alt = re.search("[^_\s]+", add_text_cp)
                        sp = m4alt.span()
                        if alt_w == m4alt.group():  # same word, skip
                            continue
                        # alter the word
                        chs[sp[0]:sp[1]] = list(alt_w)
                        # mask the positions, will be ignore when getting m4alt next time
                        add_text_cp_ch_list = list(add_text_cp)
                        add_text_cp_ch_list[sp[0]:sp[1]] = ["_"] * len(alt_w)
                        add_text_cp = "".join(add_text_cp_ch_list)
                    # alternative text
                    alt_txt = "".join(chs)

                    # try to get spans
                    candidate_spans, add_text = Preprocessor.search_char_spans_fr_txt(alt_txt, text, "en")
                    spans = candidate_spans[0]
                    # cal how many words are matched this time
                    match_num = len(re.findall("_+", add_text)) - len(re.findall("_+", pre_add_text))
                    if match_num > 0:  # some words matched
                        match_num2spans[match_num] = spans
                        max_match_num = max(max_match_num, match_num)
                if max_match_num > 0:  # if there are any successful cases
                    fin_spans = match_num2spans[max_match_num]  # use the longest match

            if fin_spans is None or len(fin_spans) == 0:  # if still can not match, take partial match instead
                candidate_spans, add_text = Preprocessor.search_char_spans_fr_txt(target_str, text, "en")
                fin_spans = candidate_spans[0]
            return fin_spans

        for sample in tqdm(data, "add char span to val/test"):
            text = sample["text"]
            for spo in sample["open_spo_list"]:
                if len(spo) >= 4:
                    samples_w_tl.append(spo)
                for key, val in spo.items():
                    if key == "predicate":
                        predicate = spo["predicate"]["text"]
                        p_words = predicate.split()
                        p_lemma_words = [lemma(w) for w in p_words]
                        p_lemma = " ".join(p_lemma_words)

                        if p_lemma in predefined_p_set:
                            spo["predicate"]["predefined"] = True
                            spo["predicate"]["text"] = ""
                            spo["predicate"]["complete"] = p_lemma
                            spo["predicate"]["char_span"] = [0, 0]
                            continue

                        candidate_spans, add_text = Preprocessor.search_char_spans_fr_txt(predicate, text, "en")
                        spans = candidate_spans[0]
                        if add_text.strip("_ ") == "" and len(spans) != 0:
                            spo["predicate"]["char_span"] = spans
                            continue

                        # take prefix and suffix out
                        if re.search("[A-Za-z0-9]$", add_text):
                            for suffix in sorted(suffix_set, key=lambda a: len(a), reverse=True):
                                if re.search(" {}$".format(suffix), p_lemma):
                                    spo["predicate"]["text"] = " ".join(
                                        spo["predicate"]["text"].split()[:len(p_words) - len(suffix.split())])
                                    spo["predicate"]["suffix"] = suffix
                                    break
                        if re.search("^[A-Za-z0-9]", add_text):
                            for prefix in sorted(prefix_set, key=lambda a: len(a), reverse=True):
                                if re.search("^{} ".format(prefix), p_lemma):
                                    spo["predicate"]["text"] = " ".join(
                                        spo["predicate"]["text"].split()[len(prefix.split()):])
                                    spo["predicate"]["prefix"] = prefix
                                    break

                    elif key != "other_args":
                        arg = spo[key]
                        if arg is not None:
                            arg["char_span"] = try_best2get_spans(arg["text"], text)
                            seg_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(arg["char_span"], text, "en")
                            # if seg_extr != arg["text"]:
                            #     print(sample["text"])
                            #     print("target_seg: {}".format(arg["text"]))
                            #     print("extr_seg: {}".format(seg_extr))
                            #     pprint(spo)
                            #     print("===============")
                    else:
                        for arg in spo[key]:
                            arg["char_span"] = try_best2get_spans(arg["text"], text)
                            seg_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(arg["char_span"], text, "en")
                            # if seg_extr != arg["text"]:
                            #     print(sample["text"])
                            #     print("target_seg: {}".format(arg["text"]))
                            #     print("extr_seg: {}".format(seg_extr))
                            #     pprint(spo)
                            #     print("===============")
        return data

    valid_data = get_val_test_data(valid_path)
    test_data = get_val_test_data(test_path)
    save_as_json_lines(train_data, train_save_path)
    save_as_json_lines(valid_data, valid_save_path)
    save_as_json_lines(test_data, test_save_path)
    return train_data, valid_data, test_data


def trans2dai_dataset():
    '''
    change our data format to daixiang data format
    :return:
    '''
    in_data_dir = "../../data/normal_data/share_14_uncbase"
    out_data_dir = "../../data/ori_data/share_14_uncbase"
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    test_data_path = os.path.join(in_data_dir, "test_data.json")
    train_data_path = os.path.join(in_data_dir, "train_data.json")
    valid_data_path = os.path.join(in_data_dir, "valid_data.json")
    test_out_path = os.path.join(out_data_dir, "test.txt")
    valid_out_path = os.path.join(out_data_dir, "dev.txt")
    train_out_path = os.path.join(out_data_dir, "train.txt")

    def trans2daixiang_subwd(in_path, out_path):
        data = load_data(in_path)
        with open(out_path, "w", encoding="utf-8") as out_file:
            for sample in data:
                ent_list = []
                for ent in sample["entity_list"]:
                    ent_subwd_sp = [str(pos) if idx % 2 == 0 else str(pos - 1) for idx, pos in
                                    enumerate(ent["wd_span"])]
                    ent_list.append(",".join(ent_subwd_sp) + " " + ent["type"])
                text = sample["text"]
                ann_line = "|".join(ent_list)
                out_file.write("{}\n".format(text))
                out_file.write("{}\n".format(ann_line))
                out_file.write("\n")

    trans2daixiang_subwd(test_data_path, test_out_path)
    trans2daixiang_subwd(train_data_path, train_out_path)
    trans2daixiang_subwd(valid_data_path, valid_out_path)


def preprocess_saoke(data_path="../../data/ori_data/saoke_bk/saoke.json"):
    data = load_data(data_path)
    # fix data
    pred_fix_map = {
        '将研究领域的的重点放在': '将研究领域的重点放在',
        "NOT把X用在如何踢球上": "非把X用在如何踢球上",
        "被X荣为“ 一生只减Y ”的方法": "被X荣为“一生只减Y”的方法",
        "即将在X举行举行": "即将在X举行",
        "如阳光般有着有着": "如阳光般有着",
        "不断不断领略": "不断领略",
        "授予X诺贝尔和平奖授": "授予X诺贝尔和平奖",
        "固定使用使用": "固定使用",
        "听见X听见": "听见",
        "渗透性性能": "渗透性能",
        "将A/V数据传输传输给": "将A/V数据传输给",
        '节奏节奏': '节奏',
        '占总总面积': '占总面积',
        "主要以X主要以": "主要以",
        '根状茎茎': '根状茎',
        "实为为": "实为",
        '第一笔风险投资资金': '第一笔风险投资金',
        '无需X照看': '无须X照看',
        "[蒸发和过热]受热面": "[蒸发|过热]受热面",
        "[高稳产农田地面积|人均高稳产农田地面积": "[高稳产农田地面积|人均高稳产农田地面积]",
        "[减弱OR消失]": "[减弱|消失]",
        "整体[规划|营建|": "整体[规划|营建]",
        "争先奔走|": "争先奔走",
        "拟订|组织实施]": "[拟订|组织实施]",
        "规定|": "规定",
        "以“董氏大宗祠图”|": "以“董氏大宗祠图”",
        "健全||": "健全",
        "转移|培训": "[转移|培训]",
        "被国务院确确定为": "被国务院确定为",
        "聚众|": "聚众",
        "由X由": "由X出版",
        "以X以": "以X为目标",
        "可以像X可以像": "可以像X",
        "可使X低下|": "可使X低下",
        "一直致力于为X提供服务]": "一直致力于为X提供[设施|服务]",
        "以达到的效果X的效果": "以达到X的效果",
        "为X[化妆、扮戏]": "为X[化妆|扮戏]",
        "有利X快速[构建调整]Y": "有利X快速[构建|调整]Y",
        "依法监督检查X贯彻执行安全生产法规情况|设备设施安全情况]": "依法监督检查X贯彻执行[安全生产[法律|法规]情况|安全生产条件|设备设施安全情况]",
        "致力于帮助X解决教育、职业等等工作方方面面所遇到的难题]": "致力于帮助X解决[衣|食|住|行|娱乐|情感|教育|职业]的难题",
        "为X提供一个理性选择的平台|": "为X提供一个[客观评估|理性选择]的平台",
        "会X带来[冲击和影响]": "会给X带来[冲击|影响]",
        "以X赢得了[Y]": "以X赢得了Y",
        "依法指定X负责Y的[一审|[基层法院|检察院二审案件]的审理及法律监督工作": "依法指定X负责Y的[一审|基层[法院|检察院]二审]案件的[审理|法律监督工作]",
        "喊 X sockrrlates": "喊Xsockrrlates",
        "用X向发行中心申报[中奖等级|持票人[姓名|住址|身份证号码]": "用X向发行中心申报[中奖等级|持票人[姓名|住址|身份证号码]]",
        "对X可实现全部[主大多数炒制的菜肴[主|副]料和佐料的[一次投放|一次完成Y]": "对X可实现全部[主|副]料和佐料的[一次投放|一次完成Y]",
        "为X奉献了了一个又一个角色": "为X奉献了一个又一个角色",
        "是保护X免遭[[地震|水灾|火灾]等环境事故]|人为操作[失误|错误]|各种计算机犯罪行为]导致的破坏过程": "是保护X免遭[[地震|水灾|火灾]等环境事故|人为操作[失误|错误]|各种计算机犯罪行为]导致的破坏过程",
        "从而减轻OR避免X的压迫": "从而[减轻|避免]X的[摩擦|压迫]",
        "让X也能接触[文雅词汇OR忠孝节义故事]": "让X也能接触[文雅词汇|忠孝节义故事]",
        "负责对X的[审批发放和管理]": "负责对X的[审批发放|管理]",
        "办理X的[调任|转任|审批|工作]": "办理X的[调任|转任|审批工作]",
        "赞X[声如奔雷，势如狂涛，韵如豪歌]": "赞X[声如奔雷|势如狂涛|韵如豪歌]",
        "[拥资为商贾|[马贩|屠宰]之类]": "拥资为[商贾|马贩|屠宰]",
        "办理X的[调任|转任|审批|工作": "办理X的[调任|转任|审批|工作]",
        "体现了[的[豪华|性感|舒适|创造性]演绎的极致": "体现了X的[豪华|性感|舒适|创造性]演绎的极致",
        "勾勒了X[的[山水风光|人文历史|自然地理|民俗风情]": "勾勒了X的[山水风光|人文历史|自然地理|民俗风情]",
        "[于[一禁|一倡]之间的[疏导|劝勉|说服]|[[夺其利|与其利|与其利]的因势利导的教育工作]": "[于[一禁|一倡]之间的[疏导|劝勉|说服]|[夺其利|与其利]的因势利导的教育工作]",
    }

    subj_fix_map = {
        '[社会总需求|社会总供给|全区社会资金]': '[社会[总需求|总供给]|全区社会资金]',
        "_东部沿海渔场": "东部沿海渔场",
        "_全省": "全省",
        "[浙报传媒集团股份有限公司和北京华奥星空科技发展有限公司]": "[浙报传媒集团股份有限公司|北京华奥星空科技发展有限公司]",
        '[udp和tcp]协议在实现数据传输时的可靠性': '[udp|tcp]协议在实现数据传输时的可靠性',
        "|新华|高峰|紫岫]": "[新华|高峰|紫岫]",
        '[饥饿疾病]': '[饥饿|疾病]',
        "ThinkPad T430s 2352A31笔记本ThinkPad T430s 2352A31笔记本": "ThinkPad T430s 2352A31笔记本",
        "字字": "字",
        "《死亡游行》《死亡游行》": "《死亡游行》",
        "[城乡普通[中小学和学前教育]师资配置": "城乡普通[中小学|学前教育]师资配置",
        "神农神农": "神农",
        "国内[汽油|柴油]原油价格": "国内[汽|柴油]原油价格",
        "[劳动行政主管部门|劳动行政主管部门委托的专门机构]": "劳动行政主管部门[|委托的专门机构]",
        "许多[问题|问题解决之道]": "许多问题[|解决之道]",
        "[国务院水行政主管部门|国务院水行政主管部门授权的流域管理机构]": "国务院水行政主管部门[|授权的流域管理机构]",
        "[阿斯拉|阿斯拉相关人员]": "阿斯拉[|相关人员]",
        "[天线下倾角|天线方位角|天线高度]": "[天线[下倾角|方位角]|天线高度]",
        "[各级水行政主管部门|各级水行政主管部门授权的有关部门|流域机构]": "[各级水行政主管部门[|授权的有关部门]|流域机构]",
        '二者二者': '二者',
        '[资产和负债]的[货币结构和利率结构]': '[资产|负债]的[货币结构|利率结构]',
        '“洁瑞“': "“洁瑞”",
        '[查波|查波]': '查波',
        "[吃住全包式的学生宿舍|伙食自理式的学生宿舍|传统式的学生宿舍|农庄式的独立小院]": "[[吃住全包式的|伙食自理式的]学生宿舍|传统式的学生宿舍|农庄式的独立小院]",
        "[卓越的质量|先进的理念|优质的服务|市场的需求|企业的品牌|企业的知名度]": "[卓越的质量|先进的理念|优质的服务|市场的需求|企业的[品牌|知名度]]",
        "[大厅迎送引导服务法|重要旅客登记服务法|特殊需要预约服务法|双语式服务法|网络式畅通服务法]": "[大厅迎送引导服务法|重点旅客登记服务法|特殊需要预约服务法|双语式服务法|网络式畅通服务法]",
        "[李泽民|李泽民]": "[李光达|李泽民]",
        "“非外汇形式资产——人民币”“非外汇形式资产——人民币”借方": "“非外汇形式资产——人民币”借方",
        "[监测数据|监测资料|污源资料]": "[监测[数据|资料]|污源资料]",
        "[试点县|试点市|地级市]": "[试点[县|市]|地级市]",
        "根状茎茎": "根状茎",
        "各区人民政府指定机构机构": "各区人民政府指定机构",
        "工员": "员工",
        '向华强夫妇|端木樱子|及其他向氏家族成员]三十余人': '[向华强夫妇|端木樱子|其他向氏家族成员]',
        "[主题曲《凋零》||插曲《沉醉》]": "[主题曲《凋零》|插曲《沉醉》]",
        "[BodoL innhoff|BodoL innhoff同事]": "BodoL innhoff[同事|]",
        "[张老|张老伙伴们]": "张老[伙伴们|]",
        "[沃金|沃金的暗矛巨魔]": "沃金[的暗矛巨魔|]",
        "[放下武器的[回民士兵|平民]": "放下武器的[回民士兵|平民]",
        "[花园景|电话|平面电视|保险箱|书桌|暖气|更衣室|沙发|木质/镶木地板|衣柜/衣橱|吹风机|免费洗浴用品|卫生间|浴室|浴缸OR淋浴|唤醒服务]": "[花园景|电话|平面电视|保险箱|书桌|暖气|更衣室|沙发|木质/镶木地板|衣柜/衣橱|吹风机|免费洗浴用品|卫生间|浴室|浴缸|淋浴|唤醒服务]",
        "[《男吊》《女吊》|《跳无常》]的[恐怖|刺激]]": "[《男吊》|《女吊》|《跳无常》]的[恐怖|刺激]",
        "[省|市|地]级水行政主管部门]": "[省|市|地]级水行政主管部门",
        "[沙丘魔堡二|终极动员令系列][沙丘魔堡二|终极动员令系列]": "[沙丘魔堡二|终极动员令系列]",
        "[容量|型式|空载电流|空载损耗|短路（负载）损耗|阻抗电压容量|型式|空载电流|空载损耗|短路（负载）损耗|阻抗电压]": "[容量|型式|空载电流|空载损耗|短路（负载）损耗|阻抗电压]",
        "[[《朱砂鱼谱》|《金鱼品》|《金鱼饲养法》]": "[《朱砂鱼谱》|《金鱼品》|《金鱼饲养法》]",
        "[Denise Gimpel|Lyce.jankowski|Lyce.jankowski]": "[Denise Gimpel|Dr.james S.Edgren|Lyce.jankowski]",
        "[张家同先生|万新宇先生|赵卓然先生|曹绪龙先生|客户代表]|": "[张家同先生|万新宇先生|赵卓然先生|曹绪龙先生|客户代表]",
        "1名[年龄较大的儿童|成人]|一张加床收费": "1名[年龄较大的儿童|成人]一张加床收费",
        "[[公开|真诚]谈论重要课题|不断挑战自己思考能力]]": "[[公开|真诚]谈论重要课题|不断挑战自己思考能力]",
        "[中原文化|江淮文化|金陵文化|吴文化]": "[中原|江淮|金陵|吴]文化",
        "[陈正宗|陈荣统|陈孔雅]等": "[陈正宗|陈孔雅|陈荣统]等",
        "[陈正宗|陈荣统|陈孔雅]": "[陈正宗|陈孔雅|陈荣统]",
        "[我|那几个士兵]": "[那几个士兵|我]",
        "[他|中牟仓之助|五代友厚]": "[他|五代友厚|中牟仓之助]",
        '[[庆阳|安化|合水|宁州|正宁]难民]|[董原难民]]': '[[庆阳|安化|合水|宁州|正宁]难民|董原难民]',
        "[村提留|乡统筹费|镇统筹费]": "[村提留|[乡|镇]统筹费]",
        "[茜茜公主|晨光国际|仁康科技]]": "[茜茜公主|晨光国际|仁康科技]",
        "其[灵活|性价比|服务经验|对本土消费者|对本土消费者洞察]": "其[灵活|性价比|服务经验|对本土消费者洞察]",
        "油画[《红衣少女》|《紫罗|兰》]|": "油画[《红衣少女》|《紫罗兰》]",
        "[碳纤维OR硼纤维增强的环氧树脂基复合材料|金属基复合材料]": "[[碳纤维|硼纤维]增强的[环氧树脂基复合材料|金属基复合材料]]",
        "[创意大厦|太古汇|歌剧院": "[创意大厦|太古汇|歌剧院]",
        "[投保人|被保险人|受益人": "[投保人|被保险人|受益人]",
        "[阴虚|阳虚体质之人": "[阴虚|阳虚]体质之人",
        "[共同企业文化|基本原则": "[共同企业文化|基本原则]",
        "臭氧射流式混合杀菌装置|美国陶氏公司反渗透装置|纤维活性炭高效吸附]": "[臭氧射流式混合杀菌装置|美国陶氏公司反渗透装置|纤维活性炭高效吸附]",
        "记|传阅|归档]工作": "[收发登记|传阅|归档]工作",
        "肩膀酸痛|腰痛|颈椎痛|关节痛|风湿|失眠|气喘]": "[肩膀酸痛|腰痛|颈椎痛|关节痛|风湿|失眠|气喘]",
        "保障|服务]": "[保障|服务]",
        "[卡迪拉克汽车分公司|克莱斯勒公司": "[卡迪拉克汽车分公司|克莱斯勒公司]",
        "[社团|大连高校天文协会联盟": "[社团|大连高校天文协会联盟]",
        "饮食文化烹调方法]": "[饮食文化|烹调方法]",
        "[本网站": "本网站",
        "《唐伯虎点秋香2》|《龙凤店》]等": "[《唐伯虎点秋香2》|《龙凤店》]等",
        "巫密河漂流|南哨观音阁|十里杜鹃|老山界|古秃杉|八万山|鹅掌楸自然保护区|例定千秋碑|福建柏生物群|南哨水上度假区|久仰|久吉|返招民族村|久脸苗族分迁遗址]": "[巫密河漂流|南哨观音阁|十里杜鹃|老山界|古秃杉|八万山|鹅掌楸自然保护区|例定千秋碑|福建柏生物群|南哨水上度假区|久仰|久吉|返招民族村|久脸苗族分迁遗址]",
        "《行政管理与政府行为研究》|《社会主义市场经济论》|《可持续城市化发展研究--中国四川的实证分析》]": "[《行政管理与政府行为研究》|《社会主义市场经济论》|《可持续城市化发展研究--中国四川的实证分析》]",
        "[黄鳝|螃蟹|水蛇|小龙虾|水老鼠|青蛙|蟾蜍": "[黄鳝|螃蟹|水蛇|小龙虾|水老鼠|青蛙|蟾蜍]",
        "《唐伯虎点秋香2》|《龙凤店》]": "[《唐伯虎点秋香2》|《龙凤店》]",
        "基本特质|经验教训": "[基本特质|经验教训]",
        "[发烟罐、发烟手榴弹]": "[发烟罐|发烟手榴弹]",
        "[发烟罐、发烟手榴弹]等": "[发烟罐|发烟手榴弹]",
        "众|": "众",
        "[统一部署大幅度裁员]": "统一部署大幅度裁员",
        "[深圳市及广东省教育督导室]": "深圳市及广东省教育督导室",
        "_境内": "境内",
        "[综合管理档案调阅]工作": "[综合管理|档案调阅]工作",
        "徒步登山|岩降瀑降|攀岩探洞[]": "[徒步登山|岩降瀑降|攀岩探洞]",
        "[讲解+画风+技巧解析]": "[讲解|画风|技巧解析]",
        "[公司和人]的价值": "[公司|人]的价值",
        "昏迷|失败": "昏迷失败",
        "整个欧洲人口老年化|整个福利社会[]": "[整个欧洲人口老年化|整个福利社会]",
        "[刺激性和腐蚀性]": "[刺激性|腐蚀性]",
        "[人事部全国人才流动中心与首都经济贸易大学首经技术培训中心]": "[人事部全国人才流动中心|首都经济贸易大学首经技术培训中心]",
        "《哲学研究》|《马克思主义研究》|《当代世界与社会主义》|《科学社会主义》|《光明日报》（理论版）|《中国教育报》（理论版）": "[《哲学研究》|《马克思主义研究》|《当代世界与社会主义》|《科学社会主义》|《光明日报》（理论版）|《中国教育报》（理论版）]",
        "[知识型技能型]航空服务专门人才": "[知识型|技能型]航空服务专门人才",
        "[唯象热力学和统计力学]的理论": "[唯象热力学|统计力学]的理论",
        "GERMART”|“吉玛特”": "[“GERMART”|“吉玛特”]",
        "时尚但不张扬|机敏但不锋芒": "[时尚但不张扬|机敏但不锋芒]",
        "_专题片里出现的案例": "专题片里出现的案例",
        "社团|与大连其他高校": "[社团|大连其他高校]",
        "_母亲": "母亲",
        "_父亲": "父亲",
        "[国安局 ”与警政署]": "[“国安局 ”|警政署]",
        "[卫青霍去病]": "[卫青|霍去病]",
        "_沿海": "沿海",
        "迪达X": "迪达",
        "[大中小城市和小城镇]": "[大中小城市|小城镇]",
        "公司|": "[公司|产品]",
    }

    obj_fix_map = {
        "[]日本学术会议|加拿大皇家学会等": "[日本学术会议|加拿大皇家学会]等",
        "[汪彩霞与冯三元冯莲芳与顾慥]": "[汪彩霞|冯三元|冯莲芳|顾慥]",
        '那种奇特的发麻的感觉|': '那种奇特的发麻的感觉',
        '[药物和膳食结合]': '[药物|膳食]',
        '石头|老树梗': '[石头|老树梗]',
        'X回民全部区域': '回民全部区域',
        '[开业申请书合法的验资证明]': '开业申请书',
        '国际减灾十年||国际减灾日': '[国际减灾十年|国际减灾日]',
        "[唯象热力学和统计力学]的理论": "[唯象热力学|统计力学]的理论",
        "[节约资源和保护环境]的发展模式": "[节约资源|保护环境]的发展模式",
        "[南非和刚果]": "[南非|刚果]",
        "没有[亲缘|血缘|没地缘]": "没有[亲缘|血缘|地缘]",
        '[阳台|DVD播放机|卫星频道|平面电视|空调|书桌|客厅角|洗衣机|沙发|衣柜|衣橱|淋浴|浴缸|免费洗浴用品|卫生间|浴室|冰箱|微波炉|厨房|用餐区|电烧水壶|厨房用具|餐桌]等设施': '[阳台|DVD播放机|卫星频道|平面电视|空调|书桌|客厅角|洗衣机|沙发|衣柜|衣橱|淋浴|浴缸|免费洗浴用品|卫生间|浴室|冰箱|微波炉|用餐区|厨房|电烧水壶|厨房用具|餐桌]等设施',
        '[对应|同一种颜色|同一种形状]的': '[对应|同一种[颜色|形状]]的',
        '完成化学反应所消耗的试剂量试剂量': '完成化学反应所消耗的试剂量',
        '[红水泡|白水泡|蓝水泡|红白水泡|黑白水铁包金水泡|红顶水泡|墨水泡|五花水泡|紫蓝水泡]': '[红水泡|白水泡|蓝水泡|红白水泡|黑白水泡|铁包金水泡|红顶水泡|墨水泡|五花水泡|紫蓝水泡]',
        "印投投注说明": "印投注说明",
        "免税的的[计划内项目|登记项目]": "免税的[计划内项目|登记项目]",
        "AutoCAD2006中文版的[文件操作|绘图设置|绘制二维图形|编辑图形对象的基本命令|图案的填充方法及其设置|图层、块以及属性的定义|三维图形创建|渲染与着色|图形打印]": "AutoCAD 2006中文版的[文件操作|绘图设置|绘制二维图形|编辑图形对象的基本命令|图案的填充方法及其设置|图层、块以及属性的定义|三维图形创建|渲染与着色|图形打印]",
        "完整客户生命周期的的[发生|发展]": "完整客户生命周期的[发生|发展]",
        "攻击舰艇|主力舰": "[攻击舰艇|主力舰]",
        "[科学家及联合国粮农组织和世界卫生组织]": "[科学家及联合国粮农组织|世界卫生组织]",
        "[三桧甚至五桧]的大海船": "[三桧|五桧]的大海船",
        "[水电农村电气化县和小水电代燃料工程建设]": "[水电农村电气化县|小水电代燃料工程建设]",
        "[“和谐、个性、求真、创新”]的办学理念": "[“和谐|个性|求真|创新”]的办学理念",
        "[必要性和现实差距]": "[必要性|现实差距]",
        "[饮食文化烹调方法]": "[饮食文化|烹调方法]",
        "[撒播10公斤条播4公斤]": "[撒播10公斤|条播4公斤]",
        "[海南省体育总会和海南省民政厅]": "[海南省体育总会|海南省民政厅]",
        "教授带头作用|团队合作精神[]": "[教授带头作用|团队合作精神]",
        "[小可爱和百褶裙]": "[小可爱|百褶裙]",
        "约占全_省面积的1/2": "约占全省面积的1/2",
        "[刺激性和腐蚀性]": "[刺激性|腐蚀性]",
        "极高的[知名度和影响力]": "极高的[知名度|影响力]",
        "国企热|央企热|国进民退": "[国企热|央企热|国进民退]",
        "[运动和变化]之中": "[运动|变化]之中",
        "[卫青霍去病]": "[卫青|霍去病]",
        "国家[产业政策及改革措施]": "国家[产业政策|改革措施]",
        "[棉花企业发展规划及区域性棉花交易市场发展规划]": "[棉花企业发展规划|区域性棉花交易市场发展规划]",
        "[迷惑孤独]": "[迷惑|孤独]",
        "英国|美国": "[英国|美国]",
        "[硫化矿和氧化矿]": "[硫化矿|氧化矿]",
        "[纯化和灭活]的甲肝病毒和乙肝表面抗原的混合液": "[纯化|灭活]的甲肝病毒和乙肝表面抗原的混合液",
        "[基层党组织建设和党员教育管理]工作": "[基层党组织建设|党员教育管理]工作",
        "[候鸟来的季节]": "候鸟来的季节",
        "[黄鱼|带鱼|鲳鱼|虾类|蟹类及贝藻类]": "[黄鱼|带鱼|鲳鱼|虾类|蟹类|贝藻类]",
        "[]剑桥大学教授|英圈皇家学会会员|英闭自然历史博物馆资深占生物学专家": "[剑桥大学教授|英圈皇家学会会员|英闭自然历史博物馆资深占生物学专家]",
        "学校招生|学生就业]等种种困难": "[学校招生|学生就业]等种种困难",
        "洗面奶|保湿水|眼部精油|颈部精油|眼贴膜|颈贴膜]": "[洗面奶|保湿水|眼部精油|颈部精油|眼贴膜|颈贴膜]",
        "[党性|党风|党纪|的": "[党性|党风|党纪]的",
        "[各类军工用品做工方面的经验|": "[各类军工用品做工方面的经验|精湛的金属淬火技术]",
        "巫密河漂流|南哨观音阁|十里杜鹃|老山界|古秃杉|八万山|鹅掌楸自然保护区|例定千秋碑|福建柏生物群|南哨水上度假区|久仰|久吉|返招民族村|久脸苗族分迁遗址]": "[巫密河漂流|南哨观音阁|十里杜鹃|老山界|古秃杉|八万山|鹅掌楸自然保护区|例定千秋碑|福建柏生物群|南哨水上度假区|久仰|久吉|返招民族村|久脸苗族分迁遗址]",
        "《唐伯虎点秋香2》|《龙凤店》]等": "[《唐伯虎点秋香2》|《龙凤店》]等",
        "[黑眼圈|眼部皱纹": "[黑眼圈|眼部皱纹]",
        "[洗面奶|保湿水|眼部精油|颈部精油|眼贴膜|颈贴膜]": "洗面奶|保湿水|眼部精油|颈部精油|眼贴膜|颈贴膜]",
        "[基本特质|经验教训": "[基本特质|经验教训]",
        "[适量开水|少许白糖": "[适量开水|少许白糖]",
        "身体好|力气大]的": "[身体好|力气大]的青年人",
        "[管理学经济学|人力资源管理方面": "[管理学|经济学|人力资源管理]方面",
        "[菜地|操场|公社大会场地|": "[菜地|操场|公社大会场地]",
        "“心向中华爱国爱家笑看天下风云事”“永胜功成为侠为义书友常伴国士名”]": "[“心向中华爱国爱家笑看天下风云事”|“永胜功成为侠为义书友常伴国士名”]",
        "枣萎蔫果病|枣雾蔫病]等": "[枣萎蔫果病|枣雾蔫病]等",
        "[其对应的明文块|下一个明文块中对应位": "[其对应的明文块|下一个明文块中对应位]",
        "[感情真挚深切|沉重而不悲哀|昂扬而不虚泛": "[感情真挚深切|沉重而不悲哀|昂扬而不虚泛]",
        "[卡迪拉克汽车分公司|克莱斯勒公司": "[卡迪拉克汽车分公司|克莱斯勒公司]",
        "待机|菜单|模式切换]按键": "[待机|菜单|模式切换]按键",
        "|市委|政府直属事业单位|各部门所属事业单位]的机构改革方案": "[市委|政府直属事业单位|各部门所属事业单位]的机构改革方案",
        "“空的”|“安静的”]意识状态": "[“空的”|“安静的”]意识状态",
        "阿依怒尔|艾山]": "[阿依怒尔|艾山]",
        "这些热心参与本项工作的中外人士]": "这些热心参与本项工作的中外人士",
        "[城乡水利设施建设能力水资源利用水平与保障能力|": "[城乡水利设施建设能力|水资源利用水平与保障能力]",
        "Java|iOS]开发多年": "[Java|iOS]开发多年",
        "[一定的摩擦力|不错的质感": "[一定的摩擦力|不错的质感]",
        "[全新12色颜料墨水系统|": "[全新12色颜料墨水系统|超高密度打印头技术“FINE”]",
        "[重大决策|重要部署|《政府工作报告》|重点工作": "[重大决策|重要部署|《政府工作报告》|重点工作]",
        "英语：Marc Edworthy]]]": "英语：Marc Edworthy",
        "[孩子们自": "孩子们自己",
        "定性资料的判别分析|定量资料的判别分析[": "[定性资料的判别分析|定量资料的判别分析]",
        "阳光采购|快速采购]协同": "[阳光|快速]采购协同",
        "|各类教学实习|实验室|多功能报告厅|计算机网络中心|校园网]": "[各类教学实习|实验室|多功能报告厅|计算机网络中心|校园网]",
        "滋肾润肺|补肝明目]": "[滋肾润肺|补肝明目]",
        "公安部|上影英皇]等单位": "[公安部|上影英皇]等单位",
        "玩忽职守|违法乱纪]的": "[玩忽职守|违法乱纪]的",
        "Internet网络|IVR语音系统|WAP手机智能以|自动售票机]等多种方式": "[Internet网络|IVR语音系统|WAP手机智能以|自动售票机]等多种方式",
        '《行政管理与政府行为研究》|《社会主义市场经济论》|《可持续城市化发展研究--中国四川的实证分析》]等': '[《行政管理与政府行为研究》|《社会主义市场经济论》|《可持续城市化发展研究--中国四川的实证分析》]等',
        '肩膀酸痛|腰痛|颈椎痛|关节痛|风湿|失眠|气喘]': '[肩膀酸痛|腰痛|颈椎痛|关节痛|风湿|失眠|气喘]',
        '“专转本”|“专接本”]方式': '[“专转本”|“专接本”]方式',
        '[蛇头草|水钟流头黑南瓜、野饭瓜、南瓜三七、野南瓜、野金瓜头': '[蛇头草|水钟流头|黑南瓜|野饭瓜|南瓜三七|野南瓜|野金瓜头]',
        '辖区的[四个镇和两个开发区]': '辖区的[四个镇|两个开发区]',
        '[雨量充沛|日照充足|气候温和': '[雨量充沛|日照充足|气候温和]',
        '[公司董事|副总裁|导航事业部总经理': '[公司董事|副总裁|导航事业部总经理]',
        '传感器信号调理|数据转换和处理]解决方案方面': '[传感器信号调理|数据转换和处理]解决方案方面',
        '立案监督|侦查监督|刑罚执行监督]事项': '[立案监督|侦查监督|刑罚执行监督]事项',
        '武器装备科研生产许可证|军品科研生产承担单位保密资格]审查': '[武器装备科研生产许可证|军品科研生产承担单位保密资格]审查',
        '尼泊尔|锡金|不丹|孟加拉|印度|中国大陆的西藏自治区]等地': '[尼泊尔|锡金|不丹|孟加拉|印度|中国大陆的西藏自治区]等地',
        "美国罗德岛大学的": '美国罗德岛大学',
        '臭氧射流式混合杀菌装置|美国陶氏公司反渗透装置|纤维活性炭高效吸附]等': '[臭氧射流式混合杀菌装置|美国陶氏公司反渗透装置|纤维活性炭高效吸附]等',
        '[可爱而时尚]的': '[可爱|时尚]的',
        "何克抗主持的“跨越式发展的”实验校": '何克抗主持的“跨越式发展”的实验校',
        "中高端软体家具中高端软体家具": "中高端软体家具",
        "民民族主义": "民族主义",
        "不懈的不懈的": "不懈的",
        "许多[问题|问题解决之道]": "许多问题[|解决之道]",
        "[台湾大学哲学系主任|比利时鲁汶大学客座教授|荷兰莱顿大学客座教授]": "[台湾大学[哲学系主任|哲学研究所所长]|[比利时鲁汶大学|荷兰莱顿大学]客座教授]",
        "[宣传干事副科长|县文联副主席|县文联主席|黄冈市作家协会副主席|中国通俗文艺研究会理事|湖北省民间文艺家协会副主席]": "[宣传干事副科长|县文联[副主席|主席]|黄冈市作家协会副主席|中国通俗文艺研究会理事|湖北省民间文艺家协会副主席]",
        "[网络应用性能加速|安全内容管理|安全事件管理|用户管理|网络资源管理|网络资源优化|桌面系统管理]": "[网络应用性能加速|安全内容管理|安全事件管理|用户管理|网络资源[管理|优化]|桌面系统管理]",
        "[砂质岩砂质岩|碳酸盐岩|红岩|第四纪松散堆积物]": "[砂质岩|碳酸盐岩|红岩|第四纪松散堆积物]",
        "[苇塘内|苇塘附近地带]": "苇塘[内|附近地带]",
        "[英国浪漫主义风景画家|著名的水彩画家|著名的油画家]": "[英国浪漫主义风景画家|著名的[水彩画家|油画家]]",
        "如何才能使[组织的力量集中集中实现上升的杠杆效果|有效控制减小风险]": "如何才能使[组织的力量集中实现上升的杠杆效果|有效控制减小风险]",
        "[最新研究|最新设计|最新制造]的": "最新[研究|设计|制造]的",
        "[平素调养|亚健康状态|慢性疾病|虚弱性疾病|病后恢复]等人群": "[平素调养|亚健康状态|[慢性|虚弱性]疾病|病后恢复]等人群",
        "[党委常委|湖北大学副校长|马克思主义学院博士生导师|马克思主义学院教授|湖北省有突出贡献中青年专家]": "[湖北大学[党委常委|副校长]|马克思主义学院[博士生导师|教授]|湖北省有突出贡献中青年专家]",
        "[实验室|音乐室实验室|音乐室|科技活动室|学生机房|多功能教室|学生公寓|科技活动室|学生机房|多功能教室|学生公寓]": "[实验室|音乐室|科技活动室|学生机房|多功能教室|学生公寓]",
        "我国[青藏高原|青藏高原毗邻地区]": "我国青藏高原[|毗邻地区]",
        "[飞机用燃料|火箭用推进剂|各种润滑剂|各种液压油]": "[飞机用燃料|火箭用推进剂|各种[润滑剂|液压油]]",
        "[局域网信息传递|广域网信息传递]": "[局域网|广域网]信息传递",
        "[《公司法》|有关法律|有关法规]的规定": "[《公司法》|有关[法律|法规]的规定]",
        "[[自己|正选们]的合照|自己的球拍]": "[正选们的合照|自己的球拍]",
        "[培训学分管理||培训学分管理和社区康复协调员持证上岗]": "[培训学分管理|社区康复协调员持证上岗]",
        "[上海市体育总会第七届委员会常委|上海市体育总会第七届委员会副主席]": "上海市体育总会第七届委员会[常委|副主席]",
        "[人类化石20余件|石制品万余件|骨角器化石|哺乳动物化石]": "[人类化石20余件|石制品万余件|[骨角器|哺乳动物]化石]",
        "[由设区的市政府行使的经济社会管理权限|[省政府|省政府部门]下放给[省辖市政府|省辖市政府部门]的经济社会管理权限": "[由设区的市政府行使的经济社会管理权限|省政府[|部门]下放给省辖市政府[|部门]的经济社会管理权限]",
        "[省辖市政府|省辖市政府部门]": "[省辖市政府[|部门]]",
        "[现存|嘉靖二十七年手抄|康熙十一年手抄]": "[现存[嘉靖二十七年|康熙十一年]手抄]",
        '[各类房屋建筑|房屋建筑附属设施|市政设施抗震设防]': '[各类房屋建筑[|附属设施]|市政设施抗震设防]',
        '[大厅迎送引导服务法|重要旅客登记服务法|特殊需要预约服务法|双语式服务法|网络式畅通服务法]': '[大厅迎送引导服务法|重点旅客登记服务法|特殊需要预约服务法|双语式服务法|网络式畅通服务法]',
        "[著名数学家|沃尔夫奖得主|阿贝尔奖得主]": "[著名数学家|[沃尔夫奖|阿贝尔奖]得主]",
        "[嗜热细菌|耐热细菌|真菌|酵母菌菌株|相关分解酶]": "[[嗜热|耐热]细菌|真菌|酵母菌菌株|相关分解酶]",
        "[州人民政府|市人民政府|地区行政公署矿山企业主管部门]会同有关[部门|单位]": "[[州|市]人民政府|地区行政公署矿山企业主管部门]会同有关[部门|单位]",
        "不通不通路": "不通路",
        "杨箕村四面唯一与外界相通的通道通道": "杨箕村四面唯一与外界相通的通道",
        "[野鸡|野鸡]": "[野鸡|野鸭]",
        "一批[批骨|角器]": "一批[骨|角器]",
        "令人怦然心动的载体载体": "令人怦然心动的载体",
        "[团体比赛总分第一|个人第二名|个人第三名]": "[团体比赛总分第一|个人[第二名|第三名]]",
        '[什么可以做|什么不可以做]': '[什么可以做|不可以做]',
        '[“第二届中国评剧艺术节”|“第八届中国戏剧节”|“全国地方戏曲优秀剧目评比展演”|省文艺精品工程一等奖省文艺精品工程一等奖]': '[“第二届中国评剧艺术节”|“第八届中国戏剧节”|“全国地方戏曲优秀剧目评比展演”|省文艺精品工程一等奖]',
        '县级重点文物保护单位县级重点文物保护单位': '县级重点文物保护单位',
        '硝酸|发烟硫酸|氯磺酸、乙烯亚胺、乙烯二胺、氢氧化钠': '[硝酸|发烟硫酸|氯磺酸|乙烯亚胺|乙烯二胺|氢氧化钠]',
        '在当地的销售公司|特约经销处服务人员': '在当地的[销售公司|特约经销处服务人员]',
        '事故查处|责任追究]落实情况': '[事故查处|责任追究]落实情况',
        '现有的[防火墙|安防检测|入侵检测|负载均衡|频宽管理|网络防毒]等[设备|网络]问题': '现有的[防火墙|[安防|入侵]检测|负载均衡|频宽管理|网络防毒]等[设备|网络]问题',
        '[接入硬件|软件操作的方式]': '[接入[硬件|软件]操作的方式]',
        '[侯方域]陈贞慧]吴应箕]': '[侯方域|陈贞慧|吴应箕]',
        '[背包行囊等必须装备|选择出发日期|决定方向|路线包|包括可能的替代线路|准备好地图|或GPS|选择补给地点|事先邮寄待领的补充给养|预订起终点的交通|住宿]等': '[背包行囊等必须装备|选择出发日期|决定[方向|路线|可能的替代线路]|准备好[地图|GPS]|选择补给地点|事先邮寄待领的补充给养|预订起终点的[交通|住宿]]等',
        'IEC61032图7试具11|GB/T16842试具11|GB8898|IEC60065|IEC60598|GB7000|IEC60335|GB4706]等标准要求': '[IEC61032图7试具11|GB/T16842试具11|GB8898|IEC60065|IEC60598|GB7000|IEC60335|GB4706]等标准要求',
        '作为兴建房屋之用的': '作为兴建房屋之用',
        '项目招标|工程质量监督科技信息]工作': '[项目招标|工程质量监督|科技信息]工作',
        "[花园景|电话|平面电视|保险箱|书桌|暖气|更衣室|沙发|木质/镶木地板|衣柜/衣橱|吹风机|免费洗浴用品|卫生间|浴室|浴缸OR淋浴|唤醒服务]": "[花园景|电话|平面电视|保险箱|书桌|暖气|更衣室|沙发|木质/镶木地板|衣柜/衣橱|吹风机|免费洗浴用品|卫生间|浴室|浴缸|淋浴|唤醒服务]",
        "[获得[生存|发展|壮大|为全社会服务]": "[获得[生存|发展|壮大]|为全社会服务]",
        "[[[应用生物技术|化工|药物制剂|]相关专业全日制学生|培训机构学生]": "[[应用生物技术|化工|药物制剂|相关专业][全日制学生|培训机构学生]]",
        "[市委各部门之间|政府各部门之间|市委各部门与政府各部门之间|市直部门与乡之间|市直部门与镇之间]的职责分工]": "[市委各部门之间|政府各部门之间|市委各部门与政府各部门之间|市直部门与[乡|镇]之间]的职责分工",
        "[江河|溪流|湖泊|水塘|海岸]等水域岸边|其浅水处]": "[[江河|溪流|湖泊|水塘|海岸]等水域岸边|其浅水处]",
        "[深化改革|扩大开放|引进国内外[资金|技术|人才]": "[深化改革|扩大开放|引进国内外[资金|技术|人才]]",
        "[德国EVITA—Ⅱ全自动呼吸机|CSI—507SD多参数监护仪|日本美能自动牵引床|洁净手术室]|": "[德国EVITA—Ⅱ全自动呼吸机|CSI—507SD多参数监护仪|日本美能自动牵引床|洁净手术室]",
        "[初中|小学]|教育": "[初中|小学]教育",
        "[金属OR硬纸|塑料]": "[金属|硬纸|塑料]",
        "碱[碱2|10|25]mg": "碱[2|10|25]mg",
        "[市场|社会生产]|": "[市场|社会生产]",
        "长州藩[炮台|军舰]|": "长州藩[炮台|军舰]",
        "[果园|道旁]|[地边树丛|[房前屋后|庭院中]的树上]": "[果园|[道旁|地边]树丛|[房前屋后|庭院中]的树上]",
        "[马克思主义[民族观|宗教观|党的[民族宗教理论|政策|法律法规]]": "[马克思主义[民族观|宗教观]|党的[民族宗教理论|政策|法律法规]]",
        "党工委关于党的[思想|组织]作风建设|党员教育计划|中心组学习计划]": "[党工委关于党的[思想|组织]作风建设|党员教育计划|中心组学习计划]",
        "[[国内|省内]各市[音协|文艺界]人士": "[国内|省内各市][音协|文艺界]人士",
        "[倒转褶曲|平卧褶曲OR逆掩断层推覆构造体]": "[倒转褶曲|平卧褶曲|逆掩断层推覆构造体]",
        "[[手机|PDA|MP4|笔记本电脑|小型接收终端]": "[手机|PDA|MP4|笔记本电脑|小型接收终端]",
        "[大城市|中城市|小城市|小城镇]": "[[大|中|小]城市|小城镇]",
        "[氯氰菊酯等|特谱唑]混合喷雾]": "[氯氰菊酯等杀虫剂|特谱唑混合喷雾]",
        "[拱星卫生院重建|拱星卫生院加固项目|拱星小学重建项目|拱星幼儿园重建项目]": "[拱星卫生院[重建|加固]项目|拱星[小学|幼儿园]重建项目]",
        "[财政、[金融|其他经济|社会发展]": "[财政|金融|其他经济|社会发展]的情况",
        "[厦门市船东协会|水路运输行政主管部门OR其他有关部门]": "[厦门市船东协会|水路运输行政主管部门|其他有关部门]",
        "[消费者与企业有关的[经济|政治|社会|日常活动]范围内的[行为|需要|态度|动机]等": "消费者与企业有关的[经济|政治|社会|日常]活动范围内的[行为|需要|态度|动机]",
        "[[北京|上海|天津]等地方政府]|[中央政府[民政|社会福利|外资管理]等部门的专项咨询项目的顾问]]": "[[北京|上海|天津]等地方政府|中央政府][民政|社会福利|外资管理]等部门的专项咨询项目的顾问",
        "包括[中国[市长测评|城市开发区投资环境评估]等在内的重要项目]]": "中国[市长测评|城市开发区投资环境评估]等重要项目",
        "[机关行政事务管理|对外[联络|协调|接待]|会议的组织安排|对外信息发布]]": "[机关行政事务管理|对外[联络协调|接待]|会议的组织安排|对外信息发布]",
        "[DESC内容丰富|活动全面]": "[内容丰富|活动全面]",
        "[厦门知青在武平生活画面数十年来武平厦门间的交流活动|数十年来武平厦门间的交流活动]": "[厦门知青在武平生活画面|数十年来武平厦门间的交流活动]",
        "[国内[武术界|教育界]的[专家|教授|学者]": "国内[武术界|教育界]的[专家|教授|学者]",
        "[调节性|混合性]近视]": "[调节性|混合性]近视",
        "[四个专场|专业组]的[金奖|银奖|铜奖|优秀舞蹈新作品奖]共七十二个": "[[专场|专业组]的[金|银|铜]奖|优秀舞蹈新作品奖]",
        "[操作者的熟练程度|进操作者的熟练程度|进针次数|穿刺针与穿刺点胸膜切线位的锐角度|肺气肿]等因素针次数、穿刺针与穿刺点胸膜切线位的锐角度及肺气肿等因素": "[操作者的熟练程度|进针次数|穿刺针与穿刺点胸膜切线位的锐角度|肺气肿]等因素",
        "直流1mA电压[（U1mA）|0.75 U1mA|0.75 U1mA]": "[直流1mA电压（U1mA）|0.75 U1mA下泄漏电流]",
        "残疾人接受[康复训练|机构托养|职业培训|就业指导]|[开展文化|体育活动]]": "残疾人[接受[康复训练|机构托养|职业培训|就业指导]|开展[文化|体育]活动]",
        "[百度公司的战略规划|百度公司的运营管理]": "百度公司的[战略规划|运营管理]",
        "[川陕公路|成绵高速公路|绵广高速公路]南接成昆铁路|成渝铁路|成渝高速公路]": "[川陕公路|成绵高速公路|绵广高速公路|成昆铁路|成渝铁路|成渝高速公路]",
        "[[化工|冶金|钢厂|管道]等处必不可少的元件": "[化工|冶金|钢厂|管道]等处必不可少的元件",
        "[具备[管理|经济|法律|人力资源管理]等方面的知识和能力]|[能在[事业单位|政府部门]从事[人力资源管理]|[[教学|科研]方面工作的]工商管理学科高级专门人才": "[具备[管理|经济|法律|人力资源管理]等方面的知识和能力|能在[事业单位|政府部门]从事[人力资源管理|教学|科研]方面工作]的工商管理学科高级专门人才",
        "[[金山|中国银联|BBTV百视通|科大讯飞]等合作伙伴]|全球软件开发大赛等软件开发平台|遍布全球的研发机构|液晶面板等上游资源保障]": "[[金山|中国银联|BBTV百视通|科大讯飞]等合作伙伴|全球软件开发大赛等软件开发平台|遍布全球的研发机构|液晶面板等上游资源保障]",
        "[[HTTP|TCP|UDP（SUDP|RUDP）|网关穿透模组|全球IP表]": "[HTTP|TCP|UDP|SUDP|RUDP|网关穿透模组|UDP穿透|RPNP穿透|全球IP表]",
        "全市[经济|[资源|环境]协调发展": "全市[经济|资源|环境]协调发展",
        "[汉篆|汉隶]结合的书法变化轨迹|[汉篆|汉隶|八分体]的美的魅力]": "[[汉篆|汉隶]结合的书法变化轨迹|[汉篆|汉隶|八分体]的美的魅力]",
        "[全市党政群机关直属事业单位|各部门所属事业单位]《机构编制管理证》|新增人员的[控制|列编]等手续": "[全市党政群机关[直属事业单位|各部门所属事业单位]《机构编制管理证》|新增人员的[控制|列编]手续]",
        "[[“稳粮|增番|兴畜|扩经|促林|强工”]总体思路": "[稳粮|增番|兴畜|扩经|促林|强工]思路",
        "[[轻松|明亮|准确|圆润]": "[轻松|明亮|准确|圆润]",
        "[保险箱|熨斗|书桌|熨衣设备|暖气|淋浴|吹风机|免费洗浴用品|卫生间|浴缸OR淋浴|电视|电话|有线频道|迷你吧|冰箱]": "[保险箱|熨斗|书桌|熨衣设备|暖气|淋浴|吹风机|免费洗浴用品|卫生间|浴缸|淋浴|电视|电话|有线频道|迷你吧|冰箱]",
        "[土地抵押事项|房产抵押事项|车辆抵押事项|设备抵押事项]": "[土地|房产|车辆|设备]抵押事项",
        "[“三资企业”|旅游开发项目文件的审查|报批|协调服务]]": "[“三资企业”|旅游开发项目文件的[审查|报批|协调服务]]",
        "一部[喜剧|爱情片OR剧情片]": "一部[喜剧|爱情片|剧情片]",
        "[声音|歌唱的内容|歌唱者的风度、仪表、气质歌唱者的风度、仪表、气质]": "[声音|歌唱的内容|歌唱者的[风度|仪表|气质]]",
        "[业务培训||理论调研|宣传|信息|基层院备案材料的[收集|报送|多媒体示证]工作": "[业务培训|理论调研|宣传|信息|基层院备案材料的[收集|报送|多媒体示证]工作]",
        "[重庆能源结构中[清洁能源|可再生能源|新能源]": "重庆能源结构中[清洁能源|可再生能源|新能源]",
        "[执行STS - 41C条|在太空中记录了168个小时的]的": "[执行STS - 41C条|在太空中记录了168个小时的]",
        "[基本号码|特别号码]|": "[基本号码|特别号码]",
        "[[良好|正确]的[发声方法|发声技巧]": "[[良好|正确]的[发声方法|发声技巧]]",
        "主机的所有用户的[注册名|真名|最后登录时间|使用shell类型]等]": "主机的所有用户的[注册名|真名|最后登录时间|使用shell类型]等",
        "[国家大政方针|[政治|经济|社会生活]]中的重要问题]": "[国家大政方针|[政治|经济|社会生活]中的重要问题]",
        "[计算机科学与技术|软件工程计算机科学与技术、软件工程、网络工程、电子信息工程、通信工程、自动化、信息管理与信息系统等7个本科专业|网络工程|电子信息工程|通信工程|自动化|信息管理与信息系统]等7个本科专业": "[计算机科学与技术|软件工程|网络工程|电子信息工程|通信工程|自动化|信息管理与信息系统]",
        "[杂居少数民族|城镇少数民族]|少数民族[妇女|儿童]保护等有关事宜": "[散杂居少数民族|城镇少数民族|少数民族[妇女|儿童]保护]",
        "[中原文化|江淮文化|金陵文化|吴文化]": "[中原|江淮|金陵|吴]文化",
        "大量[军费|武器弹药]|": "大量[军费|武器弹药]",
        "[调整|缓冲OR线性化处理]": "[调整|缓冲|线性化处理]",
        "[项目建设的[程序|质量|安全|进度|资金使用（结算）|决算|竣工]等全过程": "项目建设的[程序|质量|安全|进度|资金使用（结算）|决算|竣工]等全过程",
        "[汉族|陕县人|中共党员|[中国戏剧家协会会员|河南省戏剧家协会理事|河南省艺术创作中心特约导演|三门峡市戏剧家协会主席|三门峡市文化局艺术科科长]": "[汉族|陕县人|中共党员|中国戏剧家协会会员|河南省戏剧家协会理事|河南省艺术创作中心特约导演|三门峡市戏剧家协会主席|三门峡市文化局艺术科科长]",
        "全球最大规模的[搜索引擎[营销|优化]专业会议搜索引擎战略大会": "全球最大规模的搜索引擎[营销|优化]专业会议搜索引擎战略大会",
        "[《魔法少女奈叶StrikerS THE COMICS》第一卷单行本漫画（日文）|《魔法少女奈叶StrikerS THE COMICS》的繁体中文版]": "《魔法少女奈叶StrikerS THE COMICS》[第一卷单行本漫画（日文）|繁体中文版]",
        "[民族[古籍|文物]的[抢救|收集|整理|出版规划]等工作": "[民族[古籍|文物]的[抢救|收集|整理|出版规划]等工作]",
        "[夫妻之间相濡以沫|[父|母]女之间血脉相袭的[点点滴滴|经典画面]": "[夫妻之间相濡以沫|[父|母]女之间血脉相袭的[点点滴滴|经典画面]]",
        "[阴平、[阴上|阴去|阴入|阳平|阳上|阳去|阳入]": "[阴平|阴上|阴去|阴入|阳平|阳上|阳去|阳入]",
        "[The Patriotic Front | PF]": "[The Patriotic Front|PF]",
        "[办公场所|经营场地]的[产权证明|租期1年以上的租赁合同|": "[办公场所|经营场地]的[产权证明|租期1年以上的租赁合同|合法的验资证明]",
        "周边[6个县（市）|55个乡镇|600多个天然村]": "周边[6个县（市）|55个乡镇|500多个天然村]",
        "食物的[四气|五味|归经|阴阳]属性等|人体的生理密切相关的[理论|经验]]": "[食物的[四气|五味|归经|阴阳]属性|人体的生理密切相关的[理论|经验]]",
        "党[工委|管委会]的[印章管理|机要保密]工作]": "党[工委|管委会]的[印章管理|机要保密]工作",
        "[进军演艺界|拍功夫片|获得更大的个人发展，成为明星甚至功夫巨星|获得更大的个人发展|成为明星甚至功夫巨星]": "[进军演艺界|拍功夫片|获得更大的个人发展|成为明星甚至功夫巨星]",
        "[《第五项修炼》的理论|《第五项修炼》的可操作性]": "《第五项修炼》的[理论|可操作性]",
        "[机关|事业单位人员|企业管理人员|专业技术人员]统计|[机关|事业]单位工资统计|人事信息管理]工作": "[[机关|事业单位人员|企业管理人员|专业技术人员]统计|[机关|事业]单位工资统计|人事信息管理工作]",
    }
    place_fix_map = {
        "在苏州|[工业园区|高新区]": "在苏州[工业园区|高新区]",
    }
    qua_fix_map = {
        "在专业(优质课)比赛中": "在专业（优质课）比赛中",
        "就适合相关职业(工种)": "就适合相关职业（工种）",
        "在[学业佛学思想]等方面": "在学业[|佛学思想]等方面",
        "根据《中共惠阳区委、惠阳区人民政府关于印发〈惠阳区人民政府机构改革方案〉的通知》(惠阳委发[2010]14号)精神": "根据《中共惠阳区委、惠阳区人民政府关于印发〈惠阳区人民政府机构改革方案〉的通知》（惠阳委发[2010]14号）精神",
        "按[项目技术水平的高低|经济效益的大小|社会效益的大小]": "按[项目技术水平的高低|[经济效益|社会效益]的大小]",
        "在[家园游戏|家园游戏续作]中": "在家园游戏[|续作]中",
        "[在校领导的高度重视|全体师生的共同努力]]": "在[校领导的高度重视|全体师生的共同努力]下",
        "在[零上40度高温|超低温环境]中|在[干燥|潮湿|风尘]等各个环境中": "[在[零上40度高温|超低温环境]中|在[干燥|潮湿|风尘]等各个环境中]",
        "从美国的[民主|宪法]|美国社会的问题|美国的移民历史|美国人的生活习惯]": "从[美国的[民主|宪法]|美国社会的问题|美国的移民历史|美国人的生活习惯]",
        "以[文字|彩图]]": "以[文字|彩图]",
        '一贯一贯': '一贯',
        "[": "_",
        "以林科字(2006)122号文": "以林科字（2006）122号文",
        "除了缓(控)释片以及某些特殊用途的片剂以外": "除了缓（控）释片以及某些特殊用途的片剂以外",
        "携传统(唐代)绘画研究项目": "携传统（唐代）绘画研究项目",
        "以()的名义": "以（）的名义",
        "在全国亿元乡(镇)社会经济发展经验交流会上": "在全国亿元乡（镇）社会经济发展经验交流会上",
    }
    time_fix_map = {
        "在()前": "在（）前",
        "(12月11日)": "（12月11日）",
        "(12月8日)": "（12月8日）",
    }
    text_fix_map = {
        "AGM-114反坦克导弹共发展出两代，一代于982年投产。": "AGM-114反坦克导弹共发展出两代，一代于1982年投产。",
        "5月21日，姚明的妻子叶莉于在休斯敦当地医院顺利产下一女。": "5月21日，姚明的妻子叶莉于在休斯顿当地医院顺利产下一女。",
        "科技辅导员活动辅导组织培训，使各校教师受益非浅。": "科技辅导员和活动辅导组织培训，使各校教师受益非浅。",
    }

    def clean_arg(txt):

        # txt = re.sub("\s*([\|\[\]])\s*", r"\1", txt)

        # rm blanks
        txt = txt.strip()
        txt = re.sub("([\u4e00-\u9fa5\|\[\]])\s+", r"\1", txt)
        txt = re.sub("\s+([\u4e00-\u9fa5\|\[\]])", r"\1", txt)

        # OR -> |
        txt = re.sub("([^a-zA-Z])OR([^a-zA-Z])", r"\1|\2", txt)

        # redundant characters
        txt = re.sub("\|+", "|", txt)
        txt = txt.strip("_")
        return txt

    for sample in data:
        text = sample["natural"]

        # if "国际减灾十年" in sample["natural"] and "国际减灾日" in sample["natural"]:
        #     print("!23")

        for spo in sample["logic"]:

            # fix subject
            if spo["subject"] in subj_fix_map:
                spo["subject"] = subj_fix_map[spo["subject"]]

            # fix predicate
            spo["predicate"] = spo["predicate"].strip()
            if spo["predicate"] in pred_fix_map:
                spo["predicate"] = pred_fix_map[spo["predicate"]]

            # fix object
            new_objs = []
            for obj in spo["object"]:
                if obj in obj_fix_map:
                    new_objs.append(obj_fix_map[obj])
                else:
                    new_objs.append(obj)
            spo["object"] = new_objs

            # fix place
            if spo["place"] in place_fix_map:
                spo["place"] = place_fix_map[spo["place"]]

            # fix time
            if spo["time"] in time_fix_map:
                spo["time"] = time_fix_map[spo["time"]]

            # fix qualifier
            if spo["qualifier"] in qua_fix_map:
                spo["qualifier"] = qua_fix_map[spo["qualifier"]]

            # fix text
            if text in text_fix_map:
                text = text_fix_map[text]
                sample["natural"] = text

            # special case
            if text == '又如绣石头、老树梗等，线粗，排针不必过于均匀。' and spo["predicate"] == "排针" and spo["object"][0] == '不必过于均匀':
                spo["subject"] = "排针"
                spo["predicate"] = '不必过于均匀'

            if text == '信息管理→施工项目管理是一项复杂的现代化的管理活动，更要依靠大量的信息以及对大量信息的管理，并应用电子计算机进行辅助。' and \
                    spo["object"][0] == '大量的信息|对大量信息的管理':
                spo["subject"] = "施工项目管理"
                spo["object"][0] = '[大量的信息|对大量信息的管理]'

            if text == '这种观点的大多数是具有使命感的资本家，还有主张进行自由式民主改革的人。' and spo["object"][0] == '资本家|':
                spo["object"][0] = '[资本家|主张进行自由式民主改革的人]'
            if text == '这种观点的大多数是具有使命感的资本家，还有主张进行自由式民主改革的人。' and spo["predicate"] == '主张进行':
                spo["subject"] = '人'

            if text == "前期投资只需要传统压缩机空调的一半，中期运行耗电量只需要传统空调的[/b]1/8[/b]——[/b]1/10[/b]，后期维护费用低。" and spo["predicate"] == "只需要X的[/b]1/8[/b]——[/b]1/10[/b]":
                text = "前期投资只需要传统压缩机空调的一半，中期运行耗电量只需要传统空调的1/8——1/10，后期维护费用低。"
                spo["predicate"] = "只需要X的1/8——1/10"

            if text == '全省户籍人口为70780918人。' and spo["object"][0] == '79780918人':
                spo["object"][0] = '70780918人'

            if text == "其中，60㎡内中小户型活跃，占比超过40%，增加约6%，罗湖、盐田占比超过5成，福田在5成左右。" and spo["time"] == "4月":
                spo["time"] = "_"

            if text == "还搭载了MOTOBLUR服务，这是一个聪明的交互聚合应用。" and spo["predicate"] == "MOTOBLUR服务":
                spo["predicate"] = "搭载了"
                spo["object"] = ["MOTOBLUR服务", ]
                spo["subject"] = "_"

            if text == "2017年1月3日晚间，由中央纪委宣传部、中央电视台联合制作的电视专题片《打铁还需自身硬》将在中央电视台综合频道首播。" and \
                    spo["subject"] == "《打铁还需自身硬》将在X首播" and spo["predicate"] == "中央电视台综合频道":
                spo["subject"] = "《打铁还需自身硬》"
                spo["predicate"] = "将在X首播"
                spo["object"][0] = "中央电视台综合频道"
                spo["time"] = "2017年1月3日晚间"

            if text == "2、梅予援：诗人梅庚之孙，梅立本的父亲，乾隆六年（1741）举人，次年中进士，步入仕途，官徐州教授。" \
                    and spo["time"] == "1742":
                spo["time"] = "1741"
            if text == "作法：采用花生仁、糕粉、白砂糖和精白面粉混合烘制而成。" and spo["subject"] == "采用X混合烘制而成" \
                    and spo["predicate"] == "精白面粉":
                spo["predicate"] = "采用X混合烘制而成"
                spo["object"][0] = "[花生仁|糕粉|白砂糖|精白面粉]"
                spo["subject"] = "_"

            if text == "1998年市委宣传部、市文联授予“三门峡市十佳文艺工作者”称号，1999年市政府授予“三门峡市劳动模范”称号。" and \
                    spo["subject"] == "[]" and spo["predicate"] == "_" and spo["qualifier"] == "_" and spo[
                "place"] == "_" \
                    and spo["time"] == "_" and spo["object"][0] == "_":
                spo["subject"] = "[市委宣传部|市文联]"
                spo["predicate"] = "授予X称号"
                spo["object"] = ["“三门峡市十佳文艺工作者”", ]
                spo["time"] = "1998年"
                sample["logic"].append({
                    "subject": "市政府",
                    "predicate": "授予X称号",
                    "object": ["“三门峡市劳动模范”", ],
                    "time": "1999年",
                    "place": "_",
                    "qualifier": "_",
                })

            if spo["predicate"] == '基于X对的对象' and spo["object"][0] == "要保护的":
                spo["predicate"] = '基于X的对象'
                spo["object"] = "要保护"

            if text == "这种方法虽然是企业在营销环境中进行的，但又不是纯自然的，是人们根据调查目的主动地、有目的地施加一些影响，所以，这种方法往往能够按照研究目的取得比较准确、有效的资料，是应用范围比较广泛的方法。" and \
                    spo["subject"] == "在X进行":
                spo["subject"] = "企业"

            if text == "西红柿多汁，可以利尿，肾炎病人也宜食用。" and \
                    spo["predicate"] == "西红柿" == spo["subject"]:
                spo["predicate"] = "多"
            if text == "加强各级残疾人辅助器具服务中心（站）建设，推进辅助器具服务进社区、到家庭，供应配发辅助器具9.2万件。" and \
                    spo["predicate"] == "辅助器具服务" == spo["subject"]:
                spo["predicate"] = "到"
            if text == "并且由于对弗里德兰的深入了解，他比谁都更看得透。" and \
                    spo["predicate"] == "他" == spo["subject"]:
                spo["predicate"] = "比谁都更看得透"
            if text == "该村到乡道路为土路，交通不方便；" and \
                    spo["predicate"] == "该村到乡道路" == spo["subject"]:
                spo["predicate"] = "为"
            if text == "赵州说∶「因为它有业识在。" and \
                    spo["predicate"] == "它" == spo["subject"]:
                spo["predicate"] = "有X在"

            if text == "本名：未公开（甘力事务所的惯例，旗下艺人基本都用艺名）" and \
                    spo["predicate"] == "都用" == spo["object"][0]:
                spo["object"][0] = "艺名"
            if text == "“诚信为本，质量为根”，发扬我中华千古不变的传统美德；" and \
                    spo["predicate"] == "为" == spo["object"][0]:
                spo["object"][0] = "根"
            if text == "在日本围棋最危难的时候，是依田屡次捍卫日本围棋的尊严，内战抗衡外籍棋手，外战力敌中韩军团。" and \
                    spo["predicate"] == "屡次捍卫" == spo["object"][0]:
                spo["object"][0] = "日本围棋的尊严"
            if text == "在日本围棋最危难的时候，是依田屡次捍卫日本围棋的尊严，内战抗衡外籍棋手，外战力敌中韩军团。" and \
                    spo["predicate"] == "分布" == spo["object"][0]:
                spo["object"][0] = "_"
                spo["qualifier"] = "密集"
            if text == "在日本围棋最危难的时候，是依田屡次捍卫日本围棋的尊严，内战抗衡外籍棋手，外战力敌中韩军团。" and \
                    spo["predicate"] == "人口" == spo["object"][0]:
                spo["object"][0] = "_"
                spo["qualifier"] = "鼎盛"

            if text == "3．歌唱中特别强调气息的控制，强调（连贯性）及音色的优美，要求歌唱中语气富于变化，情感表达真挚。" and \
                    spo["predicate"] == "要求X":
                spo["object"] = ["[语气富于变化|情感表达真挚]"]
            if text == "分离应用层和领域层有利于对领域模型的抽象和不断精化，也有利实施人员快速构建或调整产品，以满足企业发展变化的管理需要。" and \
                    spo["predicate"] == "有利X快速[构建|调整]Y":
                spo["object"] = ["实施人员", "产品"]
            if text == "汉武帝登极之三年，即公元前138年，“河水溢于平原，大饥，人相食”的事实，已出现于官方纪录。" and \
                    spo["predicate"] == "=公元前138年":
                spo["predicate"] = "="
                spo["object"] = ["公元前138年"]
            if text == "党委委员、副镇长谭昌兰：" and spo["subject"] == "谭昌兰" and spo["object"][0] == "[党委委员|副镇长]":
                spo["predicate"] = "ISA"
            if text == "'而对于削瘦扁平的人来说，更需要借着脂肪游移的原理，将腋下、后背、小腹的脂肪集中包裹在胸部，将大腿根部、外侧的脂肪上提固定在臀部，创造出玲珑有致的身材。" and \
                    spo["predicate"] == "将X集中Y":
                spo["predicate"] = "将X集中包裹Y"
                spo["object"] = ['[腋下|后背|小腹]的脂肪', '在胸部']
            if text == '快拿起手中的武器和你的小伙伴参加掠夺资源的战斗中吧！' and spo["predicate"] == "快拿起" == spo["object"][0]:
                spo["object"][0] = "手中的武器"
            if text == "以拱星敬老院和集镇主干道泰州路以及支线街道建设为主要内容的第二期援建项目确定，设计方案已经通过专家审查，工程可望11月中旬开工。" \
                    and spo["subject"] == "第二期援建项目" == spo["object"][0]:
                spo["object"][0] = "_"
            if text == "由于GPS技术所具有的全天候、高精度和自动测量的特点，作为先进的测量手段和新的生产力，已经融入了国民经济建设、国防建设和社会发展的各个应用领域。" \
                    and spo["subject"] == "生产力" == spo["object"][0]:
                spo["object"][0] = "新的"
            if text == "这种思想当时在物理界不但普遍存在，而且由来已久。" \
                    and spo["subject"] == "这种思想" == spo["object"][0]:
                spo["object"][0] = "[普遍存在|由来已久]"
            if text == "因为缺乏幼功，所以在从高庆奎先生之子高韵笙先生指导下每天练功。" \
                    and spo["subject"] == "高韵笙" == spo["object"][0]:
                spo["object"][0] = "高庆奎先生之子"
            if text == "建筑古朴雄健，是浙江省德清县内保存最完整的古代桥梁，现为省级文物保护单位。" \
                    and spo["subject"] == "德清县" == spo["object"][0]:
                spo["object"][0] = "浙江省"
            if text == "现在离我家已经不远了，但是要在五分钟之内到达，就要抄近路穿过一个大型的停车场。" \
                    and spo["subject"] == "我家" == spo["object"][0]:
                spo["subject"] = "_"
            if text == "若由于卡片本身质量问题造成的损坏，工作人员在票面上进行标注，乘客在有效期内由车站工作人员引导从专用通道进出站。" \
                    and spo["subject"] == "质量问题" == spo["object"][0]:
                spo["object"][0] = "卡片本身"
            if text == "的水平，所有的这些皆有助于自然分娩的母亲及生产的婴儿，婴儿的心跳率将保持在一个正常的范围内，由于子宫的适当收缩力，同时加上良好的氧气及血液供应结合着母亲的良好感觉，婴儿会比较舒适，分娩的每一阶段产道的扩张和推进都很顺畅，婴儿能以良好的位置旋转往下进入产道而出生，生产更加的自动，通过松弛的会阴，而使得母亲的身体不会产生组织、器官、肌肉不必要的损伤，由于可以旋转，特别是不会伤及婴儿的头部和身体。" \
                    and spo["subject"] == "会阴" == spo["object"][0]:
                spo["object"][0] = "松弛的"
            if text == "proav 针对专业影音应用，包括演播室设备、视频开关、录音棚、大型投影仪、数字信息发布系统、集成式家庭影院、教育场所和礼拜堂等，adi公司拥有业界最齐全的ic解决方案。" \
                    and spo["subject"] == "业界最齐全的" == spo["object"][0]:
                spo["subject"] = "ic解决方案"
            if text == "（3）传统节日（如花朝节，清明节，端午节，中秋节等）及假期穿汉服出游（服装等由外联部负责联系赞助），举行相应的传统仪式（如端午祭屈原，中秋拜月等）" \
                    and spo["subject"] == "外联部" == spo["object"][0]:
                spo["subject"] = "联系赞助"
            if text == "2011年，荣获“第五届中国机械工业优秀企业家”荣誉称号；" \
                    and spo["subject"] == "“第五届中国机械工业优秀企业家”" == spo["object"][0]:
                spo["object"][0] = "第五届"
            if text == "无论是炒纯素菜、荤菜或荤素搭配的菜，可将准备好的主料、配料和佐料全部一次投入，加盖并设定程序后，炒菜过程自动进行，无须专人翻炒或照看，无需经验，现学现会。" \
                    and spo["subject"] == "炒菜过程" == spo["object"][0]:
                spo["object"][0] = "_"
            if text == "制造业工人工资的快速上涨和有效劳动力的逐年减少，正逐渐改变30年来中国经济赖以高速增长的基础——大量廉价劳动力，人口红利逐渐消失的问题是中国企业不得不直视的一个严峻问题。" \
                    and spo["subject"] == "人口红利逐渐消失" == spo["object"][0]:
                spo["object"][0] = "严峻问题"
            if text == "总苞片多层，披针形，边缘有刺状缘毛，外层绿色，质硬而外弯，内层紫红色，开展或直立，先端具微毛；" \
                    and spo["subject"] == "外层" == spo["object"][0]:
                spo["object"][0] = "绿色"
            if text == "在2009年1月18日的预防宫颈癌疾病边城万人普查大型公益活动总结大会上，自治区人大常委会副主任杜秦瑞指出，“这次普查活动在新疆医疗卫生事业上是史无前例的，在全国也走在了前面。" \
                    and spo["subject"] == "杜秦瑞" == spo["object"][0]:
                spo["object"][0] = "自治区人大常委会副主任"
                spo["predicate"] = "ISA"
            if text == "根据省人社厅《转发人力资源社会保障部办公厅关于国家基本医疗保险、工伤保险和生育保险药品目录中部分药品进行调整规范的通知》（粤人社函〔2013〕1252号）要求，结合我市实际对部分药品进行调整规范，比如西药部分第1022号“重组人红细胞生成素（重组人促红素）”修改为“重组人红细胞生成素（CHO细胞）”，英文名称修改为“Recombinant Human Erythropoietin（CHO cell）”。" \
                    and spo["subject"] == "重组人红细胞生成素" == spo["object"][0]:
                spo["object"][0] = "重组人红细胞生成素（CHO细胞）"
                spo["subject"] = "重组人红细胞生成素（重组人促红素）"
            if text == "第四十条营业性运输船舶及非营业性运输船舶临时从事营业性运输，未按规定办理《营运证》的，及《营运证》被吊扣后仍继续营运的，由市水路运输行政主管部门或市水路运输行政主管部门委托的市水路运输管理处责令其停止营运，并按违法所得处以2倍罚款，罚款最高限额不得超过3万元。" \
                    and spo["subject"] == "市水路运输行政主管部门委托的" == spo["object"][0]:
                spo["subject"] = "市水路运输管理处"
            if text == "张恩照和戚道协这两位先后去职的企业高层人物都同FCPA有关，区别之处仅在于，张恩照是作为可能的受贿方受到处罚，而戚道协则是作为可能的行贿方受到惩罚。" \
                    and spo["subject"] == "两位先后去职的企业高层人物" == spo["object"][0]:
                spo["subject"] = "[张恩照|戚道协]"
                spo["object"][0] = "去职的企业高层人物"
            if text == "有人说，“中国没有华夏，华夏不在中国”，虽是愤慨之语，而对于华夏文明原生地的住民们，又有何言语辩驳呢？" \
                    and spo["subject"] == "住民们" == spo["object"][0]:
                spo["object"][0] = "华夏文明原生地的"
            if text == "河南文艺出版社这次进行新的开拓，必将给河南的出版事业带来新意，如果运作得好，也会带来文化与经济的双效益。" \
                    and spo["subject"] == "开拓" == spo["object"][0]:
                spo["object"][0] = "新的"
            if text == "自从去年发布以来，腾讯出品的手游《王者荣耀》每月平均新增500万日活跃用户，目前已经达到5000万，热度超过任天堂的《口袋妖怪Go》。" \
                    and spo["subject"] == "《口袋妖怪Go》" == spo["object"][0]:
                spo["subject"] = "《王者荣耀》"
            if text == "注：其中 屿头——黄岩 8元 （车程1小时左右），可使用台州公交IC卡。" \
                    and spo["subject"] == "屿头——黄岩" == spo["object"][0]:
                spo["subject"] = "屿头"
                spo["object"][0] = "黄岩"
                spo["predicate"] = "——"
            if text == "以身犯险登临绝顶，眺望莽莽苍山，平畴行千里，犹如一幅无穷尽的天然画卷，展示于茫茫大地。" \
                    and spo["subject"] == "苍山" == spo["object"][0]:
                spo["object"][0] = "莽莽"
            if text == "以身犯险登临绝顶，眺望莽莽苍山，平畴行千里，犹如一幅无穷尽的天然画卷，展示于茫茫大地。" \
                    and spo["subject"] == "印度庄园酒店" == spo["object"][0]:
                spo["object"][0] = "坎佩尔"
            if text == "作为杰出的词人，苏轼开辟了豪放词风，同杰出词人辛弃疾并称为“苏辛”。" \
                    and spo["subject"] == "辛弃疾" == spo["object"][0]:
                spo["object"][0] = "词人"
            if text == "和张恩照相同，在被迫去职之前，台湾人戚道协也在自己的职业生涯中获得了一个令人艳羡的地位——朗讯中国区总裁。" \
                    and spo["subject"] == "戚道协" == spo["object"][0]:
                spo["object"][0] = "[台湾人|朗讯中国区总裁]"
            if spo["subject"] == "推荐属性比例" and spo["predicate"] == "悟性=1：4：1：7：2":
                spo["predicate"] = "："
                spo["object"][0] = "膂力、根骨、身法、灵性、悟性=1：4：1：7：2"

        # rm blanks
        sample["natural"] = clean_arg(sample["natural"])

        for spo in sample["logic"]:
            # rm redundant blanks
            for k, v in spo.items():
                if k not in {"object", "objects"}:
                    spo[k] = clean_arg(spo[k])
                    # if len(re.findall("\[", spo[k])) != len(re.findall("\]", spo[k])):
                    #     print(spo)
                    #     print(text)
                    #     print(">>>>>>bad {}>>>>>>>".format(k))

                elif k == "object":
                    new_objs = []
                    for obj in spo[k]:
                        obj = clean_arg(obj)
                        # if len(re.findall("\[", obj)) != len(re.findall("\]", obj)):
                        #     print(spo)
                        #     print(text)
                        #     print(">>>>>>bad object>>>>>>>")

                        new_objs.append(obj)
                    spo[k] = new_objs

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

        candidate_spans, _ = Preprocessor.search_char_spans_fr_txt(search_str, text, "ch")
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

    predefined_p = dict()

    # trans predicate
    for sample in data:
        text = sample["natural"]
        for spo in sample["logic"]:
            predicate = spo["predicate"]
            # re.match("[A-Z=]+$", predicate) and predicate not in text
            if predicate != "_" and re.match("[A-Z=]+$", predicate) and predicate not in text:
                predefined_p[predicate] = predefined_p.get(predicate, 0) + 1

    # predefined_p_map = {"DESC": "描述",
    #                     "ISA": "是一种",
    #                     "IN": "位于",
    #                     "BIRTH": "生于",
    #                     "DEATH": "死于",
    #                     "=": "等于",
    #                     "NOT": "不",
    #                     }

    predefined_p_set = {"DESC", "ISA", "IN", "BIRTH", "DEATH", "=", "NOT"}
    new_data = []
    bad_spo_list = []
    for sample_id, sample in tqdm(enumerate(data), desc="transform"):
        ori_sample = copy.deepcopy(sample)
        # sample["natural"] = sample["natural"] + "[SEP]" + "，".join(predefined_p_map.values())
        text = sample["natural"]

        sample["logic"] = Preprocessor.unique_list(sample["logic"])
        new_spo_list = []
        for spo in sample["logic"]:
            # fix some wrong samples
            # if spo["object"][0] == "":
            #     print("!!!")

            if spo["predicate"] in predefined_p_set:
                if spo["object"][0] == "" and spo["predicate"] in {"DEATH", "BIRTH"}:
                    if spo["place"] != "":
                        spo["object"][0] = spo["place"]
                        spo["place"] = ""
                    elif spo["time"] != "":
                        spo["object"][0] = spo["time"]
                        spo["time"] = ""

                # # trans predicate
                # new_predicate = re.sub(k, predefined_p_map[k], spo["predicate"])
                # spo["predicate"] = new_predicate

            ori_spo = copy.deepcopy(spo)
            for key in spo:
                if spo[key] == "":
                    spo[key] = []
                elif key != "object" and key != "objects":
                    if re.search(".*\[.*\|.*\].*", spo[key]):  # need to split
                        ori_str = spo[key]
                        try:
                            split_list = parse_spe_txt2list(ori_str)
                            span_list = get_spe_txt_spans(ori_str, text, is_pred=True if key == "predicate" else False)
                        except Exception:
                            print(ori_str)
                            print(key)
                            print(text)
                            print(ori_spo)
                            print("==================error anns=======================")

                        # check spans
                        for idx, sp in enumerate(span_list):
                            extr_txt = Preprocessor.extract_ent_fr_txt_by_char_sp(sp, text, "ch")
                            try:
                                cor_str = re.sub("([^a-zA-Z]|^)[XYZU]([^a-zA-Z]|$)", r"\1\2", split_list[idx]) \
                                    if key == "predicate" else split_list[idx]
                                assert extr_txt == cor_str
                            except Exception:
                                print(text)
                                print(key)
                                print(ori_str)
                                print(extr_txt)
                                print(cor_str)
                                print("==================span search error==================")
                        comb_list = [{"char_span": [sp, ], "text": split_list[idx]} for idx, sp in enumerate(span_list)]
                        spo[key] = comb_list
                    else:
                        if key == "predicate" and spo[key] in predefined_p_set:
                            spo[key] = [
                                {"text": spo[key],
                                 "char_span": [[]],
                                 }, ]
                        else:
                            target_str = spo[key]
                            if key == "predicate":
                                spe_p_map = {
                                    "与X的Starch RX1500相当": "与的Starch RX1500相当",
                                    '将X装入UNIX服务器': '将装入UNIX服务器',
                                    '将X融入到DKNY的设计当中': '将融入到DKNY的设计当中',
                                    '与X合作生产SK-1Z02D正压防爆综合录井仪': '与合作生产SK-1Z02D正压防爆综合录井仪',
                                    '经常以X来称呼隔壁小王LYF': '经常以来称呼隔壁小王LYF',
                                }
                                if spo[key] in spe_p_map:
                                    target_str = spe_p_map[spo[key]]
                                else:
                                    target_str = re.sub("[XYZU]", "", spo[key])

                            try:
                                char_spans, _ = Preprocessor.search_char_spans_fr_txt(target_str, text, "ch")
                                spo[key] = [
                                    {"text": spo[key],
                                     "char_span": char_spans,
                                     }, ]
                                for ch_sp in char_spans:
                                    extr_txt = Preprocessor.extract_ent_fr_txt_by_char_sp(ch_sp, text, "ch")
                                    assert extr_txt == target_str
                            except Exception:
                                print(target_str)
                                print(key)
                                print(text)
                                print(ori_spo)
                                print("==================error anns=======================")

                elif key == "object":
                    new_objs = []
                    for obj in spo[key]:
                        if re.search(".*\[.*\|.*\].*", obj):  # need to split
                            ori_str = obj

                            try:
                                split_list = parse_spe_txt2list(obj)
                                span_list = get_spe_txt_spans(ori_str, text, is_pred=False)
                            except Exception:
                                print(ori_str)
                                print(key)
                                print(text)
                                print(ori_spo)
                                print("==================error anns=======================")

                            # check spans
                            for idx, sp in enumerate(span_list):
                                extr_txt = Preprocessor.extract_ent_fr_txt_by_char_sp(sp, text, "ch")
                                try:
                                    cor_str = split_list[idx]
                                    assert extr_txt == cor_str
                                except Exception:
                                    print(text)
                                    print(key)
                                    print(ori_str)
                                    print(extr_txt)
                                    print(cor_str)
                                    print("==================span search error==================")

                            comb_list = [{"char_span": [sp, ], "text": split_list[idx]} for idx, sp in
                                         enumerate(span_list)]
                            new_objs.append(comb_list)
                        else:
                            if obj == "":
                                pass
                            else:
                                try:
                                    char_spans, _ = Preprocessor.search_char_spans_fr_txt(obj, text, "ch")
                                except Exception:
                                    print(obj)
                                    print(key)
                                    print(text)
                                    print(ori_spo)
                                    print("==================error anns=======================")

                                new_objs.append([
                                    {"text": obj,
                                     "char_span": char_spans,
                                     }, ])
                    spo[key] = new_objs

            for p in spo["predicate"]:
                if re.search("[XYZU]", p["text"]) is None and len(spo["object"]) > 0 and p[
                    "text"] not in predefined_p_set:
                    p["text"] += "X"

            # align predicate and the corresponding subset of objects (by XYZU)
            ext_spo_list = []
            id_map = {"X": 0, "Y": 1, "Z": 2, "U": 3}
            spe_p_map = {
                "与X的Starch RX1500相当": "与[OBJ]的Starch RX1500相当",
                '将X装入UNIX服务器': '将[OBJ]装入UNIX服务器',
                '将X融入到DKNY的设计当中': '将[OBJ]融入到DKNY的设计当中',
                '与X合作生产SK-1Z02D正压防爆综合录井仪': '与[OBJ]合作生产SK-1Z02D正压防爆综合录井仪',
                '经常以X来称呼隔壁小王LYF': '经常以[OBJ]来称呼隔壁小王LYF',
            }

            bad_spo = False
            for p in spo["predicate"]:
                sub_objs = []
                if p["text"] in spe_p_map or p["text"] in predefined_p_set:
                    sub_objs.append(spo["object"][0])
                else:
                    for ph in re.findall("[XYZU]", p["text"]):
                        idx = id_map[ph]
                        try:
                            sub_objs.append(spo["object"][idx])
                        except Exception:
                            # print(spo)
                            # print(text)
                            # print(">>>>>>>>>>>>>>>>object list: out of index>>>>>>>>>>>>>>>>>>>>>>>>>")
                            bad_spo = True
                            bad_spo_list.append({
                                "text": text,
                                "bad_spo": ori_spo,
                                "ori_sample": ori_sample,
                            })
                            break

                if bad_spo:
                    break

                # XYZU -> [OBJ]
                if p["text"] in spe_p_map:
                    p["text"] = spe_p_map[p["text"]]
                else:
                    assert len(spo["object"]) <= 4

                    if len(spo["object"]) >= 1:
                        p["text"] = re.sub("X", "[OBJ]", p["text"])
                    if len(spo["object"]) >= 2:
                        p["text"] = re.sub("Y", "[OBJ]", p["text"])
                    if len(spo["object"]) >= 3:
                        p["text"] = re.sub("Z", "[OBJ]", p["text"])
                    if len(spo["object"]) == 4:
                        p["text"] = re.sub("U", "[OBJ]", p["text"])

                new_spo = copy.deepcopy(spo)
                new_spo["predicate"] = [p, ]
                new_spo["object"] = sub_objs
                ext_spo_list.append(new_spo)
            if len(spo["predicate"]) == 0:
                ext_spo_list.append(spo)

            # product
            open_spo_list = []
            for new_spo in ext_spo_list:
                lists4prod = []
                for k, l in new_spo.items():
                    if k in {"object", "objects"} or len(l) == 0:
                        continue
                    lists4prod.append([{"type": k, **i} for i in l])

                for objs in new_spo["object"]:
                    new_objs = []
                    for i in objs:
                        new_objs.append({"type": "object", **i})
                    lists4prod.append(new_objs)

                open_spo_list.extend([list(item) for item in itertools.product(*lists4prod)])

            # choose the best span from candidates
            filtered_open_spo_list = []
            for spo in open_spo_list:
                if any(len(arg["char_span"]) > 1 for arg in spo):
                    new_spo = []
                    for arg_idx_i, arg_i in enumerate(spo):
                        if len(arg_i["char_span"]) > 1:
                            fin_ch_sp = None
                            fin_dis_score = 9999
                            for ch_sp_i in arg_i["char_span"]:
                                dis_score = 0
                                for arg_idx_j, arg_j in enumerate(spo):
                                    if arg_idx_i == arg_idx_j:
                                        continue
                                    for ch_sp_j in arg_j["char_span"]:
                                        if len(ch_sp_j) == 0:
                                            continue
                                        dis_score += min(abs(ch_sp_i[0] - ch_sp_j[1]), abs(ch_sp_j[0] - ch_sp_i[1]))
                                if dis_score < fin_dis_score:
                                    fin_dis_score = dis_score
                                    fin_ch_sp = ch_sp_i

                            arg_cp = copy.deepcopy(arg_i)
                            arg_cp["char_span"] = fin_ch_sp
                            new_spo.append(arg_cp)
                        else:
                            arg_cp = copy.deepcopy(arg_i)
                            arg_cp["char_span"] = arg_cp["char_span"][0]
                            new_spo.append(arg_cp)
                else:
                    new_spo = [{**arg, "char_span": arg["char_span"][0]} for arg in spo]
                filtered_open_spo_list.append(new_spo)

            # clean
            # filter arg with blank span; strip blanks around entity
            fin_open_spo_list = []
            for spo in filtered_open_spo_list:
                if any((len(arg["char_span"]) == 0 and arg["text"] not in predefined_p_set)
                       or arg["text"].strip() == ""
                       for arg in spo):
                    # print(spo)
                    # print(text)
                    # print(">>>>>>>>>>>>>>>>>>>>invalid arg>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    bad_spo_list.append({
                        "text": text,
                        "bad_spo": ori_spo,
                        "ori_sample": ori_sample,
                    })
                    continue

                # strip blanks
                for arg in spo:
                    if arg["text"] in predefined_p_set:
                        # print("predefined predicate")
                        continue
                    utils.strip_entity(arg)

                fin_open_spo_list.append(spo)

            new_spo_list.extend(fin_open_spo_list)
        new_sample = {
            "id": sample_id,
            "text": text,
            "open_spo_list": new_spo_list,
        }
        new_data.append(new_sample)
    return new_data, predefined_p, bad_spo_list


def trans_saoke():
    new_data, predefined_pred_set, bad_spo_list = preprocess_saoke()

    for sample in new_data:
        span_list = []
        for spo in sample["open_spo_list"]:
            for arg in spo:
                span_list.append(arg["char_span"])
        tok_res = ChineseWordTokenizer.tokenize_plus(sample["text"], span_list=span_list)
        sample["word_list"] = tok_res["word_list"]
        sample["word2char_span"] = tok_res["word2char_span"]

    train_data_rate = 0.8
    val_data_rate = 0.1
    train_num = int(len(new_data) * train_data_rate)
    valid_num = int(len(new_data) * val_data_rate)
    test_num = len(new_data) - train_num - valid_num
    random.shuffle(new_data)
    train_data = new_data[:train_num]
    valid_data = new_data[train_num:train_num + valid_num]
    test_data = new_data[-test_num:]

    return train_data, valid_data, test_data


def trans_casie():
    from glob import glob
    data = []
    for file_path in glob("../../data/ori_data/casie_bk/*.json"):
        sample = load_data(file_path)
        file_name = file_path.split("/")[-1]
        idx = re.search("\d+", file_name).group()
        sample["id"] = idx
        data.append(sample)

    new_data = []
    for sample in data:
        if "cyberevent" not in sample:
            continue
        text = sample["content"]
        new_sample = {
            "id": sample["id"],
            "text": text,
            "event_list": []
        }

        span2text = {}
        char_span_list = []
        for hp in sample["cyberevent"]["hopper"]:
            for event in hp["events"]:
                trigger = event["nugget"]["text"]
                trigger_char_span = [event["nugget"]["startOffset"], event["nugget"]["endOffset"]]
                span2text[str(trigger_char_span)] = trigger

                new_event = {
                    "trigger": trigger,
                    "trigger_char_span": trigger_char_span,
                    "realis": event["realis"],
                    "trigger_type": "{}.{}".format(event["type"], event["subtype"]),
                    "argument_list": [],
                }
                if "argument" in event:
                    for arg in event["argument"]:
                        arg_char_span = [arg["startOffset"], arg["endOffset"]]
                        arg_txt = arg["text"]
                        span2text[str(arg_char_span)] = arg_txt

                        new_event["argument_list"].append({
                            "text": arg_txt,
                            "char_span": arg_char_span,
                            "type": arg["role"]["type"],
                        })
                new_sample["event_list"].append(new_event)
        # check spans
        span2text_ = dict(sorted(span2text.items(), key=lambda x: int(re.search("\d+", x[0]).group())))
        for sp_str, txt in span2text_.items():
            sp_se = re.search("\[(\d+), (\d+)\]", sp_str)
            sp = [int(sp_se.group(1)), int(sp_se.group(2))]
            char_span_list.append(sp)

            extr_txt = Preprocessor.extract_ent_fr_txt_by_char_sp(sp, text)
            try:
                assert extr_txt == txt
            except Exception:
                print(sample["sourcefile"])
                print(extr_txt)
                print(txt)
                print("===========================")

        # word2char_span and word list
        tok_res = WhiteWordTokenizer.tokenize_plus(text, span_list=char_span_list)
        new_sample["word_list"] = tok_res["word_list"]
        new_sample["word2char_span"] = tok_res["word2char_span"]
        new_data.append(new_sample)

    random.shuffle(new_data)
    test_data = new_data[-100:]
    save_as_json_lines(test_data, "../../data/ori_data/casie/test_data.json")

    valid_num = int(900 / 8) - 1
    for start_idx in range(0, 900, valid_num):
        end_idx = start_idx + valid_num
        if end_idx > 900:
            break
        valid_data = new_data[start_idx:end_idx]
        train_data = new_data[:start_idx] + new_data[end_idx:-100]
        save_as_json_lines(train_data, "../../data/ori_data/casie/train_data_{}.json".format(start_idx // valid_num))
        save_as_json_lines(valid_data, "../../data/ori_data/casie/valid_data_{}.json".format(start_idx // valid_num))
        print("{}, {}".format(len(train_data), len(valid_data)))


def trans_chfin():
    data_dir = "../../data/ori_data/chfinann_bk"
    save_dir = "../../data/ori_data/chfinann"
    train_data = load_data(os.path.join(data_dir, "train.json"))
    val_data = load_data(os.path.join(data_dir, "dev.json"))
    test_data = load_data(os.path.join(data_dir, "test.json"))

    def trans_chfinann(data):
        new_data = []
        for sample in data:
            text = " ".join(sample[1]["sentences"])

            # mention 2 spans
            all_char_span_list = []
            offset = [0, ]
            for sent in sample[1]["sentences"]:
                offset.append(offset[-1] + len(sent) + 1)
            mention2spans = {m: [[offset[sp[0]] + sp[1], offset[sp[0]] + sp[2]] for sp in sent_spans] for m, sent_spans
                             in sample[1]["ann_mspan2dranges"].items()}
            for m, char_spans in mention2spans.items():
                all_char_span_list.extend(char_spans)
                for sp in char_spans:
                    assert m == text[sp[0]:sp[1]]

            event_list = []
            for event in sample[1]["recguid_eventname_eventdict_list"]:
                event_type = event[1]
                arg_list = []
                for arg_type, mention in event[2].items():
                    if mention is None:
                        continue
                    for m_span in mention2spans[mention]:
                        arg_list.append({
                            "text": mention,
                            "char_span": m_span,
                            "type": arg_type,
                        })
                event_list.append({
                    "event_type": event_type,
                    "argument_list": arg_list,
                })

            tok_res = ChineseWordTokenizer.tokenize_plus(text, span_list=all_char_span_list)
            new_sample = {
                "id": sample[0],
                "text": text,
                **tok_res,
                "event_list": event_list,
            }
            new_data.append(new_sample)
        return new_data

    new_train_data = trans_chfinann(train_data)
    new_valid_data = trans_chfinann(val_data)
    new_test_data = trans_chfinann(test_data)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_as_json_lines(new_train_data, os.path.join(save_dir, "train_data.json"))
    save_as_json_lines(new_valid_data, os.path.join(save_dir, "valid_data.json"))
    save_as_json_lines(new_test_data, os.path.join(save_dir, "test_data.json"))


def preprocess_duie():
    data_dir = "../../data/ori_data/duie_comp2021_bk"
    save_dir = "../../data/ori_data/duie_comp2021"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data = load_data(os.path.join(data_dir, "train_data.json"))
    valid_data = load_data(os.path.join(data_dir, "valid_data.json"))
    test_data = load_data(os.path.join(data_dir, "test_data_1.json"))

    # fix data
    for sample in tqdm(train_data + valid_data):
        text = sample["text"]

        if text == "2  朱美音21967出生在江西南昌，1989年毕业于江西科技师范大学外语系英语专业，后来分配至鹰潭铁路一中任英语教师":
            sample["text"] = "2  朱美音 1967出生在江西南昌，1989年毕业于江西科技师范大学外语系英语专业，后来分配至鹰潭铁路一中任英语教师"
            # sample["postag"] = [{"word": w, } for w in jieba.cut(sample["text"])]
        if text == "人物简介王珊2，女，19441，1968年毕业于北京大学物理系":
            sample["text"] = "人物简介王珊2，女，1944，1968年毕业于北京大学物理系"
            # sample["postag"] = [{"word": w, } for w in jieba.cut(sample["text"])]
        if text == "影片信息电视剧影片名称：舞动芝加哥第二季  影片类型：欧美剧  影片语言：英语  上映年份：20121演员表剧情介绍美国芝加哥，单亲女孩CeCe（Bella Thorne饰）和闺蜜Rocky（Zendaya Coleman饰）原本只是两个爱跳舞的普通初中生":
            sample[
                "text"] = "影片信息电视剧影片名称：舞动芝加哥第二季  影片类型：欧美剧  影片语言：英语  上映年份：2012 演员表剧情介绍美国芝加哥，单亲女孩CeCe（Bella Thorne饰）和闺蜜Rocky（Zendaya Coleman饰）原本只是两个爱跳舞的普通初中生"
            # sample["postag"] = [{"word": w, } for w in jieba.cut(sample["text"])]
        if text == "http://news.sohu.com/20081221/n261333381.shtml 12月20日，西北政法大学动物保护法研究中心挂牌成立 同时，由西北政法大学和中国社会科学院法学研究所共同主办的“中国《动物保护法》研究项目”正式启动 　　图为西北政法大学动物保护法研究中心主任孙江(左一)从西北政法大学校长贾宇教授(左二)手中接过“西北政法大学动物保护研究中心”的牌匾":
            sample["text"] = "2008，西北政法大学动物保护法研究中心挂牌成立 同时，由西北政法大学和中国社会科学院法学研究所共同主办的“中国《动物保护法》研究项目”正式启动 　　图为西北政法大学动物保护法研究中心主任孙江(左一)从西北政法大学校长贾宇教授(左二)手中接过“西北政法大学动物保护研究中心”的牌匾"
        if text in {"于卫涛，1976年出生于河南省通许县，2013年携手刘洋创办河南欣赏网络科技集团，任该集团董事长，专注于我国大中小型企业提供专业的网络服务，带动很多企业网络方向的转型",
                    "1962年周明牂在中国植物保护学会成立大会上，作了《我国害虫农业防治研究现状和展望》的学术报告，1963年在《人民日报》上发表《结合耕作防治害虫》一文"}:
            new_spo_list = [spo for spo in sample["spo_list"] if spo["predicate"] != "所属专辑"]
            sample["spo_list"] = new_spo_list

        # if text in {
        #     "该片获1988年第八届中国电影金鸡奖最佳女主角奖（潘虹）",
        #             }:
        #     print("2 fix")

        for spo in sample["spo_list"]:
            if text == '比如张艺谋的两届威尼斯国际电影节金狮奖 ，第38届柏林国际电影节金熊奖 ，两届英国电影学院奖最佳外语片 ，第55届台湾电影金马奖最佳导演奖 ，第05届中国电影华表奖最佳导演奖 ，2008影响世界华人大奖' and \
                spo["object"]["@value"] == '中国电影华表奖最佳导演奖' and spo["object"]["period"] == "5":
                spo["object"]["period"] = "05"
            if text == '陈思诚，曾获得第三届英国万像国际华语电影节优秀男配角奖，第21届北京大学生电影节最佳导演处女作奖，第23届北京大学生电影节最佳编剧奖等奖项，前两年背叛佟丽娅一事也是闹得人尽皆知' \
                    and spo["object"]["@value"] == '英国万像国际华语电影节优秀男配角奖' and spo["object"]["period"] == "3":
                spo["object"]["period"] = "三"
            if text == '1982年，许冠文凭《摩登保镖》，勇夺第一届香港电影金像奖最佳男主角奖' \
                    and spo["object"]["@value"] == '香港电影金像奖最佳男主角奖' and spo["object"]["period"] == "1":
                spo["object"]["period"] = "一"
            if text == '亚洲电影大奖终身成就奖:第02届，2008年:山田洋次第04届，2010年:阿米达巴彻第05届，2011年:邹文怀第06届，2012年:许鞍华第08届，2014年:侯孝贤第09届，2015年:林权泽第10届，2016年:树木希林&袁和平第11届，2017年:徐克第12届，2018年:张艾嘉第13届，2019年:李沧东' \
                    and spo["object"]["@value"] == '亚洲电影大奖终身成就奖' and "period" in spo["object"] and spo["object"]["period"] == "9":
                spo["object"]["period"] = "09"
            if text == '亚洲电影大奖终身成就奖:第02届，2008年:山田洋次第04届，2010年:阿米达巴彻第05届，2011年:邹文怀第06届，2012年:许鞍华第08届，2014年:侯孝贤第09届，2015年:林权泽第10届，2016年:树木希林&袁和平第11届，2017年:徐克第12届，2018年:张艾嘉第13届，2019年:李沧东' \
                    and spo["object"]["@value"] == '亚洲电影大奖终身成就奖' and "period" in spo["object"] and spo["object"]["period"] == "6":
                spo["object"]["period"] = "06"
            if text == '《007》、《谍影重重》作为曾经的特工片，燃爆了整个好莱坞，前者至今收获约70亿美元，后者至今收获约11亿美元，在收获不菲票房的同时，也赢尽了口碑' \
                    and spo["object"]["@value"] == '70亿美元' and spo["subject"] == "7":
                spo["subject"] = "007"
            if text == '1987年，由作家张弦编剧、李亚林导演的电影《井》完成，潘虹因在该片中扮演徐丽莎而获得第八届中国电影金鸡奖最佳女主角奖、第18届意大利陶尔米纳国际电影节最佳女主角奖' \
                    and spo["object"]["@value"] == '中国电影金鸡奖最佳女主角' and spo["object"]["period"] == "8":
                spo["object"]["period"] = "八"
            if text == '该片获1988年第八届中国电影金鸡奖最佳女主角奖（潘虹）' \
                    and spo["object"]["@value"] == '中国电影金鸡奖最佳女主角' and spo["object"]["period"] == "8":
                spo["object"]["period"] = "八"
            if text == "出生于1984年9月26日，河南郑州，学历：本科，特长：主持、朗诵、表演、国画、平面模特、童声模仿 毕业院校/专业：中国传媒大学/播音与主持艺术专业，管文君，中央电视台体育频道（CCTV5）体育晨报《天气体育》主持人，同时，还主持中央电视台经济频道（CCTV2）《第一印象》、中央电视台农业频道《农业气象》和中国气象频道《天气直播间》等栏目" \
                and spo["object"]["@value"] == "" and spo["predicate"] == "毕业院校":
                spo["object"]["@value"] = "中国传媒大学"
            if text == "《白色梦幻》是一部于1998年1月1日出品的电视剧，由太纲导演，由田岷、许亚军、盖丽丽 和何晴主演，一共有20集，每集48分钟" \
                and spo["object"]["@value"] == "" and spo["predicate"] == "上映时间":
                spo["object"]["@value"] = "1998年1月1日"
                spo["subject"] = "白色梦幻"
            if text == "《闪点行动第五季》是一部由编剧Stephanie Morgenstern编写的一部动作剧情电视剧，出品时间为2012年09月20日" \
                and spo["object"]["@value"] == "" and spo["predicate"] == "上映时间":
                spo["object"]["@value"] = "2012年09月20日"
                spo["subject"] = "闪点行动第五季"

            if text == "第六名 刘悦 1982.1.25 江苏淮安 ——2001年10月 首届百事全国新星大赛江苏地区选拔赛一等奖 最佳激情奖 明日之星称号，2002年6月 中韩新星选秀大赛独唱组特等奖，2004年6月 中央电视台《非常6+1》， 湖南卫视超级女生成都赛区前10 ， 2011年 第一届华人星光大道第七名 用灵魂唱歌的歌手，唱的无数听众流泪，不愧“小刘欢”，因此成功拜师大刘欢 9 S" \
                and spo["predicate"] == "获奖" and "period" in spo["object"] and spo["object"]["period"] == "":
                spo["object"]["period"] = "首"
            if text == "2010年获Music Radio中国Top排行榜内地最佳创作歌手奖，第八届东南劲爆音乐榜颁奖典礼劲爆内地最佳唱作歌手奖、劲爆最佳作曲人奖李健以重庆为起点，开启2018-2020“不止 是李健”世界巡回演唱会" \
                and spo["predicate"] == "获奖" and "period" in spo["object"] and spo["object"]["period"] == "":
                spo["object"]["period"] = "八"

            if spo["predicate"] in {"专业代码", "邮政编码"} and text[
                re.search(spo["object"]["@value"], text).span()[0] - 1] == "0":
                spo["object"]["@value"] = "0" + spo["object"]["@value"]

            # strip redundant whitespaces and unknown characters
            spo["subject"] = clean_entity(spo["subject"])
            for k, item in spo["object"].items():
                spo["object"][k] = clean_entity(item)

        new_spo_list = []
        for spo in sample["spo_list"]:
            if spo["subject"].lower() not in text.lower() or spo["object"]["@value"].lower() not in text.lower():
                # drop wrong spo
                continue

            if spo["subject"] not in text:  # if not in, try recover upper case
                m = re.search(re.escape(spo["subject"].lower()), text.lower())
                # print("{}----{}".format(spo["subject"], text[m.span()[0]:m.span()[1]]))
                spo["subject"] = text[m.span()[0]:m.span()[1]]

            if spo["object"]["@value"] not in text:
                m = re.search(re.escape(spo["object"]["@value"].lower()), text.lower())
                # print("{}----{}".format(spo["object"], text[m.span()[0]:m.span()[1]]))
                spo["object"]["@value"] = text[m.span()[0]:m.span()[1]]
            new_spo_list.append(spo)

        filtered_spo_list = []
        for spo in new_spo_list:
            assert spo["subject"] in text

            if spo["subject"].strip() == "":
                continue

            bad_spo = False
            for item in spo["object"].values():
                assert item in text
                if item.strip() == "":
                    bad_spo = True
                    break

            if not bad_spo:
                filtered_spo_list.append(spo)

        sample["spo_list"] = filtered_spo_list

    train_data_path = os.path.join(save_dir, "train_data.json")
    valid_data_path = os.path.join(save_dir, "valid_data.json")
    test_data_path = os.path.join(save_dir, "test_data_1.json")
    save_as_json_lines(train_data, train_data_path)
    save_as_json_lines(valid_data, valid_data_path)
    save_as_json_lines(test_data, test_data_path)


if __name__ == "__main__":
    # preprocess_duie()

    train_data, valid_data, test_data = trans_saoke()
    save_dir = "../../data/ori_data/saoke"

    train_data_path = os.path.join(save_dir, "train_data.json")
    valid_data_path = os.path.join(save_dir, "valid_data.json")
    test_data_path = os.path.join(save_dir, "test_data.json")
    save_as_json_lines(train_data, train_data_path)
    save_as_json_lines(valid_data, valid_data_path)
    save_as_json_lines(test_data, test_data_path)


