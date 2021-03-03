import json
import os
from InfExtraction.modules.preprocess import Preprocessor, WhiteWordTokenizer, ChineseWordTokenizer
from InfExtraction.modules.utils import load_data, save_as_json_lines, merge_spans
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
            spans, add_text = Preprocessor.search_char_spans_fr_txt(target_str, text, "en")
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
                    spans, add_text = Preprocessor.search_char_spans_fr_txt(alt_txt, text, "en")
                    # cal how many words are matched this time
                    match_num = len(re.findall("_+", add_text)) - len(re.findall("_+", pre_add_text))
                    if match_num > 0:  # some words matched
                        match_num2spans[match_num] = spans
                        max_match_num = max(max_match_num, match_num)
                if max_match_num > 0:  # if there are any successful cases
                    fin_spans = match_num2spans[max_match_num]  # use the longest match

            if fin_spans is None or len(fin_spans) == 0:  # if still can not match, take partial match instead
                spans, add_text = Preprocessor.search_char_spans_fr_txt(target_str, text, "en")
                fin_spans = spans
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

                        spans, add_text = Preprocessor.search_char_spans_fr_txt(predicate, text, "en")
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


def preprocess_saoke(data_path = "../../data/ori_data/saoke_bk/saoke.json"):
    data = load_data(data_path)
    # fix data
    subj_fix_map = {
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
    }
    pred_fix_map = {
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
        "从而减轻|避免X的压迫": "从而[减轻|避免]X的[摩擦|压迫]",
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
    obj_fix_map = {
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
        "[由设区的市政府行使的经济社会管理权限|[省政府|省政府部门]下放给[省辖市政府|省辖市政府部门]的经济社会管理权限": "[由设区的市政府行使的经济社会管理权限|[省政府|省政府部门]下放给[省辖市政府|省辖市政府部门]的经济社会管理权限]",
        "[《第五项修炼》的理论|《第五项修炼》的可操作性]": "《第五项修炼》的[理论|可操作性]",
        "[机关|事业单位人员|企业管理人员|专业技术人员]统计|[机关|事业]单位工资统计|人事信息管理]工作": "[[机关|事业单位人员|企业管理人员|专业技术人员]统计|[机关|事业]单位工资统计|人事信息管理工作]",
    }
    place_fix_map = {
        "在苏州|[工业园区|高新区]": "在苏州[工业园区|高新区]",
    }
    text_fix_map = {
        "科技辅导员活动辅导组织培训，使各校教师受益非浅。": "科技辅导员和活动辅导组织培训，使各校教师受益非浅。",
    }
    qua_fix_map = {
        "[在校领导的高度重视|全体师生的共同努力]]": "在[校领导的高度重视|全体师生的共同努力]下",
        "在[零上40度高温|超低温环境]中|在[干燥|潮湿|风尘]等各个环境中": "[在[零上40度高温|超低温环境]中|在[干燥|潮湿|风尘]等各个环境中]",
        "从美国的[民主|宪法]|美国社会的问题|美国的移民历史|美国人的生活习惯]": "从[美国的[民主|宪法]|美国社会的问题|美国的移民历史|美国人的生活习惯]",
        "以[文字|彩图]]": "以[文字|彩图]",
    }
    for sample in data:
        text = sample["natural"]
        if text in text_fix_map:
            sample["natural"] = text_fix_map[text]
        for spo in sample["logic"]:
            # fix subject
            if spo["subject"] in subj_fix_map:
                spo["subject"] = subj_fix_map[spo["subject"]]
            # fix predicate
            spo["predicate"] = spo["predicate"].strip()
            spo["predicate"] = re.sub("OR", "|", spo["predicate"])
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
            # fix qualifier
            if spo["qualifier"] in qua_fix_map:
                spo["qualifier"] = qua_fix_map[spo["qualifier"]]

            # fix by text
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
        target_str = re.sub("[\]\[\|]", "", spe_txt)
        if is_pred:
            target_str = re.sub("([^a-zA-Z]|^)[XYZU]([^a-zA-Z]|$)", r"\1\2", target_str)

        spans, _ = Preprocessor.search_char_spans_fr_txt(target_str, text, "ch")
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

    predefined_p_map = {"DESC": "描述",
                        "ISA": "是一种",
                        "IN": "位于",
                        "BIRTH": "生于",
                        "DEATH": "死于",
                        "=": "等于",
                        "NOT": "不",
                        }

    new_data = []
    bad_spo_list = []
    for sample_id, sample in tqdm(enumerate(data), desc="transform"):
        ori_sample = copy.deepcopy(sample)
        sample["natural"] = sample["natural"] + "[SEP]" + "，".join(predefined_p_map.values())
        text = sample["natural"]

        new_spo_list = []
        for spo in sample["logic"]:
            # trans predicate
            for k in predefined_p_map:
                if k in spo["predicate"]:  # and spo["predicate"] not in text
                    new_predicate = re.sub(k, predefined_p_map[k], spo["predicate"])
                    # print("{} -> {}".format(spo["predicate"], new_predicate))
                    spo["predicate"] = new_predicate

            ori_spo = copy.deepcopy(spo)
            split = False
            for key in spo:
                if spo[key] == "_":
                    spo[key] = []
                elif key != "object" and key != "objects":
                    if re.search(".*\[.*\|.*\].*", spo[key]):  # need to split
                        ori_str = spo[key]
                        split = True
                        split_list = parse_spe_txt2list(ori_str)
                        span_list = get_spe_txt_spans(ori_str, text, is_pred=True if key == "predicate" else False)

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
                        comb_list = [{"char_span": sp, "text": split_list[idx]} for idx, sp in enumerate(span_list)]
                        spo[key] = comb_list
                    else:
                        char_sp, _ = Preprocessor.search_char_spans_fr_txt(spo[key], text, "ch")
                        spo[key] = [
                            {"text": spo[key],
                             "char_span": char_sp,
                             }, ]
                elif key == "object":
                    new_objs = []
                    for obj in spo[key]:
                        if re.search(".*\[.*\|.*\].*", obj):  # need to split
                            ori_str = obj
                            split_list = parse_spe_txt2list(obj)
                            span_list = get_spe_txt_spans(ori_str, text, is_pred=False)

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
                            comb_list = [{"char_span": sp, "text": split_list[idx]} for idx, sp in enumerate(span_list)]
                            new_objs.append(comb_list)
                            split = True
                            # print(text)
                            # print(spo)
                            # print(ori_str)
                            # print(split_list)
                            # print("============split {}=============".format(key))
                        else:
                            if obj == "_":
                                pass
                            else:
                                char_sp, _ = Preprocessor.search_char_spans_fr_txt(obj, text, "ch")
                                new_objs.append([
                                    {"text": obj,
                                     "char_span": char_sp,
                                     }, ])
                    spo[key] = new_objs

            for p in spo["predicate"]:
                if re.search("[XYZU]", p["text"]) is None and len(spo["object"]) > 0:
                    p["text"] += "X"

            # align predicate and the corresponding subset of objects (by XYZU)
            ext_spo_list = []
            id_map = {"X": 0, "Y": 1, "Z": 2, "U": 3}

            bad_spo = False
            for p in spo["predicate"]:
                sub_objs = []
                for ph, idx in id_map.items():
                    if ph in p["text"]:
                        if ph == "U" and "UNIX" in p["text"] or \
                                ph == "U" and "MOTOBLUR" in p["text"] or \
                                ph == "Y" and "DKNY" in p["text"] or \
                                ph == "Y" and "LYF" in p["text"] or \
                                ph == "Z" and "SK-1Z02D" in p["text"]:
                            continue
                        try:
                            sub_objs.append(spo["object"][idx])
                        except Exception:
                            bad_spo = True
                            bad_spo_list.append({
                                "text": text,
                                "bad_spo": ori_spo,
                                "ori_sample": ori_sample,
                            })
                if bad_spo:
                    break
                p["text"] = re.sub("[XYZU]", "[OBJ]", p["text"])
                new_spo = copy.deepcopy(spo)
                new_spo["predicate"] = [p, ]
                new_spo["object"] = sub_objs
                ext_spo_list.append(new_spo)
            if len(spo["predicate"]) == 0:
                assert len(spo["object"]) <= 1
                ext_spo_list.append(spo)

            # product
            open_spo_list = []
            for new_spo in ext_spo_list:
                lists4prod = []
                for k, l in new_spo.items():
                    if k in {"object", "objects"} or len(l) == 0:
                        continue
                    # try:
                    lists4prod.append([{"type": k, **i} for i in l])
                    # except Exception:
                    #     # print("!")
                for objs in new_spo["object"]:
                    new_objs = []
                    for i in objs:
                        # try:
                        new_objs.append({"type": "object", **i})
                        # except Exception:
                        #     # print("!")
                    lists4prod.append(new_objs)

                open_spo_list.extend([list(item) for item in itertools.product(*lists4prod)])

            new_spo_list.extend(open_spo_list)
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
    train_num = int(len(new_data) * 0.8)
    valid_num = int(len(new_data) * 0.1)
    test_num = len(new_data) - train_num - valid_num
    random.shuffle(new_data)
    train_data = new_data[:train_num]
    valid_data = new_data[train_num:train_num + valid_num]
    test_data = new_data[-test_num:]

    save_as_json_lines(train_data, "../../data/ori_data/saoke/train_data.json")
    save_as_json_lines(valid_data, "../../data/ori_data/saoke/valid_data.json")
    save_as_json_lines(test_data, "../../data/ori_data/saoke/test_data.json")

if __name__ == "__main__":
    pass
