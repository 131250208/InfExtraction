import json
import os
from InfExtraction.modules.preprocess import Preprocessor, WhiteWordTokenizer
from InfExtraction.work_flows import settings_tplinker_pp as settings
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


def preprocess_saoke():
    data_path = "../../data/ori_data/saoke/saoke.json"
    data = load_data(data_path)
    predefined_p = set()

    # fix data
    pred_fix_map = {
        "[蒸发和过热]受热面": "[蒸发|过热]受热面",
        "[高稳产农田地面积|人均高稳产农田地面积": "[高稳产农田地面积|人均高稳产农田地面积]",
        "[减弱OR消失]": "[减弱|消失]",
        "整体[规划|营建|": "整体[规划|营建]",
        "争先奔走|": "争先奔走",
        "拟订|组织实施]": "[拟订|组织实施]",
        "规定|": "规定",
        "以“董氏大宗祠图”|": "以“董氏大宗祠图”|",
        "健全||": "健全",
        "转移|培训": "[转移|培训]",
        "被国务院确确定为": "被国务院确定为",
        "聚众|": "聚众",
    }
    for sample in data:
        for spo in sample["logic"]:
            spo["predicate"] = spo["predicate"].strip()
            if spo["predicate"] in pred_fix_map:
                spo["predicate"] = pred_fix_map[spo["predicate"]]
    for sample in tqdm(data):
        text = sample["natural"]
        for spo in sample["logic"]:
            predicate = spo["predicate"]
            if predicate != "_" and predicate not in text and re.search("[XYZU]", predicate) is None:
                if re.match("[A-Z=]+$", predicate):
                    predefined_p.add(predicate)
                elif re.search("[\[\]\|]", predicate):
                    print(text)
                    print(predicate)
                    print(spo)
                    print("=============================")
                else:
                    spans = None
                    pattern = "(" + ").*(".join(list(predicate)) + ")"
                    se = re.search(pattern, text)
                    if se is not None:
                        spans = []
                        for i in range(len(predicate)):
                            try:
                                spans.extend([*se.span(i + 1)])
                            except:
                                print("1")
                    else:
                        spe_patt = "A-Za-z0-9\.~!@#\$%^&\*()_\+,\?:'\""
                        segs = re.findall("[{}]+|[^{}]+".format(spe_patt, spe_patt), predicate)
                        sbwd_list = []
                        for seg in segs:
                            if re.match("[{}]+".format(spe_patt), seg):
                                sbwd_list.append(seg)
                            else:
                                sbwd_list.extend(list(jieba.cut(seg, cut_all=True)))
                                sbwd_list += list(seg)
                        sbwd_list = list(set(sbwd_list))

                        m_list = []
                        for sbwd in sorted(sbwd_list, key=lambda w: len(w), reverse=True):
                            for m in re.finditer(re.escape(sbwd), text):
                                m_list.append(m)

                        m_list = sorted(m_list, key=lambda m: m.span()[1] - m.span()[0], reverse=True)
                        seg_list = [m.group() for m in m_list]
                        sps = [0] * len(predicate)
                        pred_cp = predicate[:]
                        for m in m_list:
                            se = re.search(re.escape(m.group()), pred_cp)
                            if se is not None:
                                star_idx = se.span()[0]
                                sp = se.span()
                                sps[star_idx] = [*m.span()]
                                pred_ch_list = list(pred_cp)
                                pred_ch_list[sp[0]:sp[1]] = ["_"] * (sp[1] - sp[0])
                                pred_cp = "".join(pred_ch_list)
                        spans = []
                        for sp in sps:
                            if sp != 0:
                                spans.extend(sp)

                    # merge
                    new_spans = []
                    for pos in spans:
                        if len(new_spans) == 0 or pos != new_spans[-1]:
                            new_spans.append(pos)
                        else:
                            new_spans.pop()
                    pred_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(new_spans, text, "ch")
                    try:
                        assert pred_extr == predicate
                    except Exception:
                        print(pred_extr)
                        print(predicate)
                        print(text)
                        print(spo)
                        print("=============================")


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
                alt_words_list = [[w] for w in words_list[0]] if len(words_list) == 1 else itertools.product(*words_list)

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


if __name__ == "__main__":
    pass