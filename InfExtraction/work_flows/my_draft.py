import json
import os
from InfExtraction.modules.preprocess import Preprocessor, WhiteWordTokenizer
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

def get_add_seg(type_set, seed):
    # types
    random.seed(seed)
    types = list(type_set)
    random.shuffle(types)

    additional_seg = ""
    type2char_span = {}
    add_tok_list = []
    add_entities = []
    for tp in types:
        if additional_seg != "":
            additional_seg += " ; "
            add_tok_list.append(";")
        star_idx = len(additional_seg)
        additional_seg += tp
        end_idx = star_idx + len(tp)
        type2char_span[tp] = [star_idx, end_idx]
        add_tok_list.append(tp)
        ent = {
            "text": tp,
            "type": "NER:Ontology",
            "char_span": [star_idx, end_idx],
        }
        add_entities.append(ent)

    add_tok_list.append("[SEP]")
    additional_seg += " [SEP] "
    return additional_seg, add_tok_list, type2char_span, add_entities


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
                        tok_span = [ent[0], ent[1] + 1]
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
            filename2data[out_filename] = out_data

    type_set = {ent["type"] for data in filename2data.values() for sample in data for ent in sample["entity_list"]}

    # generate new data
    for filename, data in filename2data.items():
        out_path = os.path.join(data_out_dir, filename)
        for idx, sample in enumerate(data):
            additional_seg, add_tok_list, type2char_span, add_entities = get_add_seg(type_set, idx)

            sample["text"] = additional_seg + sample["text"]
            sample["word_list"] = add_tok_list + sample["word_list"]
            new_ent_list = []
            new_ent_list.extend(sample["entity_list"])
            new_ent_list.extend(add_entities)
            rel_list = []
            for ent in sample["entity_list"]:
                obj = {
                    "text": ent["type"],
                    "type": "NER:Ontology",
                    "char_span": type2char_span[ent["type"]],
                }
                ent["char_span"] = [ent["char_span"][0] + len(additional_seg),
                                    ent["char_span"][1] + len(additional_seg)]
                assert obj["text"] == sample["text"][obj["char_span"][0]:obj["char_span"][1]]
                assert ent["text"] == sample["text"][ent["char_span"][0]:ent["char_span"][1]]

                rel_list.append({
                    "subject": ent["text"],
                    "subj_char_span": ent["char_span"],
                    "object": obj["text"],
                    "obj_char_span": obj["char_span"],
                    "predicate": "isA",
                })
            sample["entity_list"] = new_ent_list
            sample["relation_list"] = rel_list
        json.dump(data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


def trans_genia2():
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


def preprocess_duee():
    data_in_dir = "../../data/ori_data/duee_1_bk"
    data_out_dir = "../../data/ori_data/duee_1"
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    train_filename = "train.json"
    valid_filename = "dev.json"
    train_path = os.path.join(data_in_dir, train_filename)
    dev_path = os.path.join(data_in_dir, valid_filename)
    train_data = load_data(train_path)
    dev_data = load_data(dev_path)

    random.seed(2333)
    random.shuffle(train_data)
    event_type2train_data = {}
    for sample in tqdm(train_data):
        for spo in sample["event_list"]:
            if spo["event_type"] not in event_type2train_data:
                event_type2train_data[spo["event_type"]] = []
            event_type2train_data[spo["event_type"]].append(sample)

    memory = set()
    valid_rate = 0.1
    val_num = int(len(train_data) * valid_rate)

    event_type2val_num = {}
    count = 0
    for sample in dev_data:
        for event in sample["event_list"]:
            event_type = event["event_type"]
            event_type2val_num[event_type] = event_type2val_num.get(event_type, 0) + 1
            count += 1
    event_type2val_rate = {et: vn/count for et, vn in event_type2val_num.items()}

    valid_data = []
    event_type2data_no_dup = {}
    event_type2data_sorteded = sorted(event_type2train_data.items(), key=lambda x: len(x[1]))
    for event_type, subset in event_type2data_sorteded:
        new_subset = []
        for sample in subset:
            if sample["id"] not in memory:
                new_subset.append(sample)
                memory.add(sample["id"])
        event_type2data_no_dup[event_type] = new_subset

    for et, subset in event_type2data_no_dup.items():
        vr_et = event_type2val_rate[et]
        sub_val_num = int(val_num * vr_et) + 1
        random.shuffle(subset)
        valid_data.extend(subset[:sub_val_num])
    valid_data = valid_data[:val_num]  # may exceed val_num because of +1, so truncate first val_num samples
    # val_data_event_type2event_list = {}
    # for sample in valid_data:
    #     for event in sample["event_list"]:
    #         event_type = event["event_type"]
    #         if event_type not in val_data_event_type2event_list:
    #             val_data_event_type2event_list[event_type] = []
    #         val_data_event_type2event_list[event_type].append(event)

    valid_id_set = {sample["id"] for sample in valid_data}
    new_train_data = [sample for sample in train_data if sample["id"] not in valid_id_set]
    test_data = dev_data

    train_sv_path = os.path.join(data_out_dir, "train_data.json")
    valid_sv_path = os.path.join(data_out_dir, "valid_data.json")
    test_sv_path = os.path.join(data_out_dir, "test_data.json")
    save_as_json_lines(new_train_data, train_sv_path)
    save_as_json_lines(valid_data, valid_sv_path)
    save_as_json_lines(test_data, test_sv_path)

    return new_train_data, valid_data, test_data


def preprocess_duie():
    data_in_dir = "../../data/ori_data/duie_1_bk"
    data_out_dir = "../../data/ori_data/duie_1"
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    train_filename = "train_data.json"
    valid_filename = "dev_data.json"
    train_path = os.path.join(data_in_dir, train_filename)
    dev_path = os.path.join(data_in_dir, valid_filename)

    train_data = load_data(train_path)
    dev_data = load_data(dev_path)

    random.seed(2333)
    random.shuffle(train_data)
    rel2data = {}
    for ind, sample in tqdm(enumerate(train_data)):
        sample["id"] = ind
        for spo in sample["spo_list"]:
            if spo["predicate"] not in rel2data:
                rel2data[spo["predicate"]] = []
            rel2data[spo["predicate"]].append(sample)

    memory = set()
    valid_rate = 20000 / len(train_data)
    valid_data = []

    rel2data_no_dup = {}
    rel2data_ordered = sorted(rel2data.items(), key=lambda x: len(x[1]))
    for rel, subset in tqdm(rel2data_ordered):
        new_subset = []
        for sample in subset:
            if sample["id"] not in memory:
                new_subset.append(sample)
                memory.add(sample["id"])
        rel2data_no_dup[rel] = new_subset

    for subset in tqdm(rel2data_no_dup.values()):
        sub_val_num = int(len(subset) * valid_rate) + 1
        valid_data.extend(subset[:sub_val_num])
    valid_data = valid_data[:20000]
    valid_id_set = {sample["id"] for sample in valid_data}
    new_train_data = [sample for sample in train_data if sample["id"] not in valid_id_set]

    test_data = dev_data

    # fix data
    for sample in tqdm(new_train_data + valid_data + test_data):
        text = sample["text"]

        if text == "2  朱美音21967出生在江西南昌，1989年毕业于江西科技师范大学外语系英语专业，后来分配至鹰潭铁路一中任英语教师":
            sample["text"] = "2  朱美音 1967出生在江西南昌，1989年毕业于江西科技师范大学外语系英语专业，后来分配至鹰潭铁路一中任英语教师"
            sample["postag"] = [{"word": w, } for w in jieba.cut(sample["text"])]
        if text == "人物简介王珊2，女，19441，1968年毕业于北京大学物理系":
            sample["text"] = "人物简介王珊2，女，1944，1968年毕业于北京大学物理系"
            sample["postag"] = [{"word": w, } for w in jieba.cut(sample["text"])]
        if text == "影片信息电视剧影片名称：舞动芝加哥第二季  影片类型：欧美剧  影片语言：英语  上映年份：20121演员表剧情介绍美国芝加哥，单亲女孩CeCe（Bella Thorne饰）和闺蜜Rocky（Zendaya Coleman饰）原本只是两个爱跳舞的普通初中生":
            sample[
                "text"] = "影片信息电视剧影片名称：舞动芝加哥第二季  影片类型：欧美剧  影片语言：英语  上映年份：2012 演员表剧情介绍美国芝加哥，单亲女孩CeCe（Bella Thorne饰）和闺蜜Rocky（Zendaya Coleman饰）原本只是两个爱跳舞的普通初中生"
            sample["postag"] = [{"word": w, } for w in jieba.cut(sample["text"])]

        if text in {"于卫涛，1976年出生于河南省通许县，2013年携手刘洋创办河南欣赏网络科技集团，任该集团董事长，专注于我国大中小型企业提供专业的网络服务，带动很多企业网络方向的转型",
                    "1962年周明牂在中国植物保护学会成立大会上，作了《我国害虫农业防治研究现状和展望》的学术报告，1963年在《人民日报》上发表《结合耕作防治害虫》一文"}:
            new_spo_list = [spo for spo in sample["spo_list"] if spo["predicate"] != "所属专辑"]
            sample["spo_list"] = new_spo_list

        for spo in sample["spo_list"]:
            if spo["predicate"] in {"专业代码", "邮政编码"} and text[re.search(spo["object"], text).span()[0] - 1] == "0":
                spo["object"] = "0" + spo["object"]
            # strip redundant whitespaces and unknown characters
            spo["subject"] = clean_entity(spo["subject"])
            spo["object"] = clean_entity(spo["object"])

        new_spo_list = []
        for spo in sample["spo_list"]:
            if spo["subject"].lower() not in text.lower() or spo["object"].lower() not in text.lower():
                # drop wrong spo
                continue

            # recover upper case
            if spo["subject"] not in text:
                m = re.search(re.escape(spo["subject"].lower()), text.lower())
                # print("{}----{}".format(spo["subject"], text[m.span()[0]:m.span()[1]]))
                spo["subject"] = text[m.span()[0]:m.span()[1]]

            if spo["object"] not in text:
                m = re.search(re.escape(spo["object"].lower()), text.lower())
                # print("{}----{}".format(spo["object"], text[m.span()[0]:m.span()[1]]))
                spo["object"] = text[m.span()[0]:m.span()[1]]
            new_spo_list.append(spo)

        sample["spo_list"] = new_spo_list

    # save
    save_as_json_lines(new_train_data, os.path.join(data_out_dir, "train_data.json"))
    save_as_json_lines(valid_data, os.path.join(data_out_dir, "valid_data.json"))
    save_as_json_lines(test_data, os.path.join(data_out_dir, "test_data.json"))


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
                # if len(extr_segs) >= 2:
                #     end_bd = extr_segs[0].split()[-1]
                #     start_bd = extr_segs[-1].split()[0]
                #     seg_start_bd.add(start_bd)
                #     seg_end_bd.add(end_bd)
                #     for sg_id in range(len(extr_segs) - 1):
                #         sg1 = extr_segs[sg_id]
                #         sg2 = extr_segs[sg_id + 1]
                #         end_bd = sg1.split()[-1]
                #         start_bd = sg2.split()[0]
                #         seg_bd.add(",".join([end_bd, start_bd]))
                #         seg_start_bd.add(start_bd)
                #         seg_end_bd.add(end_bd)

                assert ent_txt == ent_txt_extr
                ent = {
                    "text": ent_txt,
                    "type": ent_type,
                    "char_span": char_span,
                    "tok_span": tok_span,
                }
                ent_list.append(ent)

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


def preprocess_daixiang_data(data_in_dir):
    train_filename = "train.txt"
    valid_filename = "dev.txt"
    test_filename = "test.txt"

    train_path = os.path.join(data_in_dir, train_filename)
    valid_path = os.path.join(data_in_dir, valid_filename)
    test_path = os.path.join(data_in_dir, test_filename)

    train_data = trans_daixiang_data(train_path)
    valid_data = trans_daixiang_data(valid_path)
    test_data = trans_daixiang_data(test_path)
    return train_data, valid_data, test_data


def preproc_save_daixiang_data(data_in_dir="../../data/ori_data/share_14_bk", data_out_dir="../../data/ori_data/share_14"):
    train_data, valid_data, test_data = preprocess_daixiang_data(data_in_dir)

    for sample in train_data + valid_data + test_data:
        text = sample["text"]
        for ent in sample["entity_list"]:
            ori_char_span = ent["char_span"]
            ent["char_span"] = merge_spans(ori_char_span, "en", "char")
            ent_ori_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(ori_char_span, text, "en")
            ent_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(ent["char_span"], text, "en")
            assert ent_ori_extr == ent_extr == ent["text"]

    save_as_json_lines(train_data, os.path.join(data_out_dir, "train_data.json"))
    save_as_json_lines(valid_data, os.path.join(data_out_dir, "valid_data.json"))
    save_as_json_lines(test_data, os.path.join(data_out_dir, "test_data.json"))


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

        # for spo in sample["open_spo_list"]:
        #     # subj = Preprocessor.extract_ent_fr_txt_by_char_sp(spo["subject"]["char_span"], text, "en")
        #     core_predicate_extr = Preprocessor.extract_ent_fr_txt_by_char_sp(spo["predicate"]["char_span"], text, "en")
        #     if not spo["predicate"]["predefined"] and core_predicate_extr != spo["predicate"]["text"]:
        #         print("core error")
        #         print(text)
        #         print(core_predicate_extr)
        #         print(spo["predicate"]["text"])
        #         pprint(spo)
        #         print("==========================")
        #
        #     complete = " ".join([spo["predicate"]["prefix"], spo["predicate"]["text"], spo["predicate"]["suffix"]]).strip()
        #     if complete != spo["predicate"]["complete"]:
        #         print("complete error")
        #         print(text)
        #         print(complete)
        #         print(spo["predicate"]["complete"])
        #         pprint(spo)
        #         print("==========================")

def trans2dai_dataset():
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
    s = "string"
    s.span = [1, 2]

    pass
    # path = "../../data/ori_data/lsoie_data/lsoie_wiki_test.conll"
    # with open(path, "r", encoding="utf-8") as file_in:
    #     data = []
    #     word_list = []
    #     tag_list = []
    #     lines = []
    #     for line in file_in:
    #         line = line.strip("\n")
    #         line = line.strip()
    #         lines.append(line)
    #         if line != "":
    #             items = line.split("\t")
    #             word = items[1]
    #             tag = items[-1]
    #             word_list.append(word)
    #             tag_list.append(tag)
    #         else:
    #             data.append({
    #                 "text": " ".join(word_list),
    #                 "word_list": word_list,
    #                 "tag_list": tag_list,
    #                 "lines": lines,
    #             })
    #             word_list = []
    #             tag_list = []
    #             lines = []
    #     if len(word_list) > 0:
    #         data.append({
    #             "text": " ".join(word_list),
    #             "word_list": word_list,
    #             "tag_list": tag_list,
    #             "lines": lines,
    #         })
    #         word_list = []
    #         tag_list = []
    #         lines = []
    #
    #     count = 0
    #     final_data = []
    #     for idx, sample in enumerate(data):
    #         word_list = sample["word_list"]
    #         tag_list = sample["tag_list"]
    #         role_tags, bio_tags = [], []
    #         for tag in tag_list:
    #             tag_split = tag.split("-")
    #             role_tags.append(tag_split[0])
    #             bio_tags.append(tag_split[-1])
    #
    #         wrong_annotation = False
    #         argument_list = []
    #         for m in re.finditer("BI*", "".join(bio_tags)):
    #             try:
    #                 assert role_tags[m.span()[0]] == role_tags[m.span()[1] - 1]
    #             except Exception:
    #                 print("!")
    #                 count += 1
    #                 wrong_annotation = True
    #                 break
    #             role = role_tags[m.span()[0]]
    #             argument_list.append({
    #                 "text": " ".join(word_list[m.span()[0]:m.span()[1]]),
    #                 "word_span": [*m.span()],
    #                 "type": role,
    #             })
    #         if wrong_annotation:
    #             continue
    #         sample["id"] = idx
    #         sample["argument_list"] = argument_list
    #         final_data.append(sample)