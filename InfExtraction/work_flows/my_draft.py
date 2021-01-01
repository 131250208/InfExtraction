import json
import os
from InfExtraction.modules.preprocess import Preprocessor, WhiteWordTokenizer
from InfExtraction.work_flows import settings_tplinker_pp as settings
from InfExtraction.modules.utils import load_data, save_as_json_lines
from tqdm import tqdm
import random
from tqdm import tqdm
from pprint import pprint
import copy
import re
import jieba

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


def trans_openie():
    data_in_dir = "../../data/ori_data/oie4_bk"
    train_filename = "openie4_labels"
    train_path = os.path.join(data_in_dir, train_filename)
    with open(train_path, "r", encoding="utf-8") as file_in:
        text = None
        words = None
        tag_lines = []
        data = []
        for line in file_in:
            if "ARG" in line or "REL" in line:
                tag_lines.append(line.split(" "))
            else:
                if text is not None:
                    data.append({
                        "text": text,
                        "word_list": words,
                        "tag_lines": tag_lines
                    })
                    tag_lines = []
                text = line
                words = line.split(" ")
        if text is not None:
            data.append({
                "text": text,
                "word_list": words,
                "tag_lines": tag_lines
            })

    for sample in data:
        open_spo_list = []
        for tags in sample["tag_lines"]:
            type2indices = {}
            for idx, tag in enumerate(tags):
                if tag not in type2indices:
                    type2indices[tag] = []
                type2indices[tag].append(idx)
            open_spo_list.append(type2indices)
        sample["open_spo_list"] = open_spo_list

def clean_entity(ent):
    ent = re.sub("�", "", ent)
    return ent.strip()


def preprocess_duee():
    data_in_dir = "../../data/ori_data/duie_1_bk"
    data_out_dir = "../../data/ori_data/duie_1"
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    train_filename = "train_data.json"
    valid_filename = "dev_data.json"
    train_path = os.path.join(data_in_dir, train_filename)
    dev_path = os.path.join(data_in_dir, valid_filename)
    with open(train_path, "r", encoding="utf-8") as file_in:
        train_data = [json.loads(line) for line in file_in]
    with open(dev_path, "r", encoding="utf-8") as file_in:
        dev_data = [json.loads(line) for line in file_in]


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
            sample["text"] = "影片信息电视剧影片名称：舞动芝加哥第二季  影片类型：欧美剧  影片语言：英语  上映年份：2012 演员表剧情介绍美国芝加哥，单亲女孩CeCe（Bella Thorne饰）和闺蜜Rocky（Zendaya Coleman饰）原本只是两个爱跳舞的普通初中生"
            sample["postag"] = [{"word": w, } for w in jieba.cut(sample["text"])]

        if text in {"于卫涛，1976年出生于河南省通许县，2013年携手刘洋创办河南欣赏网络科技集团，任该集团董事长，专注于我国大中小型企业提供专业的网络服务，带动很多企业网络方向的转型",
                    "1962年周明牂在中国植物保护学会成立大会上，作了《我国害虫农业防治研究现状和展望》的学术报告，1963年在《人民日报》上发表《结合耕作防治害虫》一文"}:
            new_spo_list = [spo for spo in sample["spo_list"] if spo["predicate"] != "所属专辑"]
            sample["spo_list"] = new_spo_list

        for spo in sample["spo_list"]:
            if spo["predicate"] in {"专业代码", "邮政编码"} and text[re.search(spo["object"], text).span()[0] - 1] == "0":
                spo["object"] = "0" + spo["object"]
                # print(text)
                # pprint(sample["spo_list"])
                # print("====================================")
            # strip redundant whitespaces and unknown characters
            spo["subject"] = clean_entity(spo["subject"])
            spo["object"] = clean_entity(spo["object"])
        # if text == "人物简介陈胜功，男，工学博士，山东省平度市人，北京航空航天大学自动化科学与电气工程学院副教授、硕士研究生导师（硕士学科： 080202机械电子工程）":
        #     for spo in sample["spo_list"]:
        #         if spo["object"] == "80202":
        #             spo["object"] = "080202"
        # if text == "0810信息与通信工程":
        #     for spo in sample["spo_list"]:
        #         if spo["object"] == "810":
        #             spo["object"] = "0810"

        new_spo_list = []
        for spo in sample["spo_list"]:
            if spo["subject"].lower() not in text.lower() or spo["object"].lower() not in text.lower():
                # print(text)
                # pprint(spo)
                # print("===========================================")

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


def preprocess_cadec():
    data_in_dir = "../../data/ori_data/cadec_bk"
    data_out_dir = "../../data/ori_data/cadec"
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    train_filename = "train.txt"
    valid_filename = "dev.txt"
    test_filename = "test.txt"

    train_path = os.path.join(data_in_dir, train_filename)
    valid_path = os.path.join(data_in_dir, valid_filename)
    test_path = os.path.join(data_in_dir, test_filename)

    def load_data(path):
        with open(path, "r", encoding="utf-8") as file_in:
            lines = [line.strip("\n") for line in file_in]
            data = []
            for i in range(0, len(lines), 3):
                sample = lines[i: i+3]
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
                    for idx in range(0, len(offsets), 2):
                        wd_sp = [offsets[idx], offsets[idx + 1]]
                        ch_sp_list = word2char_span[wd_sp[0]:wd_sp[1]]
                        ch_sp = [ch_sp_list[0][0], ch_sp_list[-1][1]]
                        seg_wd = " ".join(word_list[wd_sp[0]: wd_sp[1]])
                        seg_ch = text[ch_sp[0]:ch_sp[1]]
                        assert seg_ch == seg_wd

                        char_span.extend(ch_sp)
                        extr_segs.append(seg_ch)
                    ent = {
                        "text": " ".join(extr_segs),
                        "type": ent_type,
                        "char_span": char_span,
                        # "extr_segs": extr_segs,
                    }
                    ent_list.append(ent)
                    if len(char_span) > 4:
                        print(len(char_span)) # 6, 8
                data.append({
                    "text": sample[0],
                    "word_list": word_list,
                    "word2char_span": word2char_span,
                    "entity_list": ent_list,
                })
        return data

    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    test_data = load_data(test_path)
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

if __name__ == "__main__":
    # import networkx as nx
    #
    # G = nx.Graph()
    # cl1 = [1, 2, 3, 4]
    # cl2 = [4, 5, 6]
    # edges = [(i, j) for i in cl1 for j in cl1 if i != j]
    # edges.extend([(i, j) for i in cl2 for j in cl2 if i != j])
    # G.add_edges_from(edges)
    # print(G.number_of_nodes())
    # print(list(G.adj[4]))
    # print(list(nx.find_cliques(G)))

    from itertools import permutations
    # st = "我是王雨城"
    # st_per = permutations(st)
    # idx_per = permutations(list(range(len(st))))
    # print(list(st_per))
    # print(list(idx_per))

    data_path = "../../data/ori_data/saoke/saoke.json"
    data = load_data(data_path)
    predefined_p = set()
    for sample in data:
        text = sample["natural"]
        for spo in sample["logic"]:
            predicate = spo["predicate"]
            if predicate != "_" and predicate not in text and re.search("[XYZU]", predicate) is None:
                # print(text)
                # print(predicate)
                # print("=============================")
                if re.match("[A-Z]+$", predicate):
                    predefined_p.add(predicate)
                else:
                    try:
                        pattern = "(" + ").*(".join(list(predicate)) + ")"
                        se = re.search(pattern, text)
                        if se is not None:
                            spans = []
                            for i in range(len(predicate)):
                                spans.extend([*se.span(i + 1)])
                            # merge
                            new_spans = []
                            for pos in spans:
                                if len(new_spans) == 0 or pos != new_spans[-1]:
                                    new_spans.append(pos)
                                else:
                                    new_spans.pop()
                        else:
                            idx_permu = list(permutations(list(range(len(predicate)))))
                            ch_permu = list(permutations(predicate))
                            fin_ids = [0] * len(predicate)

                            for idx, ch_p in enumerate(ch_permu):
                                if "".join(ch_p) == "年平均气温一月":
                                    print("!")
                                new_pattern = "(" + ").*(".join(ch_p) + ")"
                                se = re.search(new_pattern, text)
                                if se is not None:
                                    ids = []
                                    map_ = idx_permu[idx]
                                    for i in range(len(predicate)):
                                        ids.append(se.span(i + 1)[0])
                                    for idx_j, pos in enumerate(ids):
                                        fin_ids[map_[idx_j]] = pos
                                    break
                            new_spans = []
                            pre = -10
                            for idx in fin_ids:
                                if idx - 1 != pre:
                                    new_spans.append(pre + 1)
                                    new_spans.append(idx)
                                pre = idx
                            new_spans.append(pre + 1)
                            new_spans = new_spans[1:]
                            print("!")
                    except Exception:
                        print(text)
                        print(predicate)
                        print("=============================")


