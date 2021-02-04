import json
import os
from InfExtraction.modules.utils import load_data, save_as_json_lines
import random
from tqdm import tqdm
import re
import jieba


def clean_entity(ent):
    ent = re.sub("�", "", ent)
    return ent.strip()


def split_duee(random_seed=2333):
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

    random.seed(random_seed)
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


def split_duie(random_seed=2333):
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

    random.seed(random_seed)
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