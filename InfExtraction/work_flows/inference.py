'''
Train the model

# load data
# Split the data
# Index the data
# Tag the data
# Init model
# Put the data into a Dataloaders
# Set optimizer and trainer
'''

import sys, getopt
import importlib
import copy

# settings
try:
    opts, args = getopt.getopt(sys.argv[1:], "s:", ["settings="])
except getopt.GetoptError:
    print('train_valid.py -s <settings_file> --settings <settings_file>')
    sys.exit(2)

settings_name = "settings_default"
for opt, arg in opts:
    if opt in ("-s", "--settings"):
        settings_name = arg

module_name = "InfExtraction.work_flows.{}".format(settings_name)
settings = importlib.import_module(module_name)

import os
import wandb
from pprint import pprint
import logging
import re
from glob import glob
import numpy as np
from shutil import copyfile
import torch
from torch.utils.data import DataLoader
from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.modules import taggers, models
from InfExtraction.modules.workers import Trainer, Evaluator
from InfExtraction.modules.metrics import MetricsCalculator
from InfExtraction.modules.utils import (DefaultLogger,
                                         save_as_json_lines,
                                         MyMappingDataset,
                                         MyIterableDataset,
                                         MyLargeFileReader,
                                         MyLargeJsonlinesFileReader,
                                         )
import itertools
from tqdm import tqdm
import math
import json


def worker_init_fn4map(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def worker_init_fn4iter(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset  # the dataset copy in this worker process
        overall_start = dataset.start
        overall_end = dataset.end
        per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        dataset.start = overall_start + worker_id * per_worker
        dataset.end = min(dataset.start + per_worker, overall_end)


def prepare_data(data,
                 language,
                 do_lower_case,
                 data_type,
                 token_level,
                 max_seq_len,
                 sliding_len,
                 combine,
                 key2dict,
                 task_type,
                 wdp_prefix=None,
                 max_char_num_in_tok=None,
                 split_early_stop=True,
                 drop_neg_samples=False,
                 additional_preprocess=None,
                 ):
    data = Preprocessor.choose_features_by_token_level_gen(data, token_level, do_lower_case)
    data = Preprocessor.choose_spans_by_token_level_gen(data, token_level)

    if additional_preprocess is not None:
        data = additional_preprocess(data, data_type)

    data = Preprocessor.split_into_short_samples(data,
                                                 max_seq_len,
                                                 sliding_len,
                                                 data_type,
                                                 token_level,
                                                 task_type,
                                                 wordpieces_prefix=wdp_prefix,
                                                 early_stop=split_early_stop,
                                                 drop_neg_samples=drop_neg_samples)

    if combine:
        data = Preprocessor.combine(data, max_seq_len)

    # inexing
    indexed_data = Preprocessor.index_features(data,
                                               language,
                                               key2dict,
                                               max_seq_len,
                                               max_char_num_in_tok)

    return list(indexed_data)


def get_dataloader(indexed_data, batch_size, collate_fn):
    if type(indexed_data) is list:
        dataloader = DataLoader(MyMappingDataset(indexed_data),
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                drop_last=False,
                                collate_fn=collate_fn,
                                worker_init_fn=worker_init_fn4map,
                                )
    else:
        assert type(indexed_data) is MyLargeFileReader
        indexed_data.shuffle_line_offsets()
        jsreader = MyLargeJsonlinesFileReader(indexed_data)

        # dataloader
        dataloader = DataLoader(MyIterableDataset(jsreader, shuffle=True),
                                batch_size=batch_size,
                                # shuffle=True,
                                num_workers=2,
                                drop_last=False,
                                collate_fn=collate_fn,
                                worker_init_fn=worker_init_fn4iter,
                                )

    return dataloader


def get_score_fr_path(model_path):
    return float(re.search("_([\d\.]+)\.pt", model_path.split("/")[-1]).group(1))


def run():
    # task
    exp_name = settings.exp_name
    task_type = settings.task_type
    run_name = settings.run_name
    model_name = settings.model_name
    tagger_name = settings.tagger_name
    language = settings.language

    # >>>>>>>>>>>>>>>> data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data_out_dir = settings.data_out_dir

    train_data = settings.train_data
    opt_filename2test_data = settings.filename2test_data
    filename2test_data, filename2test_data_dup = {}, {}
    for filename, opt_test_data in opt_filename2test_data.items():
        test_data, test_data_dup = itertools.tee(opt_test_data, 2)
        filename2test_data[filename] = test_data
        filename2test_data_dup[filename] = test_data_dup
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    key2dict = settings.key2dict  # map from feature key to indexing dict

    # additonal preprocessing config
    try:
        addtional_preprocessing_config = settings.addtional_preprocessing_config
    except Exception as e:
        addtional_preprocessing_config = {}

    # tagger config
    try:
        tagger_config = settings.tagger_config
    except:
        tagger_config = {}


    # training settings
    token_level = settings.token_level

    batch_size_test = settings.batch_size_test
    max_seq_len_test = settings.max_seq_len_test
    sliding_len_test = settings.sliding_len_test

    combine = settings.combine
    split_early_stop = settings.split_early_stop
    drop_neg_samples = settings.drop_neg_samples

    # test settings
    model_dir_for_test = settings.model_dir_for_test
    target_run_ids = settings.target_run_ids
    model_path_ids2infer = settings.model_path_ids2infer
    metric4testing = settings.metric4testing
    cal_scores = settings.cal_scores

    # model settings
    model_settings = settings.model_settings

    wdp_prefix = None
    if token_level == "subword":
        wdp_prefix = model_settings["subwd_encoder_config"]["wordpieces_prefix"]
    max_char_num_in_tok = None
    if "char_encoder_config" in model_settings and model_settings["char_encoder_config"] is not None:
        max_char_num_in_tok = model_settings["char_encoder_config"]["max_char_num_in_tok"]

    do_lower_case = False
    if "subwd_encoder_config" in model_settings and \
            model_settings["subwd_encoder_config"] is not None and \
            "do_lower_case" in model_settings["subwd_encoder_config"]:
        do_lower_case = model_settings["subwd_encoder_config"]["do_lower_case"]

    # env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_golden_data_must(data):
        must_keys4golden_data = {"id", "text", "relation_list", "entity_list", "event_list", "open_spo_list"}
        new_data = []
        for sample in tqdm(data, "get golden data"):
            # Preprocessor.choose_features_by_token_level4sample(sample, token_level, do_lower_case)
            Preprocessor.choose_spans_by_token_level4sample(sample, token_level)
            new_sample = {}
            for key in sample.keys():
                if key in must_keys4golden_data:
                    new_sample[key] = sample[key]
            new_data.append(new_sample)
        return new_data

    #  copy ori data, maintain clean for evaluation
    filename2ori_test_data = {filename: get_golden_data_must(test_data_dup)
                              for filename, test_data_dup in filename2test_data_dup.items()}

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # tagger
    tagger_class_name = getattr(taggers, tagger_name)
    if task_type == "re+tee":
        tagger_class_name = taggers.create_rebased_ee_tagger(tagger_class_name)
    elif task_type == "re+oie":
        tagger_class_name = taggers.create_rebased_oie_tagger(tagger_class_name)
    elif task_type == "re+disc_ner":
        tagger_class_name = taggers.create_rebased_discontinuous_ner_tagger(tagger_class_name)
    elif task_type == "re+tfboys":
        tagger_class_name = taggers.create_rebased_tfboys_tagger(tagger_class_name)
    elif task_type == "re+tfboys4doc_ee":
        tagger_class_name = taggers.create_rebased_tfboys4doc_ee_tagger(tagger_class_name)


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # choose spans
    train_data = Preprocessor.choose_spans_by_token_level_gen(train_data, token_level)
    # additional preprocessing
    train_data = tagger_class_name.additional_preprocess(train_data, **addtional_preprocessing_config)
    # get anns
    data_anns = {}
    for sample in tqdm(train_data, desc="collect data anns from train data"):
        for k, v in sample.items():
            if k in {"entity_list", "relation_list", "event_list", "open_spo_list"}:
                data_anns.setdefault(k, []).extend(v)

    # init tagger
    tagger = tagger_class_name(data_anns, **tagger_config)

    # init metrics_calculator
    metrics_cal = MetricsCalculator(use_ghm=False)

    # init model
    print("init model...")
    model_class_name = getattr(models, model_name)
    model = model_class_name(tagger, metrics_cal, **model_settings)
    model = model.to(device)
    print("done!")
    print(">>>>>>>>>>>>>>>>>>>>>>>>> Model >>>>>>>>>>>>>>>>>>>>>>>")
    print(model)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # function for generating data batch
    collate_fn = model.generate_batch

    # evaluator
    evaluator = Evaluator(model, device)

    filename2test_data_loader = {}
    for filename, test_data in filename2test_data.items():
        indexed_test_data = prepare_data(test_data,
                                         language,
                                         do_lower_case,
                                         "test",
                                         token_level,
                                         max_seq_len_test,
                                         sliding_len_test,
                                         combine,
                                         key2dict,
                                         task_type,
                                         wdp_prefix,
                                         max_char_num_in_tok,
                                         split_early_stop,
                                         drop_neg_samples,
                                         )
        filename2test_data_loader[filename] = get_dataloader(indexed_test_data, batch_size_test, collate_fn)


    # get model state paths
    target_run_ids = set(target_run_ids)
    assert model_dir_for_test is not None and model_dir_for_test.strip() != "", "Please set model state directory!"

    run_id2model_state_paths = {}
    for root, dirs, files in os.walk(model_dir_for_test):
        for file_name in files:
            path_se = re.search("run-\d{8}_\d{6}-(\w{8})\/.*?(val_.*)", root)
            if path_se is None:
                continue
            run_id = path_se.group(1)
            metric = path_se.group(2)
            if metric == "val_{}".format(metric4testing) \
                    and run_id in target_run_ids \
                    and re.match(".*model_state.*\.pt", file_name):
                if run_id not in run_id2model_state_paths:
                    run_id2model_state_paths[run_id] = []
                model_state_path = os.path.join(root, file_name)
                run_id2model_state_paths[run_id].append(model_state_path)

    # predicting
    run_id2scores = {}
    for run_id, model_path_list in run_id2model_state_paths.items():
        # only top k models
        sorted_model_path_list = sorted(model_path_list, key=get_score_fr_path)
        model_path_list = []
        for idx in model_path_ids2infer:
            model_path_list.append(sorted_model_path_list[idx])

        for path in model_path_list:
            # load model
            model.load_state_dict(torch.load(path))
            print("model state loaded: {}".format("/".join(path.split("/")[-2:])))
            model.eval()
            model_name = re.sub("\.pt", "", path.split("/")[-1])
            save_dir = os.path.join(data_out_dir, exp_name, run_name, run_id, model_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # test
            for filename, test_data_loader in filename2test_data_loader.items():
                gold_test_data = filename2ori_test_data[filename]
                # predicate
                pred_samples = evaluator.predict(test_data_loader, gold_test_data)

                num = len(glob(save_dir + "/*.json"))
                # save results
                save_as_json_lines(pred_samples, os.path.join(save_dir, "{}_{}".format(num, filename)))

                # score
                if cal_scores:
                    score_dict = evaluator.score(pred_samples, gold_test_data)
                    if run_id not in run_id2scores:
                        run_id2scores[run_id] = {}
                    if model_name not in run_id2scores[run_id]:
                        run_id2scores[run_id][model_name] = {}
                    run_id2scores[run_id][model_name][filename] = score_dict

    if cal_scores:
        pprint(run_id2scores)


if __name__ == "__main__":
    run()
