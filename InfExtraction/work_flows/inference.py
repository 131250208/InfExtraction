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

# settings
try:
    opts, args = getopt.getopt(sys.argv[1:], "s:", ["settings="])
except getopt.GetoptError:
    print('test.py -s <settings_file> --settings <settings_file>')
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
import numpy as np

import torch
from torch.utils.data import DataLoader
from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.modules import taggers
from InfExtraction.modules import models
from InfExtraction.modules.workers import Evaluator
from InfExtraction.modules.metrics import MetricsCalculator
from InfExtraction.modules.utils import MyDataset, save_as_json_lines, load_data
import json


def get_dataloader(data,
                   language,
                   data_type,
                   token_level,
                   max_seq_len,
                   sliding_len,
                   combine,
                   batch_size,
                   key2dict,
                   tagger,
                   collate_fn,
                   max_char_num_in_tok=None,
                   ):
    # split test data
    data = Preprocessor.split_into_short_samples(data,
                                                 max_seq_len,
                                                 sliding_len,
                                                 data_type,
                                                 token_level)

    if combine:
        data = Preprocessor.combine(data, max_seq_len)

    # check spans
    sample_id2mismatched = Preprocessor.check_spans(data, language)
    if len(sample_id2mismatched) > 0:
        logging.warning("mismatch errors in {}".format(data_type))
        pprint(sample_id2mismatched)

    # inexing
    indexed_data = Preprocessor.index_features(data,
                                               language,
                                               key2dict,
                                               max_seq_len,
                                               max_char_num_in_tok)
    # tagging
    indexed_data = tagger.tag(indexed_data)

    # dataloader
    dataloader = DataLoader(MyDataset(indexed_data),
                            batch_size=batch_size,
                            shuffle=True,
                            # num_workers=0,
                            drop_last=False,
                            collate_fn=collate_fn,
                            worker_init_fn=worker_init_fn
                            )
    return dataloader


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_score_fr_path(model_path):
    return float(re.search("_([\d\.]+)\.pt", model_path.split("/")[-1]).group(1))


def get_last_k_paths(path_list, k):
    path_list = sorted(path_list, key=get_score_fr_path)
    return path_list[-k:]


if __name__ == "__main__":
    # task
    exp_name = settings.exp_name
    task_type = settings.task_type
    run_name = settings.run_name
    model_name = settings.model_name
    tagger_name = settings.tagger_name
    language = settings.language

    # data
    data_in_dir = settings.data_in_dir
    data_out_dir = settings.data_out_dir
    # data
    data_in_dir = settings.data_in_dir
    data_out_dir = settings.data_out_dir
    train_data = load_data(settings.train_data_path)
    filename2test_data = {}
    for test_data_path in settings.test_data_path_list:
        filename = test_data_path.split("/")[-1]
        test_data = load_data(test_data_path)
        filename2test_data[filename] = test_data

    dicts = settings.dicts
    statistics = settings.statistics
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

    device_num = settings.device_num
    token_level = settings.token_level
    batch_size_test = settings.batch_size_test
    max_seq_len_test = settings.max_seq_len_test
    sliding_len_test = settings.sliding_len_test
    combine = settings.combine

    model_dir_for_test = settings.model_dir_for_test
    target_run_ids = settings.target_run_ids
    model_path_ids2infer = settings.model_path_ids2infer
    metric4testing = settings.metric4testing
    cal_scores = settings.cal_scores

    # model settings
    model_settings = settings.model_settings
    
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

    train_data = Preprocessor.choose_spans_by_token_level(train_data, token_level)
    for filename, test_data in filename2test_data.items():
        filename2test_data[filename] = Preprocessor.choose_features_by_token_level(test_data, token_level, do_lower_case)
        filename2test_data[filename] = Preprocessor.choose_spans_by_token_level(test_data, token_level)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # tagger
    tagger_class_name = getattr(taggers, tagger_name)
    if task_type == "re+ee":
        tagger_class_name = taggers.create_rebased_ee_tagger(tagger_class_name)
    elif task_type == "re+oie":
        tagger_class_name = taggers.create_rebased_oie_tagger(tagger_class_name)
    elif task_type == "re+ner":
        tagger_class_name = taggers.create_rebased_discontinuous_ner_tagger(tagger_class_name)
    elif task_type == "re+tfboys":
        tagger_class_name = taggers.create_rebased_tfboys_tagger(tagger_class_name)

    # additional preprocessing
    def additional_preprocess(data, data_type):
        return tagger_class_name.additional_preprocess(data, data_type, **addtional_preprocessing_config)

    train_data = additional_preprocess(train_data, "train")

    all_data4gen_tag_dict = []
    all_data4gen_tag_dict.extend(train_data)

    # tagger
    tagger = tagger_class_name(all_data4gen_tag_dict, **tagger_config)

    # model
    print("init model...")
    model_class_name = getattr(models, model_name)
    model = model_class_name(tagger, **model_settings)
    model = model.to(device)
    print("done!")

    # function for generating data batch
    collate_fn = model.generate_batch
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # trainer and evaluator
    evaluator = Evaluator(model, device)

    # test dataloader
    filename2test_data_loader = {}
    for filename, test_data in filename2test_data.items():
        test_dataloader = get_dataloader(test_data,
                                         language,
                                         "test",
                                         token_level,
                                         max_seq_len_test,
                                         sliding_len_test,
                                         combine,
                                         batch_size_test,
                                         key2dict,
                                         tagger,
                                         collate_fn,
                                         max_char_num_in_tok,
                                         )
        filename2test_data_loader[filename] = test_dataloader


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
                gold_test_data = filename2test_data[filename]
                # predicate
                pred_samples = evaluator.predict(test_data_loader, gold_test_data)

                # save results
                save_as_json_lines(pred_samples, os.path.join(save_dir, filename))

                # score
                if cal_scores:
                    score_dict, error_analysis_dict = evaluator.score(pred_samples, gold_test_data)
                    json.dump(error_analysis_dict,
                              open(os.path.join(save_dir, "error_analysis_dict.json"), "w", encoding="utf-8"),
                              ensure_ascii=False
                              )
                    if run_id not in run_id2scores:
                        run_id2scores[run_id] = {}
                    if model_name not in run_id2scores[run_id]:
                        run_id2scores[run_id][model_name] = {}
                    run_id2scores[run_id][model_name][filename] = score_dict

    if cal_scores:
        pprint(run_id2scores)
