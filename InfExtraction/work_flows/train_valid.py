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
                                         MyMappingDataset,
                                         MyIterableDataset,
                                         save_as_json_lines,
                                         MyLargeFileReader,
                                         MyLargeJsonlinesFileReader,
                                         )
import itertools
from tqdm import tqdm
import math
import atexit
import hashlib
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
                 load_data2memory,
                 language,
                 do_lower_case,
                 data_type,
                 token_level,
                 max_seq_len,
                 sliding_len,
                 combine,
                 run_id,
                 key2dict,
                 task_type,
                 wdp_prefix=None,
                 max_char_num_in_tok=None,
                 split_early_stop=True,
                 drop_neg_samples=False,
                 additional_preprocess=None,
                 ):
    # if combine and data_type == "train":
    #     data = Preprocessor.combine(data, 1024)

    data = Preprocessor.choose_features_by_token_level_gen(data, token_level, do_lower_case)
    data = Preprocessor.choose_spans_by_token_level_gen(data, token_level)

    if additional_preprocess is not None:
        data = additional_preprocess(data)

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

    # # check spans
    # sample_id2mismatched = Preprocessor.check_spans(data, language)
    # if len(sample_id2mismatched) > 0:
    #     logging.warning("mismatch errors in {}".format(data_type))
    #     pprint(sample_id2mismatched)

    # inexing
    indexed_data = Preprocessor.index_features(data,
                                               language,
                                               key2dict,
                                               max_seq_len,
                                               max_char_num_in_tok)
    # # tagging
    # indexed_data = tagger.tag(indexed_data)
    data_anns = {}
    if load_data2memory:
        if data_type == "train":
            res_indexed_data = []
            for sample in tqdm(indexed_data, desc="prepare n load into memory: {}".format(data_type)):
                for k, v in sample.items():
                    if k in {"entity_list", "relation_list", "event_list", "open_spo_list"}:
                        data_anns.setdefault(k, []).extend(v)
                res_indexed_data.append(sample)
        else:
            res_indexed_data = list(indexed_data)
    else:
        cache_dir = "../../data/cache"
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        cache_file_path = os.path.join(cache_dir, "indexed_{}_data_cache_{}.jsonlines".format(data_type, run_id))

        with open(cache_file_path, "w", encoding="utf-8") as out_file:
            for sample in tqdm(indexed_data, desc="prepare n save to disk: {}".format(data_type)):
                if data_type == "train":
                    for k, v in sample.items():
                        if k in {"entity_list", "relation_list", "event_list", "open_spo_list"}:
                            data_anns.setdefault(k, []).extend(v)
                line = json.dumps(sample, ensure_ascii=False)
                out_file.write("{}\n".format(line))

        res_indexed_data = MyLargeFileReader(cache_file_path, shuffle=True)
    return res_indexed_data, data_anns


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

    opt_train_data = settings.train_data
    opt_valid_data = settings.valid_data
    opt_data4checking = settings.data4checking
    opt_filename2test_data = settings.filename2test_data

    train_data, train_data4gen_tags = itertools.tee(opt_train_data, 2)
    data4checking, data4checking_dup = itertools.tee(opt_data4checking, 2)
    valid_data, valid_data_dup = itertools.tee(opt_valid_data, 2)
    filename2test_data, filename2test_data_dup = {}, {}
    for filename, opt_test_data in opt_filename2test_data.items():
        test_data, test_data_dup = itertools.tee(opt_test_data, 2)
        filename2test_data[filename] = test_data
        filename2test_data_dup[filename] = test_data_dup

    load_data2memory = settings.load_data2memory
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

    # logger settings
    use_wandb = settings.use_wandb
    config2log = settings.config_to_log
    default_log_path = settings.default_log_path
    default_run_id = settings.default_run_id
    default_dir_to_save_model = settings.default_dir_to_save_model

    # training settings
    check_tagging_n_decoding = settings.check_tagging_n_decoding
    token_level = settings.token_level
    epochs = settings.epochs
    batch_size_train = settings.batch_size_train
    max_seq_len_train = settings.max_seq_len_train
    sliding_len_train = settings.sliding_len_train

    batch_size_valid = settings.batch_size_valid
    max_seq_len_valid = settings.max_seq_len_valid
    sliding_len_valid = settings.sliding_len_valid

    batch_size_test = settings.batch_size_test
    max_seq_len_test = settings.max_seq_len_test
    sliding_len_test = settings.sliding_len_test

    combine = settings.combine
    split_early_stop = settings.split_early_stop
    drop_neg_samples = settings.drop_neg_samples

    trainer_config = settings.trainer_config
    use_ghm = settings.use_ghm
    lr = settings.lr
    model_state_dict_path = settings.model_state_dict_path  # pretrained model state

    # optimizer
    optimizer_config = settings.optimizer_config

    # save model
    metric_keyword = settings.metric_keyword
    model_bag_size = settings.model_bag_size

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
    ori_valid_data = get_golden_data_must(valid_data_dup)
    ori_data4checking = get_golden_data_must(data4checking_dup)
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

    # additional preprocessing
    def additional_preprocess(data):
        return tagger_class_name.additional_preprocess(data, **addtional_preprocessing_config)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # logger
    if use_wandb:
        # init wandb
        wandb.init(project=exp_name, name=run_name, config=config2log)
        run_id = wandb.run.id
        dir_to_save_model = wandb.run.dir
        logger = wandb
    else:
        run_id = default_run_id
        logger = DefaultLogger(default_log_path,
                               exp_name,
                               run_name,
                               default_run_id,
                               config2log)
        dir_to_save_model = default_dir_to_save_model
        if not os.path.exists(dir_to_save_model):
            os.makedirs(dir_to_save_model)
    # save settings #@
    copyfile("./{}.py".format(settings_name), os.path.join(dir_to_save_model, "{}.py".format(settings_name)))

    # index train data
    indexed_train_data, train_data_anns = prepare_data(train_data,
                                      load_data2memory,
                                      language,
                                      do_lower_case,
                                      "train",
                                      token_level,
                                      max_seq_len_train,
                                      sliding_len_train,
                                      combine,
                                      run_id,
                                      key2dict,
                                      task_type,
                                      wdp_prefix,
                                      max_char_num_in_tok,
                                      split_early_stop,
                                      drop_neg_samples,
                                      additional_preprocess,
                                      )

    def train_dataloader_fn():
        return get_dataloader(indexed_train_data, batch_size_train, collate_fn)

    # init tagger
    tagger = tagger_class_name(train_data_anns, **tagger_config)

    # init metrics_calculator
    metrics_cal = MetricsCalculator(use_ghm)

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

    indexed_valid_data, _ = prepare_data(valid_data,
                                      load_data2memory,
                                      language,
                                      do_lower_case,
                                      "valid",
                                      token_level,
                                      max_seq_len_valid,
                                      sliding_len_valid,
                                      combine,
                                      run_id,
                                      key2dict,
                                      task_type,
                                      wdp_prefix,
                                      max_char_num_in_tok,
                                      split_early_stop,
                                      drop_neg_samples
                                      )
    valid_dataloader = get_dataloader(indexed_valid_data, batch_size_valid, collate_fn)

    filename2test_data_loader = {}
    for filename, test_data in filename2test_data.items():
        indexed_test_data, _ = prepare_data(test_data,
                                         load_data2memory,
                                         language,
                                         do_lower_case,
                                         "test",
                                         token_level,
                                         max_seq_len_test,
                                         sliding_len_test,
                                         combine,
                                         run_id,
                                         key2dict,
                                         task_type,
                                         wdp_prefix,
                                         max_char_num_in_tok,
                                         split_early_stop,
                                         drop_neg_samples,
                                         )
        filename2test_data_loader[filename] = get_dataloader(indexed_test_data, batch_size_test, collate_fn)

    # debug: checking tagging and decoding
    if check_tagging_n_decoding:
        indexed_debug_data, _ = prepare_data(data4checking,
                                          load_data2memory,
                                          language,
                                          do_lower_case,
                                          "debug",
                                          token_level,
                                          max_seq_len_valid,
                                          sliding_len_valid,
                                          combine,
                                          run_id,
                                          key2dict,
                                          task_type,
                                          wdp_prefix,
                                          max_char_num_in_tok,
                                          split_early_stop,
                                          drop_neg_samples,
                                          additional_preprocess,
                                          )
        dataloader4checking = get_dataloader(indexed_debug_data, batch_size_valid, collate_fn)
        pprint(evaluator.check_tagging_n_decoding(dataloader4checking, ori_data4checking))

    # load pretrained model
    if model_state_dict_path is not None:
        model.load_state_dict(torch.load(model_state_dict_path))
        print("model state loaded: {}".format("/".join(model_state_dict_path.split("/")[-2:])))

    # optimizer
    optimizer_class_name = getattr(torch.optim, optimizer_config["class_name"])
    optimizer = optimizer_class_name(model.parameters(), lr=float(lr), **optimizer_config["parameters"])

    trainer = Trainer(model, train_dataloader_fn, device, optimizer, trainer_config, logger)

    # train and valid
    score_dict4comparing = {}
    for ep in range(epochs):
        # train
        trainer.train(ep, epochs)
        # valid
        pred_samples = evaluator.predict(valid_dataloader, ori_valid_data)
        score_dict = evaluator.score(pred_samples, ori_valid_data, "val")
        logger.log(score_dict)
        dataset2score_dict = {
            "valid_data.json": score_dict,
        }

        for metric_key, current_val_score in score_dict.items():
            if metric_keyword not in metric_key:
                continue

            if metric_key not in score_dict4comparing:
                score_dict4comparing[metric_key] = {
                    "current": 0.0,
                    "best": 0.0,
                }
            score_dict4comparing[metric_key]["current"] = current_val_score
            score_dict4comparing[metric_key]["best"] = max(current_val_score,
                                                           score_dict4comparing[metric_key]["best"])

            # save models
            if current_val_score > 0. and model_bag_size > 0:
                dir_to_save_model_this_key = os.path.join(dir_to_save_model, metric_key)
                if not os.path.exists(dir_to_save_model_this_key):
                    os.makedirs(dir_to_save_model_this_key)

                # save model state
                torch.save(model.state_dict(),
                           os.path.join(dir_to_save_model_this_key,
                                        "model_state_dict_{}_{:.5}.pt".format(ep, current_val_score * 100)))

                # all state paths
                model_state_path_list = glob("{}/model_state_*".format(dir_to_save_model_this_key))
                # sorted by scores
                sorted_model_state_path_list = sorted(model_state_path_list,
                                                      key=get_score_fr_path)
                # only save <model_bag_size> model states
                if len(sorted_model_state_path_list) > model_bag_size:
                    os.remove(sorted_model_state_path_list[0])  # remove the state dict with the minimum score

        # test
        for filename, test_data_loader in filename2test_data_loader.items():
            gold_test_data = filename2ori_test_data[filename]
            pred_samples = evaluator.predict(test_data_loader, gold_test_data)
            score_dict = evaluator.score(pred_samples, gold_test_data, filename.split(".")[0])
            logger.log(score_dict)
            dataset2score_dict[filename] = score_dict

        pprint(dataset2score_dict)
        pprint(score_dict4comparing)


if __name__ == "__main__":
    run()
