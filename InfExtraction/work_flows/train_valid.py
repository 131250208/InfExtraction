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
from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.modules import taggers
from InfExtraction.modules import models
from InfExtraction.modules.workers import Trainer, Evaluator
from InfExtraction.modules.metrics import MetricsCalculator
from InfExtraction.modules.utils import DefaultLogger, MyDataset

import os
import sys, getopt
import torch
import wandb
import json
from pprint import pprint
from torch.utils.data import DataLoader
import logging
import re
from glob import glob
import numpy as np
import importlib


def get_dataloader(data,
                   data_type,
                   token_level,
                   max_seq_len,
                   sliding_len,
                   combine,
                   batch_size,
                   key2dict,
                   tagger,
                   collate_fn,
                   task_type,
                   wdp_prefix=None,
                   max_char_num_in_tok=None,
                   split_early_stop=True
                   ):
    # split test data
    data = Preprocessor.split_into_short_samples(data,
                                                 max_seq_len,
                                                 sliding_len,
                                                 data_type,
                                                 token_level,
                                                 task_type,
                                                 wordpieces_prefix=wdp_prefix,
                                                 early_stop=split_early_stop)

    if combine:
        data = Preprocessor.combine(data, max_seq_len)

    # check spans
    sample_id2mismatched = Preprocessor.check_spans(data)
    if len(sample_id2mismatched) > 0:
        logging.warning("mismatch errors in {}".format(data_type))
        pprint(sample_id2mismatched)
    # check decoding

    # inexing
    indexed_data = Preprocessor.index_features(data,
                                               key2dict,
                                               max_seq_len,
                                               max_char_num_in_tok)
    # tagging
    indexed_data = tagger.tag(indexed_data)

    # dataloader
    dataloader = DataLoader(MyDataset(indexed_data),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
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

    # task
    exp_name = settings.exp_name
    task_type = settings.task_type
    run_name = settings.run_name
    model_name = settings.model_name
    tagger_name = settings.tagger_name
    stage = settings.stage

    # data
    data_in_dir = settings.data_in_dir
    data_out_dir = settings.data_out_dir
    train_data_path = settings.train_data
    valid_data_path = settings.valid_data
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

    # logger settings
    use_wandb = settings.use_wandb
    config2log = settings.config_to_log
    default_log_path = settings.default_log_path
    default_run_id = settings.default_run_id
    default_dir_to_save_model = settings.default_dir_to_save_model
    log_interval = settings.log_interval

    # training settings
    check_tagging_n_decoding = settings.check_tagging_n_decoding
    device_num = settings.device_num
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

    trainer_config = settings.trainer_config
    use_ghm = settings.use_ghm
    lr = settings.lr
    model_state_dict_path = settings.model_state_dict_path  # pretrained model state

    # optimizer
    optimizer_config = settings.optimizer_config

    # test settings
    model_dir_for_test = settings.model_dir_for_test
    target_run_ids = settings.target_run_ids
    top_k_models = settings.top_k_models
    cal_scores = settings.cal_scores

    # # match_pattern, only for relation extraction
    # match_pattern = settings.match_pattern if "re" in task_type else None

    # save model
    score_threshold = settings.score_threshold
    model_bag_size = settings.model_bag_size
    fin_score_key = settings.final_score_key

    # model settings
    model_settings = settings.model_settings

    wdp_prefix = None
    if token_level == "subword":
        wdp_prefix = model_settings["subwd_encoder_config"]["wordpieces_prefix"]
    max_char_num_in_tok = None
    if "char_encoder_config" in model_settings and model_settings["char_encoder_config"] is not None:
        max_char_num_in_tok = model_settings["char_encoder_config"]["max_char_num_in_tok"]

    # env
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # reset settings from args
    # ...

    # load data
    print("load data...")
    ori_train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    ori_valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
    filename2ori_test_data = {}
    for test_data_path in settings.test_data_list:
        filename = test_data_path.split("/")[-1]
        ori_test_data = json.load(open(test_data_path, "r", encoding="utf-8"))
        filename2ori_test_data[filename] = ori_test_data
    print("done!")

    # choose features and spans by token level
    ori_train_data = Preprocessor.choose_features_by_token_level(ori_train_data, token_level)
    ori_train_data = Preprocessor.choose_spans_by_token_level(ori_train_data, token_level)
    ori_valid_data = Preprocessor.choose_features_by_token_level(ori_valid_data, token_level)
    ori_valid_data = Preprocessor.choose_spans_by_token_level(ori_valid_data, token_level)
    for filename, test_data in filename2ori_test_data.items():
        filename2ori_test_data[filename] = Preprocessor.choose_features_by_token_level(test_data, token_level)
        filename2ori_test_data[filename] = Preprocessor.choose_spans_by_token_level(test_data, token_level)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # tagger
    tagger_class_name = getattr(taggers, tagger_name)
    if task_type == "re+ee":
        tagger_class_name = taggers.create_rebased_ee_tagger(tagger_class_name)
    elif task_type == "re+ner":
        tagger_class_name = taggers.create_rebased_ner_tagger(tagger_class_name)

    # additional preprocessing
    def additional_preprocess(data, data_type):
        return tagger_class_name.additional_preprocess(data, data_type, **addtional_preprocessing_config)
    
    all_data4gen_tag_dict = []
    train_data = additional_preprocess(ori_train_data, "train")
    all_data4gen_tag_dict.extend(train_data)
    valid_data = additional_preprocess(ori_valid_data, "valid")
    all_data4gen_tag_dict.extend(additional_preprocess(ori_valid_data, "train"))
    filename2test_data = {}
    for filename, ori_test_data in filename2ori_test_data.items():
        filename2test_data[filename] = additional_preprocess(ori_test_data, "test")
        all_data4gen_tag_dict.extend(additional_preprocess(ori_test_data, "train"))

    # tagger
    tagger = tagger_class_name(all_data4gen_tag_dict, **tagger_config)

    # metrics_calculator
    metrics_cal = MetricsCalculator(task_type,
                                    # match_pattern,
                                    use_ghm)

    # model
    print("init model...")
    model_class_name = getattr(models, model_name)
    model = model_class_name(tagger, metrics_cal, **model_settings)
    model = model.to(device)
    print("done!")

    # function for generating data batch
    collate_fn = model.generate_batch
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # logger
    if use_wandb:
        # init wandb
        wandb.init(project=exp_name, name=run_name, config=config2log)
        dir_to_save_model = wandb.run.dir
        logger = wandb
    else:
        logger = DefaultLogger(default_log_path,
                               exp_name,
                               run_name,
                               default_run_id,
                               config2log)
        dir_to_save_model = default_dir_to_save_model
        if not os.path.exists(dir_to_save_model):
            os.makedirs(dir_to_save_model)

    # trainer and evaluator
    evaluator = Evaluator(model, device)

    # test dataloader
    filename2test_data_loader = {}
    for filename, test_data in filename2test_data.items():
        test_dataloader = get_dataloader(test_data,
                                         "test",
                                         token_level,
                                         max_seq_len_test,
                                         sliding_len_test,
                                         combine,
                                         batch_size_test,
                                         key2dict,
                                         tagger,
                                         collate_fn,
                                         task_type,
                                         wdp_prefix,
                                         max_char_num_in_tok,
                                         split_early_stop
                                         )
        filename2test_data_loader[filename] = test_dataloader

    if stage == "train":
        train_dataloader = get_dataloader(train_data,
                                          "train",
                                          token_level,
                                          max_seq_len_train,
                                          sliding_len_train,
                                          combine,
                                          batch_size_train,
                                          key2dict,
                                          tagger,
                                          collate_fn,
                                          task_type,
                                          wdp_prefix,
                                          max_char_num_in_tok,
                                          split_early_stop
                                          )
        valid_dataloader = get_dataloader(valid_data,
                                          "valid",
                                          token_level,
                                          max_seq_len_valid,
                                          sliding_len_valid,
                                          combine,
                                          batch_size_valid,
                                          key2dict,
                                          tagger,
                                          collate_fn,
                                          task_type,
                                          wdp_prefix,
                                          max_char_num_in_tok,
                                          split_early_stop
                                          )
        # debug: checking tagging and decoding
        if check_tagging_n_decoding:
            # for checking, take valid data as train data, do additional preprocessing
            # but take original valid data as golden dataset to evaluate
            valid_data4checking = additional_preprocess(ori_valid_data, "train")
            valid_dataloader4checking = get_dataloader(valid_data4checking,
                                                       "train",  # only train data will be set a tag sequence
                                                       token_level,
                                                       max_seq_len_valid,
                                                       sliding_len_valid,
                                                       combine,
                                                       batch_size_valid,
                                                       key2dict,
                                                       tagger,
                                                       collate_fn,
                                                       task_type,
                                                       wdp_prefix,
                                                       max_char_num_in_tok,
                                                       split_early_stop
                                                       )
            pprint(evaluator.check_tagging_n_decoding(valid_dataloader4checking, ori_valid_data))

        # load pretrained model
        if model_state_dict_path is not None:
            model.load_state_dict(torch.load(model_state_dict_path))
            print("model state loaded: {}".format("/".join(model_state_dict_path.split("/")[-2:])))

        # optimizer
        optimizer_class_name = getattr(torch.optim, optimizer_config["class_name"])
        optimizer = optimizer_class_name(model.parameters(), lr=float(lr), **optimizer_config["parameters"])

        trainer = Trainer(model, train_dataloader, device, optimizer, trainer_config, logger)
        # train and valid
        best_val_score = 0.
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
            current_val_fin_score = score_dict["{}_{}".format("val", fin_score_key)]

            # test
            for filename, test_data_loader in filename2test_data_loader.items():
                gold_test_data = filename2ori_test_data[filename]
                pred_samples = evaluator.predict(test_data_loader, gold_test_data)
                score_dict = evaluator.score(pred_samples, gold_test_data, filename.split(".")[0])
                logger.log(score_dict)
                dataset2score_dict[filename] = score_dict

            pprint(dataset2score_dict)

            if current_val_fin_score > score_threshold:
                # save model state
                torch.save(model.state_dict(),
                           os.path.join(dir_to_save_model,
                                        "model_state_dict_{}_{:.5}.pt".format(ep, current_val_fin_score * 100)))

                # all state paths
                model_state_path_list = glob("{}/model_state_*".format(dir_to_save_model))
                # sorted by scores
                sorted_model_state_path_list = sorted(model_state_path_list,
                                                      key=get_score_fr_path)
                # best score in the bag
                model_path_max_score = sorted_model_state_path_list[-1]
                best_val_score = get_score_fr_path(model_path_max_score)
                # only save <model_bag_size> model states
                if len(sorted_model_state_path_list) > model_bag_size:
                    os.remove(sorted_model_state_path_list[0])  # drop the state with minimum score
            print("Current val score: {:.5}, Best val score: {:.5}".format(current_val_fin_score * 100, best_val_score))

    elif stage == "inference":
        # get model state paths
        target_run_ids = set(target_run_ids)
        assert model_dir_for_test is not None and model_dir_for_test.strip() != ""
        run_id2model_state_paths = {}
        for root, dirs, files in os.walk(model_dir_for_test):
            for file_name in files:
                run_id = root[-8:]
                if re.match(".*model_state.*\.pt", file_name) and run_id in target_run_ids:
                    if run_id not in run_id2model_state_paths:
                        run_id2model_state_paths[run_id] = []
                    model_state_path = os.path.join(root, file_name)
                    run_id2model_state_paths[run_id].append(model_state_path)

        # predicting
        run_id2scores = {}
        for run_id, model_path_list in run_id2model_state_paths.items():

            # only top k models
            model_path_list = get_last_k_paths(model_path_list, top_k_models)
            for path in model_path_list:
                # load model
                model.load_state_dict(torch.load(path))
                print("model state loaded: {}".format("/".join(path.split("/")[-2:])))
                model.eval()
                model_name = re.sub("\.pt", "", path.split("/")[-1])
                save_dir = os.path.join(data_out_dir, task_type, run_id, model_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # test
                for filename, test_data_loader in filename2test_data_loader.items():
                    gold_test_data = filename2test_data[filename]
                    # predicate
                    pred_samples = evaluator.predict(test_data_loader, gold_test_data)

                    # save results
                    json.dump(pred_samples, open(os.path.join(save_dir, filename), "w", encoding="utf-8"))

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
