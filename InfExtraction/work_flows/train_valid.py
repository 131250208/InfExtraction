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

from InfExtraction.modules.preprocess import Preprocessor, MyDataset
from InfExtraction.modules import taggers
from InfExtraction.modules import models
from InfExtraction.modules.taggers import HandshakingTaggerEE, MatrixTaggerEE
from InfExtraction.modules.workers import Trainer, Evaluator
from InfExtraction.modules.models import TPLinkerPlus, TriggerFreeEventExtractor
from InfExtraction.work_flows import settings_train_val_test as settings
from InfExtraction.work_flows.utils import DefaultLogger

import os
import torch
import wandb
import json
from pprint import pprint
from torch.utils.data import DataLoader
import logging
import re
from glob import glob


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
                   wdp_prefix=None,
                   max_char_num_in_tok=None,
                   ):
    # split test data
    data = Preprocessor.split_into_short_samples(data,
                                                       max_seq_len,
                                                       sliding_len,
                                                       data_type,
                                                       token_level=token_level,
                                                       wordpieces_prefix=wdp_prefix)

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
                            )
    return dataloader


def get_score_fr_path(model_path):
    return float(re.search("_([\d\.]+)\.pt", model_path.split("/")[-1]).group(1))


if __name__ == "__main__":
    # task
    exp_name = settings.exp_name
    task_type = settings.task_type
    run_name = settings.run_name
    model_name = settings.model_name
    tagger_name = settings.tagger_name

    # data
    data_in_dir = settings.data_in_dir
    train_data_path = os.path.join(data_in_dir, exp_name, settings.train_data)
    valid_data_path = os.path.join(data_in_dir, exp_name, settings.valid_data)
    dicts = settings.dicts
    statistics = settings.statistics
    key2dict = settings.key2dict # map from feature key to indexing dict

    # logger settings
    use_wandb = settings.use_wandb
    config2log = settings.config_to_log
    default_log_path = settings.default_log_path
    default_run_id = settings.default_run_id
    default_dir_to_save_model = settings.default_dir_to_save_model
    log_interval = settings.log_interval

    # training settings
    device_num = settings.device_num
    token_level = settings.token_level
    seed = settings.seed
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

    trainer_config = settings.trainer_config
    lr = settings.lr
    model_state_dict_path = settings.model_state_dict_path # pretrained model state

    # save model
    score_threshold = settings.score_threshold
    model_bag_size = settings.model_bag_size

    # model settings
    model_settings = settings.model_settings

    wdp_prefix = None
    if token_level == "subword":
        wdp_prefix = model_settings["subwd_encoder_config"]["wordpieces_prefix"]
    max_char_num_in_tok=None
    if "char_encoder_config" in model_settings and model_settings["char_encoder_config"] is not None:
        max_char_num_in_tok = model_settings["char_encoder_config"]["max_char_num_in_tok"]

    # env
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)  # pytorch random seed
    torch.backends.cudnn.deterministic = True  # for reproductivity

    # reset settings from args
    # ...

    # load data
    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
    filename2test_data = {}
    for filename in settings.test_data_list:
        test_data_path = os.path.join(data_in_dir, exp_name, filename)
        test_data = json.load(open(test_data_path, "r", encoding="utf-8"))
        filename2test_data[filename] = test_data

    # choose features and spans by token level
    train_data = Preprocessor.choose_features_by_token_level(train_data, token_level)
    train_data = Preprocessor.choose_spans_by_token_level(train_data, token_level)
    valid_data = Preprocessor.choose_features_by_token_level(valid_data, token_level)
    valid_data = Preprocessor.choose_spans_by_token_level(valid_data, token_level)
    for filename, test_data in filename2test_data.items():
        filename2test_data[filename] = Preprocessor.choose_features_by_token_level(test_data, token_level)
        filename2test_data[filename] = Preprocessor.choose_spans_by_token_level(test_data, token_level)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    all_data = train_data + valid_data
    for filename, test_data in filename2test_data.items():
        all_data.extend(test_data)
    # tagger
    tagger_class_name = getattr(taggers, tagger_name)
    tagger = tagger_class_name(all_data)  # HandshakingTaggerEE
    tag_size = tagger.get_tag_size()
    # model
    print("init model...")
    model_class_name = getattr(models, model_name)
    model = model_class_name(tag_size, **model_settings) # TPLinkerPlus
    model = model.to(device)
    print("done!")
    # function for generating data batch
    collate_fn = model.generate_batch
    # optional: additional preprocessing on relation/entity/events
    train_data = tagger.additional_preprocess(train_data)
    valid_data = tagger.additional_preprocess(valid_data)
    for filename, test_data in filename2test_data.items():
        filename2test_data[filename] = tagger.additional_preprocess(test_data)
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

    # dataloader
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
                                      wdp_prefix,
                                      max_char_num_in_tok,
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
                                      wdp_prefix,
                                      max_char_num_in_tok,
                                      )
    # # # have a look at dataloader
    # train_data_iter = iter(train_dataloader)
    # batch_data = next(train_data_iter)
    # print(batch_data)

    filename2test_data_loader = {}
    for filename, test_data in filename2test_data.items():
        filename2test_data_loader[filename] = get_dataloader(test_data,
                                                             "test",
                                                             token_level,
                                                             max_seq_len_test,
                                                             sliding_len_test,
                                                             combine,
                                                             batch_size_test,
                                                             key2dict,
                                                             tagger,
                                                             collate_fn,
                                                             wdp_prefix,
                                                             max_char_num_in_tok,
                                                             )

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    # load pretrained model
    if model_state_dict_path is not None:
        model.load_state_dict(torch.load(model_state_dict_path))
        print("model state loaded: {}".format("/".join(model_state_dict_path.split("/")[-2:])))

    # trainer
    trainer = Trainer(model, tagger, device, optimizer, trainer_config, logger)
    evaluator = Evaluator(task_type, model, tagger, device, match_pattern=None)

    # debug: checking data and decoding
    pprint(evaluator.check_tagging_n_decoding(valid_dataloader, valid_data))

    # train and eval
    best_val_score = 0.
    for ep in range(epochs):
        # train
        trainer.train(train_dataloader, ep, epochs)
        # valid
        pred_samples = evaluator.predict(valid_dataloader, valid_data)
        score_dict = evaluator.score(pred_samples, valid_data, "val")
        logger.log(score_dict)
        dataset2score_dict = {
            "valid_data.json": score_dict,
        }
        current_val_fin_score = score_dict["{}_{}".format("val", "trigger_class_f1")] # 将trigger_class_f1设为参数

        # test
        for filename, test_data_loader in filename2test_data_loader.items():
            gold_test_data = filename2test_data[filename]
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
