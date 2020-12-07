'''
Evaluate the performance on the test set

# Load data
# Split the data
# Index the data
# Init model
# Put the data into a Dataloaders
# Predict and evaluate
'''
from InfExtraction.modules.preprocess import Preprocessor, MyDataset
from InfExtraction.modules.taggers import HandshakingTaggerEE4TPLPlus
from InfExtraction.modules.workers import Evaluator
from InfExtraction.modules.models import TPLinkerPlus
from InfExtraction.work_flows import settings_default as settings

import os
import torch
import json
from pprint import pprint
from torch.utils.data import DataLoader
import logging
import re


def get_data_loaders(test_data_dict, collate_fn):
    filename2data_loader = {}
    for filename, test_data in test_data_dict.items():
        # splitting
        split_test_data = Preprocessor.split_into_short_samples(test_data, max_seq_len, sliding_len, "test",
                                                                feature_list_key=feature_list_key)

        # indexing and padding
        key2dict = {
            "char_list": dicts["char2id"],
            "word_list": dicts["word2id"],
            "subword_list": dicts["bert_dict"],
            "pos_tag_list": dicts["pos_tag2id"],
            "ner_tag_list": dicts["ner_tag2id"],
            "dependency_list": dicts["deprel_type2id"],
        }
        pretrained_model_padding = model_settings[
            "pretrained_model_padding"] if "pretrained_model_padding" in model_settings else 0
        indexed_test_data = Preprocessor.index_features(split_test_data,
                                                        key2dict,
                                                        max_seq_len,
                                                        model_settings["char_encoder_config"]["max_char_num_in_tok"],
                                                        pretrained_model_padding)
        # dataset
        test_dataset = MyDataset(indexed_test_data)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     drop_last=False,
                                     collate_fn=collate_fn,
                                     )
        filename2data_loader[filename] = test_dataloader
    return filename2data_loader


def predict(evaluator, test_data_dict, filename2data_loader):
    filename2res = {}
    for filename, test_data in test_data_dict.items():
        # prediction
        test_dataloader = filename2data_loader[filename]
        final_pred_samples = evaluator.predict(test_dataloader, test_data)
        filename2res[filename] = final_pred_samples

    return filename2res


def score(evaluator, test_data_dict, filename2res):
    filename2score_dict = {}
    for filename, pred_samples in filename2res.items():
        test_data = test_data_dict[filename]
        score_dict = evaluator.score(pred_samples, test_data)
        filename2score_dict[filename] = score_dict
    return filename2score_dict


def get_score_fr_path(model_path):
    return float(re.search("_([\d\.]+)\.pt", model_path.split("/")[-1]).group(1))


def get_last_k_paths(path_list, k):
    path_list = sorted(path_list, key=get_score_fr_path)
    return path_list[-k:]


if __name__ == "__main__":
    exp_name = settings.exp_name

    # testing settings
    task_type = settings.task_type
    match_pattern = settings.match_pattern
    device_num = settings.device_num
    use_bert = settings.use_bert
    token_level = settings.token_level
    batch_size = settings.batch_size_test
    max_seq_len = settings.max_seq_len_test
    sliding_len = settings.sliding_len_test
    trainer_config = settings.trainer_config
    lr = settings.lr
    model_dir_for_test = settings.model_dir_for_test
    target_run_ids = settings.target_run_ids
    top_k_models = settings.top_k_models
    cal_scores = settings.cal_scores

    # data
    data_in_dir = os.path.join(settings.data_in_dir, exp_name)
    data_out_dir = settings.data_out_dir
    if data_out_dir is not None:
        data_out_dir = os.path.join(settings.data_out_dir, exp_name)
        if not os.path.exists(data_out_dir):
            os.makedirs(data_out_dir)

    test_data_dict = {}
    for filename in settings.test_data_list:
        test_data_path = os.path.join(data_in_dir, filename)
        test_data = json.load(open(test_data_path, "r", encoding="utf-8"))
        test_data = Preprocessor.choose_spans_by_token_level(test_data, token_level=token_level)
        test_data_dict[filename] = test_data
    dicts = settings.dicts
    statistics = settings.statistics

    # model settings
    model_settings = settings.model_settings

    # env
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True

    # splitting
    if use_bert:
        max_seq_len_statistics = statistics["max_subword_seq_length"]
        feature_list_key = "subword_level_features"
    else:
        max_seq_len_statistics = statistics["max_word_seq_length"]
        feature_list_key = "word_level_features"

    if max_seq_len > max_seq_len_statistics:
        logging.warning("since max_seq_len_train is larger than the longest sample in the data, " +
                        "reset it to {}".format(max_seq_len_statistics))
        max_seq_len = max_seq_len_statistics

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # tagger and model
    tagger = HandshakingTaggerEE4TPLPlus(dicts["rel_type2id"], dicts["ent_type2id"])
    tag_size = tagger.get_tag_size()
    model = TPLinkerPlus(tag_size, **model_settings)
    model = model.to(device)
    collate_fn = model.generate_batch
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # data loaders
    filename2data_loaders = get_data_loaders(test_data_dict, collate_fn)

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
    run_id2results = {}
    for run_id, path_list in run_id2model_state_paths.items():
        if run_id not in run_id2results:
            run_id2results[run_id] = []
        # only top k models
        path_list = get_last_k_paths(path_list, top_k_models)
        for path in path_list:
            # load model
            model_name = re.sub("\.pt", "", path.split("/")[-1])
            model.load_state_dict(torch.load(path))
            print("model state loaded: {}".format("/".join(path.split("/")[-2:])))
            model.eval()

            # evaluator
            evaluator = Evaluator(task_type, model, tagger, device, match_pattern=None)

            # predict
            filename2res = predict(evaluator, test_data_dict, filename2data_loaders)

            # save results
            save_dir = os.path.join(data_out_dir, task_type, run_id, model_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for filename, res in filename2res.items():
                json.dump(res, open(os.path.join(save_dir, filename), "w", encoding="utf-8"))

            # score
            if cal_scores:
                filename2scores = score(evaluator, test_data_dict, filename2res)
                run_id2results[run_id].append({
                    "model_name": model_name,
                    "filename2scores": filename2scores,
                })

    if len(run_id2results) > 0:
        pprint(run_id2results)
