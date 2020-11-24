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
from InfExtraction.modules.taggers import HandshakingTagger4EE
from InfExtraction.modules.workers import Evaluator
from InfExtraction.modules.models import TPLinkerPlus
from InfExtraction.work_flows import settings_train_val_test as settings

import os
import torch
import json
from pprint import pprint
from torch.utils.data import DataLoader
import logging

if __name__ == "__main__":
    exp_name = settings.exp_name

    # data
    data_in_dir = settings.data_in_dir
    data_out_dir = settings.data_out_dir
    if data_out_dir is not None and not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    test_data_dict = {}
    for filename in settings.test_data_list:
        test_data_path = os.path.join(data_in_dir, exp_name, filename)
        test_data_dict[filename] = json.load(open(test_data_path, "r", encoding="utf-8"))
    dicts_path = os.path.join(data_in_dir, exp_name, settings.dicts)
    dicts = json.load(open(dicts_path, "r", encoding="utf-8"))
    statistics = settings.statistics

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
    model_state_dict_path = settings.model_state_dict_path

    # model settings
    model_settings = settings.model_settings

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
    # tagger for decoding
    tagger = HandshakingTagger4EE(dicts["rel_type2id"], dicts["ent_type2id"])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # model
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    model_settings["word_encoder_config"]["word2id"] = dicts["word2id"] # set word2id dict
    tag_size = tagger.get_tag_size()
    model = TPLinkerPlus(tag_size, **model_settings)
    model = model.to(device)
    assert model_state_dict_path is not None
    model.load_state_dict(torch.load(model_state_dict_path))
    print("model state loaded: {}".format("/".join(model_state_dict_path.split("/")[-2:])))

    model.eval()
    collate_fn = model.generate_batch
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # evaluator
    evaluator = Evaluator(task_type, model, tagger, token_level, device, match_pattern=None)
    filename2score_dict = {}
    for filename, test_data in test_data_dict.items():
        # splitting
        split_test_data = Preprocessor.split_into_short_samples(test_data, max_seq_len, sliding_len, "test",
                                                                feature_list_key=feature_list_key)

        # indexing and padding
        key2dict = {
            "char_list": dicts["char2id"],
            "word_list": dicts["word2id"],
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

        # prediction
        final_pred_samples = evaluator.predict(test_dataloader, test_data)

        # score
        score_dict = evaluator.score(final_pred_samples, test_data)
        filename2score_dict[filename] = score_dict

        # save
        if data_out_dir is not None:
            res_data_save_path = os.path.join(data_out_dir, exp_name, filename)
            json.dump(final_pred_samples, open(res_data_save_path, "w", encoding="utf-8"))

    pprint(filename2score_dict)