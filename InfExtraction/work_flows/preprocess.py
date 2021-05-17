'''
Prepare data for training
'''

from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.work_flows import settings_preprocess as settings
from InfExtraction.modules.utils import load_data, save_as_json_lines
from InfExtraction.modules.prepare_extr_info import gen_ddp_data
import os
import json
import re
import logging
from pprint import pprint
from InfExtraction.modules.utils import SpoSearcher, MyLargeFileReader, MyLargeJsonlinesFileReader, merge_gen
import itertools


# settings
data_in_dir = settings.data_in_dir
data_out_dir = settings.data_out_dir
word_tokenizer_type = settings.word_tokenizer_type
language = settings.language
pretrained_model_tokenizer_path = settings.pretrained_model_tokenizer_path
do_lower_case = settings.do_lower_case
ori_data_format = settings.ori_data_format
add_id = settings.add_id
add_char_span = settings.add_char_span
ignore_subword_match = settings.ignore_subword_match
parser = settings.parser
extracted_ent_rel_by_dicts = settings.extracted_ent_rel_by_dicts

max_word_dict_size = settings.max_word_dict_size
min_word_freq = settings.min_word_freq
if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)

# entitiy and spo extractor
ent_spo_extractor = None
if extracted_ent_rel_by_dicts:
    ent_list = settings.ent_list
    spo_list = settings.spo_list
    ent_type_map = settings.ent_type_map
    ent_type_mask = settings.ent_type_mask
    min_ent_len = settings.min_ent_len
    ent_spo_extractor = SpoSearcher(spo_list, ent_list,
                                    ent_type_map=ent_type_map,
                                    ent_type_mask=ent_type_mask,
                                    min_ent_len=min_ent_len)

# preprocessor
preprocessor = Preprocessor(language, pretrained_model_tokenizer_path, do_lower_case)

# load data
file_name2data = {}
save_paths = []
for path, folds, files in os.walk(data_in_dir):
    for file_name in files:
        file_path = os.path.join(path, file_name)
        data_path = os.path.join(data_out_dir, file_name)
        save_paths.append(data_path)

        # train_jsreader = MyLargeJsonlinesFileReader(MyLargeFileReader(file_path))
        # data = train_jsreader.get_jsonlines_generator()
        #
        # data_type = None
        # if "train" in file_name:
        #     data_type = "train"
        # if "valid" in file_name:
        #     data_type = "valid"
        # if "test" in file_name:
        #     data_type = "test"
        # # transform data
        # data = preprocessor.transform_data(data, ori_format=ori_data_format, dataset_type=data_type, add_id=add_id)
        # # add char spans
        # if add_char_span:
        #     data = preprocessor.add_char_span(data, ignore_subword_match=ignore_subword_match)
        # # check char span and list alignment
        # data = preprocessor.pre_check_data_annotation(data, language)
        # # create features
        # if parser == "ddp":
        #     parse_results = gen_ddp_data(file_path)
        # else:
        #     parse_results = None
        # data = preprocessor.create_features(data, word_tokenizer_type,
        #                                     parse_format=parser,
        #                                     parse_results=parse_results,
        #                                     ent_spo_extractor=ent_spo_extractor)
        # # add token level spans
        # data = preprocessor.add_tok_span(data)
        #
        # # check tok spans
        # data = preprocessor.check_tok_span(data, language)
        #
        # # save
        # data_size = save_as_json_lines(data, data_path)
        # statistics[file_name] = data_size


# generate supporting data: word and character dict, relation type dict, entity type dict, ...

dicts, statistics = preprocessor.generate_supporting_data(save_paths, max_word_dict_size, min_word_freq)
dicts["bert_dict"] = preprocessor.get_subword_tokenizer().get_vocab()

# save supporting data
print("save data!")
dicts_path = os.path.join(data_out_dir, "dicts.json")
json.dump(dicts, open(dicts_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
statistics_path = os.path.join(data_out_dir, "statistics.json")
json.dump(statistics, open(statistics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
print("done!")