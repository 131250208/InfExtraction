'''
Prepare data for training
'''

from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.work_flows import settings_preprocess as settings
from InfExtraction.modules.utils import load_data, save_as_json_lines
import os
import json
import re
import logging
from pprint import pprint

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
max_word_dict_size = settings.max_word_dict_size
min_word_freq = settings.min_word_freq
if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)

# load data
file_name2data = {}
for path, folds, files in os.walk(data_in_dir):
    for file_name in files:
        file_path = os.path.join(path, file_name)
        file_name = re.match("(.*?)\.json", file_name).group(1)
        file_name2data[file_name] = load_data(file_path)

# preprocessor
preprocessor = Preprocessor(language, pretrained_model_tokenizer_path, do_lower_case)

# transform data from CasRel, ETL_span, et.
for file_name, data in file_name2data.items():
    data_type = None
    if "train" in file_name:
        data_type = "train"
    if "valid" in file_name:
        data_type = "valid"
    if "test" in file_name:
        data_type = "test"

    data = preprocessor.transform_data(data, ori_format=ori_data_format, dataset_type=data_type, add_id=add_id)
    file_name2data[file_name] = data

# process
for filename, data in file_name2data.items():
    # add char spans
    if add_char_span:
        data = preprocessor.add_char_span(data, ignore_subword_match=ignore_subword_match)
    # check char span and list alignment
    preprocessor.pre_check_data_annotation(data, language)
    # create features
    data = preprocessor.create_features(data, word_tokenizer_type)
    # add token level spans
    data = preprocessor.add_tok_span(data)
    file_name2data[filename] = data
    # additional preprocessing


all_data = []
for data in file_name2data.values():
    all_data.extend(data)

# check word level and subword level spans
sample_id2mismatch = preprocessor.check_tok_span(all_data, language)
if len(sample_id2mismatch) > 0:
    error_info = "spans error!"
    logging.warning(error_info)
    pprint(sample_id2mismatch)

# generate supporting data: word and character dict, relation type dict, entity type dict, ...
dicts, statistics = preprocessor.generate_supporting_data(all_data, max_word_dict_size, min_word_freq)
# dicts["bert_dict"] = preprocessor.get_subword_tokenizer().get_vocab()
for filename, data in file_name2data.items():
    statistics[filename] = len(data)

# save main data
for filename, data in file_name2data.items():
    data_path = os.path.join(data_out_dir, "{}.json".format(filename))
    save_as_json_lines(data, data_path)

# save supporting data
dicts_path = os.path.join(data_out_dir, "dicts.json")
json.dump(dicts, open(dicts_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
statistics_path = os.path.join(data_out_dir, "statistics.json")
json.dump(statistics, open(statistics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


