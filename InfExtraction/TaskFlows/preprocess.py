'''
Prepare data for training
0. tokenize (word level, subword level)
1. add character level spans (if needed)
2. add token level spans
3. generate a dict mapping id to tag
4. generate a dict mapping id to token (for LSTM if needed)
5. generate other files if needed
'''

from InfExtraction.Components.preprocess import WordTokenizer, BertTokenizerAlignedWithStanza, Preprocessor
from InfExtraction.TaskFlows import preprocess_settings as settings
import os
import json
import re
import stanza
import logging
from pprint import pprint

# settings
exp_name = settings.exp_name
data_in_dir = os.path.join(settings.data_in_dir, exp_name)
data_out_dir = os.path.join(settings.data_out_dir, exp_name)
if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)
task_type = settings.task_type
language = settings.language
pretrained_model_path = settings.pretrained_model_path
ori_data_format = settings.ori_data_format
add_char_span = settings.add_char_span
ignore_subword_match = settings.ignore_subword_match
max_word_num = settings.max_word_num
min_word_freq = settings.min_word_freq

# load data
file_name2data = {}
for path, folds, files in os.walk(data_in_dir):
    for file_name in files:
        file_path = os.path.join(path, file_name)
        file_name = re.match("(.*?)\.json", file_name).group(1)
        file_name2data[file_name] = json.load(open(file_path, "r", encoding="utf-8"))

# preprocessor
stanza_nlp = stanza.Pipeline(settings.language)
word_tokenizer = WordTokenizer(stanza_nlp)
subword_tokenizer = BertTokenizerAlignedWithStanza.from_pretrained(pretrained_model_path,
                                                                   add_special_tokens=False,
                                                                   do_lower_case=False,
                                                                   stanza_nlp=stanza_nlp)
preprocessor = Preprocessor(word_tokenizer, subword_tokenizer)

# transform data
if ori_data_format != "tplinker": # if tplinker, skip transforming
    for file_name, data in file_name2data.items():
        data_type = None
        if "train" in file_name:
            data_type = "train"
        if "valid" in file_name:
            data_type = "valid"
        if "test" in file_name:
            data_type = "test"
        data = preprocessor.transform_data(data, ori_format=ori_data_format, dataset_type=data_type, add_id=True)
        file_name2data[file_name] = data

# process and save main data
for filename, data in file_name2data.items():
    # process data
    data = preprocessor.process_main_data(data, task_type, add_char_span, ignore_subword_match)
    file_name2data[filename] = data

# generate supporting data: word and character dict, relation type dict, entity type dict, ...
all_data = []
for data in file_name2data.values():
    all_data.extend(data)

# check word level and subword level spans
sample_id2mismatch = preprocessor.check_tok_span(all_data)
if len(sample_id2mismatch) > 0:
    error_info = "Some spans do not match the text! " \
                 "It might because that you set ignore_subword_match to false and " \
                 "the tokenizer of BERT can not handle some tokens. e.g. tokens: [ab, ##cde], text: abcd."
    logging.warning(error_info)
    pprint(sample_id2mismatch)

dicts, statistics = preprocessor.generate_supporting_data(all_data, max_word_num, min_word_freq)
for filename, data in file_name2data.items():
    statistics[filename] = len(data)

# save main data
for filename, data in file_name2data.items():
    data_path = os.path.join(data_out_dir, "{}.json".format(filename))
    json.dump(data, open(data_path, "w", encoding="utf-8"), ensure_ascii=False)

# save supporting data
dicts_path = os.path.join(data_out_dir, "dicts.json")
json.dump(dicts, open(dicts_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
statistics_path = os.path.join(data_out_dir, "statistics.json")
json.dump(statistics, open(statistics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

