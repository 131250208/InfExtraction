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

# settings
exp_name = settings.exp_name
data_in_dir = os.path.join(settings.data_in_dir, exp_name)
data_out_dir = os.path.join(settings.data_out_dir, exp_name)
task_type = settings.task_type
language = settings.language
pretrained_model_path = settings.pretrained_model_path
ori_data_format = settings.ori_data_format

if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)

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

for filename, data in file_name2data.items():
    preprocessor.build_data(data, task_type)

# #
# # clean, add char span, tok span
# # generate dicts: relation, entity type
# # check tok spans
# rel_set = set()
# ent_set = set()
# pos_tag_set = set()
# ent_tag_set = set()
# error_statistics = {}
#
# for file_name, data in file_name2data.items():
#     assert len(data) > 0
#     if "relation_list" not in data[0] and "event_list" not in data[0]:  # skip unannotated test set
#         continue
#     #     # rm redundant whitespaces
#     #     # separate by whitespaces
#     #     data = preprocessor.clean_data_wo_span(data, separate = config["separate_char_by_white"])
#
#     error_statistics[file_name] = {}
#
#     # add char span
#     if config["add_char_span"]:
#         data, miss_sample_list = preprocessor.add_char_span(data, config["ignore_subword"])
#         error_statistics[file_name]["miss_samples"] = len(miss_sample_list)
#
#     # build data for a specific task
#     preprocessor.build_data(data, task)
#
#     # collect relation types and entity types
#     for sample in tqdm(data, desc="building relation type set and entity type set"):
#         for ent in sample["entity_list"]:
#             ent_set.add(ent["type"])
#
#         for rel in sample["relation_list"]:
#             rel_set.add(rel["predicate"])
#
#         for pos_tag in sample["pos_tag_list"]:
#             pos_tag_set.add(pos_tag)
#
#         for ent_tag in sample["ent_tag_list"]:
#             ent_tag_set.add(ent_tag)
#
#     # add tok span
#     data = preprocessor.add_tok_span(data)
#
#     # check tok span
#     span_error_memory = preprocessor.check_tok_span(data)
#     if len(span_error_memory) > 0:
#         print(span_error_memory)
#     error_statistics[file_name]["tok_span_error"] = len(span_error_memory)
#
#     file_name2data[file_name] = data
# pprint(error_statistics)