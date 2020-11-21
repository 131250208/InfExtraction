data_in_dir = "../../data/ori_data"
data_out_dir = "../../data/normal_data"
exp_name = "ace2005_lu"
task_type = "ee"
language = "en"
pretrained_model_path = "../../data/pretrained_models/bert-base-uncased"
ori_data_format = "tplinker"
add_char_span = False # for data without annotated character level spans
ignore_subword_match = True # whether add whitespaces around the entities
                            # when matching and adding character level spans,
                            # e.g. if set true, " home " will not match the subword "home" in "hometown"
                            # it should be set to False when you are handling Chinese,
                            # cause no whitespaces around words in Chinese
max_word_dict_size = 50000 # the max size of word2id dict
min_word_freq = 3 # the minimum number of words in word2id
