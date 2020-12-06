data_in_dir = "../../data/ori_data"
data_out_dir = "../../data/normal_data"
exp_name = "nyt_star"
language = "en"
pretrained_model_tokenizer_path = "../../data/pretrained_models/bert-base-cased"
ori_data_format = "casrel" # casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt), tplinker (see readme)
add_char_span = True # for data without annotated character level spans
ignore_subword_match = True # whether add whitespaces around the entities
                            # when matching and adding character level spans,
                            # e.g. if set true, " home " will not match the subword "home" in "hometown"
                            # it should be set to False when you are handling Chinese,
                            # cause no whitespaces around words in Chinese
max_word_dict_size = 30000 # the max size of word2id dict
min_word_freq = 1 #
