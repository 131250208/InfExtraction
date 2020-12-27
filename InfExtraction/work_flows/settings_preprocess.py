data_in_dir = "../../data/ori_data"
data_out_dir = "../../data/normal_data"
exp_name = "genia"
language = "en"
generate_word_level_features_by_stanza = False
pretrained_model_tokenizer_path = "../../data/pretrained_models/biobert-large-cased-pubmed-58k" # biobert-large-cased-pubmed-58k, bert-base-uncased
ori_data_format = "tplinker"  # casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt), tplinker (see readme)
add_char_span = False  # for data without annotated character level spans

ignore_subword_match = True  # whether add whitespaces around the entities
# when matching and adding character level spans,
# e.g. if set true, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets,
# cause there are usually no whitespace around words in Chinese

max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 2
