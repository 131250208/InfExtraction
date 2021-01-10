data_in_dir = "../../data/ori_data/cadec"
data_out_dir = "../../data/normal_data/cadec4yelp"

# used only if "word_list" or "word2char_span" are not in data
word_tokenizer_type = "white"  # stanza, white, normal_chinese;
language = "en"  # used to init stanza for tokenizing, used only if word_tokenizer_type = "stanza"

pretrained_model_tokenizer_path = "../../data/pretrained_models/yelpbert"
do_lower_case = True  # only for bert tokenizer (and subword_list), it does not change the original text or word_list
ori_data_format = "tplinker"  # casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt), tplinker (see readme)

add_id = True

add_char_span = False  # for data without annotated character level spans
ignore_subword_match = True  # whether add whitespaces around the entities, valid only if add_char_span = True
# when matching and adding character level spans,
# e.g. if set true, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets,
# cause there are usually no whitespace around words in Chinese

max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
