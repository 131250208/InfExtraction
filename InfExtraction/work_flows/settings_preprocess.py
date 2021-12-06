data_in_dir = "../../data/normal_data/cadec"
data_out_dir = "../../data/preprocessed_data/cadec4yelp"

# >>>>>>>>>>> used if "word_list" or "word2char_span" are provided >>>>>>>>>>>>>>>>>>>>>>>
word_tokenizer_type = "white"  # white, stanza, normal_chinese;
language = "en"  # used to init stanza for tokenizing, valid only if word_tokenizer_type = "stanza"

# >>>>>>>>>>>>>> bert >>>>>>>>>>>>>>>>>>
pretrained_model_tokenizer_path = "../../data/pretrained_models/yelpbert"
do_lower_case = True  # only for bert tokenizer (subword_list), it does not change the original text or word_list

# >>>>>>>>>>>>> data format >>>>>>>>>>>>>
ori_data_format = "normal"  # casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt)
add_id = True  # set True if ids are not provided in data

# >>>>>>>>>>>>> annotate spans by searching entities >>>>>>>>>>
add_char_span = False  # for data without annotated character-level spans (offsets)
ignore_subword_match = True  # whether add whitespaces around the entities when searching spans, valid only if add_char_span = True
# when matching and adding character level spans,
# e.g. if ignore_subword_match is set True, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets,
# cause there are no whitespaces around words in Chinese

# >>>>>>>>>>> for word dict used for word embedding >>>>>>>>>>>>>>
max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
