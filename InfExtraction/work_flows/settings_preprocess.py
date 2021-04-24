data_in_dir = "../../data/ori_data/oie4"
data_out_dir = "../../data/normal_data/oie4"

language = "en"  # en, ch

# used only if "word_list" or "word2char_span" are not given in the data
word_tokenizer_type = "white"  # stanza, white, normal_chinese;

pretrained_model_tokenizer_path = "../../data/pretrained_models/bert-base-cased"
do_lower_case = False  # transform to lower case and rm accents, only for bert tokenizer and lower the subword_list,
                    # it does not change the original text or word_list,
                    # set True if use uncased bert

ori_data_format = "normal"  # normal, casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt),
                              # duie_1, duie_2, duee_1, duee_fin

add_id = True  # set True if id is not given in the data

add_char_span = False  # for data without annotated character level spans
ignore_subword_match = False  # used only if add_char_span is True, whether add whitespaces around the entities
# when matching and adding character level spans,
# e.g. if ignore_subword_match is set true, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets

# only for word embedding, do not matter if only use bert
max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
