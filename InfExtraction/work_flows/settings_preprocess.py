data_in_dir = "../../data/normal_data/ace2005_dygiepp_span_times_values"
data_out_dir = "../../data/preprocessed_data/ace2005_dygiepp_span_times_values"

# used only if "word_list" or "word2char_span" are not in data
word_tokenizer_type = "white"  # white, stanza, normal_chinese;
language = "en"  # used to init stanza for tokenizing, valid only if word_tokenizer_type = "stanza"

pretrained_model_tokenizer_path = "../../data/pretrained_models/bert-base-cased"
do_lower_case = False  # only for bert tokenizer (and subword_list), it does not change the original text or word_list
ori_data_format = "normal"  # casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt)

add_id = False

add_char_span = False  # for data without annotated character-level spans (offsets)
ignore_subword_match = True  # whether add whitespaces around the entities, valid only if add_char_span = True
# when matching and adding character level spans,
# e.g. if set true, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets,
# cause there are usually no whitespace around words in Chinese

max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
