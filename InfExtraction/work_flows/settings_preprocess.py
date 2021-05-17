from InfExtraction.modules.utils import load_data

data_in_dir = "../../data/ori_data/duee_fin_comp2021_bk"
data_out_dir = "../../data/normal_data/duee_fin_comp2021"

language = "ch"  # en, ch

# used only if "word_list" or "word2char_span" are not given in the data
word_tokenizer_type = "normal_chinese"  # stanza, white, normal_chinese;

pretrained_model_tokenizer_path = "../../data/pretrained_models/macbert-large"
do_lower_case = True  # transform to lower case and rm accents, only for bert tokenizer and lower the subword_list,
                    # it does not change the original text or word_list,
                    # set True if use uncased bert

ori_data_format = "duee_fin"  # normal, casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt),
                              # duie_1, duie_2, duee_1, duee_fin
add_id = False  # set True if id is not given in the data

add_char_span = True  # for data without annotated character level spans
ignore_subword_match = False  # used only if add_char_span is True, whether add whitespaces around the entities
# when matching and adding character level spans,
# e.g. if ignore_subword_match is set true, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets

# add pos tags, ner tags, dependency relation
parser = "ddp"  # stanza (for en), ddp (for ch), None

# extract entities and relations by dicts
extracted_ent_rel_by_dicts = False
if extracted_ent_rel_by_dicts:
    ent_list = load_data("../../data/duie_spo_dict/entity_ske.json") + load_data("../../data/duie_spo_dict/entity.json")
    spo_list = load_data("../../data/duie_spo_dict/spo_ske.json") + load_data("../../data/duie_spo_dict/spo.json")
    ent_type_map = {
        "人物": "PER",
        "地点": "LOC",
        "机构": "ORG",
    }
    ent_type_mask = {"Number", "Text", "Date"}
    min_ent_len = 2

# only for word embedding, do not matter if only use bert
max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
