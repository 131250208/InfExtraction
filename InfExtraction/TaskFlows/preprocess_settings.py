data_in_dir = "../../data/ori_data"
data_out_dir = "../../data/normal_data"
exp_name = "ace2005_lu"
task_type = "ee"
language = "en"
pretrained_model_path = "../../data/pretrained_models/bert-base-cased"
ori_data_format = "tplinker"

import stanza
stanza_nlp = stanza.Pipeline("en")
text = "Its favored cities include Boston , Washington , Los Angeles , Seattle , San Francisco and Oakland ."
import time

text_long = ""
for _ in range(10):
    text_long += text * 5
    text_long += "\n\n"
t1 = time.time()
print([w.id for sent in stanza_nlp(text_long).sentences for w in sent.words])
print(time.time() - t1)
# from InfExtraction.Components.preprocess import WordTokenizer, BertTokenizerAlignedWithStanza, Preprocessor
# import stanza
# stanza_nlp = stanza.Pipeline("en")
# word_tokenizer = WordTokenizer(stanza_nlp)
# subword_tokenizer = BertTokenizerAlignedWithStanza.from_pretrained(pretrained_model_path,
#                                                                    add_special_tokens=False,
#                                                                    do_lower_case=False,
#                                                                    stanza_nlp=stanza_nlp)
# preprocessor = Preprocessor(word_tokenizer, subword_tokenizer)
#
# data =