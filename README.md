# Information Extraction

## Discontinuous NER
### download datasets
* CADEC: https://data.csiro.au/dap/landingpage?pid=csiro:10948&v=3&d=true
* ShARe 13: https://physionet.org/content/shareclefehealth2013/1.0/
* ShARe 14: https://physionet.org/content/shareclefehealth2014task2/1.0/
Preprocess the datasets by the open-sourced code of [Trans](https://github.com/daixiangau/acl2020-transition-discontinuous-ner).

### convert data format
Use `InfExtraction/workflows/format_conv.py` to transform the datasets to our format
```python
train_filename = "train.txt"
valid_filename = "dev.txt"
test_filename = "test.txt"

train_path = os.path.join(data_in_dir, train_filename)
valid_path = os.path.join(data_in_dir, valid_filename)
test_path = os.path.join(data_in_dir, test_filename)

train_data = trans_daixiang_data(train_path)
valid_data = trans_daixiang_data(valid_path)
test_data = trans_daixiang_data(test_path)
```

### preprocess
Do this part under `InfExtraction/workflows`
1. Set settings_preprocess.py as below:
```
data_in_dir = "../../data/ori_data/share_14"
data_out_dir = "../../data/normal_data/share_14_clinic"

# used only if "word_list" or "word2char_span" are not in original data
word_tokenizer_type = "white"  # stanza, white, normal_chinese;
language = "en"  # used to init stanza for tokenizing, used only if word_tokenizer_type = "stanza"

pretrained_model_tokenizer_path = "../../data/pretrained_models/bio_clinical_bert"
do_lower_case = False  # only for bert tokenizer and lower the subword_list, 
                    # it does not change the original text or word_list, 
                    # set True if use uncased bert
ori_data_format = "normal"  # casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt)

add_id = True

add_char_span = False  # for data without annotated character level spans
ignore_subword_match = True  # whether add whitespaces around the entities, valid only if add_char_span = True
# when matching and adding character level spans,
# e.g. if ignore_subword_match is set true, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets

# only for word embedding, do not matter if only use bert
max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
```
2. Run
```
python preprocess.py
```

### train and valid

At training stage, set settings_rain4disc.py:
```
stage = "train" 
exp_name = "cadec4yelp" # share_13_clinic, share_14_clinic
pretrained_model_name = "yelpbert" # bio_clinical_bert
...
```

At inference, set settings_rain4disc.py:
```
stage = "train" 
exp_name = "cadec4yelp" # share_13_clinic, share_14_clinic
pretrained_model_name = "yelpbert" # bio_clinical_bert
...

model_dir_for_test = "./wandb"  # "./default_log_dir", "./wandb"
target_run_ids = ["eyu8cm6x", ]
top_k_models = 1  # use top k models (at validation set) to eval
metric4testing = "ent_exact_offset_f1"  # consider which metric to choose top k models 
main_test_set_name = "test_data.json"  # we also output the median results of k models for this test set. Note that this is not the median scores reported in the paper, for that results, you should reset the seed and repeat the training for 5 times.
cal_scores = True  # set False if the test sets are not annotated, then only output the result files without scoring.
...
```

For both training and inference, you can run:
```
python train_valid.py -s settings_rain4disc
```



