# News
This repository contains the code for the paper: [Discontinuous Named Entity Recognition as Maximal Clique Discovery](https://aclanthology.org/2021.acl-long.63.pdf), which is in the proceedings of [ACL 2021](https://2021.aclweb.org/). 

# Information Extraction
## Requirements

```
torch
transformers
wandb
stanza
networkx
pattern (only for the evaluation of Open Information Extraction tasks)
python-Levenshtein
pyahocorasick
```

## Data
Download datasets
* CADEC: https://data.csiro.au/dap/landingpage?pid=csiro:10948&v=3&d=true
* ShARe 13: https://physionet.org/content/shareclefehealth2013/1.0/
* ShARe 14: https://physionet.org/content/shareclefehealth2014task2/1.0/

Download in-field BERTs, links are included in each paper:
* Clinical BERT: https://www.aclweb.org/anthology/W19-1909/
* YelpBERT: https://www.aclweb.org/anthology/2020.findings-emnlp.151/

You can preprocess the datasets by the open-sourced code of [transition-discontinuous-ner](https://github.com/daixiangau/acl2020-transition-discontinuous-ner) and then change the data format by following code:
```python
import os
from  InfExtraction.work_flows.format_conv import trans_daixiang_data

train_data = trans_daixiang_data("train.txt")
valid_data = trans_daixiang_data("dev.txt")
test_data = trans_daixiang_data("test.txt")
```
Or you can convert the data to fit in the normal format directly:
```json
{"text": "Within 15 - 20 minutes my stomach felt empty and hollow .", "word_list": ["Within", "15", "-", "20", "minutes", "my", "stomach", "felt", "empty", "and", "hollow", "."], "word2char_span": [[0, 6], [7, 9], [10, 11], [12, 14], [15, 22], [23, 25], [26, 33], [34, 38], [39, 44], [45, 48], [49, 55], [56, 57]], "entity_list": [{"text": "my stomach felt empty and hollow", "type": "ADR", "char_span": [23, 55]}]}
```

### Preprocess
Put the datasets under `data/ori_data` and edit `settings_preprocess.py` as follows:
```python
data_in_dir = "../../data/ori_data/cadec"
data_out_dir = "../../data/normal_data/cadec4yelp"

# used only if "word_list" or "word2char_span" are not in data
word_tokenizer_type = "white"  # white, stanza, normal_chinese;
language = "en"  # used to init stanza for tokenizing, valid only if word_tokenizer_type = "stanza"

pretrained_model_tokenizer_path = "../../data/pretrained_models/yelpbert"
do_lower_case = True  # only for bert tokenizer (and subword_list), it does not change the original text or word_list
ori_data_format = "normal"  # casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt)

add_id = True

add_char_span = False  # for data without annotated character-level spans (offsets)
ignore_subword_match = True  # whether add whitespaces around the entities, valid only if add_char_span = True
# when matching and adding character level spans,
# e.g. if set true, " home " will not match the subword "home" in "hometown"
# it should be set to False when handling Chinese datasets,
# cause there are usually no whitespace around words in Chinese

max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
```

Run `InfExtraction/workflows/preprocess.py` to do preprocessing:
```
python preprocess.py
```

## Train

Make sure your datasets are put under `data/normal_data`

If use BERT, put it under `data/pretrained_models` and set `pretrained_model_name`.

Do not forget to set the BERT configuration file `config.json`:
```
"output_hidden_states": true,
"output_attentions": true,
```

If use word embedding, put it under `data/pretrained_emb` and set `pretrained_emb_name`. 
We support both `*.txt` and `*.bin`.

Set `stage = "train"`  in `settings_rain4disc.py` and run `train_valid.py`:
```
python train_valid.py -s settings_rain4disc.py
```

## Evaluation
Set `stage = "inference"`  in `settings_rain4disc.py` and run `train_valid.py`:
```
python train_valid.py -s settings_rain4disc.py
```
It will do evaluation on the datasets in `test_path_list`. The result data will be saved under `data/res_data` after evaluation.


## Statistics
Run `InfExtraction/workflows/others/statistics.py` 

## Analysis
Run `InfExtraction/workflows/further_analysis.py` 

