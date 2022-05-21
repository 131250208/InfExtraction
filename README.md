# News
This repository contains the code for the paper: [Discontinuous Named Entity Recognition as Maximal Clique Discovery](https://aclanthology.org/2021.acl-long.63.pdf), which is in the proceedings of [ACL 2021](https://2021.aclweb.org/). 

# Information Extraction
## 1. Requirements

```
torch
transformers
wandb
stanza
sklearn
networkx
gensim
```

At the root directory, run:
```
pip install -e .
```

## 2. Data
### 2.1 Download Datasets
* CADEC: https://data.csiro.au/dap/landingpage?pid=csiro:10948&v=3&d=true
* ShARe 13: https://physionet.org/content/shareclefehealth2013/1.0/
* ShARe 14: https://physionet.org/content/shareclefehealth2014task2/1.0/

Download in-field BERTs, links are included in each paper:
* Clinical BERT: https://www.aclweb.org/anthology/W19-1909/
* YelpBERT: https://www.aclweb.org/anthology/2020.findings-emnlp.151/

### 2.2 Convert Format
We preprocess the data following [transition-discontinuous-ner](https://github.com/daixiangau/acl2020-transition-discontinuous-ner) and change the data format to our style by following code:
```python
import os
from  InfExtraction.work_flows.format_conv import convert_daixiang_data

train_data = convert_daixiang_data("train.txt")
valid_data = convert_daixiang_data("dev.txt")
test_data = convert_daixiang_data("test.txt")
```
If you did not bother to download and convert, download our version from [here](https://drive.google.com/drive/folders/1w1OOHeM6p38LM-0aZ9htO6UO63SaHvNB?usp=sharing). We only provide CADEC but not the other two, since they are not freely available.

If you use other datasets, convert the data format to fit in our style as below:
```json
{"text": "Within 15 - 20 minutes my stomach felt empty and hollow .", "word_list": ["Within", "15", "-", "20", "minutes", "my", "stomach", "felt", "empty", "and", "hollow", "."], "word2char_span": [[0, 6], [7, 9], [10, 11], [12, 14], [15, 22], [23, 25], [26, 33], [34, 38], [39, 44], [45, 48], [49, 55], [56, 57]], "entity_list": [{"text": "my stomach felt empty and hollow", "type": "ADR", "char_span": [23, 55]}]}

```
### 2.3 Preprocess
Put the datasets in `data/normal_data` and edit `settings_preprocess.py` as follows:
```python
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
```

Run `InfExtraction/workflows/preprocess.py` to do preprocessing:
```
python preprocess.py
```

## 3. Train
- Edit `settings_rain4disc.py` for training.
```python
exp_name = "cadec4yelp" # same as the data folder
pretrained_model_name = "yelpbert" # same as the BERT folder
pretrained_emb_name = "glove.6B.100d.txt" # word embedding file
model_bag_size = 5 # top k model states will be saved in each metrics
```

- Make sure your datasets are put under `data/preprocessed_data`.

- If use BERT, put it under `data/pretrained_models` and set `pretrained_model_name`.

- Set the BERT configuration file `config.json`:
```
"output_hidden_states": true,
"output_attentions": true,
```

- If use word embedding, put it under `data/pretrained_emb` and set `pretrained_emb_name`. 
We support both `*.txt` and `*.bin`.

- run `train_valid.py`:
```
python train_valid.py -s settings_rain4disc.py
```

## 4. Evaluation
- Edit `evaluation part` in `settings_rain4disc.py`:
```python
# for inference and evaluation
model_dir_for_test = "./wandb"  # "./default_log_dir" or "./wandb"
target_run_ids = ["0kQIoiOs", ]  # set run ids
metric4testing = "ent_offset_f1" # use model states on which metric
model_path_ids2infer = [0, 2, -1] # model states are sorted by performance on above metric
cal_scores = True  # set False if golden annotations are not give in data
```
- Keep `model settings` the same as in training

- Run `inference.py`:
```
python inference.py -s settings_rain4disc.py
```
- It will do evaluation on all datasets in `test_path_list`. The results will be saved under `data/res_data` after evaluation.


## 5. Analysis
### 5.1 Discontinuous NER
- Run `InfExtraction/analysis/disc_ner/statistics.py` to get Table 6/7 in the paper.
- Run `InfExtraction/analysis/disc_ner/further_analysis.py` to get Figure 8 in the paper.

