import copy
import re
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.modules import utils
from InfExtraction.modules.utils import Indexer
import numpy as np
import networkx as nx
from InfExtraction.modules.metrics import MetricsCalculator
import logging
import random
import torch


class Tagger(metaclass=ABCMeta):
    @classmethod
    def additional_preprocess(cls, data, **kwargs):
        return data

    @abstractmethod
    def get_tag_size(self):
        pass

    @abstractmethod
    def get_tag_points(self, sample):
        '''
        This function is for generating tag points

        sample: an example
        return points for tagging
        point: (start_pos, end_pos, tag_id)
        '''
        pass

    @abstractmethod
    def tag(self, data):
        '''
        This function is for generating tag points in batch

        data: examples
        return: data with points
        '''
        pass

    @abstractmethod
    def decode(self, sample, pred_tags):
        '''
        decoding function: to extract results by the predicted tag

        :param sample: an example (to offer text, tok2char_span for decoding)
        :param pred_tags: predicted tag id tensors converted from the outputs of the forward function,
                          it is a tuple or a single tag tensor
        :return: predicted example
        '''
        pass

    # def decode_batch(self, sample_list, batch_pred_tags):
    #     pred_sample_list = []
    #     for ind in range(len(sample_list)):
    #         sample = sample_list[ind]
    #         pred_tags = [batch_pred_tag[ind] for batch_pred_tag in batch_pred_tags]
    #         pred_sample = self.decode(sample, pred_tags)  # decoding one sample
    #         pred_sample_list.append(pred_sample)
    #     return pred_sample_list

    def decode_batch(self, sample_list, batch_pred_tags, batch_pred_outputs=None):
        pred_sample_list = []
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            pred_tags = [batch_pred_tag[ind] for batch_pred_tag in batch_pred_tags]
            pred_outs = [batch_pred_out[ind] for batch_pred_out in batch_pred_outputs] \
                if batch_pred_outputs is not None else None
            pred_sample = self.decode(sample, pred_tags, pred_outs)  # decoding one sample
            pred_sample_list.append(pred_sample)
        return pred_sample_list


class HandshakingTagger4TPLPlus(Tagger):
    def __init__(self, data, **kwargs):
        '''
        :param data: all data, used to generate entity type and relation type dicts
        '''
        super().__init__()
        # generate entity type and relation type dicts
        rel_type_set = set()
        ent_type_set = set()
        for sample in data:
            # entity type
            ent_type_set |= {ent["type"] for ent in sample["entity_list"]}
            # relation type
            rel_type_set |= {rel["predicate"] for rel in sample["relation_list"]}
        rel_type_set = sorted(rel_type_set)
        ent_type_set = sorted(ent_type_set)
        self.rel2id = {rel: ind for ind, rel in enumerate(rel_type_set)}
        self.ent2id = {ent: ind for ind, ent in enumerate(ent_type_set)}

        self.id2rel = {ind: rel for rel, ind in self.rel2id.items()}

        self.separator = "\u2E80"
        self.rel_link_types = {"SH2OH",  # subject head to object head
                               "ST2OT",  # subject tail to object tail
                               # "OH2SH",  # object head to subject head
                               # "OT2ST",  # object tail to subject tail
                               }

        self.add_o2s_links = False
        if "add_o2s_links" in kwargs and kwargs["add_o2s_links"] is True:
            self.add_o2s_links = True
            self.rel_link_types = self.rel_link_types.union({
                "OH2SH",  # object head to subject head
                "OT2ST",  # object tail to subject tail
            })

        self.add_h2t_n_t2h_links = False
        if "add_h2t_n_t2h_links" in kwargs and kwargs["add_h2t_n_t2h_links"] is True:
            self.rel_link_types = self.rel_link_types.union({
                "SH2OT",  # subject head to object tail
                "ST2OH",  # subject tail to object head
                # "OT2SH",  # object tail to subject head
                # "OH2ST",  # object head to subject tail
            })
            self.add_h2t_n_t2h_links = True
            if self.add_o2s_links:
                self.rel_link_types = self.rel_link_types.union({
                    "OT2SH",  # object tail to subject head
                    "OH2ST",  # object head to subject tail
                })

        self.classify_entities_by_relation = kwargs["classify_entities_by_relation"]

        self.tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.tags |= {self.separator.join([ent, "EH2ET"]) for ent in
                      self.ent2id.keys()}  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)
        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}

    def get_tag_size(self):
        return len(self.tag2id)

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            sample["tag_points"] = self.get_tag_points(sample)
        return data

    def get_tag_points(self, sample):
        '''
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        matrix_points = []
        point_memory_set = set()

        def add_point(point):
            memory = "{},{},{}".format(*point)
            if memory not in point_memory_set:
                matrix_points.append(point)
                point_memory_set.add(memory)

        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                add_point(
                    (ent["tok_span"][0], ent["tok_span"][1] - 1,
                     self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))

        if "relation_list" in sample:
            for rel in sample["relation_list"]:
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                rel = rel["predicate"]

                # add related boundaries
                if subj_tok_span[0] <= obj_tok_span[0]:
                    add_point((subj_tok_span[0], obj_tok_span[0], self.tag2id[self.separator.join([rel, "SH2OH"])]))
                else:
                    add_point((obj_tok_span[0], subj_tok_span[0], self.tag2id[self.separator.join([rel, "OH2SH"])]))
                if subj_tok_span[1] <= obj_tok_span[1]:
                    add_point(
                        (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
                else:
                    add_point(
                        (obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))

                if self.add_h2t_n_t2h_links:
                    add_point((subj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "SH2OT"])]))
                    add_point((obj_tok_span[1] - 1, subj_tok_span[0], self.tag2id[self.separator.join([rel, "OT2SH"])]))
                    add_point((subj_tok_span[1] - 1, obj_tok_span[0], self.tag2id[self.separator.join([rel, "ST2OH"])]))
                    add_point((obj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OH2ST"])]))

        return matrix_points

    def decode(self, sample, pred_tags):
        '''
        sample: to provide tok2char_span map and text
        pred_tags: predicted tags
        '''
        rel_list, ent_list = [], []
        predicted_shaking_tag = pred_tags[0]
        shk_points = Indexer.shaking_seq2points(predicted_shaking_tag)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # entity
        head_ind2entities = {}
        for sp in shk_points:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator)
            # for an entity, the start position can not be larger than the end pos.
            if link_type != "EH2ET" or sp[0] > sp[1]:
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            ent_list.append(entity)

            head_key = "{},{}".format(ent_type, str(sp[0])) if self.classify_entities_by_relation else str(sp[0])
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)

        # tail link
        tail_link_memory_set = set()
        for sp in shk_points:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join([rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join([rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link
        for sp in shk_points:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)

            if link_type == "SH2OH":
                if self.classify_entities_by_relation:
                    subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(sp[0])), "REL:{},{}".format(rel,
                                                                                                          str(sp[1]))
                else:
                    subj_head_key, obj_head_key = str(sp[0]), str(sp[1])
            elif link_type == "OH2SH":
                if self.classify_entities_by_relation:
                    subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(sp[1])), "REL:{},{}".format(rel,
                                                                                                          str(sp[0]))
                else:
                    subj_head_key, obj_head_key = str(sp[1]), str(sp[0])
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue

            # all entities start with this subject head
            subj_list = Preprocessor.unique_list(head_ind2entities[subj_head_key])
            # all entities start with this object head
            obj_list = Preprocessor.unique_list(head_ind2entities[obj_head_key])

            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = self.separator.join(
                        [rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                        "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                        "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                        "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                        "predicate": rel,
                    })

        if self.add_h2t_n_t2h_links:
            # fitler wrong relations by Head 2 Tail and Tail to Head tags
            head2tail_link_set = set()
            tail2head_link_set = set()
            for pt in shk_points:
                tag = self.id2tag[pt[2]]
                rel, link_type = tag.split(self.separator)
                if link_type == "SH2OT":
                    head2tail_link_set.add(self.separator.join([rel, str(pt[0]), str(pt[1])]))
                elif link_type == "OT2SH":
                    head2tail_link_set.add(self.separator.join([rel, str(pt[1]), str(pt[0])]))
                if link_type == "ST2OH":
                    tail2head_link_set.add(self.separator.join([rel, str(pt[0]), str(pt[1])]))
                elif link_type == "OH2ST":
                    tail2head_link_set.add(self.separator.join([rel, str(pt[1]), str(pt[0])]))
            filtered_rel_list = []
            for spo in rel_list:
                subj_tok_span = spo["subj_tok_span"]
                obj_tok_span = spo["obj_tok_span"]
                h2t = self.separator.join([spo["predicate"], str(subj_tok_span[0]), str(obj_tok_span[1] - 1)])
                t2h = self.separator.join([spo["predicate"], str(subj_tok_span[1] - 1), str(obj_tok_span[0])])
                if h2t not in head2tail_link_set or t2h not in tail2head_link_set:
                    continue
                filtered_rel_list.append(spo)
            rel_list = filtered_rel_list

        pred_sample = copy.deepcopy(sample)
        # filter extra relations
        pred_sample["relation_list"] = [rel for rel in rel_list if "EXT:" not in rel["predicate"]]
        # filter extra entities
        ent_types2filter = {"REL:", "EXT:"}
        ent_filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = [ent for ent in ent_list if re.search(ent_filter_pattern, ent["type"]) is None]
        return pred_sample


class Tagger4RAIN(HandshakingTagger4TPLPlus):
    def __init__(self, data, **kwargs):
        '''
        :param data: all data, used to generate entity type and relation type dicts
        '''
        super(Tagger4RAIN, self).__init__(data, **kwargs)

        # self.rel_tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}
        # ent_link_types = {"EH2ET"}
        #
        # self.ent_tags = {self.separator.join([ent, lt]) for ent in self.ent2id.keys() for lt in ent_link_types}
        #
        # self.rel_tag2id = {t: idx for idx, t in enumerate(sorted(self.rel_tags))}
        # self.id2rel_tag = {idx: t for t, idx in self.rel_tag2id.items()}
        #
        # self.ent_tag2id = {t: idx for idx, t in enumerate(sorted(self.ent_tags))}
        # self.id2ent_tag = {idx: t for t, idx in self.ent_tag2id.items()}

        self.rel_tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}
        self.rel_tag2id = {t: idx for idx, t in enumerate(sorted(self.rel_tags))}
        self.id2rel_tag = {idx: t for t, idx in self.rel_tag2id.items()}

        self.ent_tag2id = self.ent2id
        self.id2ent_tag = {idx: t for t, idx in self.ent_tag2id.items()}

        print(">>>>>>>>>>>>>>>> ent_tag_size: {}; rel_tag_size: {} >>>>>>>>>>>>>>>>>>>>".format(len(self.ent_tag2id),
                                                                                                len(self.rel_tag2id)))

    def get_rel_link_types(self):
        return self.rel_link_types

    def get_rel2id(self):
        return self.rel2id

    def get_tag_size(self):
        return len(self.ent_tag2id), len(self.rel_tag2id)

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            self.get_tag_points(sample)
        return data

    def get_ent_points(self, ent_list):
        ent_matrix_points = set()
        for ent in ent_list:
            tag = ent["type"]
            if tag not in self.ent_tag2id:
                logging.warning("ent_type: {} is not in training set".format(tag))
                continue
            point = (ent["tok_span"][0], ent["tok_span"][-1] - 1,
                     self.ent_tag2id[tag],
                     )
            ent_matrix_points.add(point)
        return ent_matrix_points

    def get_rel_points(self, rel_list):
        rel_matrix_points = set()
        for rel in rel_list:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel_str = rel["predicate"]

            if rel_str not in self.rel2id:
                logging.warning("rel: {} is not in the training set".format(rel_str))
                continue

            # add related boundaries
            rel_matrix_points.add(
                (subj_tok_span[0], obj_tok_span[0], self.rel_tag2id[self.separator.join([rel_str, "SH2OH"])]))
            rel_matrix_points.add(
                (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel_str, "ST2OT"])]))
            if self.add_o2s_links:
                rel_matrix_points.add(
                    (obj_tok_span[0], subj_tok_span[0], self.rel_tag2id[self.separator.join([rel_str, "OH2SH"])]))
                rel_matrix_points.add(
                    (obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel_str, "OT2ST"])]))

            if self.add_h2t_n_t2h_links:
                rel_matrix_points.add(
                    (subj_tok_span[0], obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel_str, "SH2OT"])]))
                rel_matrix_points.add(
                    (subj_tok_span[1] - 1, obj_tok_span[0], self.rel_tag2id[self.separator.join([rel_str, "ST2OH"])]))
                if self.add_o2s_links:
                    rel_matrix_points.add(
                        (obj_tok_span[1] - 1, subj_tok_span[0], self.rel_tag2id[self.separator.join([rel_str, "OT2SH"])]))
                    rel_matrix_points.add(
                        (obj_tok_span[0], subj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel_str, "OH2ST"])]))
        return rel_matrix_points

    def get_tag_points(self, sample):
        '''
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        ent_matrix_points = self.get_ent_points(sample["entity_list"])
        rel_matrix_points = self.get_rel_points(sample["relation_list"])
        sample["ent_points"] = ent_matrix_points
        sample["rel_points"] = rel_matrix_points

    def decode(self, sample, pred_tags, pred_outs=None):
        '''
        sample: to provide tok2char_span map and text
        pred_tags: predicted tags
        '''
        rel_list, ent_list = [], []
        pred_ent_tag, pred_rel_tag = pred_tags[0], pred_tags[1]
        pred_ent_conf, pred_rel_conf = None, None
        matrix_idx2shaking_idx = None
        if pred_outs is not None:
            pred_ent_conf, pred_rel_conf = torch.sigmoid(pred_outs[0]), torch.sigmoid(pred_outs[1])
            shaking_seq_len = pred_ent_conf.size()[0]
            matrix_size = int((2 * shaking_seq_len + 0.25) ** 0.5 - 0.5)
            matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        ent_points = Indexer.shaking_seq2points(pred_ent_tag)
        rel_points = Indexer.matrix2points(pred_rel_tag)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # entity
        for pt in ent_points:
            ent_tag = self.id2ent_tag[pt[2]]
            ent_type = ent_tag
            tok_sp = [pt[0], pt[1] + 1]
            char_span_list = tok2char_span[tok_sp[0]:tok_sp[1]]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            if char_sp[1] == 0:  # if [PAD] tokens are included, char_sp would be [*, 0]
                continue
            ent_text = text[char_sp[0]:char_sp[1]]
            conf = 1.
            if pred_ent_conf is not None:
                shaking_idx = matrix_idx2shaking_idx[pt[0]][pt[1]]
                conf = pred_ent_conf[shaking_idx][pt[2]].item()
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": tok_sp,
                "char_span": char_sp,
                "conf": round(conf, 5),
            }
            ent_list.append(entity)

        # cand_ent_list4rel = []
        rel2candidate_ents = {
            "DEFAULT": []
        }
        for ent in ent_list:
            if "MASK:" in ent["type"]:
                continue
            if self.classify_entities_by_relation:
                if "REL:" in ent["type"]:
                    new_ent = copy.deepcopy(ent)
                    rel = re.sub("REL:", "", new_ent["type"])
                    if rel not in rel2candidate_ents:
                        rel2candidate_ents[rel] = []
                    rel2candidate_ents[rel].append(new_ent)
            else:
                rel2candidate_ents["DEFAULT"].append(ent)

        rel2link_type_map = {}
        for pt in rel_points:
            tag = self.id2rel_tag[pt[2]]
            rel, link_type = tag.split(self.separator)
            index_pair = "{},{}".format(pt[0], pt[1])

            if self.add_o2s_links and link_type[0] == "O":  # OH2SH, OT2ST ... -> SH2OH, ST2OT
                sph, spt = link_type.split("2")
                link_type = "2".join([spt, sph])
                index_pair = "{},{}".format(pt[1], pt[0])

            if rel not in rel2link_type_map:
                rel2link_type_map[rel] = {}
            if index_pair not in rel2link_type_map[rel]:
                rel2link_type_map[rel][index_pair] = {}

            # cal conf
            if pred_rel_conf is None:
                rel2link_type_map[rel][index_pair][link_type] = 1.
            elif link_type in rel2link_type_map[rel][index_pair]:  # confident score may be already in because of the transfer (OH2SH, OT2ST ... -> SH2OH, ST2OT
                rel2link_type_map[rel][index_pair][link_type] = max(rel2link_type_map[rel][index_pair][link_type],
                                                                    pred_rel_conf[pt[0]][pt[1]][pt[2]].item())
            else:
                rel2link_type_map[rel][index_pair][link_type] = pred_rel_conf[pt[0]][pt[1]][pt[2]].item()

        for rel, link_type_map in rel2link_type_map.items():
            cand_ent_list4rel = rel2candidate_ents.get(rel, []) \
                if self.classify_entities_by_relation else rel2candidate_ents["DEFAULT"]
            for subj in cand_ent_list4rel:
                for obj in cand_ent_list4rel:
                    h2h_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][0])
                    t2t_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    h2t_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][1] - 1)
                    t2h_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][0])

                    rel_exist = False
                    edge_conf = 1.
                    if self.add_h2t_n_t2h_links:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids] and \
                                h2t_ids in link_type_map and "SH2OT" in link_type_map[h2t_ids] and \
                                t2h_ids in link_type_map and "ST2OH" in link_type_map[t2h_ids]:
                            rel_exist = True
                            edge_conf = (link_type_map[h2h_ids]["SH2OH"] * link_type_map[t2t_ids]["ST2OT"]
                                         * link_type_map[h2t_ids]["SH2OT"] * link_type_map[t2h_ids]["ST2OH"]) ** 0.25
                    else:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids]:
                            rel_exist = True
                            edge_conf = (link_type_map[h2h_ids]["SH2OH"] * link_type_map[t2t_ids]["ST2OT"]) ** 0.5

                    if rel_exist:
                        rel_list.append({
                            "subject": subj["text"],
                            "object": obj["text"],
                            "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                            "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                            "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                            "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                            "predicate": rel,
                            "conf": round((edge_conf * subj["conf"] * obj["conf"]) ** (1 / 3), 5)
                        })

        pred_sample = sample

        # filter extra relations
        pred_sample["relation_list"] = Preprocessor.unique_list(
            [rel for rel in rel_list if "EXT:" not in rel["predicate"]])
        # filter extra entities
        ent_types2filter = {"REL:", "EXT:"}
        ent_filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = Preprocessor.unique_list(
            [ent for ent in ent_list if re.search(ent_filter_pattern, ent["type"]) is None])

        return pred_sample

    # def get_tag_points(self, sample):
    #     '''
    #     matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
    #     '''
    #     ent_matrix_points, rel_matrix_points = [], []
    #     # if self.output_ent_length:
    #     #     len_matrix_points = []
    #     if "entity_list" in sample:
    #         for ent in sample["entity_list"]:
    #             if ent["type"] not in self.ent2id:
    #                 logging.warning("entity type: {} is not in the training set".format(ent["type"]))
    #                 continue
    #
    #             point = (ent["tok_span"][0], ent["tok_span"][-1] - 1,
    #                      self.ent_tag2id[self.separator.join([ent["type"], "EH2ET"])])
    #             ent_matrix_points.append(point)
    #
    #     if "relation_list" in sample:
    #         for rel in sample["relation_list"]:
    #             subj_tok_span = rel["subj_tok_span"]
    #             obj_tok_span = rel["obj_tok_span"]
    #             rel = rel["predicate"]
    #
    #             if rel not in self.rel2id:
    #                 logging.warning("rel: {} is not in the training set".format(rel))
    #                 continue
    #
    #             # add related boundaries
    #
    #             rel_matrix_points.append(
    #                 (subj_tok_span[0], obj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "SH2OH"])]))
    #             rel_matrix_points.append(
    #                 (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "ST2OT"])]))
    #             rel_matrix_points.append(
    #                 (obj_tok_span[0], subj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "OH2SH"])]))
    #             rel_matrix_points.append(
    #                 (obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "OT2ST"])]))
    #
    #             if self.add_h2t_n_t2h_links:
    #                 rel_matrix_points.append(
    #                     (subj_tok_span[0], obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "SH2OT"])]))
    #                 rel_matrix_points.append(
    #                     (subj_tok_span[1] - 1, obj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "ST2OH"])]))
    #                 rel_matrix_points.append(
    #                     (obj_tok_span[0], subj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "OH2ST"])]))
    #                 rel_matrix_points.append(
    #                     (obj_tok_span[1] - 1, subj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "OT2SH"])]))
    #
    #     return Preprocessor.unique_list(ent_matrix_points), Preprocessor.unique_list(rel_matrix_points)
    #
    # def decode(self, sample, pred_tags):
    #     '''
    #     sample: to provide tok2char_span map and text
    #     pred_tags: predicted tags
    #     '''
    #     rel_list, ent_list = [], []
    #     pred_ent_tag, pred_rel_tag = pred_tags[0], pred_tags[1]
    #     ent_points = Indexer.shaking_seq2points(pred_ent_tag)
    #     rel_points = Indexer.matrix2points(pred_rel_tag)
    #
    #     sample_idx, text = sample["id"], sample["text"]
    #     tok2char_span = sample["features"]["tok2char_span"]
    #
    #     # entity
    #     head_ind2entities = {}
    #     for pt in ent_points:
    #         ent_tag = self.id2ent_tag[pt[2]]
    #         ent_type, link_type = ent_tag.split(self.separator)
    #         # for an entity, the start position can not be larger than the end pos.
    #         assert link_type == "EH2ET" and pt[0] <= pt[1]
    #         if link_type == "EH2ET":
    #             tok_sp = [pt[0], pt[1] + 1]
    #             char_span_list = tok2char_span[tok_sp[0]:tok_sp[1]]
    #             char_sp = [char_span_list[0][0], char_span_list[-1][1]]
    #             if char_sp[1] == 0:  # if [PAD] tokens are included, char_sp would be [*, 0]
    #                 continue
    #             ent_text = text[char_sp[0]:char_sp[1]]
    #
    #             entity = {
    #                 "type": ent_type,
    #                 "text": ent_text,
    #                 "tok_span": tok_sp,
    #                 "char_span": char_sp,
    #             }
    #             ent_list.append(entity)
    #
    #             head_key = "{},{}".format(ent_type, str(pt[0])) if self.classify_entities_by_relation else str(pt[0])
    #             if head_key not in head_ind2entities:
    #                 head_ind2entities[head_key] = []
    #             head_ind2entities[head_key].append(entity)
    #
    #     # tail link
    #     tail_link_memory_set = set()
    #     for pt in rel_points:
    #         tag = self.id2rel_tag[pt[2]]
    #         rel, link_type = tag.split(self.separator)
    #
    #         if link_type == "ST2OT":
    #             tail_link_memory = self.separator.join([rel, str(pt[0]), str(pt[1])])
    #             tail_link_memory_set.add(tail_link_memory)
    #         elif link_type == "OT2ST":
    #             tail_link_memory = self.separator.join([rel, str(pt[1]), str(pt[0])])
    #             tail_link_memory_set.add(tail_link_memory)
    #         else:
    #             continue
    #
    #     # head link
    #     for pt in rel_points:
    #         tag = self.id2rel_tag[pt[2]]
    #         rel, link_type = tag.split(self.separator)
    #
    #         if link_type == "SH2OH":
    #             if self.classify_entities_by_relation:
    #                 subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(pt[0])), "REL:{},{}".format(rel,
    #                                                                                                       str(pt[1]))
    #             else:
    #                 subj_head_key, obj_head_key = str(pt[0]), str(pt[1])
    #         elif link_type == "OH2SH":
    #             if self.classify_entities_by_relation:
    #                 subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(pt[1])), "REL:{},{}".format(rel,
    #                                                                                                       str(pt[0]))
    #             else:
    #                 subj_head_key, obj_head_key = str(pt[1]), str(pt[0])
    #         else:
    #             continue
    #
    #         if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
    #             # no entity start with subj_head_key or obj_head_key
    #             continue
    #
    #         # all entities start with this subject head
    #         subj_list = Preprocessor.unique_list(head_ind2entities[subj_head_key])
    #         # all entities start with this object head
    #         obj_list = Preprocessor.unique_list(head_ind2entities[obj_head_key])
    #
    #         # go over all subj-obj pair to check whether the tail link exists
    #         for subj in subj_list:
    #             for obj in obj_list:
    #                 tail_link_memory = self.separator.join(
    #                     [rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
    #                 if tail_link_memory not in tail_link_memory_set:
    #                     # no such relation
    #                     continue
    #                 rel_list.append({
    #                     "subject": subj["text"],
    #                     "object": obj["text"],
    #                     "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
    #                     "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
    #                     "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
    #                     "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
    #                     "predicate": rel,
    #                 })
    #
    #     if self.add_h2t_n_t2h_links:
    #         # fitler wrong relations by Head 2 Tail and Tail to Head tags
    #         head2tail_link_set = set()
    #         tail2head_link_set = set()
    #         for pt in rel_points:
    #             tag = self.id2rel_tag[pt[2]]
    #             rel, link_type = tag.split(self.separator)
    #             if link_type == "SH2OT":
    #                 head2tail_link_set.add(self.separator.join([rel, str(pt[0]), str(pt[1])]))
    #             elif link_type == "OT2SH":
    #                 head2tail_link_set.add(self.separator.join([rel, str(pt[1]), str(pt[0])]))
    #             if link_type == "ST2OH":
    #                 tail2head_link_set.add(self.separator.join([rel, str(pt[0]), str(pt[1])]))
    #             elif link_type == "OH2ST":
    #                 tail2head_link_set.add(self.separator.join([rel, str(pt[1]), str(pt[0])]))
    #         filtered_rel_list = []
    #         for spo in rel_list:
    #             subj_tok_span = spo["subj_tok_span"]
    #             obj_tok_span = spo["obj_tok_span"]
    #             h2t = self.separator.join([spo["predicate"], str(subj_tok_span[0]), str(obj_tok_span[1] - 1)])
    #             t2h = self.separator.join([spo["predicate"], str(subj_tok_span[1] - 1), str(obj_tok_span[0])])
    #             if h2t not in head2tail_link_set or t2h not in tail2head_link_set:
    #                 continue
    #             filtered_rel_list.append(spo)
    #         rel_list = filtered_rel_list
    #
    #     pred_sample = copy.deepcopy(sample)
    #     # filter extra relations
    #     pred_sample["relation_list"] = Preprocessor.unique_list(
    #         [rel for rel in rel_list if "EXT:" not in rel["predicate"]])
    #     # filter extra entities
    #     ent_types2filter = {"REL:", "EXT:"}
    #     ent_filter_pattern = "({})".format("|".join(ent_types2filter))
    #     pred_sample["entity_list"] = Preprocessor.unique_list(
    #         [ent for ent in ent_list if re.search(ent_filter_pattern, ent["type"]) is None])
    #
    #     return pred_sample


def create_rebased_ee_tagger(base_class):
    class REBasedEETagger(base_class):
        def __init__(self, data, *args, **kwargs):
            super(REBasedEETagger, self).__init__(data, *args, **kwargs)
            self.event_type2arg_rols = {}
            for sample in data:
                for event in sample["event_list"]:
                    # event_type = event["trigger_type"]
                    # for arg in event["argument_list"]:
                    #     if event_type not in self.event_type2arg_rols:
                    #         self.event_type2arg_rols[event_type] = set()
                    #     self.event_type2arg_rols[event_type].add(arg["type"])
                    event_type = event["event_type"]
                    for arg in event["argument_list"]:
                        if event_type not in self.event_type2arg_rols:
                            self.event_type2arg_rols[event_type] = set()
                        self.event_type2arg_rols[event_type].add(arg["type"])
        @classmethod
        def additional_preprocess(cls, data, **kwargs):

            new_data = copy.deepcopy(data)
            separator = "\u2E82"
            for sample in new_data:
                # transform event list to relation list and entity list
                fin_ent_list = []
                fin_rel_list = []
                for event in sample["event_list"]:
                    if "trigger" not in event:
                        continue

                    fin_ent_list.append({
                        "text": event["trigger"],
                        "type": "EE:{}{}{}".format("Trigger", separator, event["event_type"]),
                        "char_span": event["trigger_char_span"],
                        "tok_span": event["trigger_tok_span"],
                    })
                    for arg in event["argument_list"]:
                        fin_ent_list.append({
                            "text": arg["text"],
                            "type": "EE:{}{}{}".format("Argument", separator, arg["type"]),
                            "char_span": arg["char_span"],
                            "tok_span": arg["tok_span"],
                        })
                        fin_rel_list.append({
                            "subject": arg["text"],
                            "subj_char_span": arg["char_span"],
                            "subj_tok_span": arg["tok_span"],
                            "object": event["trigger"],
                            "obj_char_span": event["trigger_char_span"],
                            "obj_tok_span": event["trigger_tok_span"],
                            "predicate": "EE:ARG2TRI",
                        })
                        fin_rel_list.append({
                            "subject": arg["text"],
                            "subj_char_span": arg["char_span"],
                            "subj_tok_span": arg["tok_span"],
                            "object": event["trigger"],
                            "obj_char_span": event["trigger_char_span"],
                            "obj_tok_span": event["trigger_tok_span"],
                            "predicate": "EE:{}{}{}".format(arg["type"], separator, event["event_type"]),
                        })

                # extend original entity list
                if "entity_list" in sample:
                    fin_ent_list.extend(sample["entity_list"])
                # extend original relation list
                if "relation_list" in sample:
                    fin_rel_list.extend(sample["relation_list"])
                sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)
                sample["relation_list"] = Preprocessor.unique_list(fin_rel_list)
            return new_data

        # def decode(self, sample, pred_outs):
        #     pred_sample = super(REBasedEETagger, self).decode(sample, pred_outs)
        #     pred_sample["event_list"] = self._trans2ee(pred_sample["relation_list"], pred_sample["entity_list"])
        #     # filter extra entities and relations
        #
        #     return pred_sample

        def trans(self, sample):
            # filter tags with EE:
            new_rel_list, new_ent_list = [], []
            for rel in sample["relation_list"]:
                if rel["predicate"].split(":")[0] == "EE":
                    new_rel = copy.deepcopy(rel)
                    new_rel["predicate"] = re.sub(r"EE:", "", new_rel["predicate"])
                    new_rel_list.append(new_rel)
            for ent in sample["entity_list"]:
                if ent["type"].split(":")[0] == "EE":
                    new_ent = copy.deepcopy(ent)
                    new_ent["type"] = re.sub(r"EE:", "", new_ent["type"])
                    new_ent_list.append(new_ent)
            rel_list, ent_list = new_rel_list, new_ent_list

            # decoding
            separator = "\u2E82"
            type_wise_edges = []
            arg_offset2roles = {}
            arg_mark2arg = {}
            tri_offset2event_types = {}
            tri_mark2trigger = {}
            for ent in ent_list:
                arg_tri, role = ent["type"].split(separator)
                tok_offset = "{},{}".format(*ent["tok_span"])
                if arg_tri == "Argument":
                    arg_offset2roles.setdefault(tok_offset, set()).add(role)

                    arg = copy.deepcopy(ent)
                    arg["type"] = role
                    arg_mark2arg["{},{}".format(tok_offset, role)] = arg

                elif arg_tri == "Trigger":
                    event_type = role
                    tri_offset2event_types.setdefault(tok_offset, set()).add(event_type)
                    tri_mark2trigger["{},{}".format(tok_offset, event_type)] = {
                        "trigger": ent["text"],
                        "trigger_tok_span": ent["tok_span"],
                        "trigger_char_span": ent["char_span"],
                        "event_type": event_type,
                    }

            for rel in rel_list:
                if rel["predicate"] == "ARG2TRI":
                    arg_offset = "{},{}".format(*rel["subj_tok_span"])
                    tri_mark = "{},{}".format(*rel["obj_tok_span"])
                    arg_roles = arg_offset2roles.get(arg_offset, set())
                    event_types = tri_offset2event_types.get(tri_mark, set())
                    if len(arg_roles) == 1 and \
                            len(event_types) == 1:
                        arg_role = list(arg_roles)[0]
                        event_type = list(event_types)[0]
                        if arg_role in self.event_type2arg_rols[event_type]:
                            rel_cp = copy.deepcopy(rel)
                            rel_cp["predicate"] = separator.join([arg_role, event_type])
                            type_wise_edges.append(rel_cp)
                else:
                    type_wise_edges.append(rel)

            tri_mark2args = {}
            arg_used_mem = set()
            for edge in type_wise_edges:
                arg_role, event_type = edge["predicate"].split(separator)
                tri_mark = "{},{},{}".format(*edge["obj_tok_span"], event_type)
                tri_mark2trigger[tri_mark] = {
                    "trigger": edge["object"],
                    "trigger_tok_span": edge["obj_tok_span"],
                    "trigger_char_span": edge["obj_char_span"],
                    "event_type": event_type,
                }
                tri_mark2args.setdefault(tri_mark, []).append({
                    "text": edge["subject"],
                    "char_span": edge["subj_char_span"],
                    "tok_span": edge["subj_tok_span"],
                    "type": arg_role,
                })
                arg_mark = "{},{},{}".format(*edge["subj_tok_span"], arg_role)
                arg_used_mem.add(arg_mark)

            event_list = []
            for trigger_mark, trigger in tri_mark2trigger.items():
                arg_list = utils.unique_list(tri_mark2args.get(trigger_mark, []))
                if len(arg_list) == 0:  # if it is a single-node trigger, add all possible arguments
                    arg_list = []
                    for arg_mark, arg in arg_mark2arg.items():
                        if arg_mark not in arg_used_mem and \
                                arg["type"] in self.event_type2arg_rols[trigger["event_type"]]:
                            arg_list.append(arg)
                            arg_mark = "{},{},{}".format(*arg["tok_span"], arg["type"])
                            arg_used_mem.add(arg_mark)

                event_list.append({
                    **trigger,
                    "argument_list": arg_list,
                })

            sample["event_list"] = event_list
            sample["entity_list"] = [ent for ent in sample["entity_list"] if "EE:" not in ent["type"]]
            sample["relation_list"] = [rel for rel in sample["relation_list"] if
                                       "EE:" not in rel["predicate"]]

            return sample

    return REBasedEETagger


def create_rebased_tfboys_tagger(base_class):
    class REBasedTFBOYSTagger(base_class):
        def __init__(self, data, *args, **kwargs):
            super(REBasedTFBOYSTagger, self).__init__(data, *args, **kwargs)
            self.event_type2arg_rols = {}
            self.event_type2arg_rols = {}
            for sample in data:
                for event in sample["event_list"]:
                    event_type = event["event_type"]
                    for arg in event["argument_list"]:
                        self.event_type2arg_rols.setdefault(event_type, set()).add(arg["type"])

            self.dtm_arg_type_by_edges = kwargs["dtm_arg_type_by_edges"]

        @classmethod
        def additional_preprocess(cls, data, **kwargs):

            new_data = copy.deepcopy(data)
            for sample in tqdm(new_data, desc="additional preprocessing"):
                separator = "\u2E82"
                fin_ent_list = []
                fin_rel_list = []
                clique_element_list = []

                for event in sample["event_list"]:
                    event_type = event["event_type"]
                    arg_list = copy.deepcopy(event["argument_list"])

                    event_nodes_edges = {
                        "entity_list": [],
                        "relation_list": [],
                    }

                    if "trigger" in event:
                        pseudo_arg = {
                            "type": "Trigger",
                            "char_span": event["trigger_char_span"],
                            "tok_span": event["trigger_tok_span"],
                            "text": event["trigger"],
                        }

                        arg_list += [pseudo_arg]

                    for i, arg_i in enumerate(arg_list):
                        ch_sp_list_i = arg_i["char_span"]
                        tk_sp_list_i = arg_i["tok_span"]

                        if type(arg_i["char_span"][0]) is not list:
                            ch_sp_list_i = [arg_i["char_span"], ]
                            tk_sp_list_i = [arg_i["tok_span"], ]

                        for sp_idx, ch_sp in enumerate(ch_sp_list_i):
                            tk_sp = tk_sp_list_i[sp_idx]
                            event_nodes_edges["entity_list"].append({
                                "text": arg_i["text"],
                                "type": "EE:{}{}{}".format(event_type, separator, arg_i["type"]),
                                "char_span": ch_sp,
                                "tok_span": tk_sp,
                            })

                        for j, arg_j in enumerate(arg_list):
                            ch_sp_list_j = arg_j["char_span"]
                            tk_sp_list_j = arg_j["tok_span"]
                            if type(arg_j["char_span"][0]) is not list:
                                ch_sp_list_j = [arg_j["char_span"], ]
                                tk_sp_list_j = [arg_j["tok_span"], ]

                            for sp_idx_i, ch_sp_i in enumerate(ch_sp_list_i):
                                for sp_idx_j, ch_sp_j in enumerate(ch_sp_list_j):
                                    tk_sp_i = tk_sp_list_i[sp_idx_i]
                                    tk_sp_j = tk_sp_list_j[sp_idx_j]

                                    event_nodes_edges["relation_list"].append({
                                        "subject": arg_i["text"],
                                        "subj_char_span": ch_sp_i,
                                        "subj_tok_span": tk_sp_i,
                                        "object": arg_j["text"],
                                        "obj_char_span": ch_sp_j,
                                        "obj_tok_span": tk_sp_j,
                                        "predicate": "EE:{}".format(separator.join(["IN_SAME_EVENT", event_type])),
                                    })
                                    if kwargs["dtm_arg_type_by_edges"]:
                                        event_nodes_edges["relation_list"].append({
                                            "subject": arg_i["text"],
                                            "subj_char_span": ch_sp_i,
                                            "subj_tok_span": tk_sp_i,
                                            "object": arg_j["text"],
                                            "obj_char_span": ch_sp_j,
                                            "obj_tok_span": tk_sp_j,
                                            "predicate": "EE:{}".format(separator.join([arg_i["type"], arg_j["type"]])),
                                        })
                    clique_element_list.append(event_nodes_edges)
                    fin_ent_list.extend(copy.deepcopy(event_nodes_edges["entity_list"]))
                    fin_rel_list.extend(copy.deepcopy(event_nodes_edges["relation_list"]))

                # add original ents and rels
                if "entity_list" in sample:
                    fin_ent_list.extend(sample["entity_list"])
                if "relation_list" in sample:
                    fin_rel_list.extend(sample["relation_list"])

                sample["entity_list"] = fin_ent_list
                sample["relation_list"] = fin_rel_list
                sample["clique_element_list"] = clique_element_list

            return new_data

        def get_tag_points(self, sample):
            super(REBasedTFBOYSTagger, self).get_tag_points(sample)
            if "clique_element_list" in sample:
                for clique_elements in sample["clique_element_list"]:
                    clique_elements["ent_points"] = self.get_ent_points(clique_elements["entity_list"])
                    clique_elements["rel_points"] = self.get_rel_points(clique_elements["relation_list"])

        # def decode(self, sample, pred_outs):
        #     pred_sample = super(REBasedTFBOYSTagger, self).decode(sample, pred_outs)
        #     # pred_sample = self._trans(pred_sample)
        #
        #     # pred_sample["entity_list"] = [ent for ent in pred_sample["entity_list"] if "EE:" not in ent["type"]]
        #     # pred_sample["relation_list"] = [rel for rel in pred_sample["relation_list"] if
        #     #                                 "EE:" not in rel["predicate"]]
        #     return pred_sample

        def trans(self, sample):

            rel_list = sample["relation_list"]
            ent_list = sample["entity_list"]

            # choose tags with EE:
            new_rel_list, new_ent_list = [], []
            for rel in rel_list:
                if rel["predicate"].split(":")[0] == "EE":
                    new_rel = copy.deepcopy(rel)
                    new_rel["predicate"] = re.sub(r"EE:", "", new_rel["predicate"])
                    new_rel_list.append(new_rel)
            for ent in ent_list:
                if ent["type"].split(":")[0] == "EE":
                    new_ent = copy.deepcopy(ent)
                    new_ent["type"] = re.sub(r"EE:", "", new_ent["type"])
                    new_ent_list.append(new_ent)

            # decoding
            tok2char_span = sample["features"]["tok2char_span"]
            text = sample["text"]
            separator = "\u2E82"
            event2graph = {}
            offsets2arg_pair_rel = {}
            for rel in new_rel_list:
                subj_offset_str = "{},{}".format(*rel["subj_tok_span"])
                obj_offset_str = "{},{}".format(*rel["obj_tok_span"])

                if "IN_SAME_EVENT" in rel["predicate"]:
                    _, event_type = rel["predicate"].split(separator)
                    if event_type not in event2graph:
                        event2graph[event_type] = nx.Graph()
                    event2graph[event_type].add_edge(subj_offset_str, obj_offset_str)
                else:
                    offset_str4arg_pair = separator.join([subj_offset_str, obj_offset_str])
                    if offset_str4arg_pair not in offsets2arg_pair_rel:
                        offsets2arg_pair_rel[offset_str4arg_pair] = set()
                    offsets2arg_pair_rel[offset_str4arg_pair].add(rel["predicate"])

            event2role_map = {}
            for ent in new_ent_list:
                event_type, role = ent["type"].split(separator)
                offset_str = "{},{}".format(*ent["tok_span"])
                if event_type not in event2role_map:
                    event2role_map[event_type] = {}
                if offset_str not in event2role_map[event_type]:
                    event2role_map[event_type][offset_str] = set()
                event2role_map[event_type][offset_str].add(role)

                if event_type not in event2graph:
                    event2graph[event_type] = nx.Graph()
                event2graph[event_type].add_node(offset_str)

            # find events (cliques) under every event type
            event_list = []
            for event_type, graph in event2graph.items():
                role_map = event2role_map.get(event_type, dict())
                cliques = list(nx.find_cliques(graph))  # all maximal cliques

                for cli in cliques:
                    event = {
                        "event_type": event_type,
                    }
                    arguments = []
                    for offset_str in cli:
                        start, end = offset_str.split(",")
                        tok_span = [int(start), int(end)]
                        char_span = Preprocessor.tok_span2char_span(tok_span, tok2char_span)
                        arg_text = Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text)
                        role_set = role_map.get(offset_str, set())

                        if self.dtm_arg_type_by_edges:
                            role_set_fin = set()
                            if len(role_set) == 1:
                                role_set_fin.add(list(role_set)[0])
                            else:  # determine the role by the edge
                                min_edge_num = 1 << 31
                                can_role_set = set()
                                for offset_str_j in cli:
                                    arg_p_set = offsets2arg_pair_rel.get(separator.join([offset_str, offset_str_j]),
                                                                         set())
                                    if len(arg_p_set) != 0 and len(arg_p_set) < min_edge_num:
                                        min_edge_num = len(arg_p_set)
                                        can_role_set = {arg_p.split(separator)[0] for arg_p in arg_p_set}
                                role_set_fin = can_role_set

                            if len(role_set_fin) == 1:
                                role_set = role_set_fin

                        for role in role_set:
                            if role in self.event_type2arg_rols[event_type] or role == "Trigger":
                                arguments.append({
                                    "text": arg_text,
                                    "type": role,
                                    "char_span": char_span,
                                    "tok_span": tok_span,
                                })

                    # find trigger
                    new_argument_list = []
                    triggers = []
                    for arg in arguments:
                        if arg["type"] == "Trigger":
                            triggers.append(arg)
                        else:
                            new_argument_list.append(arg)

                    if len(triggers) > 0:
                        trigger = random.choice(triggers)
                        event["trigger"] = trigger["text"]
                        event["trigger_tok_span"] = trigger["tok_span"]
                        event["trigger_char_span"] = trigger["char_span"]
                        event["trigger_list"] = triggers

                    # if the role sets corresponding to the nodes are all empty,
                    # this clique is invalid.
                    # The corresponding event without argument list and triggers
                    # will not be appended into the event list.
                    if len(new_argument_list) > 0 or "trigger" in event:
                        event["argument_list"] = new_argument_list
                        event_list.append(event)

            sample["event_list"] = event_list

            sample["entity_list"] = [ent for ent in sample["entity_list"] if "EE:" not in ent["type"]]
            sample["relation_list"] = [rel for rel in sample["relation_list"] if
                                       "EE:" not in rel["predicate"]]
            # del sample["entity_list"]
            # del sample["relation_list"]
            return sample

    return REBasedTFBOYSTagger


def create_rebased_discontinuous_ner_tagger(base_class):
    class REBasedDiscontinuousNERTagger(base_class):
        def __init__(self, *arg, **kwargs):
            super(REBasedDiscontinuousNERTagger, self).__init__(*arg, **kwargs)
            self.language = kwargs["language"]
            self.use_bound = kwargs["use_bound"]

        @classmethod
        def additional_preprocess(cls, data, **kwargs):
            use_bound = kwargs["use_bound"]

            new_tag_sep = "\u2E82"
            new_data = []
            for sample in data:
                new_sample = copy.deepcopy(sample)
                text = sample["text"]
                new_ent_list = []
                new_rel_list = []
                clique_element_list = []

                for ent in sample["entity_list"]:
                    assert len(ent["char_span"]) == len(ent["tok_span"])
                    ent_type = ent["type"]

                    clique_nodes_edges = {
                        "entity_list": [],
                        "relation_list": [],
                    }

                    # boundary
                    ch_sp = [ent["char_span"][0], ent["char_span"][-1]]
                    tok_sp = [ent["tok_span"][0], ent["tok_span"][-1]]
                    if use_bound:
                        clique_nodes_edges["entity_list"].append({
                            "text": text[ch_sp[0]:ch_sp[1]],
                            "type": new_tag_sep.join([ent_type, "BOUNDARY"]),
                            "char_span": ch_sp,
                            "tok_span": tok_sp,
                        })

                    for idx_i in range(0, len(ent["char_span"]), 2):
                        seg_i_ch_span = [ent["char_span"][idx_i], ent["char_span"][idx_i + 1]]
                        seg_i_tok_span = [ent["tok_span"][idx_i], ent["tok_span"][idx_i + 1]]

                        position_tag = "B" if idx_i == 0 else "I"
                        if len(ent["char_span"]) == 2:
                            position_tag = "S"
                        new_ent_type = "{}{}{}".format(ent_type, new_tag_sep, position_tag)

                        clique_nodes_edges["entity_list"].append({
                            "text": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                            "type": new_ent_type,
                            "char_span": seg_i_ch_span,
                            "tok_span": seg_i_tok_span,
                        })
                        for idx_j in range(idx_i + 2, len(ent["char_span"]), 2):
                            seg_j_ch_span = [ent["char_span"][idx_j], ent["char_span"][idx_j + 1]]
                            seg_j_tok_span = [ent["tok_span"][idx_j], ent["tok_span"][idx_j + 1]]
                            clique_nodes_edges["relation_list"].append({
                                "subject": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                "subj_char_span": seg_i_ch_span,
                                "subj_tok_span": seg_i_tok_span,
                                "object": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                                "obj_char_span": seg_j_ch_span,
                                "obj_tok_span": seg_j_tok_span,
                                "predicate": "{}{}{}".format(ent_type, new_tag_sep, "SAME_ENT"),
                            })
                            # ============= 共存 0113 ===============
                            clique_nodes_edges["relation_list"].append({
                                "subject": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                                "subj_char_span": seg_j_ch_span,
                                "subj_tok_span": seg_j_tok_span,
                                "object": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                "obj_char_span": seg_i_ch_span,
                                "obj_tok_span": seg_i_tok_span,
                                "predicate": "{}{}{}".format(ent_type, new_tag_sep, "SAME_ENT"),
                            })
                            # ================================================
                    if len(clique_nodes_edges["relation_list"]) > 0:
                        clique_element_list.append(clique_nodes_edges)
                    new_ent_list.extend(copy.deepcopy(clique_nodes_edges["entity_list"]))
                    new_rel_list.extend(copy.deepcopy(clique_nodes_edges["relation_list"]))

                new_sample["entity_list"] = new_ent_list
                new_sample["relation_list"] = new_rel_list
                new_sample["clique_element_list"] = clique_element_list
                new_data.append(new_sample)
            return new_data

        def get_tag_points(self, sample):
            super(REBasedDiscontinuousNERTagger, self).get_tag_points(sample)
            if "clique_element_list" in sample:
                for clique_elements in sample["clique_element_list"]:
                    clique_elements["ent_points"] = self.get_ent_points(clique_elements["entity_list"])
                    clique_elements["rel_points"] = self.get_rel_points(clique_elements["relation_list"])

        def trans(self, ori_sample):
            # decoding
            ent_list = ori_sample["entity_list"]
            rel_list = ori_sample["relation_list"]
            text = ori_sample["text"]
            tok2char_span = ori_sample["features"]["tok2char_span"]

            new_ent_list = []
            new_tag_sep = "\u2E82"
            ent_type2anns = {}

            # map boudaries by entity type
            # map entities by type
            for ent in ent_list:
                ent_type, pos_tag = ent["type"].split(new_tag_sep)
                ent["type"] = pos_tag
                if ent_type not in ent_type2anns:
                    ent_type2anns[ent_type] = {
                        "seg_list": [],
                        "rel_list": [],
                        "boundaries": [],
                    }

                if ent["type"] == "BOUNDARY":
                    ent_type2anns[ent_type]["boundaries"].append(ent)
                else:
                    ent_type2anns[ent_type]["seg_list"].append(ent)

            # map relations by entity type
            for rel in rel_list:
                ent_type, rel_tag = rel["predicate"].split(new_tag_sep)
                rel["predicate"] = rel_tag
                assert rel_tag == "SAME_ENT"
                # =========== 共存 0113 ====================
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                if utils.span_contains(subj_tok_span, obj_tok_span) or \
                        utils.span_contains(obj_tok_span, subj_tok_span):
                    continue
                # ======================
                if ent_type in ent_type2anns:
                    ent_type2anns[ent_type]["rel_list"].append(rel)

            for ent_type, anns in ent_type2anns.items():
                def extr(bd_span):
                    sub_seg_list = anns["seg_list"]
                    sub_rel_list = anns["rel_list"]
                    if bd_span is not None:
                        # select nodes and edges in this region
                        sub_seg_list = [seg for seg in anns["seg_list"] if
                                        utils.span_contains(bd_span, seg["tok_span"])]
                        sub_rel_list = [rel for rel in anns["rel_list"]
                                        if utils.span_contains(bd_span, rel["subj_tok_span"])
                                        and utils.span_contains(bd_span, rel["obj_tok_span"])]

                    offset2seg_types = {}  # "1,2" -> {B, I}
                    graph = nx.Graph()
                    for seg in sub_seg_list:
                        offset_key = "{},{}".format(*seg["tok_span"])
                        if offset_key not in offset2seg_types:
                            offset2seg_types[offset_key] = set()
                        offset2seg_types[offset_key].add(seg["type"])
                        graph.add_node(offset_key)  # add a segment (a node)

                    for rel in sub_rel_list:
                        subj_offset_key = "{},{}".format(*rel["subj_tok_span"])
                        obj_offset_key = "{},{}".format(*rel["obj_tok_span"])
                        if rel["predicate"] == "SAME_ENT":
                            graph.add_edge(subj_offset_key, obj_offset_key)  # add an edge between 2 segments

                    cliques = []
                    if bd_span is not None:
                        for cli in nx.find_cliques(graph):  # find all maximal cliques,
                            # filter invalid ones that do not include boundary tokens
                            if any(int(n.split(",")[0]) == bd_span[0] for n in cli) and \
                                    any(int(n.split(",")[1]) == bd_span[1] for n in cli):
                                cliques.append(cli)
                    else:
                        cliques = nx.find_cliques(graph)

                    for cli in cliques:
                        spans = []
                        for n in cli:
                            start, end = n.split(",")
                            spans.append([int(start), int(end)])
                        tok_span = []
                        last_end = -10
                        for sp in sorted(spans, key=lambda sp: sp[0]):
                            if sp[0] < last_end:
                                continue
                            tok_span.extend(sp)
                            last_end = sp[1]

                        tok_span = utils.merge_spans(tok_span)
                        char_span = Preprocessor.tok_span2char_span(tok_span, tok2char_span)
                        new_ent_list.append({
                            "text": Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text, self.language),
                            "type": ent_type,
                            "char_span": char_span,
                            "tok_span": tok_span,
                        })

                if self.use_bound:
                    for boundary in anns["boundaries"]:
                        bound_span = boundary["tok_span"]
                        extr(bound_span)
                else:
                    extr(None)

            pred_sample = copy.deepcopy(ori_sample)
            pred_sample["entity_list"] = new_ent_list
            del pred_sample["relation_list"]
            return pred_sample

    return REBasedDiscontinuousNERTagger


def create_rebased_oie_tagger(base_class):
    class REBasedOIETagger(base_class):
        def __init__(self, *arg, **kwargs):
            super(REBasedOIETagger, self).__init__(*arg, **kwargs)
            self.language = kwargs["language"]
            self.add_next_link = kwargs["add_next_link"]

        @classmethod
        def additional_preprocess(cls, data, **kwargs):

            add_next_link = kwargs["add_next_link"]
            new_tag_sep = "\u2E82"
            new_data = []
            for sample in data:
                new_sample = copy.deepcopy(sample)
                text = sample["text"]
                new_ent_list = []
                new_rel_list = []

                for spo in sample["open_spo_list"]:
                    ent_list = []
                    tok_span_clique = []
                    char_span_clique = []
                    clique_type = "normal_spo"
                    if spo["predicate"]["predefined"]:
                        clique_type = spo["predicate"]["complete"]

                    for key, val in spo.items():
                        if key == "other_args":
                            continue
                        ent_list.append({
                            "text": val["text"],
                            "type": key,
                            "char_span": val["char_span"],
                            "tok_span": val["tok_span"],
                        })
                        if key == "predicate":
                            if spo[key]["prefix"] != "":
                                ent_list.append({
                                    "text": val["text"],
                                    "type": "PREFIX:{}".format(spo[key]["prefix"]),
                                    "char_span": val["char_span"][:2],
                                    "tok_span": val["tok_span"][:2],
                                })
                            if spo[key]["suffix"] != "":
                                ent_list.append({
                                    "text": val["text"],
                                    "type": "SUFFIX:{}".format(spo[key]["suffix"]),
                                    "char_span": val["char_span"][-2:],
                                    "tok_span": val["tok_span"][-2:],
                                })
                        tok_span_clique.extend(val["tok_span"])
                        char_span_clique.extend(val["char_span"])
                    for arg in spo["other_args"]:
                        ent_list.append({
                            "text": arg["text"],
                            "type": arg["type"],
                            "char_span": arg["char_span"],
                            "tok_span": arg["tok_span"],
                        })
                        tok_span_clique.extend(arg["tok_span"])
                        char_span_clique.extend(arg["char_span"])

                    # add next links and entity types
                    for ent in ent_list:
                        ent_type = ent["type"]
                        for idx_i in range(0, len(ent["char_span"]), 2):
                            seg_i_ch_span = [ent["char_span"][idx_i], ent["char_span"][idx_i + 1]]
                            seg_i_tok_span = [ent["tok_span"][idx_i], ent["tok_span"][idx_i + 1]]
                            if add_next_link and "SUFFIX:" not in ent_type and "PREFIX:" not in ent_type:
                                position_tag = "B" if idx_i == 0 else "I"
                                new_ent_type = "{}{}{}".format(ent_type, new_tag_sep, position_tag)
                            else:
                                new_ent_type = ent_type

                            new_ent_list.append({
                                "text": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                "type": new_ent_type,
                                "char_span": seg_i_ch_span,
                                "tok_span": seg_i_tok_span,
                            })
                            if add_next_link and idx_i + 2 < len(ent["char_span"]):
                                idx_j = idx_i + 2
                                seg_j_ch_span = [ent["char_span"][idx_j], ent["char_span"][idx_j + 1]]
                                seg_j_tok_span = [ent["tok_span"][idx_j], ent["tok_span"][idx_j + 1]]

                                new_rel_list.append({
                                    "subject": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                    "subj_char_span": seg_i_ch_span,
                                    "subj_tok_span": seg_i_tok_span,
                                    "object": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                                    "obj_char_span": seg_j_ch_span,
                                    "obj_tok_span": seg_j_tok_span,
                                    "predicate": "{}{}{}".format(ent_type, new_tag_sep, "NEXT"),
                                })

                    # add edges for each clique
                    for idx_i in range(0, len(tok_span_clique), 2):
                        seg_i_ch_span = [char_span_clique[idx_i], char_span_clique[idx_i + 1]]
                        seg_i_tok_span = [tok_span_clique[idx_i], tok_span_clique[idx_i + 1]]
                        for idx_j in range(idx_i + 2, len(tok_span_clique), 2):
                            seg_j_ch_span = [char_span_clique[idx_j], char_span_clique[idx_j + 1]]
                            seg_j_tok_span = [tok_span_clique[idx_j], tok_span_clique[idx_j + 1]]

                            new_rel_list.append({
                                "subject": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                "subj_char_span": seg_i_ch_span,
                                "subj_tok_span": seg_i_tok_span,
                                "object": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                                "obj_char_span": seg_j_ch_span,
                                "obj_tok_span": seg_j_tok_span,
                                "predicate": "{}{}{}".format(clique_type, new_tag_sep, "SAME_SPO"),
                            })
                            new_rel_list.append({
                                "subject": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                                "subj_char_span": seg_j_ch_span,
                                "subj_tok_span": seg_j_tok_span,
                                "object": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                "obj_char_span": seg_i_ch_span,
                                "obj_tok_span": seg_i_tok_span,
                                "predicate": "{}{}{}".format(clique_type, new_tag_sep, "SAME_SPO"),
                            })
                new_sample["entity_list"] = Preprocessor.unique_list(new_ent_list)
                new_sample["relation_list"] = Preprocessor.unique_list(new_rel_list)
                new_data.append(new_sample)
            return new_data

        def decode(self, sample, pred_outs):
            pred_sample = super(REBasedOIETagger, self).decode(sample, pred_outs)
            return self._trans(pred_sample)

        def _trans(self, ori_sample):
            # decoding
            ent_list = ori_sample["entity_list"]
            rel_list = ori_sample["relation_list"]
            text = ori_sample["text"]
            tok2char_span = ori_sample["features"]["tok2char_span"]

            new_tag_sep = "\u2E82"

            cli_type2graph = {}
            ent_type2link_map = {}
            for rel in rel_list:
                pre_tag, rel_tag = rel["predicate"].split(new_tag_sep)
                subj_offset_key = "{},{}".format(*rel["subj_tok_span"])
                obj_offset_key = "{},{}".format(*rel["obj_tok_span"])

                if rel_tag == "SAME_SPO":
                    if pre_tag not in cli_type2graph:
                        cli_type2graph[pre_tag] = nx.Graph()
                    cli_type2graph[pre_tag].add_edge(subj_offset_key, obj_offset_key)
                if rel_tag == "NEXT" and self.add_next_link:
                    if pre_tag not in ent_type2link_map:
                        ent_type2link_map[pre_tag] = {}
                    ent_type2link_map[pre_tag][subj_offset_key] = obj_offset_key

            offset2ents = {}
            for ent_i in ent_list:
                offset_str = "{},{}".format(*ent_i["tok_span"])
                if offset_str not in offset2ents:
                    offset2ents[offset_str] = []
                offset2ents[offset_str].append(ent_i)

            spo_list = []
            for cli_type, graph in cli_type2graph.items():
                # a clique is a spo triplet
                for cli in nx.find_cliques(graph):
                    temp_ent_list = []
                    cli = set(cli)
                    ents_in_cli = []
                    for offset_str in cli:
                        assert offset_str in offset2ents  # debug for ground truth, 删
                        # if offset_str not in offset2ents:
                        #     continue
                        ents_in_cli.extend(offset2ents[offset_str])

                    if self.add_next_link:
                        for ent_i in ents_in_cli:
                            if "SUFFIX:" in ent_i["type"] or "PREFIX:" in ent_i["type"]:
                                temp_ent_list.append(ent_i)
                            elif "{}{}".format(new_tag_sep, "B") in ent_i["type"]:  # concatenate segments (spans)
                                ent_type_i, pos_tag_i = ent_i["type"].split(new_tag_sep)
                                point_offset_key = "{},{}".format(*ent_i["tok_span"])
                                tok_span = [*ent_i["tok_span"]]

                                link_map = ent_type2link_map.get(ent_type_i, {})
                                while point_offset_key in link_map:  # has a next link to another node
                                    point_offset_key = link_map[point_offset_key]  # point to next node
                                    if point_offset_key not in cli:
                                        break
                                    inside_seg = False
                                    for ent_j in offset2ents[point_offset_key]:
                                        ent_type_split = ent_j["type"].split(new_tag_sep)
                                        ent_type_j, pos_tag_j = ent_type_split[0], ent_type_split[-1]
                                        if ent_type_j == ent_type_i and pos_tag_j == "I":
                                            tok_span.extend(ent_j["tok_span"])
                                            inside_seg = True
                                            break
                                    if inside_seg is False:
                                        break

                                char_span = Preprocessor.tok_span2char_span(tok_span, tok2char_span)
                                temp_ent_list.append({
                                    "text": Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text, self.language),
                                    "type": ent_type_i,
                                    "char_span": char_span,
                                    "tok_span": tok_span,
                                })

                    # generate spo object
                    spo = {"predicate": {"text": "",
                                         "complete": "",
                                         "predefined": False,
                                         "prefix": "",
                                         "suffix": "",
                                         "char_span": [0, 0],
                                         },
                           "subject": {"text": "", "char_span": [0, 0]},
                           "object": {"text": "", "char_span": [0, 0]},
                           "other_args": []}
                    other_args = []
                    for ent_t in temp_ent_list:
                        if ent_t["type"] in {"subject", "object", "predicate"}:
                            spo[ent_t["type"]] = {**spo[ent_t["type"]], **ent_t}
                        elif "SUFFIX:" not in ent_t["type"] and "PREFIX:" not in ent_t["type"]:
                            other_args.append(ent_t)
                    spo["other_args"] = other_args

                    if cli_type == "normal_spo":
                        for ent_t in temp_ent_list:
                            if "SUFFIX:" in ent_t["type"]:
                                spo["predicate"]["suffix"] = re.sub("SUFFIX:", "", ent_t["type"])
                            if "PREFIX:" in ent_t["type"]:
                                spo["predicate"]["prefix"] = re.sub("PREFIX:", "", ent_t["type"])

                        sep = " " if self.language == "en" else ""
                        spo["predicate"]["complete"] = sep.join([spo["predicate"]["prefix"],
                                                                 spo["predicate"]["text"],
                                                                 spo["predicate"]["suffix"], ]).strip()
                        spo["predicate"]["predefined"] = False
                    else:
                        spo["predicate"] = {"text": cli_type,
                                            "complete": cli_type,
                                            "predefined": True,
                                            "prefix": "",
                                            "suffix": "",
                                            "char_span": [0, 0],
                                            },
                    spo_list.append(spo)

            pred_sample = copy.deepcopy(ori_sample)
            pred_sample["open_spo_list"] = spo_list
            # del pred_sample["relation_list"]
            # del pred_sample["entity_list"]

            # auc, prfc, _ = OIEMetrics.compare([pred_sample], [ori_sample], OIEMetrics.binary_linient_tuple_match)
            # if auc != 1. or prfc[2] != 1.:
            #     print("1")
            return ori_sample

    return REBasedOIETagger