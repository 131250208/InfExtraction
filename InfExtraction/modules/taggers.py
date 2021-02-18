import copy
import re
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from InfExtraction.modules.preprocess import Indexer, Preprocessor
from InfExtraction.modules import utils
import numpy as np
import networkx as nx
from InfExtraction.modules.metrics import MetricsCalculator
# from InfExtraction.modules.ancient_eval4oie import OIEMetrics
import logging


class Tagger(metaclass=ABCMeta):
    @classmethod
    def additional_preprocess(cls, data, data_type, **kwargs):
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

    def decode_batch(self, sample_list, batch_pred_tags, batch_pred_outputs=None):
        pred_sample_list = []
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            pred_tags = [batch_pred_tag[ind] for batch_pred_tag in batch_pred_tags]
            pred_sample = self.decode(sample, pred_tags)  # decoding one sample
            pred_sample_list.append(pred_sample)
        return pred_sample_list


class HandshakingTagger4TPLPlus(Tagger):
    @classmethod
    def is_additional_ent_type(cls, ent_type):
        if re.search("(EXT:|REL:)", ent_type) is None:
            return False
        else:
            return True

    @classmethod
    def additional_preprocess(cls, data, data_type, **kwargs):
        if data_type not in {"train", "debug"}:
            return data

        new_data = copy.deepcopy(data)
        for sample in new_data:
            assert "entity_list" in sample
            fin_ent_list = copy.deepcopy(sample["entity_list"])
            fin_rel_list = copy.deepcopy(sample["relation_list"]) if "relation_list" in sample else []

            # add default entity type
            add_default_entity_type = kwargs["add_default_entity_type"]
            # if len(fin_ent_list) == 0:  # entity list is empty, generate default entities by the relation list.
            #     add_default_entity_type = True
            if add_default_entity_type is True:
                for ent in sample["entity_list"]:
                    fin_ent_list.append({
                        "text": ent["text"],
                        "type": "EXT:DEFAULT",
                        "char_span": ent["char_span"],
                        "tok_span": ent["tok_span"],
                    })

            add_nested_relation = kwargs["add_nested_relation"]
            add_same_type_relation = kwargs["add_same_type_relation"]
            # add additional relations
            for idx_i, ent_i in enumerate(fin_ent_list):
                for idx_j, ent_j in enumerate(fin_ent_list):
                    if idx_i == idx_j:
                        continue
                    # nested
                    if add_nested_relation:
                        if (ent_i["tok_span"][1] - ent_i["tok_span"][0]) < (
                                ent_j["tok_span"][1] - ent_j["tok_span"][0]) \
                                and ent_i["tok_span"][0] >= ent_j["tok_span"][0] \
                                and ent_i["tok_span"][1] <= ent_j["tok_span"][1]:
                            fin_rel_list.append({
                                "subject": ent_i["text"],
                                "subj_char_span": [*ent_i["char_span"]],
                                "subj_tok_span": [*ent_i["tok_span"]],
                                # "subj_type": ent_i["type"],
                                "object": ent_j["text"],
                                "obj_char_span": [*ent_j["char_span"]],
                                "obj_tok_span": [*ent_j["tok_span"]],
                                # "obj_type": ent_j["type"],
                                "predicate": "EXT:NESTED_IN",
                            })

                    # same type co-occurrence
                    if add_same_type_relation:
                        if ent_j["type"] == ent_i["type"] and not cls.is_additional_ent_type(ent_i["type"]):
                            fin_rel_list.append({
                                "subject": ent_i["text"],
                                "subj_char_span": [*ent_i["char_span"]],
                                "subj_tok_span": [*ent_i["tok_span"]],
                                "object": ent_j["text"],
                                "obj_char_span": [*ent_j["char_span"]],
                                "obj_tok_span": [*ent_j["tok_span"]],
                                "predicate": "EXT:SAME_TYPE",
                            })

            classify_entities_by_relation = kwargs["classify_entities_by_relation"]
            if classify_entities_by_relation:
                for rel in fin_rel_list:
                    # add rel types to entities
                    fin_ent_list.append({
                        "text": rel["subject"],
                        "type": "REL:{}".format(rel["predicate"]),
                        "char_span": rel["subj_char_span"],
                        "tok_span": rel["subj_tok_span"],
                    })
                    fin_ent_list.append({
                        "text": rel["object"],
                        "type": "REL:{}".format(rel["predicate"]),
                        "char_span": rel["obj_char_span"],
                        "tok_span": rel["obj_tok_span"],
                    })

            sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)
            sample["relation_list"] = Preprocessor.unique_list(fin_rel_list)
        return new_data

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
                               "OH2SH",  # object head to subject head
                               "ST2OT",  # subject tail to object tail
                               "OT2ST",  # object tail to subject tail
                               }

        self.add_h2t_n_t2h_links = False
        if "add_h2t_n_t2h_links" in kwargs and kwargs["add_h2t_n_t2h_links"] is True:
            self.rel_link_types = self.rel_link_types.union({
                "SH2OT",  # subject head to object tail
                "OT2SH",  # object tail to subject head
                "ST2OH",  # subject tail to object head
                "OH2ST",  # object head to subject tail
            })
            self.add_h2t_n_t2h_links = True

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

        self.rel_tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}
        self.rel_tag2id = {t: idx for idx, t in enumerate(sorted(self.rel_tags))}
        self.id2rel_tag = {idx: t for t, idx in self.rel_tag2id.items()}

        # ent_link_types = {"EH2ET"}
        # self.ent_tags = {self.separator.join([ent, lt]) for ent in self.ent2id.keys() for lt in ent_link_types}
        # self.ent_tag2id = {t: idx for idx, t in enumerate(sorted(self.ent_tags))}
        self.ent_tag2id = self.ent2id
        self.id2ent_tag = {idx: t for t, idx in self.ent_tag2id.items()}

    def get_tag_size(self):
        return len(self.ent_tag2id), len(self.rel_tag2id)

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            ent_points, rel_points = self.get_tag_points(sample)
            sample["ent_points"] = ent_points
            sample["rel_points"] = rel_points
        return data

    def get_tag_points(self, sample):
        '''
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        ent_matrix_points, rel_matrix_points = [], []
        # if self.output_ent_length:
        #     len_matrix_points = []
        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                # tag = self.separator.join([ent["type"], "EH2ET"])
                tag = ent["type"]
                point = (ent["tok_span"][0], ent["tok_span"][-1] - 1,
                         self.ent_tag2id[tag],
                         )
                ent_matrix_points.append(point)

        if "relation_list" in sample:
            for rel in sample["relation_list"]:
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                rel = rel["predicate"]

                if rel not in self.rel2id:
                    continue

                # add related boundaries

                rel_matrix_points.append(
                    (subj_tok_span[0], obj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "SH2OH"])]))
                rel_matrix_points.append(
                    (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "ST2OT"])]))
                # rel_matrix_points.append(
                #     (obj_tok_span[0], subj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "OH2SH"])]))
                # rel_matrix_points.append(
                #     (obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "OT2ST"])]))

                if self.add_h2t_n_t2h_links:
                    rel_matrix_points.append(
                        (subj_tok_span[0], obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "SH2OT"])]))
                    rel_matrix_points.append(
                        (subj_tok_span[1] - 1, obj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "ST2OH"])]))
                    # rel_matrix_points.append(
                    #     (obj_tok_span[1] - 1, subj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "OT2SH"])]))
                    # rel_matrix_points.append(
                    #     (obj_tok_span[0], subj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "OH2ST"])]))

        return Preprocessor.unique_list(ent_matrix_points), Preprocessor.unique_list(rel_matrix_points)

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
    #         # ent_type, link_type = ent_tag.split(self.separator)
    #         # for an entity, the start position can not be larger than the end pos.
    #         # assert link_type == "EH2ET" and pt[0] <= pt[1]
    #         # if link_type == "EH2ET":
    #         ent_type = ent_tag
    #         tok_sp = [pt[0], pt[1] + 1]
    #         char_span_list = tok2char_span[tok_sp[0]:tok_sp[1]]
    #         char_sp = [char_span_list[0][0], char_span_list[-1][1]]
    #         ent_text = text[char_sp[0]:char_sp[1]]
    #
    #         entity = {
    #             "type": ent_type,
    #             "text": ent_text,
    #             "tok_span": tok_sp,
    #             "char_span": char_sp,
    #         }
    #         ent_list.append(entity)
    #
    #         head_key = "{},{}".format(ent_type, str(pt[0])) if self.classify_entities_by_relation else str(pt[0])
    #         if head_key not in head_ind2entities:
    #             head_ind2entities[head_key] = []
    #         head_ind2entities[head_key].append(entity)
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

    def decode(self, sample, pred_tags):
        '''
        sample: to provide tok2char_span map and text
        pred_tags: predicted tags
        '''
        rel_list, ent_list = [], []
        pred_ent_tag, pred_rel_tag = pred_tags[0], pred_tags[1]
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
            ent_text = text[char_sp[0]:char_sp[1]]

            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": tok_sp,
                "char_span": char_sp,
            }
            ent_list.append(entity)

        cand_ent_list4rel = []
        for ent in ent_list:
            if self.classify_entities_by_relation and "REL:" in ent["type"]:
                new_ent = copy.deepcopy(ent)
                new_ent["type"] = re.sub("REL:", "", new_ent["type"])
                cand_ent_list4rel.append(new_ent)
            else:
                cand_ent_list4rel.append(ent)

        rel2link_type_map = {}
        for pt in rel_points:
            tag = self.id2rel_tag[pt[2]]
            rel, link_type = tag.split(self.separator)
            if rel not in rel2link_type_map:
                rel2link_type_map[rel] = {}
            index_pair = "{},{}".format(pt[0], pt[1])
            if index_pair not in rel2link_type_map[rel]:
                rel2link_type_map[rel][index_pair] = set()
            rel2link_type_map[rel][index_pair].add(link_type)

        for rel, link_type_map in rel2link_type_map.items():
            for subj in cand_ent_list4rel:
                for obj in cand_ent_list4rel:
                    h2h_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][0])
                    t2t_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    h2t_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][1] - 1)
                    t2h_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][0])

                    rel_exist = False
                    if self.add_h2t_n_t2h_links:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids] and \
                                h2t_ids in link_type_map and "SH2OT" in link_type_map[h2t_ids] and \
                                t2h_ids in link_type_map and "ST2OH" in link_type_map[t2h_ids]:
                            rel_exist = True
                    else:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids]:
                            rel_exist = True
                    if rel_exist:
                        rel_list.append({
                            "subject": subj["text"],
                            "object": obj["text"],
                            "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                            "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                            "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                            "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                            "predicate": rel,
                        })

        pred_sample = copy.deepcopy(sample)
        # filter extra relations
        pred_sample["relation_list"] = Preprocessor.unique_list(
            [rel for rel in rel_list if "EXT:" not in rel["predicate"]])
        # filter extra entities
        ent_types2filter = {"REL:", "EXT:"}
        ent_filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = Preprocessor.unique_list(
            [ent for ent in ent_list if re.search(ent_filter_pattern, ent["type"]) is None])

        return pred_sample


def create_rebased_ee_tagger(base_class):
    class REBasedEETagger(base_class):
        def __init__(self, data, *args, **kwargs):
            super(REBasedEETagger, self).__init__(data, *args, **kwargs)
            self.event_type2arg_rols = {}
            for sample in data:
                for event in sample["event_list"]:
                    event_type = event["trigger_type"]
                    for arg in event["argument_list"]:
                        if event_type not in self.event_type2arg_rols:
                            self.event_type2arg_rols[event_type] = set()
                        self.event_type2arg_rols[event_type].add(arg["type"])

        @classmethod
        def additional_preprocess(cls, data, data_type, **kwargs):
            if data_type not in {"train", "debug"}:
                return data

            new_data = copy.deepcopy(data)
            separator = "\u2E82"
            for sample in new_data:
                # transform event list to relation list and entity list
                fin_ent_list = []
                fin_rel_list = []
                for event in sample["event_list"]:
                    fin_ent_list.append({
                        "text": event["trigger"],
                        "type": "EE:{}{}{}".format("Trigger", separator, event["trigger_type"]),
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
                            # "subj_type": "EE:{}{}{}".format("Argument", separator, arg["type"]),
                            "object": event["trigger"],
                            "obj_char_span": event["trigger_char_span"],
                            "obj_tok_span": event["trigger_tok_span"],
                            # "obj_type": "EE:{}{}{}".format("Trigger", separator, event["trigger_type"]),
                            "predicate": "EE:{}{}{}".format(arg["type"], separator, event["trigger_type"]),
                        })
                sample["relation_list"] = Preprocessor.unique_list(fin_rel_list)
                # extend original entity list
                if "entity_list" in sample:
                    fin_ent_list.extend(sample["entity_list"])
                sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)
            new_data = super().additional_preprocess(new_data, data_type, **kwargs)
            return new_data

        def decode(self, sample, pred_outs):
            pred_sample = super(REBasedEETagger, self).decode(sample, pred_outs)
            pred_sample["event_list"] = self._trans2ee(pred_sample["relation_list"], pred_sample["entity_list"])
            # filter extra entities and relations
            pred_sample["entity_list"] = [ent for ent in pred_sample["entity_list"] if "EE:" not in ent["type"]]
            pred_sample["relation_list"] = [rel for rel in pred_sample["relation_list"] if
                                            "EE:" not in rel["predicate"]]
            return pred_sample

        def _trans2ee(self, rel_list, ent_list):
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
            rel_list, ent_list = new_rel_list, new_ent_list

            # decoding
            separator = "\u2E82"
            trigger_offset2vote = {}
            trigger_offset2trigger_text = {}
            trigger_offset2trigger_char_span = {}

            # get candidate trigger types from relations
            for rel in rel_list:
                trigger_offset = rel["obj_tok_span"]
                trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
                trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
                trigger_offset2trigger_char_span[trigger_offset_str] = rel["obj_char_span"]
                _, event_type = rel["predicate"].split(separator)

                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                    event_type, 0) + 1

            # get candidate trigger types from entity tags
            for ent in ent_list:
                t1, t2 = ent["type"].split(separator)
                if t1 == "Trigger":
                    event_type = t2
                    trigger_span = ent["tok_span"]
                    trigger_offset_str = "{},{}".format(trigger_span[0], trigger_span[1])
                    trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                    trigger_offset2trigger_char_span[trigger_offset_str] = ent["char_span"]
                    if trigger_offset_str not in trigger_offset2vote:
                        trigger_offset2vote[trigger_offset_str] = {}
                    trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                        event_type, 0) + 1

            # choose the final trigger type by votes
            trigger_offset2event_types = {}
            for trigger_offet_str, event_type2score in trigger_offset2vote.items():
                # # choose types with the top score
                # top_score = sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][1]
                # winer_event_types = {et for et, sc in event_type2score.items() if sc == top_score}

                # ignore draw, choose only the first type
                # winer_event_types = {sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][0],}

                # save all event types
                winer_event_types = set(event_type2score.keys())

                trigger_offset2event_types[trigger_offet_str] = winer_event_types  # final event types

            # aggregate arguments by event type and trigger_offset
            trigger_offset2event2arguments = {}
            for rel in rel_list:
                trigger_offset = rel["obj_tok_span"]
                argument_role, et = rel["predicate"].split(separator)
                trigger_offset_str = "{},{}".format(*trigger_offset)
                if et not in trigger_offset2event_types[trigger_offset_str]:  # filter false relations
                    continue
                # append arguments
                if trigger_offset_str not in trigger_offset2event2arguments:
                    trigger_offset2event2arguments[trigger_offset_str] = {}
                if et not in trigger_offset2event2arguments[trigger_offset_str]:
                    trigger_offset2event2arguments[trigger_offset_str][et] = []

                trigger_offset2event2arguments[trigger_offset_str][et].append({
                    "text": rel["subject"],
                    "type": argument_role,
                    "event_type": et,
                    "char_span": rel["subj_char_span"],
                    "tok_span": rel["subj_tok_span"],
                })

            if len(trigger_offset2event_types) == 1:
                for trig_offset_str, event_types in trigger_offset2event_types.items():
                    if len(event_types) == 1:
                        et = list(event_types)[0]
                        for ent in ent_list:
                            t1, t2 = ent["type"].split(separator)
                            if t1 == "Argument":
                                arg_role = t2
                                if arg_role not in self.event_type2arg_rols[et]:
                                    continue

                                if trig_offset_str not in trigger_offset2event2arguments:
                                    trigger_offset2event2arguments[trig_offset_str] = {}
                                if et not in trigger_offset2event2arguments[trig_offset_str]:
                                    trigger_offset2event2arguments[trig_offset_str][et] = []

                                trigger_offset2event2arguments[trig_offset_str][et].append({
                                    "text": ent["text"],
                                    "type": arg_role,
                                    "event_type": et,
                                    "char_span": ent["char_span"],
                                    "tok_span": ent["tok_span"],
                                })

            # generate event list
            event_list = []
            for trigger_offset_str, event_types in trigger_offset2event_types.items():
                for et in event_types:
                    arguments = []
                    if trigger_offset_str in trigger_offset2event2arguments and \
                            et in trigger_offset2event2arguments[trigger_offset_str]:
                        arguments = trigger_offset2event2arguments[trigger_offset_str][et]

                    trigger_offset = trigger_offset_str.split(",")
                    event = {
                        "trigger": trigger_offset2trigger_text[trigger_offset_str],
                        "trigger_char_span": trigger_offset2trigger_char_span[trigger_offset_str],
                        "trigger_tok_span": [int(trigger_offset[0]), int(trigger_offset[1])],
                        "trigger_type": et,
                        "argument_list": Preprocessor.unique_list(arguments),
                    }
                    event_list.append(event)
            return event_list

    return REBasedEETagger


def create_rebased_tfboys_tagger(base_class):
    class REBasedTFBoysTagger(base_class):
        def __init__(self, data, *args, **kwargs):
            super(REBasedTFBoysTagger, self).__init__(data, *args, **kwargs)
            self.event_type2arg_rols = {}
            for sample in data:
                for event in sample["event_list"]:
                    event_type = event["trigger_type"]
                    for arg in event["argument_list"]:
                        if event_type not in self.event_type2arg_rols:
                            self.event_type2arg_rols[event_type] = set()
                        self.event_type2arg_rols[event_type].add(arg["type"])

        @classmethod
        def additional_preprocess(cls, data, data_type, **kwargs):
            if data_type not in {"train", "debug"}:
                return data

            new_data = copy.deepcopy(data)
            separator = "\u2E82"
            for sample in tqdm(new_data, desc="additional preprocessing"):
                fin_ent_list = []
                fin_rel_list = []
                for event in sample["event_list"]:
                    pseudo_arg = {
                        "type": "Trigger",
                        "char_span": event["trigger_char_span"],
                        "tok_span": event["trigger_tok_span"],
                        "text": event["trigger"],
                    }
                    event_type = event["trigger_type"]
                    arg_list = [pseudo_arg] + event["argument_list"]
                    for i, arg_i in enumerate(arg_list):
                        fin_ent_list.append({
                            "text": arg_i["text"],
                            "type": "EE:{}{}{}".format(event_type, separator, arg_i["type"]),
                            "char_span": arg_i["char_span"],
                            "tok_span": arg_i["tok_span"],
                        })
                        for j, arg_j in enumerate(arg_list):
                            fin_rel_list.append({
                                "subject": arg_i["text"],
                                "subj_char_span": arg_i["char_span"],
                                "subj_tok_span": arg_i["tok_span"],
                                "object": arg_j["text"],
                                "obj_char_span": arg_j["char_span"],
                                "obj_tok_span": arg_j["tok_span"],
                                "predicate": "EE:{}".format(separator.join(["IN_SAME_EVENT", event_type])),
                            })

                # if "relation_list" in sample:
                #     fin_rel_list.extend(sample["relation_list"])
                # if "entity_list" in sample:
                #     fin_ent_list.extend(sample["entity_list"])
                sample["entity_list"] = fin_ent_list
                sample["relation_list"] = fin_rel_list
            new_data = super().additional_preprocess(new_data, data_type, **kwargs)
            return new_data

        def decode(self, sample, pred_outs):
            pred_sample = super(REBasedTFBoysTagger, self).decode(sample, pred_outs)
            pred_sample = self._trans(pred_sample)

            # pred_sample["entity_list"] = [ent for ent in pred_sample["entity_list"] if "EE:" not in ent["type"]]
            # pred_sample["relation_list"] = [rel for rel in pred_sample["relation_list"] if
            #                                 "EE:" not in rel["predicate"]]
            del pred_sample["entity_list"]
            del pred_sample["relation_list"]
            return pred_sample

        def _trans(self, sample):
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
            for rel in new_rel_list:
                subj_offset_str = "{},{}".format(*rel["subj_tok_span"])
                obj_offset_str = "{},{}".format(*rel["obj_tok_span"])

                if "IN_SAME_EVENT" in rel["predicate"]:
                    _, event_type = rel["predicate"].split(separator)
                    if event_type not in event2graph:
                        event2graph[event_type] = nx.Graph()
                    event2graph[event_type].add_edge(subj_offset_str, obj_offset_str)

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
                    event = {}
                    arguments = []
                    for offset_str in cli:
                        start, end = offset_str.split(",")
                        tok_span = [int(start), int(end)]
                        char_span = Preprocessor.tok_span2char_span(tok_span, tok2char_span)
                        arg_text = Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text)
                        role_set = role_map.get(offset_str, set())
                        for role in role_set:
                            if role == "Trigger":
                                event["trigger"] = arg_text
                                event["trigger_tok_span"] = tok_span
                                event["trigger_type"] = event_type
                                event["trigger_char_span"] = char_span
                            else:
                                arguments.append({
                                    "text": arg_text,
                                    "type": role,
                                    "char_span": char_span,
                                    "tok_span": tok_span,
                                    "event_type": event_type,
                                })
                    event["argument_list"] = arguments
                    event_list.append(event)

            pred_sample = copy.deepcopy(sample)
            pred_sample["event_list"] = event_list
            return pred_sample
    return REBasedTFBoysTagger


def create_rebased_discontinuous_ner_tagger(base_class):
    # 0129
    class REBasedDiscontinuousNERTagger(base_class):
        def __init__(self, *arg, **kwargs):
            super(REBasedDiscontinuousNERTagger, self).__init__(*arg, **kwargs)
            self.language = kwargs["language"]
            self.use_bound = kwargs["use_bound"]
            self.seg_tag_scheme = kwargs["seg_tag_scheme"]

        @classmethod
        def additional_preprocess(cls, data, data_type, **kwargs):
            if data_type not in {"train", "debug"}:
                return data

            use_bound = kwargs["use_bound"]
            seg_tag_scheme = kwargs["seg_tag_scheme"]

            new_tag_sep = "\u2E82"
            new_data = []
            for sample in data:
                new_sample = copy.deepcopy(sample)
                text = sample["text"]
                new_ent_list = []
                new_rel_list = []
                for ent in sample["entity_list"]:
                    assert len(ent["char_span"]) == len(ent["tok_span"])
                    ent_type = ent["type"]

                    ch_sp = [ent["char_span"][0], ent["char_span"][-1]]
                    tok_sp = [ent["tok_span"][0], ent["tok_span"][-1]]

                    # boundary
                    if use_bound:
                        new_ent_list.append({
                            "text": text[ch_sp[0]:ch_sp[1]],
                            "type": new_tag_sep.join([ent_type, "BOUNDARY"]),
                            "char_span": ch_sp,
                            "tok_span": tok_sp,
                        })

                    for idx_i in range(0, len(ent["char_span"]), 2):
                        seg_i_ch_span = [ent["char_span"][idx_i], ent["char_span"][idx_i + 1]]
                        seg_i_tok_span = [ent["tok_span"][idx_i], ent["tok_span"][idx_i + 1]]

                        position_tag = None
                        if seg_tag_scheme == "BIS":
                            if idx_i == 0:
                                position_tag = "B"
                            else:
                                position_tag = "I"
                            if len(ent["char_span"]) == 2:
                                position_tag = "S"
                        elif seg_tag_scheme == "I":
                            position_tag = "I"
                        assert position_tag is not None

                        new_ent_type = "{}{}{}".format(ent_type, new_tag_sep, position_tag)

                        new_ent_list.append({
                            "text": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                            "type": new_ent_type,
                            "char_span": seg_i_ch_span,
                            "tok_span": seg_i_tok_span,
                        })
                        for idx_j in range(idx_i + 2, len(ent["char_span"]), 2):
                            seg_j_ch_span = [ent["char_span"][idx_j], ent["char_span"][idx_j + 1]]
                            seg_j_tok_span = [ent["tok_span"][idx_j], ent["tok_span"][idx_j + 1]]
                            new_rel_list.append({
                                "subject": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                "subj_char_span": seg_i_ch_span,
                                "subj_tok_span": seg_i_tok_span,
                                "object": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                                "obj_char_span": seg_j_ch_span,
                                "obj_tok_span": seg_j_tok_span,
                                "predicate": "{}{}{}".format(ent_type, new_tag_sep, "SAME_ENT"),
                            })
                            # ============= 0113 ===============
                            new_rel_list.append({
                                "subject": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                                "subj_char_span": seg_j_ch_span,
                                "subj_tok_span": seg_j_tok_span,
                                "object": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                                "obj_char_span": seg_i_ch_span,
                                "obj_tok_span": seg_i_tok_span,
                                "predicate": "{}{}{}".format(ent_type, new_tag_sep, "SAME_ENT"),
                            })
                            # ================================================
                new_sample["entity_list"] = new_ent_list
                new_sample["relation_list"] = new_rel_list
                new_data.append(new_sample)
            return new_data

        def decode(self, sample, pred_outs):
            pred_sample = super(REBasedDiscontinuousNERTagger, self).decode(sample, pred_outs)
            return self._trans(pred_sample)

        def _trans(self, ori_sample):
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
                        "continuous_entity_list": [],
                    }

                if ent["type"] == "BOUNDARY":
                    ent_type2anns[ent_type]["boundaries"].append(ent)
                elif ent["type"] in {"B", "I"}:
                    ent_type2anns[ent_type]["seg_list"].append(ent)
                else:
                    assert ent["type"] == "S"
                    ent_type2anns[ent_type]["continuous_entity_list"].append(ent)

            # map relations by entity type
            for rel in rel_list:
                ent_type, rel_tag = rel["predicate"].split(new_tag_sep)
                rel["predicate"] = rel_tag
                assert rel_tag == "SAME_ENT"
                # =========== 0113 ====================
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                if utils.span_contains(subj_tok_span, obj_tok_span) or \
                        utils.span_contains(obj_tok_span, subj_tok_span):
                    continue
                # ======================
                if ent_type in ent_type2anns:
                    ent_type2anns[ent_type]["rel_list"].append(rel)

            for ent_type, anns in ent_type2anns.items():

                if self.seg_tag_scheme == "BIS":
                    for c_ent in anns["continuous_entity_list"]:
                        c_ent["type"] = ent_type
                        new_ent_list.append(c_ent)

                def extr_disc(bd_span):
                    sub_seg_list = anns["seg_list"]
                    sub_rel_list = anns["rel_list"]
                    if bd_span is not None:
                        # select nodes and edges in this region
                        sub_seg_list = [seg for seg in anns["seg_list"] if
                                        utils.span_contains(bd_span, seg["tok_span"])]
                        sub_rel_list = [rel for rel in anns["rel_list"]
                                        if utils.span_contains(bd_span, rel["subj_tok_span"])
                                        and utils.span_contains(bd_span, rel["obj_tok_span"])]

                    offset2seg_types = {}
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
                        # 0129
                        if self.seg_tag_scheme == "BIS" and \
                                not any(n in offset2seg_types and "B" in offset2seg_types[n] for n in cli):
                            continue
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
                        extr_disc(bound_span)
                else:
                    extr_disc(None)

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
        def additional_preprocess(cls, data, data_type, **kwargs):
            if data_type not in {"train", "debug"}:
                return data

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


class TableFillingTagger(Tagger):
    def __init__(self, data):
        '''
        :param data: all data, used to generate entity type and relation type dicts
        '''
        super().__init__()
        # generate unified tag
        tag_set = set()
        self.separator = "\u2E80"
        for sample in tqdm(data, desc="generating tag set"):
            tag_triplets = self._get_tags(sample)
            tag_set |= {t[-1] for t in tag_triplets}

        self.tag2id = {tag: ind for ind, tag in enumerate(sorted(tag_set))}
        self.id2tag = {id_: t for t, id_ in self.tag2id.items()}

    def get_tag_size(self):
        return len(self.tag2id)

    def get_tag_points(self, sample):
        tag_list = self._get_tags(sample)
        new_tag_list = []
        for tag in tag_list:
            if tag[-1] in self.tag2id:
                tag[-1] = self.tag2id[tag[-1]]
                new_tag_list.append(tag)
            else:
                logging.warning("out of tag set: {} not in tag2id dict".format(tag[-1]))
        return new_tag_list

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            sample["tag_points"] = self.get_tag_points(sample)
        return data

    def _get_tags(self, sample):
        pass


# class TriggerFreeLu(TableFillingTagger):
#     def _get_tags(self, sample):
#         tag_list = []
#         event_list = sample["event_list"]
#
#         for event in event_list:
#             event_type = event["trigger_type"]
#             pseudo_argument = {
#                 "type": "Trigger",
#                 "tok_span": event["trigger_tok_span"],
#             }
#             argument_list = [pseudo_argument, ] + event["argument_list"]
#             for guide_arg in argument_list:
#                 for arg in argument_list:
#                     arg_type = arg["type"]
#                     ea_tag = "{}{}{}".format(event_type, self.separator, arg_type)
#                     for i in range(*guide_arg["tok_span"]):
#                         for j in range(*arg["tok_span"]):
#                             pos_tag = "I"
#                             if j == arg["tok_span"][0]:
#                                 pos_tag = "B"
#                             eap_tag = "{}{}{}".format(ea_tag, self.separator, pos_tag)
#                             tag_list.append([i, j, eap_tag])
#         return tag_list
#
#     def decode(self, sample, pred_tags):
#         predicted_matrix_tag = pred_tags[0]
#         matrix_points = Indexer.matrix2points(predicted_matrix_tag)
#
#         # matrix_points = [(58, 58, 87), (58, 59, 88), (58, 60, 88), (58, 65, 89), (58, 68, 91), (58, 75, 82), (58, 82, 84),
#         #                  (59, 58, 87), (59, 59, 88), (59, 60, 88), (59, 65, 89), (59, 68, 91), (59, 75, 82), (59, 82, 84),
#         #                  (60, 58, 87), (60, 59, 88), (60, 60, 88), (60, 65, 89), (60, 68, 91), (60, 75, 82), (60, 82, 84),
#         #                  (65, 58, 87), (65, 59, 88), (65, 60, 88), (65, 65, 89), (65, 68, 91), (65, 75, 82), (65, 82, 84),
#         #                  (68, 58, 87), (68, 59, 88), (68, 60, 88), (68, 65, 89), (68, 68, 91), (68, 75, 82), (68, 82, 84),
#         #                  (74, 74, 89), (74, 75, 91),
#         #                  (75, 74, 89), (75, 75, 91),
#         #                  (75, 58, 87), (75, 59, 88), (75, 60, 88), (75, 65, 89), (75, 68, 91), (75, 75, 82), (75, 82, 84),
#         #                  (82, 58, 87), (82, 59, 88), (82, 60, 88), (82, 65, 89), (82, 68, 91), (82, 75, 82), (82, 82, 84)]
#         # matrix_points = set([(7, 7, 26), (7, 9, 36), (7, 24, 30),
#         #                  (9, 7, 26), (9, 9, 36), (9, 24, 30),
#         #                  (24, 7, 26), (24, 9, 36), (24, 24, 30),
#         #
#         #                  (7, 7, 32), (7, 19, 36), (7, 24, 30),
#         #                  (19, 7, 32), (19, 19, 36), (19, 24, 30),
#         #                  (24, 7, 32), (24, 19, 36), (24, 24, 30),
#         #
#         #                  (7, 7, 82), (7, 11, 89), (7, 24, 85),
#         #                  (11, 7, 82), (11, 11, 89), (11, 24, 85),
#         #                  (24, 7, 82), (24, 11, 89), (24, 24, 85),
#         #
#         #                  (70, 70, 89), (70, 81, 84), (70, 84, 91),
#         #                  (81, 70, 89), (81, 81, 84), (81, 84, 91),
#         #                  (84, 70, 89), (84, 81, 84), (84, 84, 91),
#         #
#         #                  (74, 74, 157), (74, 81, 152), (74, 84, 159),
#         #                  (81, 74, 157), (81, 81, 152), (81, 84, 159),
#         #                  (84, 74, 157), (84, 81, 152), (84, 84, 159),
#         #
#         #                  (79, 79, 36), (79, 81, 28), (79, 84, 32), (79, 87, 32),
#         #                  (81, 79, 36), (81, 81, 28), (81, 84, 32), (81, 87, 32),
#         #                  (84, 79, 36), (84, 81, 28), (84, 84, 32), (84, 87, 32),
#         #                  (87, 79, 36), (87, 81, 28), (87, 84, 32), (87, 87, 32)])
#
#         # tags = [[p[0], p[1], self.id2tag[p[2]]] for p in matrix_points]
#         tags = matrix_points
#         sample_idx, text = sample["id"], sample["text"]
#         tok2char_span = sample["features"]["tok2char_span"]
#
#         # decoding
#
#         center_tags = []
#         for tag in tags:
#             if tag[0] == tag[1]:
#                 center_tags.append(tag)
#
#         event_pieces = []
#         num_centers = len(center_tags)
#         index = 0
#
#         def center_match(ct_1, ct_2):
#             # return predicted_matrix_tag[ct_1[0],ct_2[0],ct_2[2]].item() == 1
#             if ct_1 == ct_2:
#                 return True
#             if abs(ct_1[2] - ct_2[2]) > 20:
#                 return False
#             left_match, right_match = False, False
#             for tag in tags:
#                 if tag[0] == ct_1[0] and tag[1] == ct_2[0] and tag[2] == ct_2[2] and tag[0] != tag[1]:
#                     # print(tag)
#                     left_match = True
#                 if tag[0] == ct_2[0] and tag[1] == ct_1[0] and tag[2] == ct_1[2] and tag[0] != tag[1]:
#                     # print(tag)
#                     right_match = True
#             return right_match and left_match
#
#         while True:
#             if num_centers == 0 or index >= len(center_tags):
#                 break
#             if index == 0:
#                 event_pieces.append([center_tags[index]])
#                 index += 1
#             else:
#                 # pdb.set_trace()
#                 temp_event_pieces = []
#                 single_node = True
#                 for event_piece in event_pieces:
#                     connected_centers = []
#                     match_all = True
#                     for center_tag_ in event_piece:
#                         if center_match(center_tag_, center_tags[index]):
#                             connected_centers.append(center_tag_)
#                             single_node = False
#                         else:
#                             match_all = False
#                     if match_all:
#                         # pdb.set_trace()
#                         event_piece.append(center_tags[index])
#                     else:
#                         if len(connected_centers) > 0:
#                             temp_event_pieces.append(connected_centers + [center_tags[index]])
#                 event_pieces += temp_event_pieces
#                 if single_node:
#                     event_pieces.append([center_tags[index]])
#                 index += 1
#
#         deleted_events = []
#         for i in range(len(event_pieces)):
#             for j in range(i + 1, len(event_pieces)):
#                 if len(set(event_pieces[i]) - set(event_pieces[j])) == 0:
#                     deleted_events.append(i)
#                 elif len(set(event_pieces[j]) - set(event_pieces[i])) == 0:
#                     deleted_events.append(j)
#
#         event_pieces_ = []
#         for i in range(len(event_pieces)):
#             if i in deleted_events:
#                 continue
#             else:
#                 event_piece_list_ = []
#                 for event_piece in event_pieces[i]:
#                     event_piece_list_.append((event_piece[0], self.id2tag[event_piece[2]]))
#                 event_pieces_.append(event_piece_list_)
#         event_pieces = event_pieces_
#         # pdb.set_trace()
#         # event_piece = [offset, tag]
#         # event_type_set = set([tag.split(self.separator)[0] for tag in self.tag2id])
#         # tags_ = [[] for _ in range(predicted_matrix_tag.shape[0])]
#         # for tag in tags:
#         #     tags_[tag[0]].append([tag[1],tag[2]])
#         # event_pieces = []
#         event_list = []
#         # for i in range(len(tags_)):
#         #     tags_considered = [False] * len(tags_[i])
#         #     if len(tags_considered) == 0:
#         #         continue
#         #     temp_event_pieces = []
#         #     for event_piece_list in event_pieces:
#         #         all_connect = True
#         #         event_type = event_piece_list[0][1].split(self.separator)[0]
#         #         connected_pieces = []
#         #         for event_piece in event_piece_list:
#         #             connect = False
#         #             # pdb.set_trace()
#         #             for j in range(len(tags_[i])):
#         #                 tag = tags_[i][j]
#         #                 if tag[0] > i:
#         #                     continue
#         #                 if tag[1] == event_piece[1] and tag[0] == event_piece[0]:
#         #                     connect = True
#         #                     connected_pieces.append(event_piece)
#         #                     tags_considered[j] = True
#         #                     break
#         #             # pdb.set_trace()
#         #             if not connect:
#         #                 all_connect = False
#         #         if all_connect:
#         #             # event_piece_list = []
#         #             for j in range(len(tags_[i])):
#         #                 piece = tags_[i][j]
#         #                 if piece[0] != i:
#         #                     continue
#         #                 match_all = True
#         #                 for event_piece in event_piece_list:
#         #                     offset = event_piece[0]
#         #                     match = False
#         #                     for piece_ in tags_[offset]:
#         #                         if piece[0] == piece_[0] and piece[1] == piece_[1]:
#         #                             match = True
#         #                     if not match:
#         #                         match_all = False
#         #                 # pdb.set_trace()
#         #                 if match_all:
#         #                     event_piece_list.append([i, piece[1]])
#         #                     tags_considered[j] = True
#         #
#         #         else:
#         #             if len(connected_pieces) > 0:
#         #                 has_B = False
#         #                 for j in range(len(tags_[i])):
#         #                     piece = tags_[i][j]
#         #                     if piece[0] != i:
#         #                         continue
#         #                     if piece[1].split(self.separator)[0] == event_type and piece[0] <= i:
#         #                         if 'B' == piece[1][-1]:
#         #                             has_B = True
#         #
#         #                         match_all = True
#         #                         for event_piece in connected_pieces:
#         #                             offset = event_piece[0]
#         #                             match = False
#         #                             for piece_ in tags_[offset]:
#         #                                 if piece[0] == piece_[0] and piece[1] == piece_[1]:
#         #                                     match = True
#         #                             if not match:
#         #                                 match_all = False
#         #                         # pdb.set_trace()
#         #                         if match_all:
#         #                             connected_pieces.append([i, piece[1]])
#         #                             tags_considered[j] = True
#         #
#         #
#         #
#         #                 if has_B:
#         #                     # pdb.set_trace()
#         #                     temp_event_pieces.append(connected_pieces)
#         #
#         #     event_pieces += temp_event_pieces
#         #
#         #     for event_type in event_type_set:
#         #         connected_pieces = []
#         #         for j in range(len(tags_considered)):
#         #             if tags_considered[j]:
#         #                 continue
#         #             tag = tags_[i][j]
#         #             if tag[0] == i and tag[1].split(self.separator)[0] == event_type:
#         #                 connected_pieces.append([i, tag[1]])
#         #         if len(connected_pieces) > 0:
#         #             event_pieces.append(connected_pieces)
#
#         for event_piece_list in event_pieces:
#             event = {
#             }
#             event['argument_list'] = []
#             for i in range(len(event_piece_list)):
#                 event_piece = event_piece_list[i]
#                 tag = event_piece[1]
#                 tag_type = tag[:-2]  # wyc 1208
#                 event_type, role = tag_type.split(self.separator)  # wyc 1208
#                 if tag[-1] == 'B':
#                     end_piece = event_piece
#                     if 'Trigger' in tag:
#                         for j in range(i + 1, len(event_piece_list)):
#                             if 'Trigger' in event_piece_list[j][1]:
#                                 if event_piece_list[j][1][-1] == 'I' and tag_type in event_piece_list[j][1]:
#                                     end_piece = event_piece_list[j]
#                                 else:
#                                     break
#                         event['trigger'] = ' '.join(text.split()[event_piece[0]: end_piece[0] + 1])
#                         event['trigger_tok_span'] = [event_piece[0], end_piece[0] + 1]
#                         event['trigger_char_span'] = [tok2char_span[event_piece[0]][0], tok2char_span[end_piece[0]][1]]
#                         event['trigger_type'] = event_type  # wyc 1208
#                     else:
#                         for j in range(i + 1, len(event_piece_list)):
#                             if event_piece_list[j][1][-1] == 'I' and tag_type in event_piece_list[j][1]:
#                                 end_piece = event_piece_list[j]
#                             elif (end_piece[0] + 1) < event_piece_list[j][0]:
#                                 break
#
#                         event['argument_list'].append({'text': ' '.join(text.split()[event_piece[0]:end_piece[0] + 1]),
#                                                        "event_type": event_type,  # wyc 1208
#                                                        'tok_span': [event_piece[0], end_piece[0] + 1],
#                                                        'char_span': [tok2char_span[event_piece[0]][0],
#                                                                      tok2char_span[end_piece[0]][1]],
#                                                        'type': role,  # wyc 1208
#                                                        })
#             # try:
#             # assert 'trigger' in event
#             # except:
#             #     pdb.set_trace()
#             #     pass
#             event_list.append(event)  # predicted event list
#
#         pred_sample = copy.deepcopy(sample)
#         # change to the predicted one, if not, will use ground truth to score
#         pred_sample["event_list"] = event_list
#         # pdb.set_trace()
#         return pred_sample

class Tagger4TFBoys(Tagger):
    def __init__(self, data, **kwargs):
        node_tag_set, edge_tag_set, role_tag_set = set(), set(), set()
        self.separator = "\u2E80"
        self.edge_link_types = {
            "SH2OH",  # subject head to object head
            "ST2OT",  # subject tail to object tail
        }
        self.add_h2t_n_t2h_links = False
        if "add_h2t_n_t2h_links" in kwargs and kwargs["add_h2t_n_t2h_links"] is True:
            self.edge_link_types = self.edge_link_types.union({
                "SH2OT",  # subject head to object tail
                "ST2OH",  # subject tail to object head
            })
            self.add_h2t_n_t2h_links = True
        for sample in data:
            for event in sample["event_list"]:
                node_tag_set.add(event["trigger_type"])
                edge_tag_set.union(
                    {self.separator.join([event["trigger_type"], link]) for link in self.edge_link_types})
                role_tag_set.add(self.separator.join(["Trigger", event["trigger_type"]]))
                for arg in event["argument_list"]:
                    role_tag_set.add(arg["type"])

        self.node_tag2id = {tag: idx for idx, tag in enumerate(sorted(node_tag_set))}
        self.edge_tag2id = {tag: idx for idx, tag in enumerate(sorted(edge_tag_set))}
        self.role_tag2id = {tag: idx for idx, tag in enumerate(sorted(role_tag_set))}

    # def additional_preprocess(cls, data, data_type, **kwargs):
    #     return data

    def get_tag_size(self):
        return len(self.node_tag2id), len(self.edge_tag2id), len(self.role_tag2id)

    def get_tag_points(self, sample):
        '''
       matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
       '''
        node_matrix_points, edge_matrix_points, role_matrix_points = [], [], []

        for event in sample["event_list"]:
            event_type = event["trigger_type"]
            pseudo_arg = {
                "type": self.separator.join(["Trigger", event_type]),
                "tok_span": event["trigger_tok_span"],
            }

            node_list = event["argument_list"] + [pseudo_arg, ]

            role_matrix_points_event = []
            # nodes and roles
            for node in node_list:
                point = (event["tok_span"][0],
                         event["tok_span"][-1] - 1,
                         self.node_tag2id[event_type],
                         )
                node_matrix_points.append(point)
                point = (event["tok_span"][0],
                         event["tok_span"][-1] - 1,
                         self.node_tag2id[node["type"]],
                         )
                role_matrix_points_event.append(point)
            role_matrix_points.append(Preprocessor.unique_list(role_matrix_points_event))

            # edges
            for n_i in node_list:
                for n_j in node_list:
                    subj_tok_span = n_i["tok_span"]
                    obj_tok_span = n_j["tok_span"]

                    edge_matrix_points.append(
                        (subj_tok_span[0], obj_tok_span[0],
                         self.edge_tag2id[self.separator.join([event_type, "SH2OH"])]))
                    edge_matrix_points.append(
                        (subj_tok_span[1] - 1, obj_tok_span[1] - 1,
                         self.edge_tag2id[self.separator.join([event_type, "ST2OT"])]))

                    if self.add_h2t_n_t2h_links:
                        edge_matrix_points.append(
                            (subj_tok_span[0], obj_tok_span[1] - 1,
                             self.edge_tag2id[self.separator.join([event_type, "SH2OT"])]))
                        edge_matrix_points.append(
                            (subj_tok_span[1] - 1, obj_tok_span[0],
                             self.edge_tag2id[self.separator.join([event_type, "ST2OH"])]))

        return Preprocessor.unique_list(node_matrix_points), \
               Preprocessor.unique_list(edge_matrix_points), \
               role_matrix_points

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            node_points, edge_points, role_points = self.get_tag_points(sample)
            sample["node_points"] = node_points
            sample["edge_points"] = edge_points
            sample["role_points"] = role_points
        return data

    # to do
    def decode(self, sample, pred_tags):
        rel_list, ent_list = [], []
        pred_ent_tag, pred_rel_tag = pred_tags[0], pred_tags[1]
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
            ent_text = text[char_sp[0]:char_sp[1]]

            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": tok_sp,
                "char_span": char_sp,
            }
            ent_list.append(entity)

        cand_ent_list4rel = []
        for ent in ent_list:
            if self.classify_entities_by_relation and "REL:" in ent["type"]:
                new_ent = copy.deepcopy(ent)
                new_ent["type"] = re.sub("REL:", "", new_ent["type"])
                cand_ent_list4rel.append(new_ent)
            else:
                cand_ent_list4rel.append(ent)

        rel2link_type_map = {}
        for pt in rel_points:
            tag = self.id2rel_tag[pt[2]]
            rel, link_type = tag.split(self.separator)
            if rel not in rel2link_type_map:
                rel2link_type_map[rel] = {}
            index_pair = "{},{}".format(pt[0], pt[1])
            if index_pair not in rel2link_type_map[rel]:
                rel2link_type_map[rel][index_pair] = set()
            rel2link_type_map[rel][index_pair].add(link_type)

        for rel, link_type_map in rel2link_type_map.items():
            for subj in cand_ent_list4rel:
                for obj in cand_ent_list4rel:
                    h2h_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][0])
                    t2t_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    h2t_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][1] - 1)
                    t2h_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][0])

                    rel_exist = False
                    if self.add_h2t_n_t2h_links:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids] and \
                                h2t_ids in link_type_map and "SH2OT" in link_type_map[h2t_ids] and \
                                t2h_ids in link_type_map and "ST2OH" in link_type_map[t2h_ids]:
                            rel_exist = True
                    else:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids]:
                            rel_exist = True
                    if rel_exist:
                        rel_list.append({
                            "subject": subj["text"],
                            "object": obj["text"],
                            "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                            "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                            "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                            "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                            "predicate": rel,
                        })

        pred_sample = copy.deepcopy(sample)
        # filter extra relations
        pred_sample["relation_list"] = Preprocessor.unique_list(
            [rel for rel in rel_list if "EXT:" not in rel["predicate"]])
        # filter extra entities
        ent_types2filter = {"REL:", "EXT:"}
        ent_filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = Preprocessor.unique_list(
            [ent for ent in ent_list if re.search(ent_filter_pattern, ent["type"]) is None])
        return 0

    def decode_batch(self, sample_list, batch_pred_tags, batch_pred_outputs=None):
        pred_sample_list = []
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            pred_tags = [batch_pred_tag[ind] for batch_pred_tag in batch_pred_tags]
            pred_sample = self.decode(sample, pred_tags)  # decoding one sample
            pred_sample_list.append(pred_sample)
        return pred_sample_list


class NERTagger4RAIN(Tagger):
    def __init__(self, data, *args, **kwargs):
        '''
        :param data: all data, used to generate tags
        '''
        self.language = kwargs["language"]

        self.separator = "\u2E80"
        shaking_link_types = {
            "EH2ET",  # continuous entity head to tail
            "BH2BT",  # discontinuous entity boundary head to boundary tail
            # "SH2ST-B",  # segment head to segment tail, Begin
            # "SH2ST-I",  # segment head to segment tail, Inside
        }
        matrix_link_types = {"CLIQUE"}
        ent_type_set = set()
        for sample in data:
            for ent in sample["entity_list"]:
                ent_type_set.add(ent["type"])
        self.shaking_tags = {self.separator.join([ent_type, lt]) for ent_type in ent_type_set for lt in
                             shaking_link_types}
        self.matrix_tags = {self.separator.join([ent_type, lt]) for ent_type in ent_type_set for lt in
                            matrix_link_types}

        self.shk_tag2id = {t: idx for idx, t in enumerate(sorted(self.shaking_tags))}
        self.id2shk_tag = {idx: t for t, idx in self.shk_tag2id.items()}

        self.mt_tag2id = {t: idx for idx, t in enumerate(sorted(self.matrix_tags))}
        self.id2mt_tag = {idx: t for t, idx in self.mt_tag2id.items()}

    def get_tag_size(self):
        return len(self.shk_tag2id), len(self.mt_tag2id)

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            ent_points, rel_points = self.get_tag_points(sample)
            sample["ent_points"] = ent_points
            sample["rel_points"] = rel_points
        return data

    def get_tag_points(self, sample):
        '''
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        shk_points, mt_points = [], []
        for ent in sample["entity_list"]:
            ent_type = ent["type"]
            if len(ent["tok_span"]) == 2:
                # continuous entity
                point = (ent["tok_span"][0], ent["tok_span"][-1] - 1,
                         self.shk_tag2id[self.separator.join([ent_type, "EH2ET"])])
                shk_points.append(point)
            else:
                # disc boundary
                point = (ent["tok_span"][0], ent["tok_span"][-1] - 1,
                         self.shk_tag2id[self.separator.join([ent_type, "BH2BT"])])
                shk_points.append(point)

                # # discontinuous segments
                # for idx_i in range(0, len(ent["char_span"]), 2):
                #     seg_tok_span = [ent["tok_span"][idx_i], ent["tok_span"][idx_i + 1]]
                #
                #     position_tag = "SH2ST-B" if idx_i == 0 else "SH2ST-I"
                #     tag = self.separator.join([ent_type, position_tag])
                #     point = (seg_tok_span[0], seg_tok_span[-1] - 1, self.shk_tag2id[tag])
                #     shk_points.append(point)

            # relations
            tok_ids = utils.spans2ids(ent["tok_span"])
            for i in tok_ids:
                for j in tok_ids:
                    # if i == j:
                    #     continue
                    point = (i, j, self.mt_tag2id[self.separator.join([ent_type, "CLIQUE"])])
                    mt_points.append(point)

        return Preprocessor.unique_list(shk_points), Preprocessor.unique_list(mt_points)

    def decode(self, sample, pred_tags):
        '''
        sample: to provide tok2char_span map and text
        pred_tags: predicted tags
        '''

        pred_ent_tag, pred_rel_tag = pred_tags[0], pred_tags[1]
        shk_points = Indexer.shaking_seq2points(pred_ent_tag)
        mt_points = Indexer.matrix2points(pred_rel_tag)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        ent_list = []
        ent_type2boundaries = {}
        ent_type2graph = {}
        memory = set()
        for pt in shk_points:
            shk_tag = self.id2shk_tag[pt[2]]
            ent_type, link_type = shk_tag.split(self.separator)

            if link_type == "BH2BT":
                boundary = [pt[0], pt[1] + 1]
                if ent_type not in ent_type2boundaries:
                    ent_type2boundaries[ent_type] = []
                ent_type2boundaries[ent_type].append(boundary)
            else:
                assert link_type == "EH2ET"
                tok_span = [pt[0], pt[1] + 1]
                char_span = Preprocessor.tok_span2char_span(tok_span, tok2char_span)
                ent_txt = Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text, self.language)
                mem = "{},{}".format(ent_type, str(char_span))
                if mem not in memory:
                    ent_list.append({
                        "text": ent_txt,
                        "char_span": char_span,
                        "tok_span": tok_span,
                        "type": ent_type,
                    })
                    memory.add(mem)

        for pt in mt_points:
            mt_tag = self.id2mt_tag[pt[2]]
            ent_type, link_type = mt_tag.split(self.separator)
            if link_type == "CLIQUE":
                if ent_type not in ent_type2graph:
                    ent_type2graph[ent_type] = nx.Graph()
                ent_type2graph[ent_type].add_edge(pt[0], pt[1])

        for ent_type, boundaries in ent_type2boundaries.items():
            if ent_type not in ent_type2graph:
                continue
            graph = ent_type2graph[ent_type]
            for bd in boundaries:
                tok_ids = list(range(*bd))
                cliques = list(nx.find_cliques(nx.induced_subgraph(graph, tok_ids)))
                cliques = [cli for cli in cliques if tok_ids[0] == min(cli) and tok_ids[-1] == max(cli)]

                for cli in cliques:
                    cli = sorted(cli)
                    tok_span = utils.ids2span(cli)
                    char_span = Preprocessor.tok_span2char_span(tok_span, tok2char_span)
                    ent_txt = Preprocessor.extract_ent_fr_txt_by_char_sp(char_span, text, self.language)
                    mem = "{},{}".format(ent_type, str(char_span))
                    if mem not in memory:
                        ent_list.append({
                            "text": ent_txt,
                            "char_span": char_span,
                            "tok_span": tok_span,
                            "type": ent_type,
                        })
                        memory.add(mem)

        pred_sample = copy.deepcopy(sample)
        pred_sample["entity_list"] = ent_list

        # sc_dict = MetricsCalculator.get_ent_cpg_dict([pred_sample], [sample])
        # for sck, sc in sc_dict.items():
        #     if sc[0] != sc[2] or sc[0] != sc[1]:
        #         print("1")
        return pred_sample
