import torch
class EETagger:
    def __init__(self, tags, max_seq_len):
        super().__init__()
        self.tag2id = {t: idx for idx, t in enumerate(sorted(tags))}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len

    def get_spots(self, sample):
        '''
        This function is for indexing data

        sample: an example of data
        return spots for tagging
        spot: (start_pos, end_pos, tag_id)
        '''
        pass

    def spots2tag_batch(self, spots_batch):
        '''
        This function is for generation tag tensors for training

        convert spots to a tag matrix
        spots_batch:
            [batch1, batch2, ....]
            batch1: [(start_pos, end_pos, tag_id), ]
        return: tag matrix (a tensor)
        '''
        pass

    def get_spots_fr_tag_matrix(self, tag_matrix):
        '''
        This function if for supporting the decoding function below

        tag_matrix: the tag matrix
        return matrix_spots: [(start_pos, end_pos, tag_id), ]
        '''
        matrix_spots = []
        for point in torch.nonzero(tag_matrix,
                                   as_tuple=False):  # shaking_tag.nonzero() -> torch.nonzero(shaking_tag, as_tuple = False)
            tag_id = point[2].item()
            spot = (point[0], point[1], tag_id)
            matrix_spots.append(spot)
        return matrix_spots

    def decode(self, text, tag_matrix, tok2char_span, tok_offset=0, char_offset=0):
        '''
        decoding function: to extract results by the predicted tag

        :param text: text
        :param tag_matrix: tag
        :param tok2char_span: the spots in the tag matrix is set by token level spans,
                            so this map is needed to get character level spans.
                            We need character level spans to extract entities from the text.
        :param tok_offset: if text is a subtext of test data, tok_offset and char_offset must be set to recover the spans in the original text
        :param char_offset: the same as above
        :return: extracted event list
        '''
        pass