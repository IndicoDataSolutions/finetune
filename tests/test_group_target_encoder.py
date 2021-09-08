import numpy as np
import json

from finetune.target_models.seq2seq import HFS2S
from finetune.base_models.huggingface.models import HFT5
from finetune.encoding.input_encoder import EncodedOutput
from finetune.encoding.group_target_encoders import (
    GroupSequenceLabelingEncoder, 
    PipelineSequenceLabelingEncoder, 
    MultiCRFGroupSequenceLabelingEncoder,
    BROSEncoder,
    JointBROSEncoder,
    GroupRelationEncoder,
    SequenceLabelingTextEncoder,
    GroupLabelingTextEncoder,
    JointLabelingTextEncoder,
)

def test_nest_group_sequence_label():
    encoder = GroupSequenceLabelingEncoder(pad_token="<PAD>", bio_tagging=True)
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 12, 'label': 'z', 'text': 'percent'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 5, 'end': 17, 'text': 'percent (5%)'},
        ], 'label': None}
    ]
    label = (labels, groups)
    encoder.fit([label])
    assert len(encoder.classes_) == 9
    out = EncodedOutput(
        token_ids=np.array([   0, 9583,  139,   40,  249, 8875,    2]), 
        tokens=np.array(['0', 'five', ' per', 'cent', ' (', '5', '%)', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  4, 8, 12, 14, 15, 17, -1]), 
        token_starts=np.array([-1,  0,  5, 8, 13, 14, 15, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["five percent (5%)"],
    )
    label_arr = encoder.transform(out, label)
    # Classes are [PAD, BG-PAD, IG-PAD, B-z, BG-B-z, IG-B-z, I-z, BG-I-z, IG-I-z]
    assert label_arr == [0, 3, 4, 8, 5, 8, 8, 0]

def test_multi_crf_group_sequence_label():
    encoder = MultiCRFGroupSequenceLabelingEncoder(pad_token="<PAD>", bio_tagging=True)
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 12, 'label': 'z', 'text': 'percent'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 5, 'end': 17, 'text': 'percent (5%)'},
        ], 'label': None}
    ]
    label = (labels, groups)
    encoder.fit([label])
    assert len(encoder.classes_) == 3
    assert len(encoder.group_classes_) == 3
    out = EncodedOutput(
        token_ids=np.array([   0, 9583,  139,   40,  249, 8875,    2]), 
        tokens=np.array(['0', 'five', ' per', 'cent', ' (', '5', '%)', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  4, 8, 12, 14, 15, 17, -1]), 
        token_starts=np.array([-1,  0,  5, 8, 13, 14, 15, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["five percent (5%)"],
    )
    label_arr = encoder.transform(out, label)
    # First list is NER information ([PAD, B-z, I-z])
    # Second list is group information ([Not in group, BG-, IG-])
    assert label_arr == [[0, 1, 1, 2, 1, 2, 2, 0], [0, 0, 1, 2, 2, 2, 2, 0]]

def test_pipeline_group_sequence_label():
    encoder = PipelineSequenceLabelingEncoder(pad_token="<PAD>", bio_tagging=True)
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 12, 'label': 'z', 'text': 'percent'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 13, 'end': 17, 'text': '(5%)'},
        ], 'label': None}
    ]
    label = (labels, groups)
    encoder.fit([label])
    assert len(encoder.classes_) == 3
    out = EncodedOutput(
        token_ids=np.array([   0, 9583,  139,   40,  249, 8875,    2]), 
        tokens=np.array(['0', 'five', ' per', 'cent', ' (', '5', '%)', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  4, 8, 12, 14, 15, 17, -1]), 
        token_starts=np.array([-1,  0,  5, 8, 13, 14, 15, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["five percent (5%)"],
    )
    label_arr = encoder.transform(out, label)
    # Just group information ([Not in group, BG-, IG-])
    assert label_arr == [0, 0, 0, 0, 1, 2, 2, 0]

def test_BROS_sequence_label():
    encoder = BROSEncoder(pad_token="<PAD>")
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 12, 'label': 'z', 'text': 'percent'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 5, 'end': 8, 'text': 'per'},
            {'start': 13, 'end': 17, 'text': '(5%)'},
        ], 'label': None},
        {"spans": [
            {'start': 8, 'end': 12, 'text': 'cent'},
        ], 'label': None}
    ]
    label = (labels, groups)
    encoder.fit([label])
    assert len(encoder.classes_) == 4
    out = EncodedOutput(
        token_ids=np.array([   0, 9583,  139,   40,  249, 8875,    2]), 
        tokens=np.array(['0', 'five', ' per', 'cent', ' (', '5', '%)', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  4, 8, 12, 14, 15, 17, -1]), 
        token_starts=np.array([-1,  0,  5, 8, 13, 14, 15, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["five percent (5%)"],
    )
    label_arr = encoder.transform(out, label)
    # First list is start token information (Binary indicator of group start)
    # Second list is next token information (Index of next token in group)
    assert label_arr == [[0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 4, 0, 5, 6, 0, 0]]

def test_joint_BROS_sequence_label():
    encoder = JointBROSEncoder(pad_token="<PAD>", bio_tagging=True)
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 12, 'label': 'z', 'text': 'percent'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 5, 'end': 8, 'text': 'per'},
            {'start': 13, 'end': 17, 'text': '(5%)'},
        ], 'label': None},
        {"spans": [
            {'start': 8, 'end': 12, 'text': 'cent'},
        ], 'label': None}
    ]
    label = (labels, groups)
    encoder.fit([label])
    print(encoder.classes_)
    assert len(encoder.ner_classes_) == 3
    assert len(encoder.group_classes_) == 4
    out = EncodedOutput(
        token_ids=np.array([   0, 9583,  139,   40,  249, 8875,    2]), 
        tokens=np.array(['0', 'five', ' per', 'cent', ' (', '5', '%)', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  4, 8, 12, 14, 15, 17, -1]), 
        token_starts=np.array([-1,  0,  5, 8, 13, 14, 15, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["five percent (5%)"],
    )
    label_arr = encoder.transform(out, label)
    # First list is standard sequence labeling label ([PAD, B-z, I-z])
    # Second list is start token information (Binary indicator of group start)
    # Third list is next token information (Index of next token in group)
    assert label_arr == [
        [0, 1, 1, 2, 1, 2, 2, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 4, 0, 5, 6, 0, 0],
    ]


def test_group_relation_sequence_label():
    encoder = GroupRelationEncoder(pad_token="<PAD>", n_groups=5)
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 8, 'label': 'z', 'text': 'per'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 0, 'end': 4, 'text': 'five'},
            {'start': 13, 'end': 17, 'text': '(5%)'},
        ], 'label': None},
        {"spans": [
            {'start': 5, 'end': 12, 'text': 'percent'},
        ], 'label': None},
    ]
    label = (labels, groups)
    encoder.fit([label])
    out = EncodedOutput(
        token_ids=np.array([   0, 9583,  139,   40,  249, 8875,    2]), 
        tokens=np.array(['0', 'five', ' per', 'cent', ' (', '5', '%)', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  4, 8, 12, 14, 15, 17, -1]), 
        token_starts=np.array([-1,  0,  5, 8, 13, 14, 15, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["five percent (5%)"],
    )
    label_arr = encoder.transform(out, label)
    # Should be of shape [n_groups, seq_len]
    # Is 1 when the token belongs to the respective group, 0 otherwise
    assert label_arr == [
        [0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1]
    ]

def test_seq2seq_sequence_label():
    model = HFS2S(base_model=HFT5)
    pipeline = HFS2S._get_input_pipeline(model)
    encoder = SequenceLabelingTextEncoder(pipeline, 512)
    text = "five percent (5%)"
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 8, 'label': 'z', 'text': 'per'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    label_arr = encoder.transform([labels])[0][0]
    assert isinstance(label_arr, np.ndarray) # token ids
    labels_out = encoder.inverse_transform([label_arr], [text])[0]
    assert labels_out == labels # Round trip back to labels

def test_seq2seq_group_label():
    model = HFS2S(base_model=HFT5)
    pipeline = HFS2S._get_input_pipeline(model)
    encoder = GroupLabelingTextEncoder(pipeline, 512)
    text = "five percent (5%) \n test"
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 8, 'label': 'z', 'text': 'per'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 5, 'end': 8, 'text': 'per'},
            {'start': 13, 'end': 24, 'text': '(5%) \n test'},
        ], 'label': None},
        {"spans": [
            {'start': 8, 'end': 12, 'text': 'cent'},
        ], 'label': None}
    ]
    label_arr = encoder.transform([(labels, groups)])[0][0]
    assert isinstance(label_arr, np.ndarray) # token ids                                                                                              
    groups_out = encoder.inverse_transform([label_arr], [text])[0] # Takes labels and groups in but only encodes groups?
    assert groups_out == groups

def test_seq2seq_joint_label():
    model = HFS2S(base_model=HFT5)
    pipeline = HFS2S._get_input_pipeline(model)
    encoder = JointLabelingTextEncoder(pipeline, 512)
    text = "five percent (5%)"
    labels = [
        {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
        {'start': 5, 'end': 8, 'label': 'z', 'text': 'per'},
        {'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'},
    ]
    groups = [
        {"spans": [
            {'start': 5, 'end': 8, 'text': 'per'},
            {'start': 13, 'end': 17, 'text': '(5%)'},
        ], 'label': None},
        {"spans": [
            {'start': 8, 'end': 12, 'text': 'cent'},
        ], 'label': None}
    ]
    label_arr = encoder.transform([(labels, groups)])[0][0]
    assert isinstance(label_arr, np.ndarray)
    labels_out, groups_out = encoder.inverse_transform([label_arr], [text])[0]
    assert labels_out == labels
    assert groups_out == groups
