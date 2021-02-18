import numpy as np

from finetune.encoding.input_encoder import EncodedOutput
from finetune.encoding.target_encoders import SequenceLabelingEncoder

def test_sequence_label_encoder():
    encoder = SequenceLabelingEncoder(pad_token="<PAD>")
    labels = [{'start': 13, 'end': 17, 'label': 'z', 'text': '(5%)'}]
    encoder.fit([labels])
    out = EncodedOutput(
        token_ids=np.array([   0, 9583,  139,   40,  249, 8875,    2]), 
        tokens=np.array(['0', 'five', ' percent', ' (', '5', '%)', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  4, 12, 14, 15, 17, -1]), 
        token_starts=np.array([-1,  0,  5, 13, 14, 15, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["five percent (5%)"]
    )
    label_arr = encoder.transform(out, labels)
    assert label_arr == [0, 0, 0, 1, 1, 1, 0]
    

def test_sequence_label_encoder_does_not_exceed_half():
    encoder = SequenceLabelingEncoder(pad_token="<PAD>")
    labels = [{'start': 0, 'end': 2, 'label': 'z', 'text': 'to'}]
    encoder.fit([labels])
    out = EncodedOutput(
        token_ids=np.array([    0, 46661,     2]), 
        tokens=np.array(['0', 'token', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  5, -1]), 
        token_starts=np.array([-1,  0, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["token"]
    )
    label_arr = encoder.transform(out, labels)
    assert label_arr == [0, 0, 0]
    

def test_sequence_label_encoder_exceeds_half():
    encoder = SequenceLabelingEncoder(pad_token="<PAD>")
    labels = [{'start': 0, 'end': 3, 'label': 'z', 'text': 'tok'}]
    encoder.fit([labels])
    out = EncodedOutput(
        token_ids=np.array([    0, 46661,     2]), 
        tokens=np.array(['0', 'token', '2'], dtype='<U21'), 
        token_ends=np.array([-1,  5, -1]), 
        token_starts=np.array([-1,  0, -1]), 
        useful_start=0, 
        useful_end=512,
        input_text=["token"]

    )
    label_arr = encoder.transform(out, labels)
    assert label_arr == [0, 1, 0]
    

def test_sequence_label_encoder_half():
    encoder = SequenceLabelingEncoder(pad_token="<PAD>")
    labels = [{'start': 0, 'end': 3, 'label': 'z', 'text': 'str'}]
    encoder.fit([labels])
    out = EncodedOutput(
        token_ids=np.array([   0, 8660,    2]), 
        tokens=np.array(['0', 'stream', '2'], dtype='<U21'),
        token_ends=np.array([-1,  6, -1]),
        token_starts=np.array([-1,  0, -1]), 
        useful_start=0,
        useful_end=512,
        input_text=["stream"]
    )
    label_arr = encoder.transform(out, labels)
    assert label_arr == [0, 1, 0]
