import pytest
import unittest

from finetune import SequenceLabeler
from finetune.base_models import RoBERTa

class Test2dChunk(unittest.TestCase):
    def setUp(self):
        self.x = ["Loss Total Cost 63 67 68 48 44 85 20 42 30 63 40 17"]
        self.context = [[{"top": 157, "left": 237, "bottom": 187, "right": 315, "start": 0, "end": 4, "token": "Loss"}, {"top": 157, "left": 523, "bottom": 187, "right": 617, "start": 5, "end": 10, "token": "Total"}, {"top": 157, "left": 635, "bottom": 187, "right": 713, "start": 11, "end": 15, "token": "Cost"}, {"top": 208, "left": 237, "bottom": 237, "right": 277, "start": 16, "end": 18, "token": "63"}, {"top": 207, "left": 523, "bottom": 237, "right": 565, "start": 19, "end": 21, "token": "67"}, {"top": 258, "left": 237, "bottom": 287, "right": 277, "start": 22, "end": 24, "token": "68"}, {"top": 257, "left": 523, "bottom": 287, "right": 563, "start": 25, "end": 27, "token": "48"}, {"top": 307, "left": 237, "bottom": 337, "right": 277, "start": 28, "end": 30, "token": "44"}, {"top": 308, "left": 523, "bottom": 337, "right": 562, "start": 31, "end": 33, "token": "85"}, {"top": 360, "left": 238, "bottom": 388, "right": 277, "start": 34, "end": 36, "token": "20"}, {"top": 358, "left": 523, "bottom": 387, "right": 562, "start": 37, "end": 39, "token": "42"}, {"top": 410, "left": 237, "bottom": 438, "right": 277, "start": 40, "end": 42, "token": "30"}, {"top": 410, "left": 523, "bottom": 438, "right": 562, "start": 43, "end": 45, "token": "63"}, {"top": 458, "left": 237, "bottom": 488, "right": 277, "start": 46, "end": 48, "token": "40"}, {"top": 458, "left": 526, "bottom": 488, "right": 565, "start": 49, "end": 51, "token": "17"}]]

        self.seq_labels = [[{"start": 19, "end": 21, "label": "0000ff", "text": "67"}, {"start": 25, "end": 27, "label": "0000ff", "text": "48"}, {"start": 31, "end": 33, "label": "0000ff", "text": "85"}, {"start": 37, "end": 39, "label": "0000ff", "text": "42"}, {"start": 43, "end": 45, "label": "0000ff", "text": "63"}, {"start": 49, "end": 51, "label": "0000ff", "text": "17"}]]

    def test_sequence_merging(self):
        model = SequenceLabeler(
            base_model=RoBERTa,
            chunk_long_sequences=True,
            subtoken_predictions=False,
            use_auxiliary_info=True,
            context_dim=4,
            default_context={
                'left': 0,
                'right': 0,
                'top': 0,
                'bottom': 0,
            },
            n_context_embed_per_channel=48,
            context_in_base_model=True,
            n_layers_with_aux=-1,
            val_size=0,
            val_interval=0,
            batch_size=4,
            predict_span_threshold=1.0
        )
        model.fit(self.x * 10, self.seq_labels * 10, context=self.context * 10)
        pred = model.predict(self.x, context=self.context)[0]
        self.assertEqual(len(pred), 1)
        self.assertEqual(pred[0]["text"], '67 48 85 42 63 17')
        self.assertEqual(pred[0]["start_segments"], [19, 25, 31, 37, 43, 49])
        self.assertEqual(pred[0]['end_segments'], [21, 27, 33, 39, 45, 51])
        
                
