import os
import json

import unittest
import numpy as np
import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModel, BertTokenizer
from finetune import SequenceLabeler, DocumentLabeler
from finetune.base_models import LayoutLM
from finetune.base_models.huggingface.models import (
    HFBert,
    HFElectraGen,
    HFElectraDiscrim,
    HFXLMRoberta,
    HFT5,
    HFAlbert,
)
from finetune.target_models.seq2seq import HFS2S
from sklearn.model_selection import train_test_split
from finetune.encoding.sequence_encoder import finetune_to_indico_sequence
try:
    import torch
    from finetune.base_models.huggingface.hf_layoutlm import LayoutlmModel
    TORCH_SUPPORT = True
except ImportError:
    HFLayoutLM = None
    LayoutlmModel = None
    TORCH_SUPPORT = False


class TestHuggingFace(unittest.TestCase):
    def setUp(self):
        with open('tests/data/weird_text.txt') as f:
            weird_text = ''.join(f.readlines())
        self.text = weird_text[:1000]

    def check_embeddings_equal(self, finetune_base_model, hf_model_path, **kwargs):
        finetune_model = SequenceLabeler(
            base_model=finetune_base_model,
            train_embeddings=False,
            n_epochs=1,
            batch_size=1,
        )
        finetune_seq_features = finetune_model.featurize_sequence([self.text])[0]
        hf_seq_features = self.huggingface_embedding(self.text, hf_model_path, **kwargs)[0]
        if len(finetune_seq_features) + 2 == len(hf_seq_features):
            hf_seq_features = hf_seq_features[1:-1]
        np.testing.assert_array_almost_equal(
            finetune_seq_features, hf_seq_features, decimal=5,
        )
        finetune_model.fit([self.text], [[{"start": 0, "end": 4, "label": "class_a"}]])

    def huggingface_embedding(self, text, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModel.from_pretrained(model_path)
        input_ids = tf.constant(tokenizer.encode(self.text))[None, :]  # Batch size 1

        if model.config.is_encoder_decoder:
            outputs = model.encoder(input_ids)
        else:
            outputs = model(
                input_ids,
            )  # outputs is tuple where first element is sequence features
        last_hidden_states = outputs[0]
        return last_hidden_states.numpy()

    def test_xlm_roberta(self):
        self.check_embeddings_equal(HFXLMRoberta, "jplu/tf-xlm-roberta-base")

    def test_bert(self):
        self.check_embeddings_equal(HFBert, "bert-base-uncased")

    def test_t5_s2s(self):
        text = "sequence test text"
        finetune_model = HFS2S(
            base_model=HFT5,
            n_epochs=30,
            batch_size=2,
        )
        finetune_model.fit([text] * 5, [text] * 5)
        finetune_model.save("test.jl")
        self.assertEqual(finetune_model.predict([text]), [text])
        self.assertEqual(len(finetune_model.predict([text, "Some other text"])), 2)
        loaded_model = HFS2S.load("test.jl")
        self.assertEqual(loaded_model.predict([text]), [text])

    def test_t5_s2s_ner(self):
        with open(os.path.join('Data', 'Sequence', 'reuters.json'), "rt") as fp:
            texts, labels = json.load(fp)
        raw_docs = ["".join(text) for text in texts]
        texts, annotations = finetune_to_indico_sequence(raw_docs, texts, labels, none_value="<PAD>")
        labels = [" | ".join(x["text"] for x in a) for a in annotations]
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, labels, test_size=0.1, random_state=42
        )
        finetune_model = HFS2S(
            base_model=HFT5,
            n_epochs=3,
            batch_size=2,
        )
        finetune_model.fit(train_texts, train_annotations)
        seps_included = False
        pred_correct = False
        for t, p, l in zip(test_texts, finetune_model.predict(test_texts), test_annotations):
            seps_included = seps_included or " | " in p # check it predicts separators
            pred_correct = pred_correct or any(li in p for li in l.split(" | ")) # at least one extraction is predicted correctly.

        self.assertTrue(seps_included)
        self.assertTrue(pred_correct)

    def test_t5(self):
        self.check_embeddings_equal(HFT5, "t5-base")


    def test_albert(self):
        self.check_embeddings_equal(HFAlbert, "albert-base-v2")

    @unittest.skipIf(not TORCH_SUPPORT, reason="Pytorch not installed")
    def test_layoutlm(self):
        def format_ondoc_for_hf(document, tokenizer):
            input_dict = {
                "input_ids": [],
                "bbox": []
            }
            tokens = []
            token_boxes = []
            for page in doc:
                for token in page["tokens"]:
                    word_tokens = tokenizer.tokenize(token["text"])
                    tokens.extend(word_tokens)
                    box = [
                        int(token["position"]["left"] / page["pages"][0]["size"]["width"] * 1000),
                        int(token["position"]["top"] / page["pages"][0]["size"]["height"] * 1000),
                        int(token["position"]["right"] / page["pages"][0]["size"]["width"] * 1000),
                        int(token["position"]["bottom"] / page["pages"][0]["size"]["height"] * 1000),
                    ]
                    token_boxes.extend([box] * len(word_tokens))
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            pad_len = len(input_ids)
            input_ids = np.pad(input_ids, (0, pad_len))
            token_boxes = token_boxes + [[0, 0, 0, 0]] * pad_len
            print(len(input_ids))
            input_dict["input_ids"] = torch.LongTensor(input_ids).unsqueeze(0)
            input_dict["bbox"] = torch.LongTensor(token_boxes).unsqueeze(0)
            return input_dict

        def subset_doc(doc):
            """ Take approximately the first 512 tokens from just the first page """
            for page in doc:
                page["tokens"] = page["tokens"][:60]
                page_idx_end = page["tokens"][-1]["page_offset"]["end"]
                page["pages"][0]["text"] = page["pages"][0]["text"][:page_idx_end]
                page["pages"][0]["doc_offset"]["end"] = page["pages"][0]["doc_offset"]["start"] + page_idx_end
                return [page]

        with open("tests/data/test_ocr_documents.json", "rt") as fp:
            documents = json.load(fp)
        # hack so we don't have to do chunking for the HF model
        documents = [subset_doc(doc) for doc in documents]
        tokenizer = BertTokenizer.from_pretrained("finetune/model/layoutlm-base-uncased/")
        model = LayoutlmModel.from_pretrained("finetune/model/layoutlm-base-uncased/")
        # turn off dropout
        model.eval()
        finetune_model = DocumentLabeler(base_model=LayoutLM, config=dict(
            embed_p_drop=0.0, attn_p_drop=0.0, resid_p_drop=0.0, clf_p_drop=0.0
        ))
        # we run a single doc at a time so the output shapes match
        # and we avoid having to write padding and pad removal code for HF
        for doc in documents:
            input_dict = format_ondoc_for_hf(doc, tokenizer)
            outputs = model(**input_dict)
            hf_seq_features = outputs[0].detach().numpy()

            finetune_seq_features = finetune_model.featurize_sequence([doc])
            seq_len = finetune_seq_features.shape[1]
            np.testing.assert_array_almost_equal(
                finetune_seq_features, hf_seq_features[:, :seq_len, :], decimal=5,
            )
