from abc import ABCMeta, abstractmethod

COMMON_SETTINGS = {
    'max_length': 512,
    'batch_size': 2,
    'n_epochs': 3,
    'predict_batch_size': 20,
    'chunk_context': None
}


class SourceModel(metaclass=ABCMeta):
    is_bidirectional = True

    @classmethod
    def get_optimal_params(cls, config):
        settings = dict(COMMON_SETTINGS)
        settings.update(cls.settings)
        return settings

    @property
    @abstractmethod
    def encoder(cls):
        raise NotImplementedError

    @property
    @abstractmethod
    def featurizer(cls):
        raise NotImplementedError

    @property
    @abstractmethod
    def settings(cls):
        raise NotImplementedError

    @classmethod
    def get_encoder(cls, config=None, **kwargs):
        return cls.encoder(**kwargs)

    @classmethod
    def get_featurizer(cls, X, encoder, config, train=False, reuse=None, **kwargs):
        return cls.featurizer(X, encoder, config, train=train, reuse=reuse, **kwargs)

    @classmethod
    def translate_base_model_format(cls):
        pass


from finetune.base_models.gpt.model import GPTModel, GPTModelSmall
from finetune.base_models.gpt2.model import GPT2Model, GPT2Model345, GPT2Model762, GPT2Model1558
from finetune.base_models.textcnn.model import TextCNNModel, FastTextCNNModel
from finetune.base_models.bert.model import (
    BERTModelCased,
    BERTModelLargeCased,
    RoBERTa,
    FusedRoBERTa,
    RoBERTaLarge,
    DistilBERT,
    DistilRoBERTa,
    DocRep,
    FusedDocRep,
    LayoutLM
)
from finetune.base_models.tcn.model import TCNModel
from finetune.base_models.oscar.model import GPCModel

# Aliases
GPT = GPTModel
GPT2 = GPT2Small = GPT2Model
GPT2Medium = GPT2Model345
GPT2Large = GPT2Model762
GPT2XL = GPT2Model1558
TextCNN = TextCNNModel
FastTextCNN = FastTextCNNModel
BERT = BERTModelCased
BERTLarge = BERTModelLargeCased
ROBERTA = RoBERTa
ROBERTALarge = RoBERTaLarge
DistilROBERTA = DistilRoBERTa
TCN = TCNModel
OSCAR = GPCModel
