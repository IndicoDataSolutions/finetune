from abc import ABCMeta, abstractmethod


class SourceModel(metaclass=ABCMeta):
    is_bidirectional = True

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
    def get_encoder(cls, **kwargs):
        return cls.encoder(**kwargs)

    @classmethod
    def get_featurizer(cls, X, encoder, config, train=False, reuse=None, **kwargs):
        return cls.featurizer(X, encoder, config, train=train, reuse=reuse, **kwargs)


from finetune.base_models.gpt.model import GPTModel, GPTModelSmall
from finetune.base_models.gpt2.model import GPT2Model, GPT2Model345, GPT2Model762, GPT2Model1558
from finetune.base_models.textcnn.model import TextCNNModel
from finetune.base_models.bert.model import BERTModelCased, BERTModelLargeCased, RoBERTa, RoBERTaLarge, DistilBERT, DistilRoBERTa
from finetune.base_models.tcn.model import TCNModel

# Aliases
GPT = GPTModel
GPT2 = GPT2Small = GPT2Model
GPT2Medium = GPT2Model345
GPT2Large = GPT2Model762
GPT2XL = GPT2Model1558
TextCNN = TextCNNModel
BERT = BERTModelCased
BERTLarge = BERTModelLargeCased
ROBERTA = RoBERTa
ROBERTALarge = RoBERTaLarge
DistilROBERTA = DistilRoBERTa
TCN = TCNModel
