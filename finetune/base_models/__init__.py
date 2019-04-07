from abc import ABCMeta, abstractmethod

class SourceModel(metaclass=ABCMeta):

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
from finetune.base_models.gpt2.model import GPT2Model
