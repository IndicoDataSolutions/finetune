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
    def get_featurizer(cls, X, encoder, config, train=False, reuse=None):
        return cls.featurizer(X, encoder, config, train=train, reuse=reuse)
