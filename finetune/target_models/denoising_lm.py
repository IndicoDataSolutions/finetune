from finetune.errors import FinetuneError
from finetune.target_models.masked_language_model import MaskedLanguageModel, MaskedLanguageModelPipeline


class DenoisingLanguage&AuxModel(MaskedLanguageModelPipeline):
    def feed_shape_type_def(self):
        if not self.config.use_auxiliary_info:
            raise FinetuneError('Denoising regressor assumes context witih continuous values is present.')
        return super().feed_shape_type_def(self)
    
    def text_to_tokens_mask(self, X, Y=None, context=None, forced_mask=None):
        feats = super().text_to_tokens_mask(self, X=X, Y=Y, context=context, forced_mask=forced_mask)
        # mask ids should be for the context as opposed to the text
        mlm_ids = 