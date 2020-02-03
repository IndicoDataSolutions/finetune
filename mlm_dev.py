from pprint import pprint
from finetune import MaskedLanguageModel
from finetune.base_models import RoBERTa


def load_model():
    model = MaskedLanguageModel(base_model=RoBERTa)
    model._cached_predict = False
    return model


model = load_model()

text_to_display = model.display_text("This is an extremely boring sentence, the point of which is to generate enough masks to test my display formatting, and give me a false sense of productivity by exercising my creative faculties in composiing it.")
pprint(len(text_to_display))
pprint(text_to_display)
