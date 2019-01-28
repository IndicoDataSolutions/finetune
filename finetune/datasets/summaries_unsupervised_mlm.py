import logging

from newsroom import jsonl
import random
from finetune import Classifier
import ujson as _json
import glob

logging.basicConfig(level=logging.DEBUG)
DATASET_PATH = "/root/code/Data/newsroom/train_half.data"
#WIKI_PATHS = glob.glob("/root/code/Downloads/wiki/Wikipedia_corpus_*.txt")

class jsonl_fixed(jsonl.open):
    def readlines(self, ignore_errors=False):
        """
        Adds correct handling of GeneratorExit exception.
        """
        if not ignore_errors:
            for line in self._readfile():
                yield _json.loads(line)
        else:
            for ln, line in enumerate(self._readfile()):
                try:
                    yield _json.loads(line)
                except GeneratorExit:
                    self.close()
                    break
                except:
                    print("Decoding error on line", ln)


def get_dataset():
    def get_summaries_lines():
        with jsonl_fixed(DATASET_PATH, gzip=True) as fp:
            for i, sample in enumerate(fp.readlines(ignore_errors=True)):
                yield sample["text"].replace("/n", " ")

    for item in get_summaries_lines():
        tokens = item.split(" ")
        sample = " ".join(tokens[random.randint(0, max(1, len(tokens) - 512)):])
        if sample:
            yield sample


if __name__ == "__main__":
    model = Classifier(low_memory_mode=True, verbose=True, n_epochs=10, batch_size=16, max_grad_norm=0.1, tensorboard_folder='summaries_mlm_run', dataset_size=450992, shuffle_buffer_size=50, mlm=True, lr=6.25e-5, keep_best_model=True, take_snapshots=True)
    model.fit(get_dataset)
    model.save("SummariesMLM.jl")
