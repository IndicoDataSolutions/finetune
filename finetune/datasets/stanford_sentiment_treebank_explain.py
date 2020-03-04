import logging
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors

from finetune import Classifier
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank
from finetune.base_models.gpt.model import GPTModel

logging.basicConfig(level=logging.DEBUG)


def sentences_heatmap(sentences, probs, clf):
    fig, axes = plt.subplots(len(sentences), 1, figsize=(20, len(sentences) * 0.95), dpi=160, squeeze=False)
    axes = axes[:, 0]

    axes[0].axis("off")

    for i, ax in enumerate(axes):
        ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                               color="r" if clf[i] == 0 else "g", linewidth=2))

        word_pos = 0.00
        cmap = plt.cm.copper
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

        for j, (word, prob) in enumerate(zip(sentences[i], probs[i])):
            ax.text(word_pos, 0.5, word,
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=8, color=cmap(norm(prob)),
                    transform=ax.transAxes, fontweight=700)
            word_pos += .0036 * (len(word) + 1)  # to move the word for the next iter
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig("test_sentences.png")


if __name__ == "__main__":
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    model = Classifier(
        n_epochs=1,
        batch_size=2,
        lr_warmup=0.1,
        val_size=0.0,
        max_length=64,
        base_model=GPTModel
    )

    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3,
                                                    random_state=42)
    model.fit(trainX, trainY)
    samples = model.explain(testX)
    words = []
    probs = []
    clfs = []

    for sample, text in zip(samples, testX):
        sample_words = []
        sample_probs = []
        for s, e, p in zip(sample["token_starts"], sample["token_ends"], sample['explanation'][1]):
            sample_words.append(text[s:e])
            sample_probs.append(p)
        print(list(zip(sample_words, sample_probs)), text)
        words.append(sample_words)
        probs.append(sample_probs)
        clfs.append(sample["prediction"])
    sentences_heatmap(words, probs, clfs)
    accuracy = np.mean(testY == [s["prediction"] for s in samples])
    print('Test Accuracy: {:0.2f}'.format(accuracy))
