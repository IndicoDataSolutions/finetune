import os
import unittest
import warnings

# prevent excessive warning logs 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from finetune import Comparison
from finetune.config import get_config
from finetune.utils import list_transpose
import random

SST_FILENAME = "SST-binary.csv"


class TestComparison(unittest.TestCase):
    n_sample = 20
    n_hidden = 768
    dataset_path = os.path.join(
        'Data', 'Classify', 'SST-binary.csv'
    )

    def default_config(self, **kwargs):
        return get_config(
            batch_size=2,
            max_length=128,
            n_epochs=1,
            verbose=False,
            **kwargs
        )

    def setUp(self):
        tf.reset_default_graph()

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Comparison(config=self.default_config())
        model.fit(["Indico is the best"] * 10, ["Indico is the bestestestest"] * 10, np.array([0] * 10))

        predictions = model.predict(["Is indico the best?"], ["Indico is the bestestestest"])
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

    def test_reasonable_predictions(self):
        model = Comparison(config=self.default_config(), n_epochs=3)
        animals = ["The cat says Meow.",
                   "The chicken says sqwark.",
                   "The fish says bloop.",
                   "The dog says Woof.",
                   "The frog says \"ribbet\"."]

        prince_philip_quotes = ["When a man opens a car door for his wife, it's either a new car or a new wife.",
                                "All money nowadays seems to be produced with a natural homing instinct for the Treasury",
                                "Ah you’re the one who wrote the letter. So you can write then? Ha, ha! Well done",
                                "You were playing your instruments? Or do you have tape recorders under your seats?",
                                "Oh! You are the people ruining the rivers and the environment"]

        same = []
        different = []
        n = min(len(animals), len(prince_philip_quotes))
        for j in range(n):
            for i in range(n):
                different.append([animals[i], prince_philip_quotes[j]])
                different.append([prince_philip_quotes[i], animals[j]])
                if i == j:
                    pass
                same.append([animals[i], animals[j]])
                same.append([prince_philip_quotes[i], prince_philip_quotes[j]])

        targets = np.array(["S"] * len(same) + ["D"] * len(different))
        data = same + different

        x_tr, x_te, t_tr, t_te = train_test_split(data, targets, train_size=0.03)

        model.finetune(*list_transpose(x_tr), t_tr)
        accuracy = np.mean(model.predict(*list_transpose(x_te)) == t_te)
        print("Accuracy = {}".format(accuracy))
        self.assertGreater(accuracy, max(np.mean(targets == "S"), np.mean(targets == "D")))
