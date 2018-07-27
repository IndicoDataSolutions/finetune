import os
import unittest
import warnings
import random

# prevent excessive warning logs 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from finetune import Comparison
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
        d = dict(
            batch_size=2,
            max_length=128,
            n_epochs=1,
            verbose=False,
        )
        d.update(kwargs)
        return d

    def setUp(self):
        tf.reset_default_graph()

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """

        model = Comparison(**self.default_config())
        n_samples = 10
        model.fit(["Indico is the best"] * n_samples, ["Indico is the bestestestest"] * n_samples, ['yes'] * n_samples)

        predictions = model.predict(["Is indico the best?"], ["Indico is the bestestestest"])
        for prediction in predictions:
            self.assertIsInstance(prediction, (str, bytes))

    def test_reasonable_predictions(self):
        model = Comparison(**self.default_config(n_epochs=5))

        similar = [
            ["What is the meaning of life?", "What does life mean?"],
            ["Why did the chicken cross the road", "Why did the chicken walk across the road"],
            ["What was the cause of the economic crisis?", "What caused the economic crash?"],
            ["Whats the name of the current Queen of England", "What is the current Queen of England's name?"],
            ["Is fish good for your brain?", "Ive heard fish is good for your brain, is this true?"],
            ["Why is it so hard to come up with similar questions?", "What is the reason that it is so hard to come up with similar questions?"]
        ] 

        different = [
            ["What is the air speed velocity of an unlaiden swallow?", "How many miles from Plymouth UK, to Boston"],
            ["Why is Plymouth in Boston called Plymouth", "Why is it so dificult to come up with fake questions?"],
            ["Does america love fake news or does fake news love america?", "What came first the chicken or the egg?"],
            ["Why does this test keep failing?", "What is madison doing right now?"],
            ["How long until Artificial Intelligence kills all of humankind?", "Am I a good person?"],
            ["How long would it take me to walk from the moon to the sun", "Is this enough questions to get some results?"]
        ]

        targets = np.asarray(["S"] * len(similar) + ["D"] * len(different))
        data = similar + different

        x_tr, x_te, t_tr, t_te = train_test_split(data, targets, train_size=0.3)
        model.finetune(*list_transpose(x_tr), t_tr)
        predictions = model.predict(*list_transpose(x_te))
        accuracy = np.mean([pred == true for pred, true in zip(predictions, t_te)])
        naive_baseline = max(np.mean(targets == "S"), np.mean(targets == "D"))
        self.assertGreater(accuracy, naive_baseline)
