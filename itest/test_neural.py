from unittest import TestCase

from bt import config, instance
from speaker import SpeakerLearner
from listener import ListenerLearner


class TestModels(TestCase):
    def test_speaker(self):
        options = config.options()
        options.train_iters = 2
        options.train_epochs = 3

        speaker = SpeakerLearner()
        train_data = [instance.Instance((0, 255, 0), 'green')]
        speaker.train(train_data)
        predictions, scores = speaker.predict_and_score(train_data)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], basestring)
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)

    def test_listener(self):
        options = config.options()
        options.train_iters = 2
        options.train_epochs = 3

        listener = ListenerLearner()
        train_data = [instance.Instance('green', (0, 255, 0))]
        listener.train(train_data)
        predictions, scores = listener.predict_and_score(train_data)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(predictions[0]), 3)
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)
