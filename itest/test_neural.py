import contextlib
import mock
import os
import png
import StringIO
import sys
from numbers import Number
from unittest import TestCase

from stanza.unstable import config, instance, summary
from speaker import SpeakerLearner, AtomicSpeakerLearner
from listener import ListenerLearner


TEST_DIR = '/shouldnotexist'


def yields(thing):
    yield thing


class MockOpen(object):
    def __init__(self, test_dir):
        self.files = {}
        self.old_open = open
        self.test_dir = test_dir

    def __call__(self, filename, *args, **kwargs):
        print('MockOpen called')
        if filename.startswith(self.test_dir):
            if filename not in self.files:
                print('Mocking %s' % filename)
                self.files[filename] = StringIO.StringIO()
            return contextlib.contextmanager(yields)(self.files[filename])
        else:
            return self.old_open(filename, *args, **kwargs)


class MockSummaryWriter(summary.SummaryWriter):
    def add_event(self, *args, **kwargs):
        super(MockSummaryWriter, self).add_event(*args, **kwargs)
        self.flush()
        print('Event added and flushed')


def mock_get_file_path(test_dir):
    def get_file_path(filename):
        print('mock_get_file_path called')
        return os.path.join(test_dir, filename)
    return get_file_path


class TestModels(TestCase):
    def test_speaker(self):
        self.run_speaker(SpeakerLearner)

    def test_atomic_speaker(self):
        self.run_speaker(AtomicSpeakerLearner)

    def run_speaker(self, speaker_class):
        sys.argv = []
        options = config.options()
        options.train_iters = 2
        options.train_epochs = 3

        mo = MockOpen(TEST_DIR)
        mgfp = mock_get_file_path(TEST_DIR)
        with mock.patch('stanza.unstable.summary.open', mo), \
                mock.patch('stanza.unstable.summary.SummaryWriter', MockSummaryWriter), \
                mock.patch('stanza.unstable.config.open', mo), \
                mock.patch('stanza.unstable.config.get_file_path', mgfp):
            speaker = speaker_class()
            train_data = [instance.Instance((0, 255, 0), 'green')]
            speaker.train(train_data)
            predictions, scores = speaker.predict_and_score(train_data)

        # predictions = ['somestring']
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], basestring)
        # scores = [123.456]
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)

        lossesfile = mo.files[mgfp('losses.tfevents')]
        lossesfile.seek(0)
        events = list(summary.read_events(lossesfile))
        self.assertEqual(len(events), 12)
        for e in events:
            for v in e.summary.value:
                self.assertIsInstance(v.simple_value, float)

    def test_listener(self):
        sys.argv = []
        options = config.options()
        options.train_iters = 2
        options.train_epochs = 3

        mo = MockOpen(TEST_DIR)
        mgfp = mock_get_file_path(TEST_DIR)
        with mock.patch('stanza.unstable.summary.open', mo), \
                mock.patch('stanza.unstable.summary.SummaryWriter', MockSummaryWriter), \
                mock.patch('stanza.unstable.config.open', mo), \
                mock.patch('stanza.unstable.config.get_file_path', mgfp):
            listener = ListenerLearner()
            train_data = [instance.Instance('green', (0, 255, 0))]
            listener.train(train_data)
            predictions, scores = listener.predict_and_score(train_data)

        # predictions = [(123, 45, 67)]
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(predictions[0]), 3)
        self.assertIsInstance(predictions[0][0], Number)
        # scores = [123.456]
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)

        lossesfile = mo.files[mgfp('losses.tfevents')]
        lossesfile.seek(0)
        events = list(summary.read_events(lossesfile))
        num_scalars = 0
        num_images = 0
        for e in events:
            print(e)
            for v in e.summary.value:
                if v.HasField('image'):
                    fakepng = StringIO.StringIO(v.image.encoded_image_string)
                    png.Reader(fakepng).read()
                    num_images += 1
                else:
                    self.assertTrue(v.HasField('simple_value'))
                    self.assertIsInstance(v.simple_value, float)
                    num_scalars += 1
        self.assertEqual(num_scalars, 12)
        self.assertEqual(num_images, 6)
