import contextlib
import mock
import os
import png
import StringIO
import sys
from numbers import Number
from unittest import TestCase

import run_experiment  # NOQA: for command line options
from stanza.research import config, instance, summary
from learners import HistogramLearner, LookupLearner
from learners import MostCommonSpeakerLearner, RandomListenerLearner
from speaker import SpeakerLearner, AtomicSpeakerLearner
from listener import ListenerLearner, AtomicListenerLearner
from rsa import RSALearner


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

    def test_speaker_gru(self):
        self.run_speaker(SpeakerLearner, 'GRU')

    def test_atomic_speaker(self):
        self.run_speaker(AtomicSpeakerLearner)

    def test_rsa_speaker(self):
        self.run_speaker(RSALearner, images=True)  # extra images from listener

    def test_histogram_speaker(self):
        self.run_speaker(HistogramLearner, tensorboard=False)

    def test_lookup_speaker(self):
        self.run_speaker(LookupLearner, tensorboard=False)

    def test_most_common_speaker(self):
        self.run_speaker(MostCommonSpeakerLearner, tensorboard=False)

    def run_speaker(self, speaker_class, cell='LSTM', tensorboard=True, images=False):
        sys.argv = []
        options = config.options()
        options.train_iters = 2
        options.train_epochs = 3
        options.speaker_cell = cell
        options.listener = False

        mo = MockOpen(TEST_DIR)
        mgfp = mock_get_file_path(TEST_DIR)
        with mock.patch('stanza.research.summary.open', mo), \
                mock.patch('stanza.research.summary.SummaryWriter', MockSummaryWriter), \
                mock.patch('stanza.research.config.open', mo), \
                mock.patch('stanza.research.config.get_file_path', mgfp):
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

        if tensorboard:
            self.check_tensorboard(mo, mgfp, images=images)

    def check_tensorboard(self, mo, mgfp, images=False):
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
        if images:
            self.assertEqual(num_images, 6)

    def test_listener(self):
        self.run_listener()

    def test_listener_gru(self):
        self.run_listener(cell='GRU')

    def test_random_listener(self):
        self.run_listener(listener_class=RandomListenerLearner, tensorboard=False)

    def test_lookup_listener(self):
        self.run_listener(LookupLearner, tensorboard=False)

    def test_atomic_listener(self):
        self.run_listener(listener_class=AtomicListenerLearner)

    def test_rsa_listener(self):
        self.run_listener(listener_class=RSALearner)

    def run_listener(self, listener_class=ListenerLearner, cell='LSTM', tensorboard=True):
        sys.argv = []
        options = config.options()
        options.train_iters = 2
        options.train_epochs = 3
        options.listener_cell = cell
        options.listener = True

        mo = MockOpen(TEST_DIR)
        mgfp = mock_get_file_path(TEST_DIR)
        with mock.patch('stanza.research.summary.open', mo), \
                mock.patch('stanza.research.summary.SummaryWriter', MockSummaryWriter), \
                mock.patch('stanza.research.config.open', mo), \
                mock.patch('stanza.research.config.get_file_path', mgfp):
            listener = listener_class()
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

        if tensorboard:
            self.check_tensorboard(mo, mgfp, images=True)
