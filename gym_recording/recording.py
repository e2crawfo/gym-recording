import os
import time
import logging
import numpy as np
from gym.utils import atomic_write
import dill
logger = logging.getLogger(__name__)


def always_true(x):
    return True


class TraceRecording(object):
    _id_counter = 0

    def __init__(self, directory=None, episode_filter=None, frame_filter=None):
        """
        Create a TraceRecording, writing into directory
        """

        if directory is None:
            directory = os.path.join('/tmp', 'openai.gym.{}.{}'.format(time.time(), os.getpid()))
            os.mkdir(directory)

        self.directory = directory
        self.file_prefix = 'openaigym.trace.{}.{}'.format(self._id_counter, os.getpid())
        TraceRecording._id_counter += 1

        self.episode_filter = always_true if episode_filter is None else episode_filter
        assert callable(self.episode_filter)

        self.frame_filter = always_true if frame_filter is None else frame_filter
        assert callable(self.frame_filter)

        self.closed = False

        self.actions = []
        self.observations = []
        self.rewards = []
        self.episode_id = 0

        self.buffered_step_count = 0
        self.buffer_batch_size = 100

        self.episodes_first = 0
        self.episodes = []
        self.batches = []

    def add_reset(self, observation):
        assert not self.closed
        self.end_episode()
        self.observations.append(observation)

    def add_step(self, action, observation, reward):
        assert not self.closed

        self.actions.append(action)
        self.rewards.append(reward)

        frame_id = len(self.actions)
        if self.frame_filter(frame_id):
            self.observations.append(observation)
            self.buffered_step_count += 1

    def end_episode(self):
        """
        if len(observations) == 0, nothing has happened yet.
        If len(observations) == 1, then len(actions) == 0, and we have only called reset and done a null episode.
        """
        print("End of episode {}".format(self.episode_id))
        if len(self.observations) > 0:
            if len(self.episodes) == 0:
                self.episodes_first = self.episode_id

            if self.episode_filter(self.episode_id):
                print("Storing episode {}".format(self.episode_id))
                self.episodes.append({
                    'actions': optimize_list_of_ndarrays(self.actions),
                    'observations': optimize_list_of_ndarrays(self.observations),
                    'rewards': optimize_list_of_ndarrays(self.rewards),
                })

            self.actions = []
            self.observations = []
            self.rewards = []
            self.episode_id += 1

            if self.buffered_step_count >= self.buffer_batch_size:
                self.save_complete()

    def save_complete(self):
        """
        Save the latest batch and write a manifest listing all the batches.
        We save the arrays as raw binary, in a format compatible with np.load.
        We could possibly use numpy's compressed format, but the large observations we care about (VNC screens)
        don't compress much, only by 30%, and it's a goal to be able to read the files from C++ or a browser someday.
        """

        filename = '{}.ep{:09}.pkl'.format(self.file_prefix, self.episodes_first)
        print("Saving data to {}".format(filename))
        with atomic_write.atomic_write(os.path.join(self.directory, filename), True) as f:
            dill.dump({'episodes': self.episodes}, f, protocol=dill.HIGHEST_PROTOCOL, recurse=False)
            bytes_per_step = float(f.tell() + f.tell()) / float(self.buffered_step_count)

        self.batches.append({
            'first': self.episodes_first,
            'len': len(self.episodes),
            'fn': filename})

        manifest = {'batches': self.batches}
        manifest_fn = '{}.manifest.pkl'.format(self.file_prefix)
        with atomic_write.atomic_write(os.path.join(self.directory, manifest_fn), True) as f:
            dill.dump(manifest, f, protocol=dill.HIGHEST_PROTOCOL, recurse=False)

        # Adjust batch size, aiming for 5 MB per file.
        # This seems like a reasonable tradeoff between:
        #   writing speed (not too much overhead creating small files)
        #   local memory usage (buffering an entire batch before writing)
        #   random read access (loading the whole file isn't too much work when just grabbing one episode)
        self.buffer_batch_size = max(1, min(50000, int(5000000 / bytes_per_step + 1)))

        self.episodes = []
        self.episodes_first = None
        self.buffered_step_count = 0

    def close(self):
        """
        Flush any buffered data to disk and close. It should get called automatically at program exit time, but
        you can free up memory by calling it explicitly when you're done
        """
        if not self.closed:
            self.end_episode()
            if len(self.episodes) > 0:
                self.save_complete()
            self.closed = True
            logger.info('Wrote traces to %s', self.directory)


def optimize_list_of_ndarrays(x):
    """
    Replace a list of ndarrays with a single ndarray with an extra dimension.
    Should return unchanged a list of other kinds of observations or actions, like Discrete or Tuple
    """
    if type(x) == np.ndarray:
        return x
    if len(x) == 0:
        return np.array([[]])
    if type(x[0]) == float or type(x[0]) == int:
        return np.array(x)
    if type(x[0]) == np.ndarray and len(x[0].shape) == 1:
        return np.array(x)
    return x
