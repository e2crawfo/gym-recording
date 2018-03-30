import os
import glob
import logging
import dill
logger = logging.getLogger(__name__)


__all__ = ['scan_recorded_traces', 'TraceRecordingReader']


class TraceRecordingReader:
    def __init__(self, directory):
        self.directory = directory
        self.binfiles = {}

    def close(self):
        for k in self.binfiles.keys():
            if self.binfiles[k] is not None:
                self.binfiles[k].close()
                self.binfiles[k] = None

    def get_recorded_batches(self):
        ret = []
        manifest_ptn = os.path.join(self.directory, 'openaigym.trace.*.manifest.pkl')
        trace_manifest_fns = glob.glob(manifest_ptn)
        logger.debug('Trace manifests %s %s', manifest_ptn, trace_manifest_fns)

        for trace_manifest_fn in trace_manifest_fns:

            with open(trace_manifest_fn, 'rb') as f:
                trace_manifest = dill.load(f)

            ret += trace_manifest['batches']
        return ret

    def get_recorded_episodes(self, batch):
        filename = os.path.join(self.directory, batch['fn'])
        with open(filename, 'rb') as f:
            batch_d = dill.load(f)
        return batch_d['episodes']


def scan_recorded_traces(directory, episode_cb=None, max_episodes=None):
    """
    Go through all the traces recorded to directory, and call episode_cb for every episode.
    Set max_episodes to end after a certain number (or you can just throw an exception from episode_cb
    if you want to end the iteration early)
    """
    rdr = TraceRecordingReader(directory)
    added_episode_count = 0
    for batch in rdr.get_recorded_batches():
        for ep in rdr.get_recorded_episodes(batch):
            episode_cb(ep['observations'], ep['actions'], ep['rewards'])
            added_episode_count += 1
            if max_episodes is not None and added_episode_count >= max_episodes:
                return
    rdr.close()
