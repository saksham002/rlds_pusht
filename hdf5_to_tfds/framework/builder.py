import tensorflow_datasets as tfds
from tensorflow_datasets.core.dataset_metadata import DatasetMetadata


class GenericBuilder(tfds.core.GeneratorBasedBuilder):
    """Generic TFDS builder driven by a user config and a pre-sliced episode list."""

    VERSION     = tfds.core.Version('1.0.0')  # overridden dynamically by runner.py
    USER_CONFIG     = None  # set by runner.py before instantiation
    WORKER_EPISODES = None  # dict: {split: [episode_path, ...]} for this worker's slice

    @classmethod
    def get_metadata(cls):
        return DatasetMetadata(description = '', citation = '', tags = [])

    def _info(self):
        return self.dataset_info_from_configs(
            features = self.USER_CONFIG.get_features()
        )

    def _split_generators(self, dl_manager):
        return {
            split: self._generate_examples(episodes)
            for split, episodes in self.WORKER_EPISODES.items()
            if episodes
        }

    def _generate_examples(self, episodes):
        for ep in episodes:
            yield self.USER_CONFIG.parse_episode(ep)
