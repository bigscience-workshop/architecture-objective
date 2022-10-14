import functools
import seqio
import tensorflow_datasets as tfds
from t5.evaluation import metrics
from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary, add_eos=True, required=False),
    'targets': seqio.Feature(vocabulary=vocabulary, add_eos=True)
}

# ==================================== C4 ======================================
# Final pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "wikipedia.en_unsupervised",
    source=seqio.TfdsDataSource(tfds_name="wikipedia/20200301.en:1.0.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=output_features,
    metric_fns=[])
