import functools

import seqio
from seqio import feature_converters
from t5.data import preprocessors, utils
import json as js
import tensorflow as tf

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocabulary, add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=vocabulary, add_eos=True)
}

DATASET_FOLDER="gs://bigscience/pile/raw"
DATASET_SPLITS_TO_FILEPATTERN={
    "train": f"{DATASET_FOLDER}/train/*.jsonl",
    "val": f"{DATASET_FOLDER}/val.jsonl",
    "test": f"{DATASET_FOLDER}/test.jsonl"
}

@utils.map_over_dataset
def extract_text_from_json(json: str):
    return tf.py_function(js.loads(json)["text"])

seqio.TaskRegistry.add(
    'pile_t2t_span_corruption',
    source=seqio.TextLineDataSource(
        split_to_filepattern=DATASET_SPLITS_TO_FILEPATTERN,
    ),
    preprocessors=[
        extract_text_from_json,
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
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)

# Prefix language modeling pretraining task used in Raffel et al., 2019.
seqio.TaskRegistry.add(
    "pile_t2t_prefix_lm",
    source=seqio.TextLineDataSource(
        split_to_filepattern=DATASET_SPLITS_TO_FILEPATTERN,
    ),
    preprocessors=[
        extract_text_from_json,
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.prefix_lm,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)

if __name__ == "__main__":
    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    seqio.get_dataset(
        "pile_t2t_span_corruption",
        task_feature_lengths=task_feature_lengths,
        feature_converter=converter,
    )