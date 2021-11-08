import functools
import seqio
from t5.data import preprocessors, utils
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
    "train": [
        f"{DATASET_FOLDER}/train/{i:02d}.jsonl" for i in range(2)
    ],
    "val": f"{DATASET_FOLDER}/val.jsonl",
    "test": f"{DATASET_FOLDER}/test.jsonl"
}

@utils.map_over_dataset
def extract_text_from_json_tf(json: str):
    output = tf.strings.split(json, '{"text": "', maxsplit=1)[1]
    output = tf.strings.split(output, '", "meta": {', maxsplit=1)[0]
    return {"text": output}

seqio.TaskRegistry.add(
    'pile_t2t_span_corruption',
    source=seqio.TextLineDataSource(
        split_to_filepattern=DATASET_SPLITS_TO_FILEPATTERN,
    ),
    preprocessors=[
        extract_text_from_json_tf,
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

seqio.TaskRegistry.add(
    "pile_t2t_prefix_lm",
    source=seqio.TextLineDataSource(
        split_to_filepattern=DATASET_SPLITS_TO_FILEPATTERN,
    ),
    preprocessors=[
        extract_text_from_json_tf,
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
