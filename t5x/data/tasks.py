import functools

import seqio

import t5x.data.vocab
import t5x.data.utils
import t5x.data.c4_utils

from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs": 
        seqio.Feature(
            vocabulary=t5x.data.vocab.get_default_vocabulary(),
            add_eos=True,
            required=False),
    "targets":
        seqio.Feature(
            vocabulary=t5x.data.vocab.get_default_vocabulary(),
            add_eos=True)
}


# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye
TaskRegistry.add(
    'c4_eye_span_corruption',
    # source=seqio.TextLineDataSource(
    source=t5x.data.utils.CustomDataSource(
        split_to_filepattern=t5x.data.c4_utils.get_c4_files(),
    ),
    preprocessors=[
        t5x.data.utils.extract_text_from_json_tf,
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

