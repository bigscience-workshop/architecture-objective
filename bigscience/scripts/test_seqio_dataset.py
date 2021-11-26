import seqio
from t5x import models
import tensorflow as tf
from ..gins import task

def main():
    ds = seqio.get_dataset(
        "c4_v220_full_lm",
        task_feature_lengths={"targets": 626},
        feature_converter=models.DecoderOnlyModel.FEATURE_CONVERTER_CLS ,
        use_cached=True
    )
    first_element = next(iter(ds))
    print(first_element)
    print(first_element["targets"])
    print(tf.shape(first_element["targets"]))

if __name__ == "__main__":
    main()