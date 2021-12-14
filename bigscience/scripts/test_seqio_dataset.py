import seqio

from t5x import utils
import tensorflow as tf
from ..gins import task

def main():
    ds = utils.get_dataset(
        utils.DatasetConfig(
            "c4_prefix_lm_objective_decoder_architecture_with_bot_seperator",

            task_feature_lengths={
                "decoder_target_tokens": 626,
                "decoder_input_tokens": 626,
                "decoder_segment_ids": 626,
                "decoder_causal_attention": 626,
                "targets": 625 # we have to take in account an extra token between input and target
            },
            split="train",
            batch_size=2048,
            shuffle=False,
            seed=None,
            use_cached=True,
            pack=True,
            use_custom_packing_ops=False,
            use_memory_cache=False,
        ),
        0,
        1,
        seqio.PassThroughFeatureConverter(),
    )
    first_element = next(iter(ds))
    print(first_element)

    # This should output `dict_keys(['decoder_target_tokens', 'decoder_input_tokens', 'decoder_loss_weights', 'decoder_segment_ids', 'decoder_positions'])`
    print(first_element.keys())

    print(first_element["decoder_target_tokens"])
    print(first_element["decoder_loss_weights"])

    # This should all output `tf.Tensor([2048  626], shape=(2,), dtype=int32)`
    print(tf.shape(first_element["decoder_target_tokens"]))
    print(tf.shape(first_element["decoder_input_tokens"]))
    print(tf.shape(first_element["decoder_loss_weights"]))
    print(tf.shape(first_element["decoder_segment_ids"]))
    print(tf.shape(first_element["decoder_positions"]))

if __name__ == "__main__":
    main()