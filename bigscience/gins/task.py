import dataclasses
import functools

import seqio
import t5
import tensorflow as tf
from t5.data import preprocessors, get_default_vocabulary
from t5.data.preprocessors import select_random_chunk, reduce_concat_tokens, split_tokens

from t5x.partitioning import LogicalAxisRules

# --- Seqio ---
seqio.add_global_cache_dirs([
    'gs://bigscience-t5x/seqio_cached_tasks',
    'gs://bigscience-t5x/seqio_cached_tasks/t0-adapt'
])

TaskRegistry = seqio.TaskRegistry

def full_lm(dataset, sequence_length, output_features):
    """Full language modeling objective"""
    ds = dataset
    ds = select_random_chunk(ds, output_features=output_features, feature_key='targets', max_length=65536)
    ds = seqio.preprocessors.append_eos(ds, output_features)
    ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
    ds = split_tokens(ds, max_tokens_per_segment=sequence_length['targets'])
    # ds = trim_and_pad_dataset(ds, sequence_length) # I feel this should be interesting, we should use `split_tokens_to_targets_length`
    return ds

TaskRegistry.add(
    "c4_v220_full_lm",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        full_lm,
    ],
    output_features={
        "targets": seqio.Feature(
            vocabulary=get_default_vocabulary(), add_eos=True)
    },
    metric_fns=[])

# We want input and target to have an additional token between.
# Inspired by https://github.com/google-research/text-to-text-transfer-transformer/blob/9844ddb4f760ae8a1d4de410578f6211e487bbf9/t5/data/tasks.py#L445

assert get_default_vocabulary().vocab_size == 32100, "Use T5 tokenizer by default"
BOT_ID = 32000 # FIXME: this is only true for t5 tokenizer right now.
@dataclasses.dataclass(frozen=True)
class FancyFeature(seqio.Feature):
    # This token is use to seperate input and target. `bot` is the acronym for beginning of target
    add_bot: bool = False

def pack_prefix_lm_decoder_only(ds,
                                sequence_length,
                                output_features,
                                loss_on_targets_only=True,
                                pad_id=0):
    """Randomly split the tokens for the prefix LM objective."""
    packed_length = sequence_length["decoder_input_tokens"]
    assert packed_length % 2 == 0
    # "targets" is a special key
    add_bot = output_features["decoder_input_tokens"].add_bot

    assert all(l == packed_length for key, l in sequence_length.items() if (not add_bot) or key != "targets")
    assert all(l.add_bot == add_bot for key, l in output_features.items() if key != "targets")
    if add_bot:
        assert sequence_length["targets"] == packed_length - 1
    else:
        assert sequence_length["targets"] == packed_length

    @seqio.utils.map_over_dataset(num_seeds=1)
    def pack_examples(example, seed):
        split_point = tf.random.stateless_uniform((),
                                                  minval=1,
                                                  # Adding an extra token costs a bit.
                                                  maxval=packed_length if output_features["decoder_input_tokens"].add_bot else packed_length - 1,
                                                  seed=seed,
                                                  dtype=tf.int32)
        if output_features["decoder_input_tokens"].add_bot:
            decoder_target_tokens = tf.concat(
                [
                    example['targets'][:split_point - 1],
                    # bot will be the same as _<extra_id_99>. Not ideal, but the tokenizer doesn't have `bos` right now.
                    [BOT_ID],
                    example['targets'][split_point - 1:],
                ],
                axis=-1
            )
        else:
            decoder_target_tokens = example['targets']

        decoder_input_tokens = seqio.utils.make_autoregressive_inputs(decoder_target_tokens)

        if loss_on_targets_only:
          decoder_loss_weights = tf.cast(
              tf.range(packed_length) >= split_point, tf.int32)
        else:
          decoder_loss_weights = tf.ones((packed_length,), dtype=tf.int32)

        padding_mask = tf.cast(
            tf.not_equal(decoder_target_tokens, pad_id), dtype=tf.int32)
        decoder_loss_weights *= padding_mask

        decoder_causal_attention = tf.cast(
            tf.range(packed_length) <= split_point, tf.int32)

        return {
            'decoder_target_tokens': decoder_target_tokens,
            'decoder_input_tokens': decoder_input_tokens,
            'decoder_loss_weights': decoder_loss_weights,
            'decoder_causal_attention': decoder_causal_attention,
        }

    return pack_examples(ds)

TaskRegistry.add(
    "c4_prefix_lm_objective_decoder_architecture_with_bot_seperator",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.targets_for_prefix_lm_objective,
        pack_prefix_lm_decoder_only,
    ],
    output_features={
        "decoder_target_tokens": FancyFeature(vocabulary=get_default_vocabulary(), add_eos=False, add_bot=True),
        "decoder_input_tokens": FancyFeature(vocabulary=get_default_vocabulary(), add_eos=False, add_bot=True),
        "decoder_loss_weights": FancyFeature(vocabulary=get_default_vocabulary(), add_eos=False, add_bot=True),
        "decoder_causal_attention": FancyFeature(
            vocabulary=get_default_vocabulary(), add_eos=False, add_bot=True),
        # All but the last stage of the preprocessing uses "targets" as the key,
        # so this output feature is necessary. It is not marked required because
        # the final preprocessor drops it.
        "targets": seqio.Feature(vocabulary=get_default_vocabulary(), required=False),
    },
    metric_fns=[])

# --- Improve sharding ---

# def fully_sharded_logical_axis_rules() -> LogicalAxisRules:
#     """Fully sharded rules for P5X model in terms of logical axes names."""
#     return (
#       ('batch', 'data'),
#       ('vocab', 'model'),
#       ('mlp', 'model'),
#       ('heads', 'model'),
#       ('joined_kv', 'model'),
#       ('kv', None),
#       ('embed', 'model'),
#       ('embed', 'data'),
#       ('relpos_buckets', None),
#       ('length', None),
#       ('layers', None),
#       ('stack', None),
#     )
