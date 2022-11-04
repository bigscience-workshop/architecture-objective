# Copyright 2022

"""Defines the vocabulary"""
import seqio

# # TODO: Link to Eleuther's custom default tokenizer when ready.
# DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
# DEFAULT_EXTRA_IDS = 100

DEFAULT_SPM_PATH = "/fsx/lintangsutawika/t5-tokenizer/spiece.model"
DEFAULT_EXTRA_IDS = 0

def get_default_vocabulary():
  return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)