# Copyright 2022

"""Defines the vocabulary"""
import seqio

# # TODO: Link to Eleuther's custom default tokenizer when ready.
DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"  # GCS

def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH)