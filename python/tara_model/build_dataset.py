"""
python/tara_model/build_dataset.py

Tokenizes, filters, deduplicates, and splits the TARA dataset into:
- train.jsonl (80%)
- val.jsonl (10%)
- test.jsonl (10%)

Uses the canonical 344-token tokenizer.
Measures real tokens, sequence lengths, unknown token rates, and produces a verifiable manifest.
"""

import os
import sys
import json
import random
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tara_model.compile_unified_dataset import compile_unified_dataset

def build_dataset():
    return compile_unified_dataset()

if __name__ == "__main__":
    build_dataset()
