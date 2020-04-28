# Copyright (c) 2020-2021 Joppe Geluykens
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from copy import deepcopy

__all__ = ["tokenize_example", "detokenize_example"]

from proc_gen.data.to_example import TranslationExample, SPECIAL_TOKENS, REQUIREMENT_SEP


def tokenize_example(
    example: TranslationExample, tokenizer="moses"
) -> TranslationExample:
    if tokenizer not in ("moses",):
        raise NotImplementedError("Only moses tokenizer currently supported.")

    example = deepcopy(example)

    if tokenizer == "moses":
        from sacremoses import MosesTokenizer

        tokenizer = MosesTokenizer("en")
        tokenizer_kwargs = {
            "aggressive_dash_splits": True,
            "return_str": True,
            "escape": False,
            "protected_patterns": SPECIAL_TOKENS,  # Protect special tokens
        }
        tokenize = tokenizer.tokenize

    example.src = tokenize(example.src, **tokenizer_kwargs)
    example.tgt = tokenize(example.tgt, **tokenizer_kwargs)

    return example


def detokenize_example(
    example: TranslationExample, tokenizer="moses"
) -> TranslationExample:
    if tokenizer != "moses":
        raise NotImplementedError("Only moses tokenizer is currently supported.")

    example = deepcopy(example)

    if tokenizer == "moses":
        from sacremoses import MosesDetokenizer

        tokenizer = MosesDetokenizer("en")

    example.src = tokenizer.detokenize(example.src.split())
    example.tgt = tokenizer.detokenize(example.tgt.split())

    return example
