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
from proc_gen.data.schema import Procedure, Method, Requirement

__all__ = ["recipe1m_to_procedure"]


def recipe1m_to_procedure(example: dict) -> (Procedure, str):
    """

    :param example: dict containing keys 'ingredients': list, 'instructions': list, title: str
    :return: (Procedure) the parsed procedure, (str) partition ('train', 'valid', 'test')
    """
    assert (
        "title" in example.keys()
    ), f"Recipe1M example {example} did not contain required key 'title'"
    assert (
        "ingredients" in example.keys()
    ), f"Recipe1M example {example} did not contain required key 'ingredients'"
    assert (
        "instructions" in example.keys()
    ), f"Recipe1M example {example} did not contain required key 'instructions'"
    assert (
        "partition" in example.keys()
    ), f"Recipe1M example {example} did not contain required key 'partition'"

    target_product = example["title"]

    requirements = []
    for r in example["ingredients"]:
        assert "text" in r.keys(), f"Ingredient {r} didn't contain expected key 'text'"
        # TODO: use ny-times-parser
        req = Requirement(object=r["text"], quantity="")  # TODO: parse quantity
        requirements.append(req)

    tasks = []
    for t in example["instructions"]:
        assert "text" in t.keys(), f"Instruction {t} didn't contain expected key 'text'"
        tasks.append(t["text"])

    methods = [Method(requirements=requirements, tasks=tasks)]

    partition = example["partition"]
    if partition == "val":
        partition = "valid"

    return Procedure(target_product=target_product, methods=methods), partition
