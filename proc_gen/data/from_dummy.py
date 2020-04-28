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

__all__ = ["dummy_to_procedure"]


def dummy_to_procedure(index: int) -> (Procedure, str):
    """Dummy data to test pipeline.

    :return: (Procedure) the parsed procedure, (str) partition ('train', 'val', 'test')
    """

    target_product = "Dummy target product"

    requirements = []
    for _ in range(5):
        req = Requirement(object="dry dummy", quantity="1/2 teaspoon")
        requirements.append(req)

    tasks = []
    for _ in range(7):
        tasks.append("Cook dummy according to package directions; drain well.")

    methods = [Method(requirements=requirements, tasks=tasks)]

    if 0 < index < 10:
        partition = "test"
    elif 10 < index < 20:
        partition = "valid"
    else:
        partition = "train"

    return Procedure(target_product=target_product, methods=methods), partition
