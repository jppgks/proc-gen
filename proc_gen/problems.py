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
from enum import Enum, auto

__all__ = ["Problem", "TASK_TO_PROBLEMS"]


class Problem(Enum):
    # e.g. reaction prediction
    Requirements_TO_TargetProduct = auto()

    # e.g. retrosynthesis
    TargetProduct_TO_Requirements = auto()

    # e.g.: recipe generation, smiles to actions
    Requirements_TO_TargetProductAndTasks = auto()
    TargetProductAndRequirements_TO_Tasks = auto()

    # e.g. goal prediction
    Tasks_TO_TargetProduct = auto()

    # for language modeling and autoregressive language generation
    # e.g.: prompt model with target product to generate requirements
    #  OR prompt same model with requirements to generate target product
    RequirementsAndTargetProductShuffle = auto()
    # e.g.: prompt model with target product and requirements to generate tasks
    TargetProductAndRequirementsAndTasks = auto()
    RequirementsAndTargetProductAndTasks = auto()


TASK_TO_PROBLEMS = {
    "translation": [
        Problem.Requirements_TO_TargetProduct,
        Problem.TargetProduct_TO_Requirements,
        Problem.Requirements_TO_TargetProductAndTasks,
        Problem.TargetProductAndRequirements_TO_Tasks,
        Problem.Tasks_TO_TargetProduct,
    ],
    "language_modeling": [
        Problem.RequirementsAndTargetProductShuffle,
        Problem.TargetProductAndRequirementsAndTasks,
        Problem.RequirementsAndTargetProductAndTasks,
    ],
}
