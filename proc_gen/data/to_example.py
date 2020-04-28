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
from dataclasses import dataclass
from typing import Iterable, List, Union

from proc_gen.data.schema import Procedure, Requirement, Method
from proc_gen.problems import Problem

__all__ = [
    "procedure_to_example",
    "TranslationExample",
    "SPECIAL_TOKENS",
    "REQUIREMENT_SEP",
    "string_to_requirements",
    "string_to_tasks",
    "example_to_procedure",
]

TARGET_PRODUCT_SEP = "<tps>"
REQUIREMENT_SEP = "<eor>"
TASK_SEP = "<eot>"
REQUIREMENTS_TASKS_SEP = REQUIREMENTS_TP_SEP = "<rts>"
SPECIAL_TOKENS = [
    REQUIREMENT_SEP,
    TASK_SEP,
    TARGET_PRODUCT_SEP,
    REQUIREMENTS_TASKS_SEP,
    REQUIREMENTS_TP_SEP,
]


@dataclass
class TranslationExample:
    src: str
    tgt: str


def tasks_to_string(tasks: List[str], tp: str = None) -> str:
    if tp:
        return tp + f" {TARGET_PRODUCT_SEP} " + tasks_to_string(tasks)

    return f" {TASK_SEP} ".join(tasks)


def string_to_tasks(tasks_string: str, parse_tp=False) -> List[str]:
    if parse_tp:
        tp, tasks_string = tasks_string.split(f" {TARGET_PRODUCT_SEP} ")
        return [tp] + string_to_tasks(tasks_string)
    else:
        return tasks_string.split(f" {TASK_SEP} ")


def requirements_to_string(
    requirements: Iterable[Requirement], tp: str = None, tp_last=False
) -> str:
    if tp:
        if tp_last:
            return requirements_to_string(requirements) + f" {TARGET_PRODUCT_SEP} " + tp
        else:
            return tp + f" {TARGET_PRODUCT_SEP} " + requirements_to_string(requirements)

    req_str = f" {REQUIREMENT_SEP} ".join((req.to_string() for req in requirements))

    return req_str


def string_to_requirements(
    requirements_string: str, parse_tp=False
) -> List[Union[str, Requirement]]:
    if parse_tp:
        tp, requirements_string = requirements_string.split(f" {TARGET_PRODUCT_SEP} ")

        return [tp] + string_to_requirements(requirements_string)
    else:
        req_strings = requirements_string.split(f" {REQUIREMENT_SEP} ")

        return [Requirement.from_string(req_str) for req_str in req_strings]


def procedure_to_example(proc: Procedure, problem: str) -> TranslationExample:
    assert len(proc.methods) > 0, (
        f"Procedure {proc} didn't contain any methods. "
        f"Cannot convert to translation example."
    )

    method = proc.methods[0]
    if problem is Problem.Requirements_TO_TargetProductAndTasks:
        src_entry = requirements_to_string(method.requirements)
        tgt_entry = tasks_to_string(method.tasks, tp=proc.target_product)
    elif problem is Problem.RequirementsAndTargetProductAndTasks:
        src_entry = (
            requirements_to_string(method.requirements)
            + f" {REQUIREMENTS_TP_SEP} "
            + tasks_to_string(method.tasks, tp=proc.target_product)
        )
        # language modeling task has no target language
        tgt_entry = ""
    elif problem is Problem.TargetProductAndRequirements_TO_Tasks:
        tp_last = True
        src_entry = requirements_to_string(
            method.requirements, tp=proc.target_product, tp_last=tp_last
        )
        tgt_entry = tasks_to_string(method.tasks)
    elif problem is Problem.TargetProductAndRequirementsAndTasks:
        src_entry = (
            requirements_to_string(method.requirements, tp=proc.target_product)
            + f" {REQUIREMENTS_TASKS_SEP} "
            + tasks_to_string(method.tasks)
        )
        # language modeling task has no target language
        tgt_entry = ""
    elif problem is Problem.Requirements_TO_TargetProduct:
        src_entry = requirements_to_string(method.requirements)
        tgt_entry = proc.target_product
    elif problem is Problem.TargetProduct_TO_Requirements:
        src_entry = proc.target_product
        tgt_entry = requirements_to_string(method.requirements)
    elif problem is Problem.Tasks_TO_TargetProduct:
        src_entry = tasks_to_string(method.tasks)
        tgt_entry = proc.target_product
    else:
        raise NotImplementedError(
            f"Unable to parse translation example. No parsing implementation for problem {problem}."
        )

    return TranslationExample(src=src_entry, tgt=tgt_entry)


def example_to_procedure(example: TranslationExample, problem: str) -> Procedure:
    method = Method(requirements=[], tasks=[])
    proc = Procedure(target_product="", methods=[method])

    if problem is Problem.Requirements_TO_TargetProductAndTasks:
        method.requirements = string_to_requirements(example.src)
        tp_and_tasks = string_to_tasks(example.tgt, parse_tp=True)
        proc.target_product, method.tasks = tp_and_tasks[0], tp_and_tasks[1:]
    elif problem is Problem.TargetProductAndRequirements_TO_Tasks:
        tp_and_reqs = string_to_requirements(example.src, parse_tp=True)
        proc.target_product, method.requirements = tp_and_reqs[0], tp_and_reqs[1:]
        method.tasks = string_to_tasks(example.tgt)
    elif problem is Problem.Requirements_TO_TargetProduct:
        method.requirements = string_to_requirements(example.src)
        proc.target_product = example.tgt
    elif problem is Problem.TargetProduct_TO_Requirements:
        proc.target_product = example.src
        method.requirements = string_to_requirements(example.tgt)
    elif problem is Problem.Tasks_TO_TargetProduct:
        method.tasks = string_to_tasks(example.src)
        proc.target_product = example.tgt
    else:
        raise NotImplementedError(
            f"Unable to parse procedure "
            f"from translation example {example}. No parsing implementation for problem {problem} "
        )

    assert proc.target_product
    assert proc.methods[0].requirements and proc.methods[0].tasks

    return proc
