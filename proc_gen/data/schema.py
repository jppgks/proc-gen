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
from typing import Optional, List, Union

__all__ = ["Procedure", "Method", "Requirement", "PARTITIONS"]

PARTITIONS = ["train", "valid", "test"]


@dataclass
class Requirement:
    object: str
    quantity: Union[str, int, float]
    optional: Optional[bool] = None

    def __str__(self):
        return self.to_string()

    def to_string(self) -> str:
        req_str = ""
        req_str += self.object
        if self.quantity:
            req_str += " (" + self.quantity + ")"
        if self.optional:
            req_str += " - optional"
        return req_str

    @staticmethod
    def from_string(req_str: str):
        parsed_req = Requirement(object="", quantity="", optional=False)

        # Parse object
        req_str = req_str.split(" (")
        parsed_req.object = req_str[0]
        if len(req_str) == 1:
            return parsed_req
        _, req_str = req_str  # drop object part

        # Parse quantity
        req_str = req_str.split(")")
        if len(req_str) == 1:
            return parsed_req
        parsed_req.quantity = req_str[0].strip()
        _, req_str = req_str  # drop quantity part

        # Parse optional
        if req_str:
            parsed_req.optional = True

        return parsed_req


@dataclass
class Method:
    requirements: List[Requirement]
    tasks: List[str]

    def __str__(self):
        return f"Method(Requirements: {self.requirements}, Instructions: {self.tasks})"


@dataclass
class Procedure:
    target_product: str
    methods: List[Method]

    def __str__(self):
        return f"Procedure(Goal: {self.target_product}, Methods: {self.methods})"
