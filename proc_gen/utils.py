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
from pathlib import Path

__all__ = ["get_ckpt_dir", "replace_in_path"]


def get_ckpt_dir(orig_path, model_arch, version=None):
    ckpt_dir = orig_path
    ckpt_dir = replace_in_path(ckpt_dir, replace_part="data", new_part="ckpts")
    suffix = f"-{str(version)}" if version else ""
    ckpt_dir = ckpt_dir / f"ckpts-{model_arch}{suffix}"

    return ckpt_dir


def replace_in_path(orig_path: Path, replace_part: str, new_part: str):
    part_index = orig_path.parts.index(replace_part)

    all_parts = (
        list(orig_path.parts[:part_index])
        + [new_part]
        + list(orig_path.parts[part_index + 1 :])
    )

    return Path("").joinpath(*all_parts)
