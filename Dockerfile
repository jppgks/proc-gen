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

FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ENV WORKSPACE /workspace

RUN apt-get -y update \
    && apt-get install -yq --no-install-recommends \
    wget \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN conda install -y gcc_linux-64 gxx_linux-64

# Install Python package
COPY requirements.txt ${WORKSPACE}/requirements.txt
RUN conda run --no-capture-output pip install -r ${WORKSPACE}/requirements.txt

COPY . ${WORKSPACE}
RUN conda run --no-capture-output pip install -e ${WORKSPACE}

# Download GPT2-BPE English vocabulary and encoder
ENV BPE_DIR ${WORKSPACE}/bpe-files
RUN pg-bpe-download ${BPE_DIR}

WORKDIR ${WORKSPACE}

# (optional) Provide custom user/group id
ARG UG_ID=1000
RUN adduser --disabled-password --gecos '' pg-user -u ${UG_ID} && \
    usermod -aG sudo pg-user && \
    chown -R pg-user ${WORKSPACE}
USER pg-user
