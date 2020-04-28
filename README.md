# `proc-gen`: Neural Procedure Generation
This repository contains code to reproduce the results of our paper `Neural Machine Translation for Conditional Generation of Novel Procedures` as published in the *54th Hawaii International Conference on System Sciences* (HICSS '21). [[citation](#citation)]

It contains the `proc-gen` Python package which provides a wrapper over [Fairseq](https://github.com/pytorch/fairseq), introducing helpers for generating procedures.

## Installation

### tl;dr
```bash
docker build -t proc-gen .
```

### Setup conda environment
```bash
conda create -n proc-gen python=3.8.5
conda activate proc-gen
```

### Install Python package
Besides installing the `proc_gen` Python package, this also adds a number of `pg-*` command line utilities (see [./bin](./bin)) to the PATH, used further on in this documentation.
```bash
git clone https://github.com/jppgks/proc-gen.git && cd ./proc-gen
pip install -e .
# (optional) pip install -e .[mlflow]
```

`proc_gen` uses [Fairseq](https://github.com/pytorch/fairseq) under the hood. Fairseq provides [optional installation instructions](https://github.com/pytorch/fairseq#requirements-and-installation) for e.g. faster training.

### Vocabulary and encoder
Download GPT2-BPE English vocabulary and encoder:
```bash
BPE_DIR=$(pwd)/bpe-files
pg-bpe-download ${BPE_DIR}
```

## Data
### Download Recipe1M dataset
Register, understand the terms of use for the dataset and download [here](http://im2recipe.csail.mit.edu/dataset/login/). We only need the "Layers" (`recipe1M_layers.tar.gz`) file.

Untar the required data file:
```bash
DATA_DIR=/tmp/data # path to directory for storing untarred Recipe1M data
mkdir -p ${DATA_DIR}
tar -xvf recipe1M_layers.tar.gz layer1.json --directory ${DATA_DIR}
```

## Usage

### Data preprocessing
```bash
PROCESSED_DATA_DIR=/tmp/out # path to directory for storing processed data
CKPT_DIR=/tmp/ckpts # path to directory for storing checkpoints
RESULTS_DIR=/tmp/results # path to directory for storing results

DATASET={Recipe1M|dummy}
PROBLEM=Requirements_TO_TargetProductAndTasks # see all available problem types in proc_gen/problems.py
MODEL_TYPE=fairseq # only fairseq is currently supported

docker run \
  -v ${DATA_DIR}:/data/procgen/v1/source/Recipe1M \
  -v ${PROCESSED_DATA_DIR}:/data/procgen/v1/processed \
  proc-gen:latest \
    pg-prepare-data \
      --input-path /data/procgen/v1/source/Recipe1M/layer1.json \
      --dataset ${DATASET} \
      --problem ${PROBLEM} \
      --model-type ${MODEL_TYPE} \
      --output-dir /data/procgen/v1/processed \
      [--bpe-dir ${BPE_DIR}] \
      [--no-tokenize]
```

### Model training
```bash
# Task setup
MODEL_ARCH={lstm|conv|transformer|bart|gpt2}
TASK={translation|denoising|language_modeling}

# List available GPU devices
export CUDA_VISIBLE_DEVICES=0[,1[,2[,..]]]

docker run --gpus all \
  -v ${PROCESSED_DATA_DIR}:/data/procgen/v1/processed \
  -v ${CKPT_DIR}:/ckpts \
  proc-gen:latest \
    pg-train-model \
      --data_dir /data/procgen/v1/processed \
      --dataset ${DATASET} \
      --problem ${PROBLEM} \
      --model_type ${MODEL_TYPE} \
      --model_arch ${MODEL_ARCH} \
      --task ${TASK} \
      [--warm_start] \
      [--log_mlflow]
```

#### Distributed training (data parallel)
See the [`torch.distributed.launch` documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for more information about the tool used below to spawn distributed training processes (across nodes).
```bash
# Distributed training setup
read -a ALL_HOSTS <<< "<space separated list of hostnames>"
WORLD_SIZE=${#ALL_HOSTS[@]}

NUM_GPUS_PER_NODE=<number of gpus available on each host>
MASTER_ADDRESS=<IP of master host>
MASTER_PORT=1234

# Launch workers
for (( i=1; i<$WORLD_SIZE; i++ )); do
  NODE_RANK=$i
  WORKER_NODE=${ALL_HOSTS[$i]}
  ssh -f $WORKER_NODE docker run --gpus all \
    -v ${PROCESSED_DATA_DIR}:/data/procgen/v1/processed \
    -v ${CKPT_DIR}:/ckpts \
    proc-gen:latest \
      NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((${NUM_GPUS_PER_NODE}-1))) /opt/conda/bin/python -m torch.distributed.launch \
        --nproc_per_node=${NUM_GPUS_PER_NODE} \
        --nnodes=${WORLD_SIZE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDRESS} \
        --master_port=${MASTER_PORT} \
        /opt/conda/bin/pg-train-model \
          --data_dir /data/procgen/v1/processed \
          --dataset ${DATASET} \
          --problem ${PROBLEM} \
          --model_type ${MODEL_TYPE} \
          --model_arch ${MODEL_ARCH} \
          --task ${TASK} \
          [--warm_start]
done
# Launch master
NODE_RANK=0
WORKER_NODE=${ALL_HOSTS[0]}
ssh $WORKER_NODE docker run --gpus all \
  -v ${PROCESSED_DATA_DIR}:/data/procgen/v1/processed \
  -v ${CKPT_DIR}:/ckpts \
  proc-gen:latest \
    NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((${NUM_GPUS_PER_NODE}-1))) /opt/conda/bin/python -m torch.distributed.launch \
      --nproc_per_node=${NUM_GPUS_PER_NODE} \
      --nnodes=${WORLD_SIZE} \
      --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDRESS} \
      --master_port=${MASTER_PORT} \
      /opt/conda/bin/pg-train-model \
        --data_dir /data/procgen/v1/processed \
        --dataset ${DATASET} \
        --problem ${PROBLEM} \
        --model_type ${MODEL_TYPE} \
        --model_arch ${MODEL_ARCH} \
        --task ${TASK} \
        [--warm_start]
```

### Generating predictions
```bash
docker run --gpus all \
  -v ${PROCESSED_DATA_DIR}:/data/procgen/v1/processed \
  -v ${CKPT_DIR}:/ckpts \
  -v ${RESULTS_DIR}:/results \
  proc-gen:latest \
    pg-generate-predictions \
      --data_dir /data/procgen/v1/processed \
      --dataset ${DATASET} \
      --problem ${PROBLEM} \
      --model_type ${MODEL_TYPE} \
      --model_arch ${MODEL_ARCH}
```

#### Interactive generation
```bash
docker run -it \
  -v ${PROCESSED_DATA_DIR}:/data/procgen/v1/processed \
  -v ${CKPT_DIR}:/ckpts \
  -v ${RESULTS_DIR}:/results \
  proc-gen:latest \
    fairseq-interactive \
      /data/procgen/v1/processed/${PROBLEM}/${DATASET}/${MODEL_TYPE}/data-bin/tokenized/ \
      --path /ckpts/procgen/v1/processed/${PROBLEM}/${DATASET}/${MODEL_TYPE}/ckpts-transformer_iwslt_de_en/checkpoint_best.pt
```

### Evaluation
```bash
docker run \
  -v ${PROCESSED_DATA_DIR}:/data/procgen/v1/processed \
  -v ${RESULTS_DIR}:/results \
  proc-gen:latest \
    pg-evaluate-model \
        --data_dir ${WORKDIR}/data/procgen/v1/processed \
        --dataset ${DATASET} \
        --problem ${PROBLEM} \
        --model_type ${MODEL_TYPE} \
        --model_arch ${MODEL_ARCH}
```

## Citation
```bibtex
@inproceedings{geluykens2021procgen,
  title={Neural Machine Translation for Conditional Generation of Novel Procedures},
  author={Geluykens, Joppe and Mitrovi{\'c}, Sandra and Ortega V{\'a}zquez, Carlos Eduardo and Laino, Teodoro and Vaucher, Alain and De Weerdt, Jochen},
  booktitle={Proceedings of the 54th Hawaii International Conference on System Sciences},
  pages={1091},
  year={2021}
}
```
