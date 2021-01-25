# Credit

As part of my thesis we will do a comprehensive investigation of the LUKE model. Majority of the code originates from the GitHub repo: https://github.com/studio-ousia/luke. The work is done by Ikuya Yamada and his team.

If you are intersted in the original work, please find [original paper](https://arxiv.org/abs/2010.01057).

Note, as I am still working on my thesis this repo may be subject to change. 

# Getting Started 

**LUKE** (**L**anguage **U**nderstanding with **K**nowledge-based
**E**mbeddings) is a new pre-trained contextualized representation of words and
entities based on transformer. It achieves state-of-the-art results on important
NLP benchmarks including
**[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)** (extractive
question answering),
**[CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)** (named entity
recognition), **[ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)**
(cloze-style question answering),
**[TACRED](https://nlp.stanford.edu/projects/tacred/)** (relation
classification), and
**[Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)**
(entity typing).

This repository contains the source code to pre-train the model and fine-tune it
to solve downstream tasks.

## Comparison with State-of-the-Art

LUKE outperforms the previous state-of-the-art methods on five important NLP
tasks:

| Task                           | Dataset                                                                      | Metric | LUKE              | Previous SOTA                                                             |
| ------------------------------ | ---------------------------------------------------------------------------- | ------ | ----------------- | ------------------------------------------------------------------------- |
| Extractive Question Answering  | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)                    | EM/F1  | **90.2**/**95.4** | 89.9/95.1 ([Yang et al., 2019](https://arxiv.org/abs/1906.08237))         |
| Named Entity Recognition       | [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)                 | F1     | **94.3**          | 93.5 ([Baevski et al., 2019](https://arxiv.org/abs/1903.07785))           |
| Cloze-style Question Answering | [ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)                         | EM/F1  | **90.6**/**91.2** | 83.1/83.7 ([Li et al., 2019](https://www.aclweb.org/anthology/D19-6011/)) |
| Relation Classification        | [TACRED](https://nlp.stanford.edu/projects/tacred/)                          | F1     | **72.7**          | 72.0 ([Wang et al. , 2020](https://arxiv.org/abs/2002.01808))             |
| Fine-grained Entity Typing     | [Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html) | F1     | **78.2**          | 77.6 ([Wang et al. , 2020](https://arxiv.org/abs/2002.01808))             |

These numbers are reported in
[our EMNLP 2020 paper](https://arxiv.org/abs/2010.01057).

## Installation

LUKE can be installed using [Poetry](https://python-poetry.org/):

```bash
$ poetry install
```

The virtual environment automatically created by Poetry can be activated by
`poetry shell`.

## Released Models

We initially release the pre-trained model with 500K entity vocabulary based on
the `roberta.large` model.

| Name          | Base Model                                                                                          | Entity Vocab Size | Params | Download                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------- | ----------------- | ------ | ------------------------------------------------------------------------------------------ |
| **LUKE-500K** | [roberta.large](https://github.com/pytorch/fairseq/tree/master/examples/roberta#pre-trained-models) | 500K              | 483 M  | [Link](https://drive.google.com/file/d/1S7smSBELcZWV7-slfrb94BKcSCCoxGfL/view?usp=sharing) |

## Reproducing Experimental Results

The experiments were conducted using Python3.6 and PyTorch 1.2.0 installed on a
server with a single or eight NVidia V100 GPUs. We used
[NVidia's PyTorch Docker container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
19.02. For computational efficiency, we used mixed precision training based on
APEX library which can be installed as follows:

```bash
$ git clone https://github.com/NVIDIA/apex.git
$ cd apex
$ git checkout c3fad1ad120b23055f6630da0b029c8b626db78f
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

The APEX library is not needed if you do not use `--fp16` option or reproduce
the results based on the trained checkpoint files.

The commands that reproduce the experimental results are provided as follows:

### Entity Typing on Open Entity Dataset

**Dataset:** [Link](https://github.com/thunlp/ERNIE)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10F6tzx0oPG4g-PeB0O1dqpuYtfiHblZU/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-typing run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-typing run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=2 \
    --gradient-accumulation-steps=2 \
    --learning-rate=1e-5 \
    --num-train-epochs=3 \
    --fp16
```

### Relation Classification on TACRED Dataset

**Dataset:** [Link](https://nlp.stanford.edu/projects/tacred/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10XSaQRtQHn13VB_6KALObvok6hdXw7yp/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    relation-classification run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    relation-classification run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=4 \
    --gradient-accumulation-steps=8 \
    --learning-rate=1e-5 \
    --num-train-epochs=5 \
    --fp16
```

### Named Entity Recognition on CoNLL-2003 Dataset

**Dataset:** [Link](https://www.clips.uantwerpen.be/conll2003/ner/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10VFEHXMiJGQvD62QbHa8C8XYSeAIt_CP/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    ner run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli\
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    ner run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=2 \
    --gradient-accumulation-steps=2 \
    --learning-rate=1e-5 \
    --num-train-epochs=5 \
    --fp16
```

### Cloze-style Question Answering on ReCoRD Dataset

**Dataset:** [Link](https://sheng-z.github.io/ReCoRD-explorer/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10LuPIQi-HslZs_BgHxSnitGe2tw_anZp/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-span-qa run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --num-gpus=8 \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-span-qa run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=1 \
    --gradient-accumulation-steps=4 \
    --learning-rate=1e-5 \
    --num-train-epochs=2 \
    --fp16
```

### Extractive Question Answering on SQuAD 1.1 Dataset

**Dataset:** [Link](https://rajpurkar.github.io/SQuAD-explorer/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/1097QicHAVnroVVw54niPXoY-iylGNi0K/view?usp=sharing)\
**Wikipedia data files (compressed):**
[Link](https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    reading-comprehension run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-negative \
    --wiki-link-db-file=enwiki_20160305.pkl \
    --model-redirects-file=enwiki_20181220_redirects.pkl \
    --link-redirects-file=enwiki_20160305_redirects.pkl \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --num-gpus=8 \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    reading-comprehension run \
    --data-dir=<DATA_DIR> \
    --no-negative \
    --wiki-link-db-file=enwiki_20160305.pkl \
    --model-redirects-file=enwiki_20181220_redirects.pkl \
    --link-redirects-file=enwiki_20160305_redirects.pkl \
    --train-batch-size=2 \
    --gradient-accumulation-steps=3 \
    --learning-rate=15e-6 \
    --num-train-epochs=2 \
    --fp16
```

# Analysis
This has been added to the original work. 

## Confusion matrix: 
To quantify the performance of a single model using confusion matrices:
```
python luke_experiments.confusion_matrix --data-dir=<DATA_DIR> --cm-output-dir=<OUTPUT_DIR>
```

## **Plotting meta-analysis for different experimental settings:**
We have perform multiple experiments with different hyperparameter setting for LUKE, e.g. *learning rate*, *dropout*, and *batch size*.

You will have to manually adjust the experimental tags of which the experimentent  ("learning_rate"). Open the luke_experiments.meta_analysis.py and modify:
``` 
tags = ["NAME_OF_PARAMETER_1", "NAME_OF_PARAMETER_2"]
```

To generate the event files for Tensorboard run: 
```
python -m luke_experiments.meta_analysis --data-dir [path/to/experiment_output]
```

Here data-dir is the path to the experiments that contain all folders with experiments. Thus, in an experiment multiple experiment folder exists, where each of them contain a result.json file ([robust_seed_1, robust_seed_2, robust_seed_3, etc.)

The output-dir is default: 'luke_experiments' (the same as the scripts). Note, the script also provides scatter and calibration plots for each model output. You can disable a meta-analysis plot (true/false): 
- "--tensorboard-plot/--no-tensorboard-plot" 
- "--scatter-plot/--no-scatter-plot"
- "--calibration-plot/--no-calibration-plot"

The default is True.

### **Run the Tensorboard:**

Run tensorboard: 

```bash
tensorboard --logdir luke_experiments/plots_meta_analysis/runs_tensorboards
```

This is the default output path from ```meta_analysis.py```.