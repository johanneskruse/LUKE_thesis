# Hello! 

This repo is part of the Thesis project *'Entity-aware representations for natural language processing'* at the Technical University of Denmark (DTU), with all plots in regard to LUKE originating from here.

In this project we set out to explore entity-aware representations for natural language processing based on language models. On that quest, we discovered the pre-trained language model LUKE.

LUKE is a new pre-trained contextualized representation of words and entities based on transformer. It achieves state-of-the-art results on important NLP benchmarks including
**[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)** (extractive
question answering),
**[CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)** (named entity
recognition), **[ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)**
(cloze-style question answering),
**[TACRED](https://nlp.stanford.edu/projects/tacred/)** (relation
classification), and
**[Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)**
(entity typing).

Majority of the code originates from the GitHub repo: https://github.com/studio-ousia/luke. 
This includes the source code to pre-train the model and fine-tune it to solve downstream tasks.
We extend the work done by Ikuya Yamada and his team by providing a pipeline for optimizing hyperparameters, which also includes a variety visualizing tools. 
Furthermore, we also present a tool for visualizing attention in the Transformer model, supporting all models from the Hugging Face transformers library (BERT, GPT-2, RoBERTa, etc.) and LUKE.
This work is based on the NLP task *entity typing*, thus, this will be included in this repo. See the original repo for the all other NLP tasks. The tools support all NLP tasks but code needs to be adapted. 

If you are intersted in the original work by Ikuya Yamada and his team: [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057).

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

These numbers are reported in the
[EMNLP 2020 paper](https://arxiv.org/abs/2010.01057).


## Installation

LUKE can be installed using [Poetry](https://python-poetry.org/):

```bash
$ poetry install
```
or 
```bash
$ pip install -r requirements.txt
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

# Getting Started

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

# Optimizing Hyperparameters and Plotting Tools 

We have used the high performance computing (HPC) systems are DTU, using a Tesla V100-SXM2. To run experiments on the cluster we have used the ```luke_hpc.jobscript_luke.sh```. 
This file runs the ```luke_hpc.experiment_setup_file.py```, where you have the option to run multiple experiments for the hyperparameters:
*train_batch_size*, 
*gradient_accumulation_steps*, 
*learning_rate*, 
*num_train_epochs*, 
*seed*, 
*train_frac_size*, 
*weight_decay*, 
*hidden_dropout_prob*.

You don't have to run all of them. Plots are only generated for the hyperparameters which you test for. 

## Confusion matrix: 
To quantify the performance of a single experiment using confusion matrices:
```
python luke_experiments.confusion_matrix --data-dir=<DATA_DIR> --cm-output-dir=<OUTPUT_DIR>
```

Here, ```--data-dir``` and ```--cm-output-dir``` are the paths for the data and the output directory. Add ´--no-save-cm´ if you don't want to save save the confusion matrices. 

## Plotting meta-analysis for different experimental settings:
By running: 
```
python -m luke_experiments.meta_analysis --data-dir <path/to/experiment_output>
```

Here ```--data-dir``` is the path to the experiments that contain all folders with experiments. Thus, in an experiment multiple experiment folder exists, where each of them contain a result.json file ([robust_seed_1, robust_seed_2, robust_seed_3, etc.)
You will generate event files for tensorbaord in for the model progression, the static images for model progression, scatter plots for experiments performance on (x=dev, y=test) performance), [calibration plots ](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html).

The output-dir is default: 'luke_experiments' (the same as the scripts). Note, the script also provides scatter and calibration plots for each model output. You can disable a meta-analysis plot (true/false): 
- "--tensorboard-plot/--no-tensorboard-plot" 
- "--scatter-plot/--no-scatter-plot"
- "--calibration-plot/--no-calibration-plot"

The default is True.

### **Run the Tensorboard:**

Run tensorboard: 

```bash
tensorboard --logdir <luke_experiments/plots_meta_analysis/runs_tensorboards>
```

This is the default output path from ```meta_analysis.py```.


# Visualization Attention

We have implemented the *Head* and *Model* from [BertViz](https://github.com/jessevig/bertviz). We also propose an extension to existing tools that allows the user to look at token-to-token attention (attention from Token1 to Token2) throughout the model's layers. 

We provide two Jupyter Notebooks *Head View* and *Model View*, respectively. We also provide a COLAB version that hold all tools is one. 

🕹 Try the [COLAB](https://colab.research.google.com/github/JKrse/LUKE_thesis/blob/master/visual_attention/colab_head_and_model_view_bert_roberta_luke.ipynb) version, which has both Head view, Model view, and the novel tool. 

## **Generate output_attentions.p**
To generate your own output_attentions.p file for visualisation run:

```
python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=<OUTPUT_DIR> entity-typing run --data-dir=<DATA_DIR> --checkpoint-file=<CHECKPOINT_DIR> --output-attentions --eval-batch-size=1 --no-train
```

This will generate the pickle file output_attentions.p which contains the “tokens”, “sentence”, and the “attention”. 
The sentence is the input for the BERT and RoBERTa (and dropdown in tool). The attention score is the attention probability for the token-to-token relationship.  

Note when running *“--outout_attentions”* the *“—eval-batch-size”* will be set to 1. Furthermore, it will appear in the *<OUTPUT_DIR>* folder. 

*For this we recommend selecting a handful of shorter examples, thus, you should modify the test.json. To easily generating the input (e.g. test.json) in the write format simple use add examples in **from_text_to_input.json** and use **from_text_to_input.py** to convert.*