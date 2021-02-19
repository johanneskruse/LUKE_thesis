#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J LUKE_exp
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s153098@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# here follow the commands you want to execute 

source ~/.bashrc
conda activate luke

# ============================================================
#   Run experiments
# ============================================================
# python luke_hpc/experiment_setup_file.py 

# ============================================================
#   Paper reconstruction (generates checkpoint file):
# ============================================================
# python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/OpenEntity entity-typing run --data-dir=data/OpenEntity --train-batch-size=2 --gradient-accumulation-steps=2 --learning-rate=1e-5 --num-train-epochs=3 --fp16
# python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/conll2003 ner run --data-dir=data/CoNLL2003 --fp16 --train-batch-size=4 --gradient-accumulation-steps=2 --learning-rate=1e-5 --num-train-epochs=5 
# python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/tacred relation-classification run --data-dir=data/TACRED --fp16 --train-batch-size=4 --gradient-accumulation-steps=8 --learning-rate=1e-5 --num-train-epochs=5

# ============================================================
#   OpenEntity (using the checkpoint file)
# ============================================================
## Use author checkpoint file: 
# python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/OpenEntity_author_checkpoint_file entity-typing run --data-dir=data/OpenEntity --checkpoint-file=data/check_point_files/OpenEntity_author/pytorch_model.bin --no-train

## Use our checkpoint file: 
# python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/OpenEntity_our_checkpoint_file entity-typing run --data-dir=data/OpenEntity --checkpoint-file=data/check_point_files/OpenEntity_reconstructed/pytorch_model.bin --no-train