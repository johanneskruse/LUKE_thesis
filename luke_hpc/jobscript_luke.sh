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

python luke_experiment/experiment_setup_file_single.py 
# python luke_experiment/experiment_setup_file_multiple.py 

# Paper reconstruction:
#python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/OpenEntity entity-typing run --data-dir=data/OpenEntity --fp16 --dont-save-model --gradient-accumulation-steps=2 --learning-rate=1e-5 --train-batch-size=4 --num-train-epochs=3
#python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/tacred_1 relation-classification run --data-dir=data/TACRED --fp16 --train-batch-size=32 --gradient-accumulation-steps=8 --learning-rate=1e-5 --num-train-epochs=5
#python -m examples.cli --model-file=luke_large_500k.tar.gz --output-dir=data/outputs/paper_reconstruction/conll2003_1 ner run --data-dir=data/CoNLL2003 --fp16 --train-batch-size=8 --gradient-accumulation-steps=2 --learning-rate=1e-5 --num-train-epochs=5 
