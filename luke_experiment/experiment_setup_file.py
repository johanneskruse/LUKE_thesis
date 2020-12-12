'''
This script is made for running the command prompts inside a python script for running multiple experiments. 

To change parameter: data-dir = data_dir, num-train-epochs = num_train_epochs, etc. 

    # examples/utils/trainer.py: 
    @click.option("--data-dir", default="data/open_entity", type=click.Path(exists=True))
    @click.option("--do-train/--no-train", default=True)
    @click.option("--train-batch-size", default=2)
    @click.option("--do-eval/--no-eval", default=True)
    @click.option("--eval-batch-size", default=32)
    @click.option("--num-train-epochs", default=3.0)
    @click.option("--seed", default=12)
    
    # examples/entity_typing/main.py: 
    @click.option("--learning-rate", default=1e-5)
    @click.option("--lr-schedule", default="warmup_linear", type=click.Choice(["warmup_linear", "warmup_constant"]))
    @click.option("--weight-decay", default=0.01)
    @click.option("--max-grad-norm", default=0.0)
    @click.option("--adam-b1", default=0.9)
    @click.option("--adam-b2", default=0.98)
    @click.option("--adam-eps", default=1e-6)
    @click.option("--adam-correct-bias", is_flag=True)
    @click.option("--warmup-proportion", default=0.06)
    @click.option("--gradient-accumulation-steps", default=1)
    @click.option("--fp16", is_flag=True)
    @click.option("--fp16-opt-level", default="O2")
    @click.option("--fp16-min-loss-scale", default=1)
    @click.option("--fp16-max-loss-scale", default=4)
    @click.option("--save-steps", default=0)
    @click.option("--save-model/--dont-save-model", is_flag=True)
'''

import os
import subprocess

# Path parameters: 
model_file  = "luke_large_500k.tar.gz"
data_dir    = "data/OpenEntity"
output_dir  = "data/outputs/OpenEntity"

### Hyperparameters: 
train_batch_size = 4
gradient_accumulation_steps = 2 # default 1 
learning_rate = 1e-5
num_train_epochs = 10
seed = list(range(10,20))
saving_model = "dont-save-model"

# Naming: 
experiment_tag = "seed"

# # ============

# Experiment: 
for i, seed_loop in enumerate(seed): 

    temp_output_dir = os.path.join(output_dir, f"robustness_{experiment_tag}_{seed_loop}")
    
    if os.path.exists(temp_output_dir): 
        continue 
        # examples/cli.py -> makes output_dir
    
    # Terminal command: 
    subprocess.call((
        f"python", "-m",
        f"examples.cli", 
        f"--model-file={model_file}", 
        f"--output-dir={temp_output_dir}",
        f"entity-typing", "run",
        f"--data-dir={data_dir}",
        f"--fp16",
        f"--train-batch-size={train_batch_size}",
        f"--gradient-accumulation-steps={gradient_accumulation_steps}",
        f"--learning-rate={learning_rate}",
        f"--num-train-epochs={num_train_epochs}",
        f"--seed={seed_loop}",
        f"--{saving_model}"
    ))

# Move: out and err files: 
out_err_folder = "luke_experiment/out_err_folder_hpc"
if not os.path.exists(f"{out_err_folder}"):
    os.makedirs(f"{out_err_folder}")

os.system(f"mv luke_experiment/*.err {out_err_folder}")
os.system(f"mv luke_experiment/*.out {out_err_folder}")
