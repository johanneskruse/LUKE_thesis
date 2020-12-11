import os

import subprocess

'''
This script is made for running the command prompts inside a python script for running multiple experiments. 

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

### Parameters of interest: 
model_file = "luke_large_500k.tar.gz"
output_dir = "data/output/OpenEntity"

data_dir = "data/OpenEntity"
train_batch_size = 4
gradient_accumulation_steps = 2 # default 1 
learning_rate = 1e-5
num_train_epochs = 5
seed = [12,13,14,15,16]
saving_model = "dont-save-model"

# Executable (** OBS " " [space] for each line): 
for i, seed_loop in enumerate(seed): 

    print(f"Robustness for seed: {i+1}/{len(seed)}")

    temp_output_dir = os.path.join(output_dir, f"robustness_seed_{seed_loop}")

    if not os.path.exists(temp_output_dir):
        os.mkdir(temp_output_dir)

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

    #os.system(f"python -m examples.cli --model-file={model_file} --output-dir={output_dir} entity-typing run --data-dir={data_dir} --fp16 --train-batch-size={train_batch_size} --gradient-accumulation-steps={gradient_accumulation_steps} --learning-rate={learning_rate} --num-train-epochs={num_train_epochs} --seed={seed_loop} --{saving_model}")

