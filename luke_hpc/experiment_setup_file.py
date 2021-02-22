'''
This script is made for running the command prompts inside a python script for running multiple experiments. 

To change parameter: data-dir = data_dir, num-train-epochs = num_train_epochs, etc. 

    # examples/entity_typing/main.py: 
    @click.option("--data-dir", default="data/open_entity", type=click.Path(exists=True))
    @click.option("--do-train/--no-train", default=True)
    @click.option("--train-batch-size", default=2)
    @click.option("--do-eval/--no-eval", default=True)
    @click.option("--eval-batch-size", default=32)
    @click.option("--num-train-epochs", default=3.0)
    @click.option("--seed", default=12)
    @click.option("--do-evaluate-prior-train/--no-evaluate-prior-train", default=True)
    
    # examples/utils/trainer.py: 
    @click.option("--learning-rate", default=1e-5)
    @click.option("--lr-schedule", default="warmup_linear", type=click.Choice(["warmup_linear", "warmup_constant"]))
    @click.option("--weight-decay", default=0.01)
    @click.option("--hidden-dropout-prob", default=0.1) # Added by us
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



# ========================================================================
import os
import subprocess
import numpy as np
import inspect


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def experiment(output_dir=".", **kwags):
    # Path parameters: 
    model_file  = "luke_large_500k.tar.gz"
    data_dir    = "data/OpenEntity"
    output_dir  = output_dir 
    saving_model= "dont-save-model"

    # Hyperparameter: 
    train_batch_size            = 2
    gradient_accumulation_steps = 2
    learning_rate               = 1e-5
    num_train_epochs            = 10
    seed                        = 12
    train_frac_size             = 1.0
    weight_decay                = 0.01
    hidden_dropout_prob         = 0.1

    experiment_tag = list(kwags.keys())[0]
    loop_items = list(kwags.values())[0]

    # Experiment: 
    for loop_item in loop_items: 
        # Naming:         
        if experiment_tag == "gradient_accumulation_steps":
            gradient_accumulation_steps = loop_item
        if experiment_tag == "num_train_epochs":
            num_train_epochs = loop_item
        if experiment_tag == "train_batch_size":
            train_batch_size = loop_item
            gradient_accumulation_steps = 1
        if experiment_tag == "learning_rate":
            learning_rate = loop_item
        if experiment_tag == "seed":
            seed = loop_item
        if experiment_tag == "train_frac_size":
            train_frac_size = loop_item
        if experiment_tag == "weight_decay":
            weight_decay = loop_item
        if experiment_tag == "hidden_dropout_prob":
            hidden_dropout_prob = loop_item

        # if type(loop_item) is np.float64:
        #     loop_item_str = str(loop_item).replace(".", "_")
        #     temp_output_dir = os.path.join(output_dir, f"robust_{experiment_tag}_{loop_item_str}")
        # else:
        
        temp_output_dir = os.path.join(output_dir, f"robust_{experiment_tag}_{loop_item}")
        
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
            f"--do-evaluate-prior-train",
            f"--seed={seed}",
            f"--{saving_model}",
            f"--num-train-epochs={num_train_epochs}",
            f"--gradient-accumulation-steps={gradient_accumulation_steps}",
            f"--train-batch-size={train_batch_size}",
            f"--learning-rate={learning_rate}",
            f"--train-frac-size={train_frac_size}",
            f"--weight-decay={weight_decay}",
            f"--hidden-dropout-prob={hidden_dropout_prob}"
        ))


# ========================================================================


output_dir = "data/outputs/seed_lr_wd_batch_train_size_dropout_with_eval_no_train_dropout"
# experiment(output_dir=output_dir, seed = list(range(10,21,1)))                                                      # Default: 12     
# experiment(output_dir=output_dir, learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])    # Default: 1e-5  
# experiment(output_dir=output_dir, weight_decay = [1000, 100, 75, 50, 25, 0, 1e-1, 1e-2, 1e-3, 1e-4])          # Default: 0.01
# experiment(output_dir=output_dir, train_batch_size = [1, 2, 4, 8, 16, 32, 64])                                      # Default: 4 [2 + gradient_accumulation_steps = 2]
# experiment(output_dir=output_dir, train_batch_size = [1]) # set gradient_accumulation_steps = 1
# experiment(output_dir=output_dir, train_frac_size = [0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])       # Default: 1.0
# experiment(output_dir=output_dir, hidden_dropout_prob = np.round(np.arange(0, 1, 0.1),2))                         # Default: 0.01


# Move: out and err files: 
out_err_folder = "luke_hpc/out_err_folder_hpc"
if not os.path.exists(f"{out_err_folder}"):
    os.makedirs(f"{out_err_folder}")


os.system(f"mv *.err {out_err_folder}")
os.system(f"mv *.out {out_err_folder}")

