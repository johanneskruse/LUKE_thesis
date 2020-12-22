from torch.utils.tensorboard import SummaryWriter
import torch
import json
import os
import numpy as np

from argparse import Namespace
import click


from functions import *

# ==============================================================================
# Meta analysis for experiments 
# ==============================================================================

class config():
    tags = ["learning_rate", "seed", "train_batch_size", "train_frac_size"]
    eval_sets = ["dev", "test"]

# ==============================================================================


@click.command()
@click.option("--data-dir", default="data/outputs/OpenEntity", type=click.Path(exists=True))
@click.option("--output-dir", default="luke_meta_analysis")
@click.option("--tensorboard-plot/--no-tensorboard-plot", default=True)
@click.option("--scatter-plot/--no-scatter-plot", default=True)
@click.option("--calibration-plot/--no-calibration-plot", default=True)
def run(**task_args):
    args = Namespace(**task_args)

    data_dir = args.data_dir    
    output_dir = args.output_dir
    tensorboard_plot =  args.tensorboard_plot
    scatter_plot = args.scatter_plot
    calibration_plot = args.calibration_plot
    
    ####### COPY START #######

    tensorboard_event_folder = os.path.join(output_dir, "runs_tensorboards") 

    # Tag used for the tensorboard: 
    experiment_tags = config.tags

    eval_results = {}
    pred_logts = {}

    for experiment_tag in experiment_tags:

        f1_eval = torch.tensor([])
        precision_eval = torch.tensor([])
        recall_eval = torch.tensor([])
        
        pred_logts[experiment_tag] = {}
        
        for root, dirs, files in os.walk(data_dir):

            result_json = root + "/results.json"
            base_root = os.path.basename(root)
            
            if os.path.exists(result_json) and experiment_tag in base_root:
                
                # Load data:     
                with open(f"{result_json}") as json_file:
                    data = json.load(json_file)
                
                log_parameters = data["experimental_configurations"]["log_parameters"]
                par = log_parameters[f"{experiment_tag}"]
                gradient_accumulation_steps = log_parameters["gradient_accumulation_steps"]

                # Get parameters to model: 
                if len(data["training_loss"]) % gradient_accumulation_steps != 0: 
                    print(f"Odd number of global_steps and training_loss: {base_root}")
                    data["training_loss"].append(data["training_loss"][-1])
                    training_loss = np.mean(np.array(data["training_loss"]).reshape(-1, gradient_accumulation_steps), axis=1)
                else: 
                    training_loss = np.mean(np.array(data["training_loss"]).reshape(-1, gradient_accumulation_steps), axis=1)

                ######################################################
                        #### Meta Analysis ####
                ######################################################

                ###########################
                # During training: 
                training_loss   = torch.tensor(training_loss)
                dev_f1          = torch.tensor([data[z] for z in data if "dev_f1_epoch" in z])
                dev_precision   = torch.tensor([data[z] for z in data if "dev_precision_epoch" in z])
                dev_recall      = torch.tensor([data[z] for z in data if "dev_recall_epoch" in z])


                ###########################
                # Development and Test evalution:
                ###########################
                # F1, Precision and Recall: 
                dev_f1_eval         = torch.tensor([data[z] for z in data if "dev_f1" in z and "_epoch" not in z])
                test_f1_eval        = torch.tensor([data[z] for z in data if "test_f1" in z])
                
                dev_precision_eval  = torch.tensor([data[z] for z in data if "dev_precision" in z and "_epoch" not in z])
                test_precision_eval = torch.tensor([data[z] for z in data if "test_precision" in z])
                
                dev_recall_eval     = torch.tensor([data[z] for z in data if "dev_recall" in z and "_epoch" not in z])
                test_recall_eval    = torch.tensor([data[z] for z in data if "test_recall" in z])


                # Data for Scatter Plot: 
                f1_eval         = np.append(f1_eval, [dev_f1_eval, test_f1_eval])
                precision_eval  = np.append(precision_eval, [dev_precision_eval, test_precision_eval])
                recall_eval     = np.append(recall_eval, [dev_recall_eval, test_recall_eval])


                # For Calibration plots:   
                pred_logts[experiment_tag][base_root] = {}
                pred_logts[experiment_tag][base_root]["dev"] = data["evaluation_predict_label"]["dev"]
                pred_logts[experiment_tag][base_root]["test"] = data["evaluation_predict_label"]["test"]


                ######################################################
                    # TensorBoards: experimental based
                ######################################################
                
                # If event file already exists or not plotting tensorboards: 
                if not tensorboard_plot or os.path.exists(f"{tensorboard_event_folder}/{base_root}"):
                    continue
                else:
                    # Experimental based: 
                    tb = SummaryWriter(log_dir = f"{tensorboard_event_folder}/{base_root}")

                    for step, loss in enumerate(training_loss):
                        tb.add_scalar(f"{experiment_tag}/00.loss/Train", scalar_value=training_loss[step], global_step=step)
                    
                    for epoch, recall in enumerate(dev_f1):
                        tb.add_scalar(f"{experiment_tag}/01.f1/development", scalar_value=dev_f1[epoch], global_step=epoch)
                    
                    for epoch, precision in enumerate(dev_precision):
                        tb.add_scalar(f"{experiment_tag}/02.precision/development", scalar_value=dev_precision[epoch], global_step=epoch)
                        
                    for epoch, recall in enumerate(dev_recall):
                        tb.add_scalar(f"{experiment_tag}/03.recall/development", scalar_value=dev_recall[epoch], global_step=epoch)

                    tb.close()
            
        ######################################################
        # Evaluation: Experimental based
        ######################################################
        
        # Scatter plot for dev and test:
        if len(f1_eval) != 0 and scatter_plot: 
            eval_results[experiment_tag] = {"f1" : f1_eval.reshape(-1,2), 
                                            "recall": precision_eval.reshape(-1,2),
                                            "precision" : recall_eval.reshape(-1,2)}

            scatter_plt = plot_scatter(eval_results[experiment_tag], experiment_tag)

            if not os.path.exists("luke_meta_analysis/scatter_plots"):
                os.makedirs("luke_meta_analysis/scatter_plots")
                
            scatter_plt.savefig(f"luke_meta_analysis/scatter_plots/dev_test_{experiment_tag}")
        

        # Calibration Plot for dev and test seperately: 
        if pred_logts[experiment_tag] and calibration_plot: 
            
            if not os.path.exists("luke_meta_analysis/calibration_plots"):
                os.makedirs("luke_meta_analysis/calibration_plots")

            for eval_set in config.eval_sets:
        
                true = []
                pred = []
            
                labels = []
                true_temp = np.array([])
                pred_temp = np.array([])
                
                for experiment in pred_logts[experiment_tag]:
                    labels.append(experiment)

                    true_temp = np.concatenate(pred_logts[experiment_tag][experiment][eval_set]["true_labels"])
                    pred_temp = np.concatenate(pred_logts[experiment_tag][experiment][eval_set]["predict_logits"])
                    pred_temp = logit2prob(pred_temp)
                
                    true.append(true_temp)
                    pred.append(pred_temp)

                cal_plot = plot_calibration_curve(true, pred, model_name=labels, title=f"{eval_set}", n_bins=10)

                cal_plot.savefig(f"luke_meta_analysis/calibration_plots/{experiment_tag}_{eval_set}")
                
####### COPY END #######

# ==============================================================================


if __name__ == '__main__':
    run()