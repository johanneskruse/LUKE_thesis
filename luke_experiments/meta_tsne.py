import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from confusion_matrix import one_hot_encoding, add_reject_entry, split_multi_single

# ================================================================================================


flatten = lambda t: [item for sublist in t for item in sublist]


def tsne_compoments(matrix, title):
    """
    Returns a matplotlib figure containing the 2D and 3D T-SNE plots.
    
    Args:
        matrix: ndarray of (n_samples, n_features)
    """
    # Note, input should be: X: 
    data_matrix = TSNE(n_components=3).fit_transform(matrix)
    # Define compoments: 
    one = data_matrix[:,0]
    two = data_matrix[:,1]
    thr = data_matrix[:,2]

    # 2D: 
    one_two = tsne_2d(one, two, title=title, labels=["1st T-SNE component", "2nd T-SNE component"])
    one_thr = tsne_2d(one, thr, title=title, labels=["1st T-SNE component", "3rd T-SNE component"])
    two_thr = tsne_2d(two, thr, title=title, labels=["2nd T-SNE component", "3rd T-SNE component"])

    # 3D: 
    one_two_thr = tsne_3d(one, two, thr, title, ["1st T-SNE component", "2nd T-SNE component", "3rd T-SNE component"])

    return one_two, one_thr, two_thr, one_two_thr


def tsne_2d(x, y, title, labels):
    """
    Returns a matplotlib figure containing the 2D T-SNE plot.
    
    Args:
        x, y, z: arrays
        title: string with name of the plot
        labels: list of strings with label names: [x, y, z]
    """
    plt.rcParams.update({'font.size': 40, 'legend.fontsize': 20})
    plt.rc('font', size=30)
    plt.rc('axes', titlesize=35)

    figure, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y)
    ax.set_title(title) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    plt.grid()
    
    plt.tight_layout()

    return figure    



def tsne_3d(x, y, z, title, labels):
    """
    Returns a matplotlib figure containing the 3D T-SNE plot.
    
    Args:
        x, y, z: arrays
        title: string with name of the plot
        labels: list of strings with label names: [x, y, z]
    """
    plt.rcParams.update({'font.size': 30, 'legend.fontsize': 20})
    plt.rc('font', size=30)
    plt.rc('axes', titlesize=35)
    labelpad = 30

    figure = plt.figure(figsize=(12,12))
    ax = figure.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.set_title(title) 
    ax.set_xlabel(labels[0], labelpad=labelpad)
    ax.set_ylabel(labels[1], labelpad=labelpad)
    ax.set_zlabel(labels[2], labelpad=labelpad)
    plt.tight_layout()

    return figure


def save_tsne(one_two, one_thr, two_thr, one_two_thr, title, save_dir = "luke_experiments"):    
    t_sne_dir = os.path.join(save_dir, "plots_meta_analysis", "plots_tsne")
    if not os.path.exists(t_sne_dir):
        os.makedirs(t_sne_dir)

    one_two.savefig(f"{t_sne_dir}/{title}_one_two")
    one_thr.savefig(f"{t_sne_dir}/{title}_one_thr")
    two_thr.savefig(f"{t_sne_dir}/{title}_two_thr")
    one_two_thr.savefig(f"{t_sne_dir}/{title}_one_two_thr")


# ================================================================================================

#
# data_dir = "../data/outputs/seed_experiment_500"
# output_dir = "."
# tsne_save = True
#

# Define path to data source: 

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", default="data/outputs/seed_experiment_500")
parser.add_argument("--output-dir", default="luke_experiments")
parser.add_argument('--save-tsne', dest='tsne_save', action='store_true')
parser.add_argument('--no-save-tsne', dest='tsne_save', action='store_false')
parser.set_defaults(tsne_save=True)

args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
tsne_save = args.tsne_save


def run():
    ##############################
    # Confusion Matrix
    eval_sets = ['dev', 'test']
    confusion_matrices = {"dev" : {}, "test" : {}}
    for root, dirs, files in os.walk(data_dir):
        
        for dir_ in dirs: 

            result_json = os.path.join(root, dir_, "results.json")
            
            try:
                with open(result_json) as json_file:
                    data = json.load(json_file)
            except:
                print(f"{dir_} does not contain a results.json")
                continue

            evaluations = data["evaluation_predict_label"]

            for eval_set in eval_sets: 

                confusion_matrices[eval_set][dir_] = {}
                
                y_true = one_hot_encoding(evaluations[eval_set]["true_labels"], add_reject=True)
                y_pred = one_hot_encoding(evaluations[eval_set]["predict_logits"], add_reject=True)

                ##############################
                ######## Multi-label #########
                confusion_matrices[eval_set][dir_]["multi_label_all"] = multilabel_confusion_matrix(y_true, y_pred)

                ##############################
                #### Splitting into multi-label and single labelled ####
                y_true_single, y_pred_single, y_true_multi, y_pred_multi = split_multi_single(y_true, y_pred)

                # The category index: 
                true_entity_single = y_true_single.argmax(axis=1)
                pred_entity_single = y_pred_single.argmax(axis=1)

                # Confusion matrix only for single labelled: 
                confusion_matrices[eval_set][dir_]["single_label_only"] = confusion_matrix(true_entity_single, pred_entity_single)

                # Confusion matrix only for multi-labelled: 
                confusion_matrices[eval_set][dir_]["multi_label_only"] = multilabel_confusion_matrix(y_true_multi, y_pred_multi)

    ##############################
    # T-SNE analysis
    for eval_set in eval_sets: 
        multi_label_all     = [confusion_matrices[eval_set][z]["multi_label_all"] for z in confusion_matrices[eval_set].keys()]
        single_label_only   = [confusion_matrices[eval_set][z]["single_label_only"] for z in confusion_matrices[eval_set].keys()]
        multi_label_only    = [confusion_matrices[eval_set][z]["multi_label_only"] for z in confusion_matrices[eval_set].keys()]
        
        features_single = len(flatten(single_label_only[0]))
        features_multi  = len(flatten(flatten(multi_label_all[0])))

        # Flatten list
        matrix_sinlge = np.array(flatten(flatten(single_label_only))).reshape(-1, features_single) 
        
        matrix_multi_label_all  = np.array(flatten(flatten(flatten(multi_label_all)))).reshape(-1, features_multi)
        matrix_multi_label_only = np.array(flatten(flatten(flatten(multi_label_only)))).reshape(-1, features_multi)

        one_two_sing, one_thr_sing, two_thr_sing, one_two_thr_sing = tsne_compoments(matrix_sinlge, title="T-SNE - Seed Experiment\nSingle-labelled")
        one_two_m_only, one_thr_m_only, two_thr_m_only, one_two_thr_m_only = tsne_compoments(matrix_multi_label_only, title="T-SNE - Seed Experiment\nMulti-labelled")
        one_two_all, one_thr_all, two_thr_all, one_two_thr_all = tsne_compoments(matrix_multi_label_all, title="T-SNE - Seed Experiment\nSingle-labelled & Multi-labelled")

        if tsne_save: 
            save_tsne(one_two_sing, one_thr_sing, two_thr_sing, one_two_thr_sing, f"{eval_set}_single_label_only")
            save_tsne(one_two_m_only, one_thr_m_only, two_thr_m_only, one_two_thr_m_only, f"{eval_set}_multi_label_only")
            save_tsne(one_two_all, one_thr_all, two_thr_all, one_two_thr_all, f"{eval_set}_multi_label_all")


if __name__ == "__main__":
    run()