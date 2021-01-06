from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import json
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 24})
plt.rc('font', size=16)
plt.rc('axes', titlesize=16)

from sklearn.calibration import calibration_curve

import argparse

# ================================================================================================

def add_reject_entry(one_hot_dataset):
    # Add reject as class: 
    for i, array in enumerate(one_hot_dataset): 
        if sum(array) > 0:
            one_hot_dataset[i] = np.append(array, 0)
        else:
            one_hot_dataset[i] = np.append(array, 1)
    return one_hot_dataset


def one_hot_encoding(dataset, add_reject=True):
    # One-hot encode and change type to np.array: 
    one_hot = [(np.array(z) > 0)*1 for z in dataset]    
    
    if add_reject:
        # Add rejection as a class
        one_hot = add_reject_entry(one_hot)

    # Make matrix: 
    num_entities = len(one_hot[0])
    one_hot = np.concatenate(one_hot)

    one_hot = np.reshape(one_hot, (-1, num_entities))

    return one_hot


def logit2prob(logits):
    prob = np.array([])
    for logit in logits: 
        prob = np.append(prob, (np.exp(logit) / (np.exp(logit) + 1)))
    return prob


def split_multi_single(y_true, y_pred):
    y_true_single = np.array([])
    y_pred_single = np.array([])
        
    y_true_multi = np.array([])
    y_pred_multi = np.array([])

    num_entities = len(y_true[0])

    for i in range(len(y_pred)):
        if sum(y_true[i,:]) <= 1 and sum(y_pred[i,:]) <=1: 
            y_true_single = np.append(y_true_single, y_true[i,:])
            y_pred_single = np.append(y_pred_single, y_pred[i,:])
        else: 
            y_true_multi = np.append(y_true_multi, y_true[i,:])
            y_pred_multi = np.append(y_pred_multi, y_pred[i,:])
    
    # Reshape: 
    y_true_single = np.reshape(y_true_single, (-1,num_entities))
    y_pred_single = np.reshape(y_pred_single, (-1,num_entities))   
    
    y_true_multi = np.reshape(y_true_multi, (-1,num_entities))
    y_pred_multi = np.reshape(y_pred_multi, (-1,num_entities)) 

    return y_true_single, y_pred_single, y_true_multi, y_pred_multi


def plot_cm(cm, class_names, normalize=False, cbar=True, font_scale=1.5, title=False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
       normalize: if cm should show normalized values
    """

    if normalize: 
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    if cbar:
        figure = plt.figure(figsize=(10,9))
    else: 
        figure = plt.figure(figsize=(7,7))
    #plt.title("Confusion matrix")
    sns.set(font_scale=font_scale) # label size
    sns.heatmap(cm, cmap="Blues", annot=True, fmt='g', linewidth=0.5, cbar=cbar) # font size
    
    if title: 
        plt.title(f"'{class_names[1]}' vs rest")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks+0.5, class_names, rotation=45)
    plt.yticks(tick_marks+0.5, class_names, rotation=0)
    
    # Labels, title and ticks
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure


# ================================================================================================

# Define path to data source: 

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", default="data/outputs/paper_reconstruction/OpenEntity/results.json")
parser.add_argument("--cm-output-dir", default="luke_experiments")
parser.add_argument('--save-cm', dest='cm_save', action='store_true')
parser.add_argument('--no-save-cm', dest='cm_save', action='store_false')
parser.set_defaults(cm_save=True)

args = parser.parse_args()

data_dir = args.data_dir
cm_output_dir = args.cm_output_dir
cm_save = args.cm_save

# data_dir = "../data/outputs/paper_reconstruction/OpenEntity/results.json"

# ================================================
def run_cm():
    with open(data_dir) as json_file:
        data = json.load(json_file)

    evaluations = data["evaluation_predict_label"]
    confusion_matrices = {"dev": {}, "test": {}}
    eval_sets = ['dev', 'test']
    labels = evaluations["label_list"]
    labels.append("reject")

    for eval_set in eval_sets: 

        y_true = one_hot_encoding(evaluations[eval_set]["true_labels"])
        y_pred = one_hot_encoding(evaluations[eval_set]["predict_logits"])

        ##############################
        ######## Multi-label #########
        confusion_matrices[eval_set]["multi_label_all"] = multilabel_confusion_matrix(y_true, y_pred)

        ##############################
        #### Splitting into multi-label and single labelled ####
        y_true_single, y_pred_single, y_true_multi, y_pred_multi = split_multi_single(y_true, y_pred)

        # The category index: 
        true_entity_single = y_true_single.argmax(axis=1)
        pred_entity_single = y_pred_single.argmax(axis=1)

        # Confusion matrix only for single labelled: 
        confusion_matrices[eval_set]["single_label_only"] = {"entity_index" : list(set(np.append(true_entity_single, pred_entity_single))),
                                                            "confusion_matrix" : confusion_matrix(true_entity_single, pred_entity_single)}

        # Confusion matrix only for multi-labelled: 
        confusion_matrices[eval_set]["multi_label_only"] = multilabel_confusion_matrix(y_true_multi, y_pred_multi)

    # ================================================================================================

    if cm_save: 
        if not os.path.exists(f"{cm_output_dir}/multi_label_all"):
            os.makedirs(f"{cm_output_dir}/plot_confusion_matrix/multi_label_all", exist_ok=True)
        if not os.path.exists(f"{cm_output_dir}/plot_confusion_matrix/multi_label_only"):
            os.makedirs(f"{cm_output_dir}/plot_confusion_matrix/multi_label_only", exist_ok=True)
        else: 
            pass

    dpi = 300

    # Example: 
    for i, cm in enumerate(confusion_matrices["dev"]["multi_label_all"]):
        multi_label_all = plot_cm(cm = cm, class_names = ["other", labels[i]], normalize=True, cbar=True, font_scale=4, title=False)
        if cm_save: 
            multi_label_all.savefig(f"{cm_output_dir}/plot_confusion_matrix/multi_label_all/multi_all_{labels[i]}.png", dpi=dpi)
    
    for i, cm in enumerate(confusion_matrices["dev"]["multi_label_only"]):
        multi_label_only = plot_cm(cm = cm, class_names = ["other", labels[i]], normalize=True, cbar=True, font_scale=4, title=False)
        if cm_save: 
            multi_label_only.savefig(f"{cm_output_dir}/plot_confusion_matrix/multi_label_only/multi_{labels[i]}.png", dpi=dpi)


    # Single label: 
    single_label = plot_cm(cm = confusion_matrices["dev"]["single_label_only"]["confusion_matrix"], 
                            class_names = labels[1:], normalize=True, cbar=True)
    if cm_save:
        single_label.savefig(f"{cm_output_dir}/plot_confusion_matrix/single_label_only.png", dpi=dpi)

    print("Done!")

# ================================================================================================
# ================================================================================================


# if __name__ == "__main__":
#     run_cm()