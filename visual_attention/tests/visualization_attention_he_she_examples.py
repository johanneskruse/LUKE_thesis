from transformers import BertTokenizer, BertModel
import torch
import pickle 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 30, 'legend.fontsize': 20})
plt.rc('font', size=25)
plt.rc('axes', titlesize=25)

import sys
sys.path.append("..")
from bertviz.util import attention_token2token, plot_attention_token2token, format_special_chars

from scipy import stats

# =============================================================== #

def he_she_attention(luke_data, sent_index):
    token_2 = "doctor"
    token_3 = "nurse"
    attn_scores_all = {}

    for i in sent_index:
        luke_data[f"sent_{i}"]["sentence"]
        attention = luke_data[f"sent_{i}"]["attention"]
        tokens = format_special_chars(luke_data[f"sent_{i}"]["tokens"])

        if "He" in tokens:
            token_main = "He"
        if "She" in tokens:
            token_main = "She"
        
        main_t2 = attention_token2token(tokens, attention, token_main, token_2)
        main_t3 = attention_token2token(tokens, attention, token_main, token_3)

        attn_scores_all[f"{token_main}"] = {}
        attn_scores_all[f"{token_main}"][f"{token_main}_{token_2}"] = main_t2
        attn_scores_all[f"{token_main}"][f"{token_main}_{token_3}"] = main_t3
        
    t1_t2_avg_he_doctor = [np.mean(layer) for layer in attn_scores_all["He"]["He_doctor"]]
    t1_t2_avg_he_nurse = [np.mean(layer) for layer in attn_scores_all["He"]["He_nurse"]]

    t1_t2_avg_she_doctor = [np.mean(layer) for layer in attn_scores_all["She"]["She_doctor"]]
    t1_t2_avg_she_nurse = [np.mean(layer) for layer in attn_scores_all["She"]["She_nurse"]]

    return t1_t2_avg_he_doctor, t1_t2_avg_he_nurse, t1_t2_avg_she_doctor, t1_t2_avg_she_nurse


def plot_multi_token_attention(he_doctor, he_nurse, she_doctor, she_nurse, title): 
    number_of_layers = len(he_doctor)    

    labels = [
        f"[He] $\longrightarrow$ [doctor] [avg. {np.mean(he_doctor):.4f}]",
        f"[He] $\longrightarrow$ [nurse] [avg. {np.mean(he_nurse):.4f}]",
        f"[She] $\longrightarrow$ [doctor] [avg. {np.mean(she_doctor):.4f}]",
        f"[She] $\longrightarrow$ [nurse] [avg. {np.mean(she_nurse):.4f}]"
        ]

    figure, ax = plt.subplots(figsize=(18,9))
    ax.plot(range(number_of_layers), he_doctor, 'o--', color="b")
    ax.plot(range(number_of_layers), he_nurse, '*-', color="g")
    ax.plot(range(number_of_layers), she_doctor, 'o--', color="k")
    ax.plot(range(number_of_layers), she_nurse, '*-', color="c")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.6, box.height])
    legend1 = ax.legend(labels, loc='lower left', bbox_to_anchor=(1, 0.5), 
                        edgecolor="white", title="Token Connction", fontsize="medium")
    ax.add_artist(legend1)


    ax.set_title(title, size="x-large") 
    ax.set_xlabel("Layer", fontsize="large")
    ax.set_ylabel("Attention score", fontsize="large")
    ax.set_xticks(range(0, number_of_layers, 2))
    ax.set_xticklabels(range(0,number_of_layers, 2))
    plt.legend()
    plt.tick_params(axis='x', labelsize="large")
    plt.tick_params(axis='y', labelsize="large")
    plt.grid()
    plt.tight_layout(rect=[0,0,0.6,1])

    return figure 


# =============================================================== #
luke_data = pickle.load(open( "../sample_data/output_attentions.p", "rb"))
output_dir = "plot_attention_visualization"
save = True

sent_index = [25,26]
for index in sent_index:
    tokens_in_sentence = format_special_chars(luke_data[f"sent_{index}"]["tokens"])
    attention_luke = luke_data[f"sent_{index}"]["attention"]
    if "He" in tokens_in_sentence:
        plot_he_doctor = plot_attention_token2token(tokens_in_sentence, attention_luke, "He", "doctor")
        plot_he_nurse = plot_attention_token2token(tokens_in_sentence, attention_luke, "He", "nurse")
    if "She" in tokens_in_sentence:
        plot_she_doctor = plot_attention_token2token(tokens_in_sentence, attention_luke, "She", "doctor")
        plot_she_nurse = plot_attention_token2token(tokens_in_sentence, attention_luke, "She", "nurse")
    

# The doctor asked the nurse a question .
he_doctor_dn, he_nurse_dn, she_doctor_dn, she_nurse_dn = he_she_attention(luke_data, sent_index=[25,26])
title = "Average attention scores\n(doctor, nurse)"
doctor_nurse = plot_multi_token_attention(he_doctor_dn, he_nurse_dn, she_doctor_dn, she_nurse_dn, title)

# The nurse asked the doctor a question .
he_doctor_nd, he_nurse_nd, she_doctor_nd, she_nurse_nd = he_she_attention(luke_data, sent_index=[27,28])
title = "Average attention scores\n(nurse, doctor)"
nurse_doctor = plot_multi_token_attention(he_doctor_nd, he_nurse_nd, she_doctor_nd, she_nurse_nd, title)

if save: 
    dpi = 300
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    plot_he_doctor.savefig(f"{output_dir}/plot_he_doctor", dpi=dpi)
    plot_he_nurse.savefig(f"{output_dir}/plot_he_nurse", dpi=dpi)            
    plot_she_doctor.savefig(f"{output_dir}/plot_she_doctor", dpi=dpi)    
    plot_she_nurse.savefig(f"{output_dir}/plot_she_nurse", dpi=dpi)    
    
    doctor_nurse.savefig(f"{output_dir}/doctor_nurse", dpi=dpi)    
    nurse_doctor.savefig(f"{output_dir}/nurse_doctor", dpi=dpi)    



# Paired t-tests: 
def significant_tests(samples_1, samples_2):
    _, p1 = stats.ttest_rel(samples_1, samples_2)
    _, p2 = stats.wilcoxon(samples_1, samples_2)
    print(f"paired p-value {p1}")
    print(f"Wilcoxon p-value {p2}")
    return p1, p2

significant_tests(he_doctor_dn, she_doctor_dn)
significant_tests(he_nurse_dn, she_nurse_dn)
significant_tests(he_doctor_nd, she_doctor_nd)
significant_tests(he_nurse_nd, she_nurse_nd)

