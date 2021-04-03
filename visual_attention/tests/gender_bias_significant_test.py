import sys
sys.path.append("..")
from bertviz.util import attention_token2token, plot_attention_token2token, format_special_chars
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30, 'legend.fontsize': 20})
plt.rc('font', size=25)
plt.rc('axes', titlesize=25)

def read_file_return_names(data_dir):
    f = open( f"{data_dir}", "r")
    file_ = f.read()
    f.close()

    file_ = file_.replace(" ", "")
    file_ = file_.split(",")

    return file_


def attention_scores_token_or_entity_heads_and_mean(data, name_list, token1=None, token2=None, token1_index=-5, token2_index=-5):

    token2token_attn = {}
    token2token_attn["layer_heads"] = {}
    token2token_attn["mean_layer"] = {}

    for i, sample in enumerate(data): 
        tokens = format_special_chars(sample["tokens"])

        if tokens[token1_index] in name_list or tokens[token1_index] in name_list:
            attention = sample["attention"]            

            if token1 is None:
                token1_ = tokens[token1_index]
            else:    
                token1_ = token1
            if token2 is None:
                token2_ = tokens[token2_index]
            else:
                token2_ = token2

            token2token_attention_temp = attention_token2token(tokens, attention, token1_, token2_)
            
            token2token_attn[f"layer_heads"][f"{i}_{token1_}_{token2_}"] = token2token_attention_temp
            layer_mean = [np.mean(layer) for layer in token2token_attention_temp]
            token2token_attn[f"mean_layer"][f"{i}_{token1_}_{token2_}"] = layer_mean
    
    return token2token_attn[f"layer_heads"], token2token_attn[f"mean_layer"]


def unparied_significant_test(samples_mean_in_layers_girls, samples_mean_in_layers_boys):
    means_girl = [np.mean(sample) for sample in samples_mean_in_layers_girls.values()]
    means_boy = [np.mean(sample) for sample in samples_mean_in_layers_boys.values()]

    mean_girl = np.mean(means_girl)
    mean_boy = np.mean(means_boy)

    var_girl = np.var(means_girl)
    var_boy = np.var(means_boy)

    _, p_val = stats.ttest_ind(means_girl, means_boy)
    return p_val, mean_girl, mean_boy, var_girl, var_boy


def plot_histrogram(samples_mean_in_layers_girls, samples_mean_in_layers_boys, title="Density Histrogram", bins = 15, loc="best", alpha=0.5, output_dir=None):
    means_boy = [np.mean(sample) for sample in samples_mean_in_layers_boys.values()]
    means_girl = [np.mean(sample) for sample in samples_mean_in_layers_girls.values()]

    figure, ax = plt.subplots(figsize=(12,9))

    ax.hist(means_boy, bins=bins, alpha=alpha, label=f'Male: ($\mu$={np.mean(means_boy):.3f}, $\sigma^2$={np.var(means_boy)*10**5:.2f}e-5)', density=True)
    ax.hist(means_girl, bins=bins, alpha=alpha, label=f'Female ($\mu$={np.mean(means_girl):.3f}, $\sigma^2$={np.var(means_girl)*10**5:.2f}e-5)', density=True)
    ax.set_xlabel("Accumulated attention scores")
    ax.set_ylabel("A.U.")
    
    ax.legend(loc=loc)
    ax.set_title(title, size="x-large") 

    if output_dir: 
        dpi = 300
        figure.savefig(output_dir, dpi=dpi)

    return figure


# =============================================================== #

data_dir = "/Users/johanneskruse/Desktop/gender_bias_output_attention"
output_dir = "plot_attention_visualization"
save = True

data = pickle.load(open( f"{data_dir}/output_attentions_test.p", "rb"))

bois_names = read_file_return_names(f"{data_dir}/boys.txt")
girl_names = read_file_return_names(f"{data_dir}/girls.txt")

# =============================================================== #

doctor_nurse_girl_names   = [data[sent] for sent in list(data)[0:100]]
doctor_nurse_boy_names    = [data[sent] for sent in list(data)[100:200]]
nurse_doctor_girl_names   = [data[sent] for sent in list(data)[200:300]]
nurse_doctor_boy_names    = [data[sent] for sent in list(data)[300:400]]


data = doctor_nurse_girl_names
name_list= girl_names
token1=None 
token2="doctor"
entity_index = -5

# =============================================================== #
# ############### (doctor, nurse) ###############
# (doctor, nurse): name -> doctor
_, dn_girl_doctor_attn = attention_scores_token_or_entity_heads_and_mean(data=doctor_nurse_girl_names, name_list=girl_names, token1=None, token2="doctor", token1_index=-5)
_, dn_boy_doctor_attn  = attention_scores_token_or_entity_heads_and_mean(data=doctor_nurse_boy_names, name_list=bois_names, token1=None, token2="doctor", token1_index=-5)
dn_gender_doctor_pval, dn_gender_doctor_mean_girl, dn_gender_doctor_mean_boy, dn_gender_doctor_var_girl, dn_gender_doctor_var_boy  = unparied_significant_test(dn_girl_doctor_attn, dn_boy_doctor_attn)
print(f"(doctor, nurse): name -> doctor\np-value: {dn_gender_doctor_pval}")
print(f"mean [girl, boy]: [{dn_gender_doctor_mean_girl}, {dn_gender_doctor_mean_boy}]")
print(f"var [girl, boy]: [{dn_gender_doctor_var_girl}, {dn_gender_doctor_var_boy}]")
dn_gender_doctor_hist = plot_histrogram(dn_girl_doctor_attn, dn_boy_doctor_attn, title = f"(doctor, nurse), [group]$\longrightarrow$[doctor]\np-value: {dn_gender_doctor_pval:.02}", output_dir="/Users/johanneskruse/Desktop/dn_gender_doctor_hist.png")

# (doctor, nurse): name -> nurse
_, dn_girl_nurse_attn = attention_scores_token_or_entity_heads_and_mean(data=doctor_nurse_girl_names, name_list=girl_names, token1=None, token2="nurse", token1_index=-5)
_, dn_boy_nurse_attn = attention_scores_token_or_entity_heads_and_mean(data=doctor_nurse_boy_names, name_list=bois_names, token1=None, token2="nurse", token1_index=-5)
dn_gender_nurse_pval, dn_gender_nurse_mean_girl, dn_gender_nurse_mean_boy, dn_gender_nurse_var_girl, dn_gender_nurse_var_boy = unparied_significant_test(dn_girl_nurse_attn, dn_boy_nurse_attn)
print(f"(doctor, nurse): name -> nurse\np-value: {dn_gender_nurse_pval}")
print(f"mean [girl, boy]: [{dn_gender_nurse_mean_girl}, {dn_gender_nurse_mean_boy}]")
print(f"var [girl, boy]: [{dn_gender_nurse_var_girl}, {dn_gender_nurse_var_boy}]")
dn_gender_nurse_hist = plot_histrogram(dn_girl_nurse_attn, dn_boy_nurse_attn, title = f"(doctor, nurse), [group]$\longrightarrow$[nurse]\np-value: {dn_gender_nurse_pval:.04}", output_dir="/Users/johanneskruse/Desktop/dn_gender_nurse_hist.png")

# =============================================================== #
# ############### (nurse, doctor) ###############
# (nurse, doctor): name -> doctor
_, nd_girl_doctor_attn = attention_scores_token_or_entity_heads_and_mean(data=nurse_doctor_girl_names, name_list=girl_names, token1=None, token2="doctor", token1_index=-5)
_, nd_boy_doctor_attn  = attention_scores_token_or_entity_heads_and_mean(data=nurse_doctor_boy_names, name_list=bois_names, token1=None, token2="doctor", token1_index=-5)
nd_gender_doctor_pval, nd_gender_doctor_mean_girl, nd_gender_doctor_mean_boy, nd_gender_doctor_var_girl, nd_gender_doctor_var_boy = unparied_significant_test(nd_girl_doctor_attn, nd_boy_doctor_attn)
print(f"(nurse, doctor): name -> doctor\np-value: {nd_gender_doctor_pval}")
print(f"mean [girl, boy]: [{nd_gender_doctor_mean_girl}, {nd_gender_doctor_mean_boy}]")
print(f"var [girl, boy]: [{nd_gender_doctor_var_girl}, {nd_gender_doctor_var_boy}]")
nd_gender_doctor_hist = plot_histrogram(nd_girl_doctor_attn, nd_boy_doctor_attn, title = f"(nurse, doctor), [group]$\longrightarrow$[doctor]\np-value: {nd_gender_doctor_pval:.01}", output_dir="/Users/johanneskruse/Desktop/nd_gender_doctor_hist.png")


# (nurse, doctor): name -> nurse
_, nd_girl_nurse_attn = attention_scores_token_or_entity_heads_and_mean(data=nurse_doctor_girl_names, name_list=girl_names, token1=None, token2="nurse", token1_index=-5)
_, nd_boy_nurse_attn = attention_scores_token_or_entity_heads_and_mean(data=nurse_doctor_boy_names, name_list=bois_names, token1=None, token2="nurse", token1_index=-5)
nd_gender_nurse_pval, nd_gender_nurse_mean_girl, nd_gender_nurse_mean_boy, nd_gender_nurse_var_girl, nd_gender_nurse_var_boy = unparied_significant_test(nd_girl_nurse_attn, nd_boy_nurse_attn)
print(f"(nurse, doctor): name -> nurse\np-value: {nd_gender_nurse_pval}")
print(f"mean [girl, boy]: [{nd_gender_nurse_mean_girl}, {nd_gender_nurse_mean_boy}]")
print(f"var [girl, boy]: [{nd_gender_nurse_var_girl}, {nd_gender_nurse_var_boy}]")
nd_gender_nurse_hist = plot_histrogram(nd_girl_nurse_attn, nd_boy_nurse_attn, title = f"(nurse, doctor), [group]$\longrightarrow$[nurse]\np-value: {nd_gender_nurse_pval:.01}", output_dir="/Users/johanneskruse/Desktop/nd_gender_nurse_hist.png")



dn_gender_doctor_hist
dn_gender_nurse_hist
nd_gender_doctor_hist
nd_gender_nurse_hist

for i, sample in enumerate(nd_boy_nurse_attn.values()): 
    if i == 0:
        means = sample
    else: 
        means = np.array(means) + np.array(sample)
    means/i
