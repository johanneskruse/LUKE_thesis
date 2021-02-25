import json 
import pandas as pd
import numpy as np

###
# Helper script useful for inspecting the output. 
# input (data_dir): test.json, test_prediction.jsonl, results.json
# outputs: {"sent", "entity", "pred", "label", "top_3_probabilities"}
###

set_ = "test"
data_file_dir = f"../data/check_point_file/{set_}.json"
predict_file_dir = f"../data/check_point_file/{set_}_predictions.jsonl"
result_file = "../data/check_point_file/results.json"


def entity_present(dict_, entity_name):
    entity = 0 
    label = []
    predict = []
    for i in dict_: 
        if dict_[i]["entity"].lower() == entity_name:
            entity += 1
            label.append(dict_[i]["label"])
            predict.append(dict_[i]["pred"])
    return entity, label, predict

def top_labels_prob(logit_list, lables):
    list_top = []
    
    for i, logits in enumerate(logit_list):
        prob = logit2prob(logits)
        top = np.argsort(prob)[-3:]
        temp_top_list = []
        for index in top: 
            temp_top_list.append([labels[index], prob[index]])
    
        list_top.append(temp_top_list)
    
    return list_top


def format_data(df, top_label_prob_list):
    assert len(df) == len(top_label_prob_list)
    
    dict_format = {}
    for i in range(len(df.index)):
        dict_format[i] = {
            "sent" : df.iloc[i]['sent'],
            "entity" : df.iloc[i]['sent'][df.iloc[i]['start']:df.iloc[i]['end']],
            "pred" : df.iloc[i]['predictions'],
            "label" : df.iloc[i]['labels_'],
            "most_likely" : top_label_prob_list[i]
        }
    return dict_format


def logit2prob(logits):
    prob = np.array([])
    for logit in logits: 
        prob = np.append(prob, (np.exp(logit) / (np.exp(logit) + 1)))
    return prob


def count_entity_types(list_entity_labels):
    labels = {}
    labels["reject"] = 0
    for label in list_entity_labels:     
        if label == []:
            labels["reject"] += 1
        
        for lab in label: 
            if lab not in labels:
                labels[lab] = 1
            else:
                labels[lab] += 1
    return labels


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# data_file_dir = f"../data/check_point_file/z_switch_data_sets/{set_}/{set_}.json"
# predict_file_dir = f"../data/check_point_file/z_switch_data_sets/{set_}/{set_}_predictions.jsonl"
# result_file = f"../data/check_point_file/z_switch_data_sets/{set_}/results.json"



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
df_results      = pd.read_json(result_file)

df_data_file    = pd.read_json(data_file_dir)
df_predict_file = pd.read_json(path_or_buf=predict_file_dir, lines=True)
df_predict_file = df_predict_file.rename(columns={"labels": "labels_"})

df = pd.concat([df_data_file, df_predict_file], axis=1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

labels      = df_results["evaluation_predict_label"]["label_list"]
logit_list  = df_results["evaluation_predict_label"][f"{set_}"]["predict_logits"]
top_label_prob_list = top_labels_prob(logit_list, labels)

dict_format_all = format_data(df, top_label_prob_list)

list_err_prob = [top_label_prob_list[i] for i in list(df.loc[df['predictions'] != df['labels_']].index)]
dict_format_err = format_data(df.loc[df['predictions'] != df['labels_']], list_err_prob)


count, label_entity, predit = entity_present(dict_format_all, "it")
count_err, label_entity_err, predict_err = entity_present(dict_format_err, "it")

count_entity_types(label_entity)
count_entity_types(label_entity_err)

sum(count_entity_types(label_entity_err).values())
sum(count_entity_types(label_entity).values())


for i in dict_format_all:
    print(f"{i}: {dict_format_all[i]}")
    print("\n")


save = False
if save == True:
    with open('error_samples.json', 'w') as outfile:
        json.dump(dict_format_err, outfile)



