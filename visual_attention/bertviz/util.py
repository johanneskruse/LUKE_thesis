import torch
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30, 'legend.fontsize': 20})
plt.rc('font', size=25)
plt.rc('axes', titlesize=25)


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def format_special_chars(tokens):
    return [t.replace('Ġ', '').replace('▁', ' ').replace('</w>', '') for t in tokens]


def drop_down(options, value=None, description='Select:', disabled=False):
    widget = widgets.Dropdown(    
        options=options,
        value=value,
        description=description,
        disabled=False,
    )
    return widget


def print_sentence(sent_a, sent_b=None):
    if sent_b is None:
        return print(f"Sentence: {sent_a}")
    else:
        print(f"Sentence: {sent_a}\nSentence b: {sent_b}")
    return 


def sentence_index(luke_data, sentence_selected, entity):
    index = []
    for i, example in enumerate(luke_data): 
        if sentence_selected == luke_data[example]["sentence"] and luke_data[example]["entity"] == entity:
            index = i
    return index


def get_entity_string(data):
    
    entity_vector = [data[sent]["entity_position_ids"][0][0] for sent in data]
    entity_index = [vector[vector > 0] for vector in entity_vector]

    tokens = [format_special_chars(data[sent]["tokens"]) for sent in data]
    
    sentences = [data[sent]["sentence"] for sent in data.keys()]

    for i, sent in enumerate(data):
        data[sent]["entity"] = " ".join(tokens[i][entity_index[i][1]:entity_index[i][-1]])
        data[sent]["sentence_with_entity"] = sentences[i] + f' [entity:{data[sent]["entity"]}]'
    
    return data 


def only_mask_attention(output_attention):
    zero_output_attention = output_attention
    for i, attention in enumerate(zero_output_attention):
        for ii, layer in enumerate(attention): 
            zero_output_attention[i][ii][:-2]   = zero_output_attention[i][ii][:-2]*0
            zero_output_attention[i][ii][-1]    = zero_output_attention[i][ii][-1]*0
    return zero_output_attention


def attention_token2token(tokens, attention, token1, token2):

    for i, token in enumerate(tokens): 
        if token == token1:
            index_token1 = i
        if token == token2:
            index_token2 = i

    attn_token2token = []
    
    for layer in attention:     
        attention_temp = []
        for head in layer:
            attention_temp.append(head[index_token1, index_token2])
        attn_token2token.append(attention_temp)

    return attn_token2token


def plot_attention_token2token(tokens, attention, token1, token2, color="blue"):

    attention_scores = attention_token2token(tokens, attention, token1, token2)
    attn_token2token_mean = [np.mean(attn_scores) for attn_scores in attention_scores]
    
    number_of_layers = len(attention_scores)
    number_of_tokens = len(attention_scores[0])

    # Begin Figure: 
    figure, ax = plt.subplots(figsize=(12,9))


    for i in range(number_of_layers): 
        ax.plot([i]*number_of_tokens, attention_scores[i], ".", color=color)    
    avg_attention, = ax.plot(range(len(attention_scores)), attn_token2token_mean, color=color)
    avg_attention.set_label(f"[{token1}] $\longrightarrow$ [{token2}]")
    ax.legend()

    ax.set_title("Token-to-Token attention", size="x-large") 

    ax.set_xlabel("Layer", fontsize="large")
    ax.set_ylabel("Attention score", fontsize="large")

    plt.tick_params(axis='x', labelsize="large")
    plt.tick_params(axis='y', labelsize="large")
    plt.grid()
    plt.tight_layout()

    return figure

