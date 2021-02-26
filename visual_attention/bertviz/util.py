import torch
import ipywidgets as widgets

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
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]


def drop_down(options, value=None, description='Select:', disabled=False):
    widget = widgets.Dropdown(    
        options=options,
        value=None,
        description='Select:',
        disabled=False,
    )
    return widget


def print_sentence(sent_a, sent_b=None):
    if sent_b is None:
        return print(f"Sentence: {sent_a}")
    else:
        print(f"Sentence: {sent_a}\nSentence b: {sent_b}")
    return 


def sentence_index(luke_data, sentence_selected):
    index = []
    for i, example in enumerate(luke_data): 
        if sentence_selected == luke_data[example]["sentence"]:
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
