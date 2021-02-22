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

def sentence_index(luke_data, sentence_selected):
    index = []
    for i, example in enumerate(luke_data): 
        if sentence_selected == luke_data[example]["sentence"]:
            index = i
    return index

def print_sentence(sent_a, sent_b=None):
    print(f"Sentence: {sent_a}")
    if sent_b:
        print(f"Sentence b: {sent_b}")