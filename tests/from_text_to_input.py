import json

# ```
# Helper script: python tests/from_text_to_input.py 
# Easily convert sentence to input for Entity Typing task. 
# input: from_text_to_input.json: 
# from_text_to_input.json format: {"sentX" : "this is a test for <entity>"} --> put < > arount entity
# ```

# data_dir = "tests/from_text_to_input_gender_bias_names.json"
# output_dir = "data/outputs/check_point_file/gender_bias"

data_dir = "tests/from_text_to_input.json"
output_dir = "data/outputs/check_point_file/OpenEntity_manipulate_input"

with open(data_dir) as json_file:
    data = json.load(json_file)

sentences = [data[key] for key in data]

format_data = []
for sentence in sentences:
    for index, letter in enumerate(sentence): 
        if letter == "<":
            start = index
        if letter == ">":
            end = index-1

    label = ["person"]
    clean_sentence = sentence.replace("<", "").replace(">", "")
    entity = clean_sentence[start:end]

    if entity[0] == " ":
        entity = entity[1:]
    if entity[-1] == " ":
        entity = entity[:-1]
    format_data.append({"sent" : clean_sentence, "start": start, "labels": label, "end": end, "entity" : entity})


with open(f'{output_dir}/test.json', 'w') as outfile:
    json.dump(format_data, outfile)