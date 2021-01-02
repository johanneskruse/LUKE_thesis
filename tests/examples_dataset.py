import json


import json
import os


class CoNLLProcessor(object):
    def get_train_examples(self, data_dir):
        return list(self._create_examples(self._read_data(os.path.join(data_dir, "eng.train")), "train"))

    def get_dev_examples(self, data_dir):
        return list(self._create_examples(self._read_data(os.path.join(data_dir, "eng.testa")), "dev"))

    def get_test_examples(self, data_dir):
        return list(self._create_examples(self._read_data(os.path.join(data_dir, "eng.testb")), "test"))

    def get_labels(self):
        return ["NIL", "MISC", "PER", "ORG", "LOC"]

    def _read_data(self, input_file):
        data = []
        words = []
        labels = []
        sentence_boundaries = []
        with open(input_file) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if words:
                        data.append((words, labels, sentence_boundaries))
                        assert sentence_boundaries[0] == 0
                        assert sentence_boundaries[-1] == len(words)
                        words = []
                        labels = []
                        sentence_boundaries = []
                    continue

                if not line:
                    if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:
                    parts = line.split(" ")
                    words.append(parts[0])
                    labels.append(parts[-1])

        if words:
            data.append((words, labels, sentence_boundaries))

        return data

    def _create_examples(self, data, fold):
        return [InputExample(f"{fold}-{i}", *args) for i, args in enumerate(data)]


class InputExample(object):
    def __init__(self, guid, words, labels, sentence_boundaries):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.sentence_boundaries = sentence_boundaries


# TACRED: 
with open("/Users/johanneskruse/Desktop/test_scripts/samples/tacred/dev.json") as j:
    data = json.load(j)

print(f"keys: {data[0].keys()}")

for z in range(10):
    print(z)
    print(f'relation: {data[z]["relation"]}')
    print(f'relation: {data[z]["relation"]}')
    print(f'subject type: {data[z]["subj_type"]}')
    print(f'object type: {data[z]["obj_type"]}')

    print(f'subject: {data[z]["token"][data[z]["subj_start"]:data[z]["subj_end"]+1]}')
    print(f'object: {data[z]["token"][data[z]["obj_start"]:data[z]["obj_end"]+1]}')

    p = ""
    for i in data[z]["token"]: 
        p =  p + " " + i
    print(p)
    print("")


# CoNLL-2003
processor = CoNLLProcessor()
train_conll = processor._read_data("/Users/johanneskruse/Desktop/test_scripts/samples/conll/eng.train")

for z in range(10):
    print(z)
    for i in train_conll[z][2][0:-1]: 
        print(train_conll[z][0][i], train_conll[z][1][i])
    print("")



# Open Entity: 


with open("/Users/johanneskruse/Desktop/test_scripts/samples/openentity/test.json") as json_file:
    data_OE = json.load(json_file)

print("Dev:")

for i in range(1000):
    print(i)
    print(f'{data_OE[i]["sent"]}') 
    print(f'Entity: {data_OE[i]["sent"][data_OE[i]["start"]:data_OE[i]["end"]]}')
    print(f'{data_OE[i]["labels"]}') 
    print("")






