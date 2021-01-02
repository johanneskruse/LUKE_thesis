import json
import os

data_dir = "data"

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

# +================================================================================+
# ================================= CoNLL-2003 ================================== 

processor = CoNLLProcessor()

train_conll = processor._read_data(os.path.join(data_dir, "CoNLL2003/eng.train"))
val_conll = processor._read_data(os.path.join(data_dir, "CoNLL2003/eng.testa"))
test_conll = processor._read_data(os.path.join(data_dir, "CoNLL2003/eng.testb"))

train_samples = sum([len(train_conll[i][2]) for i in range(len(train_conll))])
val_samples = sum([len(val_conll[i][2]) for i in range(len(val_conll))])
test_samples = sum([len(test_conll[i][2]) for i in range(len(test_conll))])

print(f"\n\nCoNLL-2003 dataset \nTraining: {train_samples} \nValidation: {val_samples} \nTest: {test_samples}")

print("\n\n")

# !! OBS !!
# 1) item on each line is a word, 2) part-of-speech (POS) tag, 
# 3) syntactic chunk tag and 4) the named entity tag. 
#   U.N.         NNP  I-NP  I-ORG 
#   official     NN   I-NP  O 
# !! OBS !!
# Author says 4 items, we only have


# 1)    train_conll[i][0]: Sentence tokenized (seperated by space) 
# 4)    train_conll[i][1]: The named entity tag
#       train_conll[i][2] = Index for train/test samples


# ================================= CoNLL-2003 ================================== 
# +================================================================================+

# Number written is output - all confirmed to be the right number. 

datasets = ["train.json", "dev.json", "test.json"]

# +================================================================================+
# ================================= OpenEntity ================================== 

# train.json:   1998
# dev.json:     1998
# test.json:    1998

print("Open Entity")
for dataset in datasets:
        
    data_dir_temp = os.path.join(data_dir, "OpenEntity", dataset)
    
    if not os.path.exists(data_dir_temp):
        continue

    with open(data_dir_temp) as json_file:
        data_OE = json.load(json_file)

    print(f"{dataset}: {len(data_OE)}" )

print("\n\n")

# ================================= OpenEntity ================================== 
# +================================================================================+

# +================================================================================+
# ================================= ReCoRD ================================== 

# train.json:     100.730
# dev.json:       10000

print("ReCoRD")
for dataset in datasets:
    
    data_dir_temp = os.path.join(data_dir, "ReCoRD", dataset)
    
    if not os.path.exists(data_dir_temp):
        continue

    with open(data_dir_temp) as json_file:
        data = json.load(json_file)

    i = 0
    for article in data["data"]:
        i += len(article["qas"])

    print(f"{dataset}: {i}" )

print("\n\n")

# ================================= ReCoRD ================================== 
# +================================================================================+

# +================================================================================+
# ================================= SQuAD ================================== 

# train.json:   87599
# dev.json:     10570

print("SQuAD")
for dataset in datasets:

    data_dir_temp = os.path.join(data_dir, "SQuAD", dataset)
    
    if not os.path.exists(data_dir_temp):
        continue

    with open(data_dir_temp) as json_file:
        data = json.load(json_file)

    i = 0
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            i += len(paragraph["qas"])

    print(f"{dataset}: {i}" )

print("\n\n")

# ================================= SQuAD ================================== 
# +================================================================================+

# +================================================================================+
# ================================= TACRED ================================== 

# TACRED
# train.json:   68124
# dev.json:     22631
# test.json:    15509

print("TACRED")
for dataset in datasets:

    data_dir_temp = os.path.join(data_dir, "TACRED", dataset)
    
    if not os.path.exists(data_dir_temp):
        continue

    with open(data_dir_temp) as json_file:
        data = json.load(json_file)

    print(f"{dataset}: {len(data)}")

print("\n\n")

# ================================= TACRED ================================== 
# +================================================================================+
