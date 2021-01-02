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

# +================================================================================+
# ================================= CoNLL-2003 ================================== 

processor = CoNLLProcessor()

data_dir = "/Users/johanneskruse/Desktop/test_scripts/samples/conll/eng.train"

train_conll = processor._read_data(data_dir)

train_samples = sum([len(train_conll[i][2]) for i in range(len(train_conll))])
val_samples = sum([len(val_conll[i][2]) for i in range(len(val_conll))])
test_samples = sum([len(test_conll[i][2]) for i in range(len(test_conll))])

print(f"\n\nCoNLL-2003 dataset \nTraining: {train_samples}")

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

# See an example: 
see = True
if see:
    for z in range(10):
        print(z)
        for i in train_conll[z][2][0:-1]: 
            print(train_conll[z][0][i], train_conll[z][1][i])
        print("")
        



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
        
    data_dir = os.path.join(os.getcwd(), "samples", "openentity", dataset)
    
    if not os.path.exists(data_dir):
        continue

    with open(data_dir) as json_file:
        data_OE = json.load(json_file)

    print(f"{dataset}: {len(data_OE)}" )

    count_place = 0
    count_location = 0  
    count_place_location = 0

    for sample in data_OE: 
        if "location" in sample["labels"] and not "place" in sample["labels"]:
            count_location +=1 
        if "location" not in sample["labels"] and "place" in sample["labels"]:
            count_place +=1
        if "location" in sample["labels"] and "place" in sample["labels"]:
            count_place_location +=1
    if see:
        print("")
        print(f"Location: {count_location}")
        print(f"Place: {count_location}")
        print(f"Location and Place: {count_place_location}")
        print("")

if see:
    print(data_OE[1])

print("\n\n")

# ================================= OpenEntity ================================== 
# +================================================================================+

# +================================================================================+
# ================================= ReCoRD ================================== 

# train.json:     100.730
# dev.json:       10000

print("ReCoRD")
for dataset in datasets:

    data_dir = os.path.join(os.getcwd(), "ReCoRD", dataset)
    
    if not os.path.exists(data_dir):
        continue

    with open(data_dir) as json_file:
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
data_dir = "/Users/johanneskruse/Desktop/test_scripts/samples/squad/dev.json"
    
with open(data_dir) as json_file:
    data = json.load(json_file)

dataset = "dev"

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

    data_dir = os.path.join(os.getcwd(), "TACRED", dataset)
    
    if not os.path.exists(data_dir):
        continue

    with open(data_dir) as json_file:
        data = json.load(json_file)

    print(f"{dataset}: {len(data)}")

print("\n\n")

# ================================= TACRED ================================== 
# +================================================================================+
