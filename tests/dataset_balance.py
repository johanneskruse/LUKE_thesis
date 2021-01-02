import json

with open("/Users/johanneskruse/Desktop/test_scripts/input/train.json") as g:
    train = json.load(g)
with open("/Users/johanneskruse/Desktop/test_scripts/input/dev.json") as g:
    dev = json.load(g)
with open("/Users/johanneskruse/Desktop/test_scripts/input/test.json") as g:
    test = json.load(g)


datasets = {"train": train, "dev": dev, "test": test}

# ================================
# Multi labels: 
for dataset in datasets: 
    single_label = 0 
    multiple_labels = 0 
    reject = 0 
    for i in datasets[dataset]:
        if len(i["labels"]) > 1: 
            multiple_labels += 1
        elif len(i["labels"]) == 1:
            single_label += 1
        else:
            #print(i["labels"])
            reject += 1

    print(f"{dataset}\n"
        f"Samples with 1 entity for {dataset}: {single_label}\n"
        f"Samples with > 1 entities for {dataset}: {multiple_labels}\n"
        f"Rejects {dataset}: {reject}\n"
        f"Sum: {multiple_labels+single_label+reject}")


# ================================
# Categories: 

for dataset in datasets: 
    entity, event, group, location, objects, organization, person, place, time, reject = [0]*10
    
    for i in datasets[dataset]:
        if "entity" in i["labels"]:
            entity += 1
        if "event" in i["labels"]:
            event += 1
        if "group" in i["labels"]: 
            group += 1
        if "location" in i["labels"]:
            location += 1  
        if "object" in i["labels"]:
            objects += 1
        if "organization" in i["labels"]:
            organization += 1
        if "person" in i["labels"]:
            person += 1
        if "place" in i["labels"]:
            place += 1
        if "time" in i["labels"]:
            time += 1   
        if not i["labels"]: 
            reject += 1

    print(f"Categories balance check for {dataset}:\n"
            f"entity: {entity}\n"
            f"event: {event}\n"
            f"group: {group}\n"
            f"location: {location}\n"
            f"objects: {objects}\n"
            f"organization: {organization}\n"
            f"person: {person}\n"
            f"place: {place}\n"
            f"time: {time}\n"
            f"reject: {reject}\n"
            )


