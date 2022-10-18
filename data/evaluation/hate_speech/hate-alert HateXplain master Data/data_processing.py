import json
from collections import Counter
from tqdm import tqdm



def process_hatexplain(filename="data/evaluation/hate_speech/hate-alert HateXplain master Data/dataset.json",
                       divisions="data/evaluation/hate_speech/hate-alert HateXplain master Data/post_id_divisions.json",
                       output_folder="data/evaluation/hate_speech/hate-alert HateXplain master Data/"):
    with open(filename, 'r') as f:
        data = json.loads(f.read())

    with open(divisions, 'r') as f:
        divisions = json.loads(f.read())

    with open(output_folder + "train.csv", "w") as train_f:
        with open(output_folder + "val.csv", "w") as val_f:
            with open(output_folder + "test.csv", "w") as test_f:

                header = "id,post,label,target\n"
                train_f.write(header)
                val_f.write(header)
                test_f.write(header)

                for d in tqdm(data.values()):
                    post_id = d["post_id"]
                    post = " ".join(d["post_tokens"])
                    label, label_count = Counter([x["label"] for x in d["annotators"]]).most_common(1)[0]
                    target, target_count = Counter([x["target"][0] if x["target"] != [] else None for x in d["annotators"] ]).most_common(1)[0]
                    if label_count == 1:
                        continue
                    if target_count == 1:
                        target = "Other"

                    if label == "normal":
                        label = 0
                    else:
                        label = 1
                    
                    line_to_write = "{},{},{},{}\n".format(post_id, post, label, target)

                    if post_id in divisions["train"]:
                        train_f.write(line_to_write)
                    elif post_id in divisions["val"]:
                        val_f.write(line_to_write)
                    else:
                        test_f.write(line_to_write)



if __name__ == "__main__":
    process_hatexplain()