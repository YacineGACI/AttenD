import argparse, math
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



def generalized_mean_of_bias(l, p=-5):
    score = 0
    for m in l:
        if m != 0.0:
            score += math.pow(m, p)
        else:
            score += math.pow(0.0000000001, p)
    score = score / len(l)
    score = math.pow(score, 1/p)
    return score




def performance(data):
    labels = []
    predictions = []
    predicted_classes = []
    for id in data.keys():
        predictions.append(data[id]["prediction"])
        predicted_classes.append(0 if data[id]["prediction"] <= 0.5 else 1)
        labels.append(float(data[id]["label"]))

    accuracy = accuracy_score(labels, predicted_classes)
    f1 = f1_score(labels, predicted_classes)
    auc = roc_auc_score(labels, predictions)

    return accuracy, f1, auc






def inference():
    print("Inference  ======>")
    for id in tqdm(data.keys()):
        # Tokenize
        input = tokenizer(data[id]["post"], truncation=True, padding="max_length", max_length=args.max_len)
        input = {k:torch.tensor(v).unsqueeze(0).to(device) for k, v in input.items()}
        for k in input.keys():
            data[id][k] = input[k]
        
        logits = model(input_ids=data[id]["input_ids"], attention_mask=data[id]["attention_mask"], token_type_ids=data[id]["token_type_ids"]).logits
        prediction = softmax(logits)[0][1].item()
        data[id]["prediction"] = prediction





def separate_pos_neg_ids(method, group):
    positive_ids = []
    negative_ids = []

    for id in data.keys():
        try:
            label = data[id]["label"]
            target = data[id]["target"]

            if method == "subgroup":
                    if target in ['None', 'Other']:
                        continue
                    if target != group:
                        continue
                    if label == "0":
                        negative_ids.append(id)
                    else:
                        positive_ids.append(id)

            elif method == "bpsn":
                if target in ['None', 'Other']:
                    continue
                if target != group:
                    if label == "1":
                        positive_ids.append(id)
                    else:
                        continue
                else:
                    if label == "0":
                        negative_ids.append(id)
                    else:
                        continue


            elif method == "bnsp":
                if target in ['None', 'Other']:
                    continue
                if target != group:
                    if label == "0":
                        negative_ids.append(id)
                    else:
                        continue
                else:
                    if label == "1":
                        positive_ids.append(id)
                    else:
                        continue
            
            else:
                raise ValueError

        except:
            pass
    

    return {"pos": positive_ids, "neg": negative_ids}

            
        




def evaluate_bias_subgroup(method, group):

    # Find pos and neg ids for this configuration
    pos_neg_ids = separate_pos_neg_ids(method, group)

    predictions = []
    ground_truth = []
    
    for sign in ["neg", "pos"]:
        for id in pos_neg_ids[sign]:
            predictions.append(data[id]["prediction"])
            ground_truth.append(float(data[id]["label"]))

    return roc_auc_score(ground_truth, predictions)



def evaluate_bias(methods, groups):
    for m in methods:
        print(m)
        mean = []
        for g in groups:
            try:
                bias_score = evaluate_bias_subgroup(m, g)
                mean.append(bias_score)
            except:
                bias_score = None
            print("==> {}: {}".format(g, bias_score))
        print("###")
        print("Mean: {}".format((1/len(mean)) * sum(mean)))
        print("GMB: {}".format(generalized_mean_of_bias(mean)))
        print("STD: {}".format(np.std(mean)))
        print("###")
        print()
            


        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--data_file", type=str, default="data/evaluation/hate_speech/hate-alert HateXplain master Data/test.csv", help="Path to test file")

    args = parser.parse_args()
    print(vars(args))

    with open(args.data_file, 'r') as f:
        data = {}
        groups = {}

        for line in f.readlines()[1:]:
            try:
                id, post, label, target = line.strip("\n ").split(',')
                data[id] = {
                    "id": id,
                    "post": post,
                    "label": label,
                    "target": target
                }
                
                if target not in groups.keys():
                    groups[target] = {"0": 0, "1": 0}

                groups[target][label] += 1

            except:
                continue

    groups = [x for x in sorted(groups.items(), key= lambda item: item[1]["0"] + item[1]["1"], reverse=True) if x[0] not in ["None", "Other"]]
    print(groups)
    print()

    methods = ["subgroup", "bpsn", "bnsp"]
    groups = [x[0] for x in groups]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    model.to(device)
    softmax = torch.nn.Softmax(dim=-1)

    inference()
    evaluate_bias(methods, groups)

    acc, f1, auc = performance(data)
    print("==============")
    print("Accuracy: {}".format(acc))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))