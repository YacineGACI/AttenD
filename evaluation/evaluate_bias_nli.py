from argparse import Namespace
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import math, argparse
import sys




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--eval_filepath", type=str, required=True, help="Filepath to evaluation templates")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum length allowed for input sentences. If longer, truncate.")

    args = parser.parse_args()
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the trained NLI model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model.to(device)

    softmax = torch.nn.Softmax(dim=1)


    net_neutral = 0
    fraction_neutral = 0
    threshold_50 = 0
    threshold_70 = 0

    # Load the data
    with open(args.eval_filepath, 'r') as f:
        lines = f.readlines()[1:]
        num_iterations = math.ceil(len(lines) / args.batch_size)

        for batch in tqdm(range(num_iterations)):
            premise = []
            hypothesis = []
            for line in lines[batch * args.batch_size: (batch + 1) * args.batch_size]:
                _, p, h = line.rsplit(",", 2)
                premise.append(p.strip())
                hypothesis.append(h.strip())

            inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding="max_length", max_length=args.max_len)
            inputs = {k:v.to(device) for k, v in inputs.items()}
            output = model(**inputs)
            logits = softmax(output.logits)
            maxes = torch.argmax(logits, dim=1)

            for i, line in enumerate(logits):
                net_neutral += line[1].item()
                fraction_neutral += 1 if maxes[i] == 1 else 0
                threshold_50 += 1 if line[1] > 0.5 else 0
                threshold_70 += 1 if line[1] > 0.7 else 0

    print("Net Neutral: {}".format(net_neutral / len(lines)))
    print("Fraction Neutral: {}".format(fraction_neutral / len(lines)))
    print("Threshold 0.5: {}".format(threshold_50 / len(lines)))
    print("Threshold 0.7: {}".format(threshold_70 / len(lines)))
