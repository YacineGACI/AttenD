from tqdm import tqdm
import random, json, argparse
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau




def tokenize_function(examples):
    random_index = random.randint(0, len(groups[args.bias_type]) - 1)
    words = groups[args.bias_type][random_index]
    tokenized_output = tokenizer(examples["text"], " ".join(words), truncation=True)
    return tokenized_output



def find_separator_position(input, tokenizer):
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    return (input == sep_id).nonzero(as_tuple=True)[-1][0].item()



def compute_correlation(input):
    mean_corr = 0
    count = 0
    for i in range(len(input)):
        for j in range(i+1, len(input)):
            mean_corr += correlation_function(input[i], input[j])[0]
            count += 1
    return mean_corr / count


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train/de-en.txt/News-Commentary.de-en.en", help="Path to data")
    parser.add_argument("--def_word_filepath", type=str, default="data/train/definition_words.json", help="filepath to the JSON file containing definition tuples per bias type")
    parser.add_argument("--tokenizer", type=str, help="Full name or path or URL to tokenizer", required=True)
    parser.add_argument("--model", type=str, help="Full name or path or URL to the model to debias", required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size. Should be left to 1")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--seed", type=int, default=41, help="Seed for reproducibility")
    parser.add_argument("--bias_type", required=True, help="Which bias type to include?")
    parser.add_argument("--agg_method", default="all", choices=["all", "separate"])
    parser.add_argument("--correlation", default="pearson", choices=["pearson", "spearman", "kendall"], help="Name of correlation fucntion to choose")
    parser.add_argument("--eval_percentage", type=float, default=0.2, help="Percentage of data to use in evaluation")

    args = parser.parse_args()

    print(vars(args))

    if args.correlation == "pearson":
        correlation_function = pearsonr
    elif args.correlation == "spearman":
        correlation_function = spearmanr
    elif args.correlation == "kendall":
        correlation_function = kendalltau
    else:
        raise ValueError


    with open(args.def_word_filepath, 'r') as f:
        groups = json.load(f)["biases"]

    torch.manual_seed(args.seed)

    configuration = AutoConfig.from_pretrained(args.model, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, return_tensors="pt")
    model = AutoModel.from_pretrained(args.model, config=configuration)


    # Load the dataset
    dataset = load_dataset('text', data_files=args.data, split={
        'train': 'train[:{}%]'.format(100 - round(args.eval_percentage * 100)),
        'eval': 'train[-{}%:]'.format(round(100 * args.eval_percentage))
    })

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"])
    tokenized_dataset.set_format(type='torch')

    dataloader = torch.utils.data.DataLoader(
                tokenized_dataset["eval"],
                batch_size=args.batch_size
            )


    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    model.eval().to(device)


    # For logging purposes
    count = 0
    attention_bias_separate = {}

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        attentions = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"]).attentions

        for layer_index, layer in enumerate(attentions):
            if layer_index not in attention_bias_separate.keys():
                attention_bias_separate[layer_index] = {}

            split_index = find_separator_position(batch["input_ids"], tokenizer)

            for head_index in range(layer.size(1)):
                attention_head = layer[0, head_index, :split_index+1, split_index+1:-1]
                if head_index not in attention_bias_separate[layer_index].keys():
                    attention_bias_separate[layer_index][head_index] = [] # List of correlations

                num_groups = attention_head.size(-1)
                local_attentions = [attention_head[:, i].tolist() for i in range(num_groups)]

                local_correlation = compute_correlation(local_attentions)
                attention_bias_separate[layer_index][head_index].append(local_correlation)

    
    # Compute the overall correlation for each head
    for layer_index in attention_bias_separate.keys():
        for head_index in attention_bias_separate[layer_index].keys():
            attention_bias_separate[layer_index][head_index] = sum(attention_bias_separate[layer_index][head_index]) / len(attention_bias_separate[layer_index][head_index])

    print(attention_bias_separate)
    