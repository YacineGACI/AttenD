import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset

from tqdm import tqdm
import pickle, argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], " ".join(group_words), truncation=True)
    return tokenized_output


def find_separator_position(input, tokenizer):
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    return (input == sep_id).nonzero(as_tuple=True)[-1][0].item()






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--layers", type=int, default=12, help="Layers used in computing attention")
    parser.add_argument("--save_every", type=int, default=2000, help="Save intermediate dictionary every number of steps")
    parser.add_argument("--corpus", type=str, default="data/train/de-en.txt/News-Commentary.de-en.en", help="Path to text corpus")
    parser.add_argument("--tmp", type=str, default="tmp.pkl", help="name of tmp file to store intermediary dictionary")
    parser.add_argument("--female_list", type=str, default="data/evaluation/qualitative/female_word_list.txt", help="Filepath to female-related words")
    parser.add_argument("--male_list", type=str, default="data/evaluation/qualitative/male_word_list.txt", help="Filepath to male-related words")
    parser.add_argument("--stereotype_list", type=str, default="data/evaluation/qualitative/stereotype_word_list.tsv", help="Filepath to stereotyped words")
    parser.add_argument("--output", type=str, required=True, help="Name of the resulting figure. Will be stored in 'evaluation/qualitative/figures'")

    args = parser.parse_args()
    print(vars(args))

    groups = {
        "gender": ["man", "woman"]
    }

    layers = list(range(args.layers))

    configuration = AutoConfig.from_pretrained(args.model, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, return_tensors="pt")
    model = AutoModel.from_pretrained(args.model, config=configuration)

    for param in model.parameters():
        param.requires_grad = False

    # Load the dataset
    dataset = load_dataset('text', data_files=args.corpus)


    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    model.to(device)


    for bias, group_words in groups.items():

        tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"])
        tokenized_dataset.set_format(type='torch')

        dataloader = torch.utils.data.DataLoader(
                    tokenized_dataset["train"],
                    batch_size=1
                )
        
        attention_scores_per_word = {}

        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            attentions = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"]).attentions
            
            # Get the index of the [SEP] token in the input ids
            second_sentence_index = find_separator_position(batch["input_ids"], tokenizer)
            
            # Get the indices of group words on the input ids
            word_positions = {}
            for count, w in enumerate(group_words):
                word_positions[w] = second_sentence_index + count + 1


            for layer_id, attn_layer in enumerate(attentions):
                if layer_id in layers:
                    for word_pos in range(second_sentence_index):
                        current_word_id = batch["input_ids"][0][word_pos].item()

                        if current_word_id in attention_scores_per_word.keys():
                            for group, group_pos in word_positions.items():
                                new_score = attn_layer[0, :, word_pos, group_pos].mean().item()
                                attention_scores_per_word[current_word_id][group]["count"] += 1

                                n = attention_scores_per_word[current_word_id][group]["count"]
                                m = attention_scores_per_word[current_word_id][group]["score"]

                                attention_scores_per_word[current_word_id][group]["score"] = ((n - 1) / n) * m + (new_score / n)
                        else:
                            attention_scores_per_word[current_word_id] = {}
                            for group, group_pos in word_positions.items():
                                attention_scores_per_word[current_word_id][group] = {
                                    "count": 1,
                                    "score": attn_layer[0, :, word_pos, group_pos].mean().item()
                                }


            if (i + 1) % args.save_every == 0:
                with open("evaluation/qualitative/intermediate_dict/{}".format(args.tmp), "wb") as f:
                    pickle.dump(attention_scores_per_word, f)


    with open("evaluation/qualitative/intermediate_dict/{}".format(args.tmp), "wb") as f:
        pickle.dump(attention_scores_per_word, f)









    # Making the figures

    # Open the attention scores
    with open("evaluation/qualitative/intermediate_dict/{}".format(args.tmp), "rb") as f:
        attentions = pickle.load(f)

    # Compute the differences in the attention scores
    for word in attentions.keys():
        attentions[word] = attentions[word]["he"]["score"] - attentions[word]["she"]["score"]



    with open(args.female_list, 'r') as f:
        female_words = [x.strip('\n') for x in f.readlines()]

    with open(args.male_list, 'r') as f:
        male_words = [x.strip('\n') for x in f.readlines()]

    with open(args.stereotype_list, 'r') as f:
        stereotype_words = []
        for line in f.readlines():
            stereotype_words += line.strip("\n").split('\t')


    # stereotype_words = ["violence"]


    female_scores = []
    male_scores = []
    stereotype_scores = []


    for word in female_words:
        word_id = tokenizer.convert_tokens_to_ids(word)
        if word_id in attentions.keys():
            female_scores.append(attentions[word_id])
        else:
            female_scores.append(None)

    for word in male_words:
        word_id = tokenizer.convert_tokens_to_ids(word)
        if word_id in attentions.keys():
            male_scores.append(attentions[word_id])
        else:
            male_scores.append(None)

    for word in stereotype_words:
        word_id = tokenizer.convert_tokens_to_ids(word)
        if word_id in attentions.keys():
            stereotype_scores.append(attentions[word_id])
        else:
            stereotype_scores.append(None)









    dataframe = [[male_words[i], male_scores[i], i,  "male-oriented"] for i in range(len(male_scores))]
    dataframe += [[female_words[i], female_scores[i], len(male_scores) + i,  "female-oriented"] for i in range(len(female_scores))]
    dataframe += [[stereotype_words[i], stereotype_scores[i], len(male_scores) + len(female_scores) + i,  "stereotype"] for i in range(len(stereotype_scores))]

    dataframe = pd.DataFrame(dataframe, columns=['word', 'score', 'rank', 'category'])


    sns.scatterplot(data=dataframe, x='score', y='rank', hue='category')

    ax = plt.gca()
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(False)

    plt.grid(False)
    plt.xlim([-0.013, 0.013])

    plt.savefig("evaluation/qualitative/figures/{}".format(args.output))
    plt.show()