from yaml import parse
from tqdm import tqdm
import random, pathlib, json, argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset


class AttentionLoss(nn.Module):
    def __init__(self, equ_lambda, heads):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.equ_lambda = equ_lambda
        self.heads = heads
        self.distillation_criterion = AttentionDistillationLoss()
        self.equalization_criterion = AttentionEqualizationLoss()


    def forward(self, attentions, teacher_attention, split_index, labels):
        if labels == 1:
            distillation_attentions = []
            equalization_attentions = []
            teacher_attentions = [] # Used to replicate the attention heads
            for layer_index, attn_layer in enumerate(attentions):
                for head_index in range(attn_layer.size(1)):
                    if (layer_index, head_index) in self.heads:
                        distillation_attentions.append(attn_layer[0, head_index:head_index+1, :split_index+1, :split_index+1])
                        equalization_attentions.append(attn_layer[0, head_index:head_index+1, :split_index+1, split_index+1:-1])
                        teacher_attentions.append(teacher_attention[layer_index][0, head_index:head_index+1, :split_index+1, :split_index+1])


            distillation_attentions = torch.stack(distillation_attentions)
            equalization_attentions = torch.stack(equalization_attentions)
            teacher_attentions = torch.stack(teacher_attentions)

            distillation_loss = self.distillation_criterion(distillation_attentions, teacher_attentions)
            equalization_loss = self.equalization_criterion(equalization_attentions)

            return distillation_loss + self.equ_lambda * equalization_loss

        else:
            # If labels=0: negative example.
            # In this case, replicate the attentions of the teacher as is
            distillation_attentions = []
            teacher_attentions = [] # Used to replicate the attention heads
            for layer_index, attn_layer in enumerate(attentions):
                for head_index in range(attn_layer.size(1)):
                    if (layer_index, head_index) in self.heads:
                        distillation_attentions.append(attn_layer[0, head_index:head_index+1, :, :])
                        teacher_attentions.append(teacher_attention[layer_index][0, head_index:head_index+1, :, :])

            distillation_attentions = torch.stack(distillation_attentions)
            teacher_attentions = torch.stack(teacher_attentions)

            distillation_loss = self.distillation_criterion(distillation_attentions, teacher_attentions)

            return distillation_loss






class AttentionDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.distance_criterion = nn.MSELoss(reduction="sum")

    def forward(self, student_attentions, teacher_attentions):
        # attentions are of shape (num_layeres, num_heads, first_seq_length, first_seq_length)
        return self.distance_criterion(student_attentions, teacher_attentions)




class AttentionEqualizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.distance_criterion = nn.MSELoss(reduction="sum")

    def forward(self, attentions):
        # "attentions" is of shape (num_layers, num_heads, first_seq_length, group_length)
        # Here num_layers = 1
        num_groups = attentions.size(-1)
        total_loss = 0
        for i in range(1, num_groups):
            total_loss += self.distance_criterion(attentions[:, :, :, 0], attentions[:, :, :, i])
        return total_loss








def tokenize_function(examples):
    is_negative_example = random.random()
    if is_negative_example < args.neg_ratio:
        num_words = random.randint(0, args.max_num_neg_words)
        words = [vocab[random.randint(0, len(vocab) - 1)] for _ in range(num_words)]
        label = 0
    else:
        random_demographic = random.choice(list(groups.keys()))
        random_index = random.randint(0, len(groups[random_demographic]) - 1)
        words = groups[random_demographic][random_index]
        label = 1

    tokenized_output = tokenizer(examples["text"], " ".join(words), truncation=True)
    tokenized_output["label"] = label
    return tokenized_output



def find_separator_position(input, tokenizer):
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    return (input == sep_id).nonzero(as_tuple=True)[-1][0].item()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train/de-en.txt/News-Commentary.de-en.en", help="Path to training data")
    parser.add_argument("--def_word_filepath", type=str, default="data/train/definition_words.json", help="filepath to the JSON file containing definition tuples per bias type")
    parser.add_argument("--vocab_filepath", type=str, default="data/train/wordlist.10000.txt", help="filepath to vocabulary used for negative examples")
    parser.add_argument("--tokenizer", type=str, help="Full name or path or URL to tokenizer", required=True)
    parser.add_argument("--model", type=str, help="Full name or path or URL to the model to debias", required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size. Should be left to 1")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--equ_lambda", type=float, default=2.0, help="equalization weight")
    parser.add_argument("--neg_ratio", type=float, default=0.8, help="ratio of negative examples")
    parser.add_argument("--max_num_neg_words", type=int, default=5,help="Maximum number of negative words randomly samples from the vocabulary")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--print_every", type=int, default=50000, help="Number of training steps after which print error")
    parser.add_argument("--eval_percentage", type=float, default=0.2, help="Percentage of data to use in evaluation")
    parser.add_argument("--seed", type=int, default=41, help="Seed for reproducibility")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation during training?")
    parser.add_argument("--output", type=str, help="Name of final debiased model. Will be stored in saved_models/encoder/", required=True)
    parser.add_argument("--num_heads", type=int, default=144, help="Number of top head showing bias to debias")
    parser.add_argument("--bias_type", type=str, default="all", help="Bias Type to mitigate")


    args = parser.parse_args()

    print(vars(args))

    if args.bias_type == "gender":
        heads = [(2, 0), (2, 9), (0, 2), (1, 1), (9, 11), (7, 4), (10, 3), (4, 9), (0, 5), (4, 10), (3, 5), (2, 3), (9, 8), (6, 3), (10, 8), (2, 8), (3, 0), (7, 11), (2, 7), (1, 11), (5, 10), (10, 10), (10, 7), (6, 10), (0, 9), (10, 0), (5, 1), (5, 4), (5, 6), (6, 11), (0, 11), (5, 5), (2, 10), (4, 11), (7, 9), (4, 6), (0, 4), (6, 0), (6, 2), (2, 6), (3, 6), (1, 9), (9, 6), (1, 8), (1, 6), (0, 6), (1, 5), (5, 3), (3, 8), (8, 1), (1, 0), (4, 0), (4, 7), (0, 7), (2, 4), (6, 7), (3, 1), (1, 2), (4, 2), (7, 0), (3, 11), (4, 8), (2, 5), (2, 2), (3, 10), (5, 2), (1, 3), (10, 9), (10, 4), (3, 7), (8, 2), (2, 11), (6, 8), (2, 1), (0, 8), (7, 8), (5, 11), (3, 2), (10, 2), (5, 9), (6, 9), (8, 4), (1, 7), (0, 1), (0, 3), (11, 8), (4, 4), (10, 6), (6, 4), (7, 2), (1, 10), (3, 3), (1, 4), (11, 9), (11, 5), (9, 9), (3, 9), (10, 5), (11, 6), (4, 3), (8, 8), (11, 4), (8, 9), (6, 6), (0, 10), (9, 0), (7, 6), (8, 6), (9, 1), (8, 11), (6, 1), (5, 7), (4, 5), (9, 3), (5, 8), (8, 5), (9, 2), (7, 10), (11, 11), (10, 1), (7, 7), (5, 0), (11, 0), (11, 2), (8, 10), (8, 7), (7, 3), (0, 0), (8, 0), (3, 4), (11, 10), (8, 3), (11, 3), (6, 5), (7, 5), (7, 1), (11, 7), (4, 1), (9, 5), (10, 11), (9, 7), (9, 4), (11, 1), (9, 10)]
    elif args.bias_type == "race":
        heads = [(0, 2), (0, 9), (1, 9), (0, 11), (0, 5), (1, 2), (2, 10), (2, 3), (9, 11), (2, 8), (1, 11), (10, 8), (7, 11), (2, 2), (0, 4), (2, 0), (2, 7), (0, 1), (3, 0), (1, 0), (9, 8), (0, 6), (6, 11), (2, 4), (7, 4), (6, 7), (4, 9), (1, 5), (1, 8), (4, 10), (4, 2), (0, 7), (4, 0), (1, 1), (2, 6), (10, 3), (3, 8), (2, 5), (2, 9), (10, 10), (0, 10), (3, 5), (5, 4), (1, 6), (3, 6), (3, 7), (10, 0), (3, 11), (3, 2), (6, 9), (3, 1), (3, 10), (10, 7), (2, 11), (5, 10), (1, 7), (5, 3), (7, 9), (6, 3), (5, 2), (6, 6), (1, 10), (10, 6), (1, 3), (8, 1), (5, 1), (0, 8), (10, 9), (2, 1), (10, 2), (4, 7), (6, 8), (8, 2), (4, 11), (9, 0), (8, 6), (9, 2), (0, 3), (0, 0), (6, 10), (10, 1), (4, 6), (5, 9), (4, 3), (5, 6), (11, 4), (5, 5), (3, 3), (6, 2), (4, 8), (8, 4), (1, 4), (11, 8), (4, 4), (7, 6), (8, 11), (3, 9), (9, 9), (7, 8), (5, 11), (10, 4), (9, 1), (8, 5), (11, 2), (11, 7), (6, 4), (9, 6), (7, 0), (11, 5), (11, 9), (9, 3), (6, 1), (7, 7), (6, 0), (10, 5), (7, 5), (7, 2), (8, 9), (3, 4), (5, 8), (8, 8), (5, 7), (4, 5), (7, 10), (11, 6), (11, 11), (9, 5), (5, 0), (9, 7), (4, 1), (7, 1), (8, 7), (8, 10), (6, 5), (8, 0), (11, 3), (8, 3), (9, 4), (11, 10), (11, 0), (10, 11), (9, 10), (7, 3), (11, 1)]
    elif args.bias_type == "religion":
        heads = [(0, 2), (0, 11), (0, 9), (2, 0), (1, 9), (2, 3), (2, 9), (3, 0), (0, 5), (9, 11), (10, 3), (4, 2), (10, 10), (1, 6), (7, 4), (1, 2), (10, 8), (4, 10), (2, 10), (2, 7), (2, 2), (3, 5), (10, 7), (6, 11), (1, 0), (5, 4), (3, 6), (1, 11), (4, 9), (0, 4), (1, 5), (7, 11), (1, 1), (9, 8), (0, 7), (2, 6), (3, 8), (1, 8), (0, 10), (2, 4), (3, 1), (5, 9), (5, 10), (6, 6), (2, 11), (0, 6), (3, 10), (3, 2), (6, 9), (10, 0), (3, 11), (0, 1), (3, 7), (5, 1), (1, 7), (4, 7), (2, 8), (4, 0), (1, 3), (4, 6), (5, 2), (10, 6), (5, 3), (7, 9), (2, 1), (0, 3), (6, 10), (6, 3), (4, 11), (8, 1), (0, 8), (7, 0), (1, 10), (11, 9), (6, 2), (10, 4), (5, 5), (1, 4), (5, 11), (6, 7), (7, 8), (0, 0), (11, 4), (3, 9), (7, 2), (4, 8), (3, 3), (9, 0), (11, 5), (10, 9), (8, 11), (9, 2), (10, 2), (5, 6), (11, 8), (11, 6), (9, 9), (8, 4), (4, 4), (6, 0), (6, 8), (4, 3), (9, 1), (11, 11), (9, 6), (6, 4), (2, 5), (10, 5), (5, 8), (7, 6), (10, 1), (8, 6), (11, 2), (5, 0), (6, 1), (7, 7), (8, 2), (8, 9), (11, 3), (7, 3), (4, 5), (8, 5), (11, 7), (11, 10), (5, 7), (4, 1), (9, 7), (11, 0), (6, 5), (8, 10), (3, 4), (7, 10), (8, 0), (10, 11), (7, 1), (9, 3), (9, 4), (8, 8), (7, 5), (9, 5), (8, 7), (9, 10), (8, 3), (11, 1)]
    elif args.bias_type == "all":
        heads = [(2, 0), (0, 2), (0, 9), (0, 11), (2, 9), (0, 5), (2, 3), (7, 4), (9, 11), (1, 9), (3, 0), (4, 10), (2, 10), (7, 11), (1, 11), (9, 8), (10, 3), (6, 11), (1, 1), (2, 8), (0, 4), (2, 7), (1, 2), (4, 9), (2, 2), (10, 8), (0, 7), (3, 5), (0, 1), (4, 2), (10, 10), (0, 6), (2, 4), (10, 7), (3, 6), (5, 4), (2, 6), (1, 0), (1, 5), (10, 0), (5, 10), (4, 0), (1, 6), (3, 8), (1, 8), (6, 7), (7, 9), (5, 1), (6, 3), (4, 6), (3, 10), (5, 3), (3, 1), (1, 7), (2, 1), (2, 11), (3, 2), (1, 3), (3, 7), (5, 5), (3, 11), (6, 10), (2, 5), (4, 7), (7, 0), (4, 11), (8, 1), (6, 9), (6, 8), (10, 9), (5, 2), (6, 2), (0, 10), (5, 6), (0, 8), (5, 9), (5, 11), (6, 6), (1, 10), (6, 0), (0, 3), (10, 6), (8, 2), (4, 8), (10, 2), (9, 6), (11, 8), (9, 0), (10, 4), (8, 4), (7, 8), (3, 3), (8, 11), (8, 6), (11, 5), (11, 9), (11, 4), (1, 4), (3, 9), (4, 3), (0, 0), (9, 2), (9, 9), (4, 4), (7, 6), (10, 1), (6, 4), (10, 5), (11, 6), (9, 1), (7, 2), (6, 1), (8, 9), (7, 7), (4, 5), (8, 5), (9, 3), (11, 2), (11, 11), (8, 8), (7, 10), (5, 8), (5, 0), (11, 7), (5, 7), (11, 10), (11, 3), (3, 4), (9, 7), (8, 10), (7, 3), (7, 5), (11, 0), (4, 1), (8, 0), (9, 5), (8, 7), (6, 5), (7, 1), (10, 11), (9, 4), (8, 3), (9, 10), (11, 1)]
    else:
        raise ValueError

    heads = heads[: args.num_heads]

    with open(args.def_word_filepath, 'r') as f:
        groups = json.load(f)["biases"]


    with open(args.vocab_filepath, 'r') as f:
        vocab = f.read().split('\n')

    torch.manual_seed(args.seed)

    configuration = AutoConfig.from_pretrained(args.model, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, return_tensors="pt")
    teacher = AutoModel.from_pretrained(args.model, config=configuration)
    student = AutoModel.from_pretrained(args.model, config=configuration)

    for param in teacher.parameters():
        param.requires_grad = False


    # Load the dataset
    dataset = load_dataset('text', data_files=args.train_data, split={
        'train': 'train[:{}%]'.format(100 - round(args.eval_percentage * 100)),
        'eval': 'train[-{}%:]'.format(round(100 * args.eval_percentage))
    })

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"])
    tokenized_dataset.set_format(type='torch')

    dataloader_train = torch.utils.data.DataLoader(
                tokenized_dataset["train"],
                batch_size=args.batch_size
            )

    dataloader_eval = torch.utils.data.DataLoader(
                tokenized_dataset["eval"],
                batch_size=args.batch_size
            )


    # Load the optimizer 
    optimizer = torch.optim.AdamW(params=student.parameters(), lr=args.lr)
    criterion = AttentionLoss(equ_lambda=args.equ_lambda, heads=heads)

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    student.train().to(device)
    teacher.eval().to(device)

    save_path = "saved_models/encoder/"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 

    for epoch in range(1, 1 + args.epochs):

        print("# Epoch ==> {}".format(epoch))

        # For logging purposes
        total_loss = 0
        count = 0

        student.train()

        for i, batch in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            student_attentions = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"]).attentions
            teacher_attentions = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"]).attentions
            
            second_sentence_index = find_separator_position(batch["input_ids"], tokenizer)
            loss = criterion(attentions=student_attentions, teacher_attention=teacher_attentions, split_index=second_sentence_index, labels=batch["label"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += args.batch_size

            if (i + 1) % args.print_every == 0:
                tqdm.write(str(total_loss / count))
                total_loss = 0
                count = 0


        if args.do_eval:
            student.eval()

            print("-"*10, " Evaluating... ", "-"*10)
            total_loss = 0
            count = 0

            for i, batch in enumerate(tqdm(dataloader_eval)):
                optimizer.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}
                student_attentions = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"]).attentions
                teacher_attentions = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"]).attentions
                
                second_sentence_index = find_separator_position(batch["input_ids"], tokenizer)
                loss = criterion(attentions=student_attentions, split_index=second_sentence_index, teacher_attention=teacher_attentions, labels=batch["label"])

                total_loss += loss.item()
                count += args.batch_size
        
            print(total_loss / count)

        student.save_pretrained(save_path + args.output)
