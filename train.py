from tqdm import tqdm
import random, pathlib, json, argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset


class AttentionLoss(nn.Module):
    def __init__(self, equ_lambda, layers):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.equ_lambda = equ_lambda
        self.layers = layers
        self.distillation_criterion = AttentionDistillationLoss()
        self.equalization_criterion = AttentionEqualizationLoss()


    def forward(self, attentions, teacher_attention, split_index, labels):
        if labels == 1:
            distillation_attentions = []
            equalization_attentions = []
            teacher_attentions = [] # Used to replicate the attention heads
            for layer_index, attn_layer in enumerate(attentions):
                if layer_index in self.layers:
                    distillation_attentions.append(attn_layer[0, :, :split_index+1, :split_index+1])
                    equalization_attentions.append(attn_layer[0, :, :split_index+1, split_index+1:-1])
                    teacher_attentions.append(teacher_attention[layer_index][0, :, :split_index+1, :split_index+1])


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
                if layer_index in self.layers:
                    distillation_attentions.append(attn_layer[0, :, :, :])
                    teacher_attentions.append(teacher_attention[layer_index][0, :, :, :])

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
    parser.add_argument("--layers", type=int, default=12, help="Layers used in debiasing")
    parser.add_argument("--equ_lambda", type=float, default=2.0, help="equalization weight")
    parser.add_argument("--neg_ratio", type=float, default=0.8, help="ratio of negative examples")
    parser.add_argument("--max_num_neg_words", type=int, default=5,help="Maximum number of negative words randomly samples from the vocabulary")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--print_every", type=int, default=50000, help="Number of training steps after which print error")
    parser.add_argument("--eval_percentage", type=float, default=0.2, help="Percentage of data to use in evaluation")
    parser.add_argument("--seed", type=int, default=41, help="Seed for reproducibility")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation during training?")
    parser.add_argument("--output", type=str, help="Name of final debiased model. Will be stored in saved_models/encoder/", required=True)


    args = parser.parse_args()

    print(vars(args))

    layers = list(range(args.layers))


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
    criterion = AttentionLoss(equ_lambda=args.equ_lambda, layers=layers)

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
