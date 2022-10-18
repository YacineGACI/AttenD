import pytorch_lightning as pl
import argparse
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AdamW,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import DataLoader, Dataset




class MaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)
        
    def load_lines(self, file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines
    
    def encode_lines(self, lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=args.max_len
        )
        return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)




class MLM(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(args.model)

    def forward(self, input_ids, labels):
        return self.model(input_ids=input_ids,labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        return {"loss": loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=args.lr)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--train_data", type=str, default="data/train/de-en.txt/News-Commentary.de-en.en", help="Filepath to training data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="Probability to mask random tokens in the input")
    parser.add_argument("--output", type=str, required=True, help="ame of trained language model. Will be saved in 'saved_models/tasks/mlm/'")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--checkpoint_callback", action="store_true", help="Checkpoint callback?")
    parser.add_argument("--logger", action="store_true", help="Do log?")

    args = parser.parse_args()
    print(vars(args))


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset = MaskedLMDataset(args.train_data, tokenizer)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )

    model = MLM()

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, checkpoint_callback=args.checkpoint_callback, logger=args.logger)
    trainer.fit(model, train_loader)

    model.model.save_pretrained("saved_models/tasks/mlm/{}".format(args.output))

