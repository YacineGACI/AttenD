import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric



def tokenize(examples):
    return tokenizer(examples['post'], truncation=True, padding="max_length", max_length=args.max_len)




def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight_decay")
    parser.add_argument("--output", type=str, required=True, help="Name of trained model (output). Will be saved in 'saved_models/tasks/$task/'")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation during training?")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--data_path", type=str, default="data/evaluation/hate_speech/hate-alert HateXplain master Data/", help="Path to folder containing files trains.csv and val.csv")

    args = parser.parse_args()
    print(vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = load_dataset("csv", data_files={"train": ["{}/train.csv".format(args.data_path)], "validation": ["{}/val.csv".format(args.data_path)]})
    print(dataset)
    dataset.set_format(columns=['id', 'post', 'label', 'target'])
    dataset = dataset.map(tokenize, batched=True, remove_columns=["target"])
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, remove_columns=["label"])
    dataset.set_format(type='torch')

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    metric = load_metric('accuracy')


    training_args = TrainingArguments(
        "saved_models/tasks/hate_speech/{}".format(args.output),
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        weight_decay=args.wd,
        no_cuda=args.no_cuda,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="no",
        num_train_epochs=args.epochs,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained("saved_models/tasks/hate_speech/{}".format(args.output))

