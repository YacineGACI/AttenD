import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from collections import namedtuple



def tokenize(task_name):

    def tokenize_sst2(examples):
        return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_cola(examples):
        return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_mrpc(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_stsb(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_mnli(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_rte(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)

    def tokenize_wnli(examples):
        return tokenizer(examples['sentence1'], examples["sentence2"], truncation=True, padding="max_length", max_length=args.max_len)



    if task_name == "sst2":
        return tokenize_sst2
    if task_name == "cola":
        return tokenize_cola
    elif task_name == "stsb":
        return tokenize_stsb
    elif task_name == "mrpc":
        return tokenize_mrpc
    elif task_name == "mnli":
        return tokenize_mnli
    elif task_name == "rte":
        return tokenize_rte
    elif task_name == "wnli":
        return tokenize_wnli
    else:
        raise ValueError





def load_metadata(task_name):
    MetadataOutput = namedtuple("MetadataOutput", ['dataset_name', 'task_name', 'short_task_name', 'num_labels', 'remove_columns', 'validation'])
    if task_name == "sst2":
        return MetadataOutput(
            dataset_name="sst2",
            task_name="Sentiment Classification",
            short_task_name="sa",
            num_labels=2,
            remove_columns=["sentence", "idx"],
            validation="validation"
        )

    elif task_name == "cola":
        return MetadataOutput(
            dataset_name="cola",
            task_name="Linguistic Acceptability",
            short_task_name="cola",
            num_labels=2,
            remove_columns=["sentence", "idx"],
            validation="validation"
        )


    elif task_name == "mrpc":
        return MetadataOutput(
            dataset_name="mrpc",
            task_name="Paraphrase Detection",
            short_task_name="pd",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation="validation"
        )

    elif task_name == "stsb":
        return MetadataOutput(
            dataset_name="stsb",
            task_name="Sementic Textual Similarity",
            short_task_name="sts",
            num_labels=1,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation="validation"
        )

    elif task_name == "mnli":
        return MetadataOutput(
            dataset_name="mnli",
            task_name="Senetence Entailment",
            short_task_name="se",
            num_labels=3,
            remove_columns=["premise", "hypothesis", "idx"],
            validation="validation_matched"
        )

    elif task_name == "rte":
        return MetadataOutput(
            dataset_name="rte",
            task_name="Sentence Entailment",
            short_task_name="rte",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation="validation"
        )

    elif task_name == "wnli":
        return MetadataOutput(
            dataset_name="wnli",
            task_name="Sentence Entailment",
            short_task_name="wnli",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation="validation"
        )
    
    else:
        raise ValueError




def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--task", type=str, required=True, help="GLUE task to train", choices=["sst2", "cola", "stsb", "mrpc", "mnli", "wnli", "rte"])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight_decay")
    parser.add_argument("--output", type=str, required=True, help="Name of trained model (output). Will be saved in 'saved_models/tasks/$task/'")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation during training?")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")

    args = parser.parse_args()
    print(vars(args))

    dataset_name, task_name, short_task_name, num_labels, remove_columns, validation = load_metadata(args.task)


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = load_dataset('glue', args.task)
    dataset = dataset.map(tokenize(args.task), batched=True, remove_columns=remove_columns)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, remove_columns=["label"])
    dataset.set_format(type='torch')

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    metric = load_metric('glue', args.task)


    training_args = TrainingArguments(
        "saved_models/tasks/{}/{}".format(args.task, args.output),
        evaluation_strategy = "no" if args.task == "mnli" or not args.do_eval else "epoch",
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
        eval_dataset=dataset[validation],
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained("saved_models/tasks/{}/{}".format(args.task, args.output))

