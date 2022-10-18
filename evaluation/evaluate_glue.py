import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
from collections import namedtuple
import math
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm




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



def compute_metrics(task_name):
    def compute_metrics_classification(eval_pred):
        logits = eval_pred.predictions[0]
        labels = eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    def compute_metrics_regression(eval_pred):
        logits = eval_pred.predictions[0]
        logits = np.array(logits).reshape((-1,))
        labels = eval_pred.label_ids
        return metric.compute(predictions=logits, references=labels)

    if task_name == "sst2":
        return compute_metrics_classification
    if task_name == "cola":
        return compute_metrics_classification
    elif task_name == "stsb":
        return compute_metrics_regression
    elif task_name == "mrpc":
        return compute_metrics_classification
    elif task_name == "mnli":
        return None
    elif task_name == "rte":
        return compute_metrics_classification
    elif task_name == "wnli":
        return compute_metrics_classification
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
            validation=["validation"]
        )

    elif task_name == "cola":
        return MetadataOutput(
            dataset_name="cola",
            task_name="Linguistic Acceptability",
            short_task_name="cola",
            num_labels=2,
            remove_columns=["sentence", "idx"],
            validation=["validation"]
        )

    elif task_name == "mrpc":
        return MetadataOutput(
            dataset_name="mrpc",
            task_name="Paraphrase Detection",
            short_task_name="pd",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation=["validation"]
        )

    elif task_name == "stsb":
        return MetadataOutput(
            dataset_name="stsb",
            task_name="Sementic Textual Similarity",
            short_task_name="sts",
            num_labels=1,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation=["validation"]
        )

    elif task_name == "mnli":
        return MetadataOutput(
            dataset_name="mnli",
            task_name="Senetence Entailment",
            short_task_name="se",
            num_labels=3,
            remove_columns=["premise", "hypothesis", "idx"],
            validation=["validation_matched", "validation_mismatched"]
        )

    elif task_name == "rte":
        return MetadataOutput(
            dataset_name="rte",
            task_name="Sentence Entailment",
            short_task_name="rte",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation=["validation"]
        )

    elif task_name == "wnli":
        return MetadataOutput(
            dataset_name="wnli",
            task_name="Sentence Entailment",
            short_task_name="wnli",
            num_labels=2,
            remove_columns=["sentence1", "sentence2", "idx"],
            validation=["validation"]
        )
    
    else:
        raise ValueError



def evaluate_mnli(test_data, batch_size=16):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    predictions = []
    labels = []
    num_steps = math.ceil(len(test_data) / batch_size)
    for i in tqdm(range(num_steps)):
        input = {k:v.to(device) for k, v in test_data[i:i+batch_size].items()}
        logits = model(**input).logits
        predictions += np.argmax(logits.cpu().detach().numpy(), axis=-1).tolist()
        labels += test_data[i:i+batch_size]['labels'].cpu().detach().tolist()
    
    assert len(predictions) == len(labels)
        
    return {'accuracy': accuracy_score(labels, predictions)}





if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Full name or path or URL to tokenizer")
    parser.add_argument("--model", type=str, required=True, help="Full name or path or URL to trained NLI model")
    parser.add_argument("--task", type=str, required=True, help="GLUE task to train", choices=["sst2", "cola", "stsb", "mrpc", "mnli", "wnli", "rte"])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--acc_steps", type=int, default=200, help="Evaluation acuumulation steps")
    parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")

    args = parser.parse_args()
    print(vars(args))
    
    dataset_name, task_name, short_task_name, num_labels, remove_columns, validation = load_metadata(args.task)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    metric = load_metric('glue', args.task)

    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        no_cuda=args.no_cuda,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="no",
        eval_accumulation_steps=args.acc_steps
    )

    for v in validation:
        dataset = load_dataset('glue', args.task, split=v)
        dataset = dataset.map(tokenize(args.task), batched=True, remove_columns=remove_columns)
        dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, remove_columns=["label"])
        dataset.set_format(type='torch')
    
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=dataset,
            compute_metrics=compute_metrics(dataset_name),
        )

        if args.task == "mnli":
            metrics = evaluate_mnli(dataset)
        else:
            metrics = trainer.evaluate()
        print("#"*5, ">  ", v)
        print(metrics)
        print()
