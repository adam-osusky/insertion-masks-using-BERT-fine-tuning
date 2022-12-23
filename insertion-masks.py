#!/usr/bin/env python3
import argparse
import sys

#imports
from datasets import load_dataset
# from pytorch-transformers import AutoTokenizer
import transformers
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import transformers
import evaluate
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--target_prob", default="")
parser.add_argument("--insert_prob", default="")
parser.add_argument("--seed", default=69, type=int, help="Random seed")
parser.add_argument("--train_dataset_name", default="train-inserted_words_dataset.jsonl", help="name")
parser.add_argument("--eval_dataset_name", default="validation-inserted_words_dataset.jsonl", help="name")
parser.add_argument("--model", default="distilbert-base-uncased")

seqeval = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


id2label = {
    0: "not inserted",
    1: "inserted"
}
label2id = {
    "not inserted": 0,
    "inserted": 1
}


def main(args: argparse.Namespace):
    access_token = "kkt"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["target"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_dataset = load_dataset("json", data_files=args.train_dataset_name, split="train")
    print(train_dataset)
    train_tokenized_data = train_dataset.map(tokenize_and_align_labels, batched=True)

    eval_dataset = load_dataset("json", data_files=args.eval_dataset_name, split="train")
    print(eval_dataset)
    eval_tokenized_data = eval_dataset.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="testrun_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_token=access_token
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_data,
        eval_dataset=eval_tokenized_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./my_model")
    # trainer.push_to_hub()

    return 0


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    x = main(args)
