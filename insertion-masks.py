#!/usr/bin/env python3
import argparse
import sys

#imports
from datasets import load_dataset
from pytorch-transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

parser = argparse.ArgumentParser()

parser.add_argument("--target-prob", default="")
parser.add_argument("--insert-prob", default="")
parser.add_argument("--seed", default=69, type=int, help="Random seed")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
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


def create_dataset():
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    dataset
    
    return 0


def train_model():
    return 0


def main(args: argparse.Namespace):
    
    # dataset = load_dataset('wikitext', 'wikitext-103-v1', split='all')
    # dataset = dataset[]

    return 0


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    x = main(args)
    print(x)