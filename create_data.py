#!/usr/bin/env python3
import argparse
import random
import sys
from tqdm import tqdm
import json

# imports
from datasets import load_dataset
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--insert_proportion", default=0.15)
parser.add_argument("--seed", default=69, type=int, help="Random seed")
parser.add_argument("--output_path", default="./data.txt", help="Where to save generated data.")
parser.add_argument("--max_vocab", default=1_000_000, type=int, help="The number of possible words to insert.")
parser.add_argument("--split", default="train", help="Which split from wiki.")
parser.add_argument("--generate_size", default=1_000_000, type=int, help="Size of the generated dataset.")
parser.add_argument("--dataset_name", default="inserted_words_dataset.jsonl", help="name")


def get_word_vocab(dataset, max_words):
    dictionary = {}
    i = 0
    for instance in dataset:
        words = instance["text"].split()
        for w in words:
            if w not in dictionary:
                dictionary[i] = w
                i += 1
                if i > max_words:
                    break
    return dictionary


def main(args: argparse.Namespace):
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test") #train
    vocab = get_word_vocab(dataset, args.max_vocab)
    vocab_size = len(vocab)  # [0, vocab_size)
    print("Vocabulary size: ", vocab_size)

    dataset = load_dataset("wikitext", "wikitext-103-v1", split=args.split)
    print(dataset)

    print("Generating data")
    count = 0
    with open(args.dataset_name, "w") as jsn:
        for instance in tqdm(dataset):
            words = instance["text"].split()
            num_words = len(words)
            num_to_insert = int(num_words * args.insert_proportion)

            insert_idxs = random.choices(range(num_words), k=num_to_insert)
            insertions = {}
            for i in insert_idxs:
                insertions[i] = vocab[random.randint(0, vocab_size - 1)]

            targets = [0] * (num_words + num_to_insert)

            inserted = 0
            new_words = []
            for i, w in enumerate(words):
                new_words.append(w)
                if i in insertions:
                    new_words.append(insertions[i])
                    targets[i + inserted + 1] = 1
                    inserted += 1
            # print("total words: ", num_words, "To insert: ", num_to_insert)
            # print("insert_idxs :", insert_idxs)
            # print(insertions)
            # print("targets len:", len(targets))
            # print(targets)
            # for i, w in zip(new_words, targets):
            #     print(i, w)
            # print("--------")

            if len(new_words) == 0:
                continue

            example = {
                "words": new_words,
                "target": targets
            }

            json.dump(example, jsn)

            count += 1
            if count > args.generate_size:
                break

    print("creating hugface object")
    generated_dataset = load_dataset("json", data_files=args.dataset_name, split="train")
    print(generated_dataset)

    return 0


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    x = main(args)
