#!/usr/bin/env python3
import argparse
import random
import sys
from tqdm import tqdm
import json

# imports
from datasets import load_dataset
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--insert_proportion", default=0.5)
parser.add_argument("--seed", default=69, type=int, help="Random seed")
parser.add_argument("--max_vocab", default=1_000_000, type=int, help="The number of possible words to insert.")
parser.add_argument("--generate_size", default=1_000, type=int, help="Number of paragraphs for the generated dataset.")
parser.add_argument("--dataset_name", default="inserted_words_dataset.jsonl", help="name")


def get_word_vocab(dataset, max_words):
    dictionary = set()
    i = 0
    for instance in dataset:
        words = instance["text"].split()
        for w in words:
            w = w.lower()
            if w not in dictionary:
                dictionary.add(w)
                i += 1
                if i > max_words:
                    return list(dictionary)


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="all")
    vocab = get_word_vocab(dataset, args.max_vocab)
    vocab_size = len(vocab)  # [0, vocab_size)
    print("Vocabulary size: ", vocab_size)

    for splt in ["train", "validation", "test"]:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=splt)
        print(dataset)

        print("Generating data for", splt, "split")
        count = 0
        with open(splt+"-"+args.dataset_name, "w") as jsn:
            for instance in tqdm(dataset):
                words = instance["text"].split()
                num_words = len(words)
                num_to_insert = int(num_words * args.insert_proportion)

                insert_idxs = np.random.choice(range(num_words), num_to_insert, replace=False)
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

                if len(new_words) == 0:
                    continue

                example = {
                    "words": new_words,
                    "target": targets
                }

                json.dump(example, jsn)
                jsn.write("\n")  # when one long line hugface breaks

                count += 1
                if count > args.generate_size:
                    break

        print("creating hugface object")
        generated_dataset = load_dataset("json", data_files=args.dataset_name, split="train")
        print(generated_dataset)
        print("---------------------------------------------")

    return 0


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    x = main(args)
