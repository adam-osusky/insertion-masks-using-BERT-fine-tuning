#!/usr/bin/env python3
import argparse
import sys

#imports
from datasets import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--insert-prob", default="")
parser.add_argument("--seed", default=69, type=int, help="Random seed")
parser.add_argument("--output-path", default="./data.txt", help="Where to save generated data.")


def main(args: argparse.Namespace):
    
    dataset = load_dataset('wikitext', 'wikitext-103-v1)
    dataset = dataset[]

    return 0


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    x = main(args)
    print(x)