import sys
import os

from config import DESCRIPTOR_LIST
from src.train import train
from src.evaluate import evaluate
from src.query import query


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py [train/eval/query] [directory] [descriptor] [output_file]")
        sys.exit(1)

    mode = sys.argv[1]
    path_name = sys.argv[2]
    descriptor_method = sys.argv[3]
    descriptor_file_name = sys.argv[4]

    if not os.path.exists(path_name):
        print(f"The directory '{path_name}' does not exist.")
        sys.exit(1)

    if descriptor_method not in list(DESCRIPTOR_LIST.keys()):
        print(f"The descriptor '{descriptor_method}' is unknown.")
        sys.exit(1)
    else:
        descriptor = DESCRIPTOR_LIST[descriptor_method]

    if mode == "train":
        train(path_name, descriptor, descriptor_file_name)
    elif mode == "eval":
        evaluate(path_name, descriptor, descriptor_file_name)
    elif mode == "query":
        query(path_name, descriptor, descriptor_file_name)
    else:
        print("The mode must be 'train', 'eval' or 'query'.")
        sys.exit(1)
