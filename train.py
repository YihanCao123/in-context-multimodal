import torch
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument()

    args = parser.parse_args()
    print(f"Args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()