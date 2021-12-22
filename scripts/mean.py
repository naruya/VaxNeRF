import argparse
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="")
args = parser.parse_args()

paths = glob.glob(args.path)

for path in paths:
    with open(path) as f:
        data = np.array(f.read().split(), dtype=np.float)
    print(path, '{:.4f}'.format(np.mean(data)))
