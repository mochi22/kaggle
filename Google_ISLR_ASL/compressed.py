import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i','--in_file', help='input file path')
parser.add_argument('-p', help = "This is save dir path")
args = parser.parse_args()

if args.in_file:
    inputs = np.load(args.in_file)

if args.p:
    np.savez_compressed(f"{args.p}", inputs)
else:
    np.savez_compressed(f"{args.in_file[:-4]}_comp.npz", inputs)