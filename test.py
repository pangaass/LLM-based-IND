import argparse
from accelerate import Accelerator
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()
accelerator = Accelerator()
print(args)
