import ncu_report
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ncu-path', type=str, required=True)
parser.add_argument('--report', type=str, required=True)
args = parser.parse_args()

sys.path.insert(1, args.ncu_path)
