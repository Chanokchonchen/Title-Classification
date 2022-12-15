import argparse
from util import *

parser = argparse.ArgumentParser(description='Python Script for Tokenize Dict')
parser.add_argument('-engine',type=str,help='Choose Tokenizer Engine',choices=['deepcut','newmm','wordcut'],default='newmm')
args = parser.parse_args()
engine = args.engine
print(f"Tokenizer Engine {engine}")
for i in range(1,SPECIAL_FEATURE+1):
    tokenize_dict(engine,f"wl{i}")