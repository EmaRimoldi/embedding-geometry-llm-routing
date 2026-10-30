import argparse
from run_prox import run
p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);run(p.parse_args().task)
