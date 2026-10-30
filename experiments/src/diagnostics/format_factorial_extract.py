"""Use the frozen original encoder implementation with isolated controlled inputs."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import encode
runtime=ROOT/'artifacts/format_factorial_runtime'
assert runtime.exists()
for name,target in [('datasets',ROOT/'datasets'),('embedders',ROOT/'embedders'),('artifacts',ROOT/'artifacts')]:
 p=runtime/name
 assert p.resolve()==target.resolve()
encode.ROOT=runtime
encode.CONFIG=dict(encode.CONFIG,qwen_layers=[21])
encode.main(sys.argv[1],'format_factorial_v1')
