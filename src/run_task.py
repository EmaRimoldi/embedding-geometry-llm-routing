import json,sys
from common import ROOT
from geometry import main
args=json.loads((ROOT/'configs/geometry_tasks.json').read_text())[int(sys.argv[1])]
main(args['encoder'],args['dataset'],args['subset'],args['fold'])
