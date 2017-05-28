import sys
from optparse import OptionParser

sys.path.append('./')

import yolo
from yolo.utils.process_config import process_config
from yolo.dataset.text_dataset import *
from yolo.net.yolo_tiny_net import *
from yolo.solver.yolo_solver import *

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please sspecify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = TextDataSet(common_params, dataset_params)
net = YoloTinyNet(common_params, net_params)
solver = YoloSolver(dataset, net, common_params, solver_params)
solver.solve()
