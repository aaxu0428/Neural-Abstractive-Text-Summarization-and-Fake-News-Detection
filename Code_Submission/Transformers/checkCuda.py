

import sys
import time
import numpy as np
import cPickle as pickle
import copy
import random
from random import shuffle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data as datar
from model import *
from utils_pg import *
from configs import *
from transformer.utils import *
from transformer.optim import Optim


print('torch.cuda.is_available()',torch.cuda.is_available())
print('torch.cuda.is_available()',torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

   