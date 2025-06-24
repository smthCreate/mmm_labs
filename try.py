import re

from algorythms import (
    momentum_method,nesterov_method,adagrad_method,adadelta_method,rmsprop_method,adam_method)

algorithms = [momentum_method, nesterov_method, adagrad_method, adadelta_method, rmsprop_method, adam_method]
print(algorithms[0].__name__)