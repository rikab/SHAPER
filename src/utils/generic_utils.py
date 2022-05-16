from __future__ import absolute_import, division, print_function

import contextlib
from functools import wraps
import gzip
from itertools import repeat
import json
import multiprocessing
import os
import sys
import time
import warnings

import six



# return argument if iterable else make repeat generator
def iter_or_rep(arg):
    if isinstance(arg, (tuple, list)):
        if len(arg) == 1:
            return repeat(arg[0])
        else:
            return arg
    elif isinstance(arg, repeat):
        return arg
    else:
        return repeat(arg)