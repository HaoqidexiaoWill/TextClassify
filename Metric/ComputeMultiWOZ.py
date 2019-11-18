import json
import numpy
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from collections import defaultdict
import operator
from sklearn.metrics import recall_score
def accuracyF1(out, labels):
    outputs = np.argmax(out, axis=1)
    return f1_score(labels, outputs, labels=[0, 1], average='macro')