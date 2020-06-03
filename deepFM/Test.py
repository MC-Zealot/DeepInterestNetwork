import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys

from pandas.core.frame import DataFrame

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512
a = "123"

with open('../din/dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  train_set_mini = train_set[0:10000]
  test_set = pickle.load(f)
  test_set_mini = train_set[0:5000]
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)


with open('train_set_mini.pkl', 'w') as outfile1:
  pickle.dump(train_set_mini, outfile1,0)

with open('test_set_mini.pkl', 'w') as outfile2:
  pickle.dump(test_set_mini, outfile2,0)
# with open(str('train_mini.pkl'), 'rb') as f:
#     train_set = pickle.load(f)
# word = pickle.load(open("train_mini.pkl", 'r'), encoding='utf-8')
with open('train_mini.pkl', 'r') as f3:
    data_dict = pickle.load(f3)
# with open('test_mini.pkl', 'rb') as f1:
#     test_set = pickle.load(f1)
#
# with open('cate.pkl', 'rb') as f2:
#     cate_list = pickle.load(f2)
#
# c = DataFrame(cate_list)
# cate_count = len(c.drop_duplicates())
# user_count, item_count, cate_count = pickle.load(f)

