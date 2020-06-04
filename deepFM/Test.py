import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
from pandas.core.frame import DataFrame

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512
def process_data_to_csv():
  with open('../din/dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    train_set_mini = train_set[0:10000]
    test_set = pickle.load(f)
    test_set_mini = train_set[0:5000]
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

    train_set_mini_df = DataFrame(train_set_mini)
    train_set_mini_df.to_csv('train_set_mini.csv')

    train_set_df = DataFrame(train_set)
    train_set_df.to_csv('train_set.csv')

    test_set_df = DataFrame(test_set)
    test_set_df.to_csv('test_set.csv')

    test_set_mini_df = DataFrame(test_set_mini)
    test_set_mini_df.to_csv('test_set_mini.csv')

    cate_list_df = DataFrame(cate_list)
    cate_list_df.to_csv('cate_list.csv')


def read_csv():
  train_set_mini = pd.read_csv('train_set_mini.csv', names=['user_id', 'viewed_item_id', 'item_id', 'label'])
  test_set_mini = pd.read_csv('test_set_mini.csv', names=['user_id', 'viewed_item_id', 'item_id', 'label'])
  cate_list = pd.read_csv('cate_list.csv', names=['user_id', 'viewed_item_id', 'item_id', 'label'])
  return train_set_mini,test_set_mini,cate_list


def get_count():
  all_item_id = []
  user_count = len(train_set_mini['user_id'].drop_duplicates())
  cate_count = len(train_set_mini['cate_list'].drop_duplicates())

  train_set_mini['viewed_item_id_list'] = train_set_mini.viewed_item_id.apply(lambda x: x[1:-1].split(','))
  train_set_mini_item_id_list = train_set_mini.viewed_item_id_list
  for item_id_list in train_set_mini_item_id_list:
    all_item_id.extend(item_id_list)
  all_item_id_df = DataFrame(all_item_id)
  item_count = len(all_item_id_df.drop_duplicates())
  return user_count, item_count, cate_count


train_set_mini, test_set_mini, cate_list = read_csv()
user_count, item_count, cate_count = get_count()

# with open('train_set_mini.pkl', 'w') as outfile1:
#   pickle.dump(train_set_mini, outfile1,0)
#
# with open('test_set_mini.pkl', 'w') as outfile2:
#   pickle.dump(test_set_mini, outfile2,0)


# with open(str('train_mini.pkl'), 'rb') as f:
#     train_set = pickle.load(f)
# word = pickle.load(open("train_mini.pkl", 'r'), encoding='utf-8')

# with open('train_mini.pkl', 'r') as f3:
#     data_dict = pickle.load(f3)

# with open('test_mini.pkl', 'rb') as f1:
#     test_set = pickle.load(f1)
#
# with open('cate.pkl', 'rb') as f2:
#     cate_list = pickle.load(f2)
#
# c = DataFrame(cate_list)
# cate_count = len(c.drop_duplicates())
# user_count, item_count, cate_count = pickle.load(f)

