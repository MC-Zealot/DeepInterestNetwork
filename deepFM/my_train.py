#-*- coding:utf-8 -*-
import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model
import pandas as pd
from pandas.core.frame import DataFrame
from utils import _eval

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512


def read_csv():
  train_set_mini = pd.read_csv('train_set_mini.csv', names=['user_id', 'viewed_item_id', 'item_id', 'label'],
                               dtype={'item_id': np.int},
                               converters={"viewed_item_id": lambda x: map(int, x.strip("[]").split(", "))})

  test_set_mini = pd.read_csv('test_set_mini.csv',
                              names=['user_id', 'viewed_item_id', 'click_and_not_click_item_id'],
                              dtype={'item_id': np.int},
                              converters={"viewed_item_id": lambda x: map(int, x.strip("[]").split(", ")),
                                          "click_and_not_click_item_id":lambda x: map(int, x.strip("()").split(", "))})
  cate_list = pd.read_csv('cate_list.csv',header=None)

  return train_set_mini, test_set_mini, cate_list


def get_count():
  cate_count = len(cate_list.drop_duplicates())
  user_count = 192403
  item_count = 63001
  return user_count, item_count, cate_count


def to_list(cate_list):
    """
    转换成模型的输入的样子
    :param cate_list:
    :return:
    """
    train_set_mini_arr = np.array(train_set_mini)
    train_set_mini_arr_list = train_set_mini_arr.tolist()

    test_set_mini_arr = np.array(test_set_mini)
    test_set_mini_list = test_set_mini_arr.tolist()

    cate_list_ids = []
    for idx, cate_id in cate_list.iterrows():
        t = cate_id[0]
        cate_list_ids.append(t)

    cate_list = cate_list_ids
    return train_set_mini_arr_list, test_set_mini_list, cate_list


train_set_mini, test_set_mini, cate_list = read_csv()

user_count, item_count, cate_count = get_count()

train_set, test_set, cate_list = to_list(cate_list)
print("train_set 0", train_set[0])
print("test_set 0", test_set[0])

best_auc = 0.0

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  model = Model(user_count, item_count, cate_count, cate_list)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  print('test_gauc: %.4f\t test_auc: %.4f\t best_auc: %.4f' % _eval(sess, model, test_set, test_batch_size, best_auc))
  sys.stdout.flush()
  lr = 0.001
  start_time = time.time()
  for _ in range(50):

    random.shuffle(train_set)

    epoch_size = round(len(train_set) / train_batch_size)
    loss_sum = 0.0
    for _, uij in DataInput(train_set, train_batch_size):
      loss = model.train(sess, uij, lr)
      loss_sum += loss

      if model.global_step.eval() % 1000 == 0:
        test_gauc, Auc, best_auc = _eval(sess, model, test_set, test_batch_size,best_auc)
        print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
              (model.global_epoch_step.eval(), model.global_step.eval(),
               loss_sum / 1000, test_gauc, Auc))
        sys.stdout.flush()
        loss_sum = 0.0

      if model.global_step.eval() % 336000 == 0:
        lr *= 0.9

    print('Epoch %d DONE\tCost time: %.2f' %
          (model.global_epoch_step.eval(), time.time()-start_time))
    sys.stdout.flush()
    model.global_epoch_step_op.eval()

  print('best test_gauc:', best_auc)
  sys.stdout.flush()
