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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512



def read_csv():
  train_set_mini = pd.read_csv('train_set_mini.csv', names=['user_id', 'viewed_item_id', 'item_id', 'label'],dtype={'label':np.int},converters={"viewed_item_id": lambda x: x.strip("[]").split(", ")})
  test_set_mini = pd.read_csv('test_set_mini.csv', names=['user_id', 'viewed_item_id', 'item_id', 'label'],dtype={'label':np.int},converters={"viewed_item_id": lambda x: x.strip("[]").split(", ")})
  cate_list = pd.read_csv('cate_list.csv',header=None)

  return train_set_mini,test_set_mini,cate_list
#看一下item count 和 cate count
def get_count():
  all_item_id = []
  user_count = len(train_set_mini['user_id'].drop_duplicates())
  cate_count = len(cate_list.drop_duplicates())
  cate_count = len(cate_list.drop_duplicates())

  # train_set_mini['viewed_item_id_list'] = train_set_mini.viewed_item_id.apply(lambda x: x[1:-1].split(','))
  train_set_mini_item_id_list = train_set_mini.viewed_item_id
  for item_id_list in train_set_mini_item_id_list:
    all_item_id.extend(item_id_list)
  all_item_id_df = DataFrame(all_item_id)
  item_count = len(all_item_id_df.drop_duplicates())
  return user_count, item_count, cate_count


def to_list(cate_list):
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

train_set_mini_arr_list, test_set_mini_list, cate_list = to_list(cate_list)

train_set = train_set_mini_arr_list
test_set = test_set_mini_list
# with open('dataset.pkl', 'rb') as f:
#   train_set = pickle.load(f)
#   test_set = pickle.load(f)
#   cate_list = pickle.load(f)
#   user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0
def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
  score_p = score[:,0]
  score_n = score[:,1]
  #print "============== p ============="
  #print score_p
  #print "============== n ============="
  #print score_n
  score_arr = []
  for s in score_p.tolist():
    score_arr.append([0, 1, s])
  for s in score_n.tolist():
    score_arr.append([1, 0, s])
  return score_arr
def _eval(sess, model):
  auc_sum = 0.0
  score_arr = []
  for _, uij in DataInputTest(test_set, test_batch_size):
    auc_, score_ = model.eval(sess, uij)
    score_arr += _auc_arr(score_)
    auc_sum += auc_ * len(uij[0])
  test_gauc = auc_sum / len(test_set)
  Auc = calc_auc(score_arr)
  global best_auc
  if best_auc < test_gauc:
    best_auc = test_gauc
    model.save(sess, 'save_path/ckpt')
  return test_gauc, Auc


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  model = Model(user_count, item_count, cate_count, cate_list)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
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
        test_gauc, Auc = _eval(sess, model)
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
