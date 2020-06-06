#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from input import DataInput, DataInputTest
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

def _eval(sess, model,test_set, test_batch_size, best_auc):
  auc_sum = 0.0
  score_arr = []
  for _, uij in DataInputTest(test_set, test_batch_size):
    auc_, score_ = model.eval(sess, uij)
    score_arr += _auc_arr(score_)
    auc_sum += auc_ * len(uij[0])
  test_gauc = auc_sum / len(test_set)
  Auc = calc_auc(score_arr)
  # global best_auc
  if best_auc < test_gauc:
    best_auc = test_gauc
    model.save(sess, 'save_path/ckpt')
  return test_gauc, Auc,best_auc

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


def get_count(cate_list):
  cate_count = len(cate_list.drop_duplicates())
  user_count = 192403
  item_count = 63001
  return user_count, item_count, cate_count


def to_list(train_set_mini, test_set_mini,cate_list):
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

def get_all_data():
    train_set_mini, test_set_mini, cate_list = read_csv()

    user_count, item_count, cate_count = get_count()

    train_set, test_set, cate_list = to_list(train_set_mini, test_set_mini,cate_list)
    return train_set, test_set, cate_list, user_count, item_count, cate_count