import pandas as pd
import numpy as np
train_set_mini = pd.read_csv('train_set_mini.csv', names=['user_id', 'viewed_item_id', 'item_id', 'label'], dtype={ 'item_id':np.int}, converters={"viewed_item_id": lambda x: map(int, x.strip("[]").split(", "))})

test_set_mini = pd.read_csv('train_set_mini.csv', names=['user_id', 'viewed_item_id', 'item_id_viewed', 'item_id_not_viewed'], dtype={ 'item_id':np.int}, converters={"viewed_item_id": lambda x: map(int, x.strip("[]").split(", "))})

print(train_set_mini.dtypes)
print(test_set_mini.dtypes)