import torch

source_file = "data/transformed_data.json"
train_data_file="data/train_data.json"
dev_data_file="data/dev_data.json"
test_data_file="data/test_data.json"

f = open(source_file, "r")
data = f.readlines()
n = len(data)
train_size, dev_size = int(0.6 * n) , int(0.2 * n)
test_size = n - train_size - dev_size
train_data, dev_data, test_data = torch.utils.data.random_split(data, [train_size, dev_size, test_size])
f.close()

with open(train_data_file, "w+") as ff :
    ff.writelines(train_data)
with open(dev_data_file, "w+") as ff :
    ff.writelines(dev_data)
with open(test_data_file, "w+") as ff :
    ff.writelines(test_data)