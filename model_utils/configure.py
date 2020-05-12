mode = "train"
data_file_path = "data/train_data.json"
## data_file_path = "data/dev_data.json"
## data_file_path = "data/test_data.json"
graph_size = 7
hop_size = 4

model_path = "model_result/model.pth"
# word_embedding_dim = 16

degree_max_size = 5
graph_size = 7

## Graph2Seq
#########
train_data_path = "../data/no_cycle/train.data"
dev_data_path = "../data/no_cycle/dev.data"
test_data_path = "../data/no_cycle/test.data"

l2_lambda = 0.000001
learning_rate = 0.001
epochs = 100

attention = True

# the following are for the graph encoding method
weight_decay = 0.0000
hidden_layer_dim = 16
feature_encode_type = "uni"
# graph_encode_method = "max-pooling" # "lstm" or "max-pooling"
graph_encode_direction = "bi" # "single" or "bi"
concat = True

########################################################
## semiNAS
data = '/Users/user/Desktop/study/SemiNAS/nasbench/data'
output_dir = 'models'
seed = 1
nodes = 7
decoder_layers = 1
dropout = 0.1
l2_reg = 1e-4
vocab_size = 8
## epochs = 1000
batch_size = 2
lr = 0.001
grad_bound = 5.0