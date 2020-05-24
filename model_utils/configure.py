## TODO graph_size, vocab_size, embedding_vocab_size 단어 정리

seed = 1

mode = "train"
data_file_path = "data/train_data.json"
model_path = "model_result/model.pth"
## data_file_path = "data/dev_data.json"
## data_file_path = "data/test_data.json"

graph_size = 7
degree_max_size = 5

hidden_layer_dim = 16
embedding_vocab_size = graph_size + 1 + 1
word_embedding_dim = hidden_layer_dim = 16
decoder_vocab_size = graph_size
vocab_size = embedding_vocab_size
adj_padding = embedding_vocab_size
length = int((graph_size+2)*(graph_size-1)/2)
hop_size = 6
graph_encode_direction = "bi" # "single" or "bi"

epochs = 2
batch_size = 50
l2_reg = 1e-4
# l2_lambda = 0.000001 # g2s
learning_rate = 0.001
grad_bound = 5.0
decoder_layers = 1
dropout = 0.1

# attention = True
concat = True

"""
########################################################
## Graph2Seq
train_data_path = "../data/no_cycle/train.data"
dev_data_path = "../data/no_cycle/dev.data"
test_data_path = "../data/no_cycle/test.data"

# the following are for the graph encoding method
weight_decay = 0.00001

feature_encode_type = "uni"
graph_encode_method = "max-pooling" # "lstm" or "max-pooling"



########################################################
## semiNAS
data = '/Users/user/Desktop/study/SemiNAS/nasbench/data'
nodes = 7

epochs = 1000
lr = 0.001
"""