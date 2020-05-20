## TODO
## configure의 값들 정리
## jupyter로 한 번 가볍게 돌려보기
## copyland에 올리기

import os
import sys
import glob
import time
import copy
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

import model_utils.configure as conf
from model import Graph2Seq
import utils

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main(mode, data_file_path):
    # mode = "train"
    # data_file_path = "data/train_data.json"
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    logging.info("conf = %s", conf)

    conf.source_length = conf.encoder_length = conf.decoder_length = (conf.graph_size + 2) * (conf.graph_size - 1) // 2
    epochs = conf.epochs

    model = Graph2Seq(mode=mode, conf=conf)

    # load data
    dataset = utils.ControllerDataset(data_file_path)
    queue = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=False, collate_fn=utils.collate_fn)

    if mode == "train":

        model.train()
        logging.info('Train data: {}'.format(len(queue)))
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.l2_reg)

        def train_step(train_queue, optimizer):
            objs = utils.AvgrageMeter()
            nll = utils.AvgrageMeter()
            for step, sample in enumerate(train_queue):
                fw_adjs = sample['fw_adjs'] 
                bw_adjs = sample['bw_adjs'] 
                operations = sample['operations'] 
                num_nodes = sample['num_nodes'] 
                sequence = sample['sequence'] 

                optimizer.zero_grad()
                log_prob, predicted_value = model(fw_adjs, bw_adjs, operations, num_nodes, targets=sequence)
                loss = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), sequence.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_bound)
                optimizer.step()
                
                n = sequence.size(0)
                objs.update(loss.data, n)
                nll.update(loss.data, n)

            return objs.avg, nll.avg
        
        for epoch in range(1, epochs + 1):
            loss, ce = train_step(queue, optimizer)
            logging.info("epoch %04d train loss %.6f ce %.6f", epoch, loss, ce)
        
        ## save trainable parameters
        torch.save(model.state_dict(), conf.model_path)
    
    if mode == "test":

        model.load_state_dict(torch.load(conf.model_path))
        model.eval()

        def test_step(test_queue):
            match = 0
            total = 0
            for step, sample in enumerate(test_queue):
                fw_adjs = sample['fw_adjs'] 
                bw_adjs = sample['bw_adjs'] 
                operations = sample['operations'] 
                num_nodes = sample['num_nodes'] 
                sequence = sample['sequence'] 

                log_prob, predicted_value = model(fw_adjs, bw_adjs, operations, num_nodes)
                
                match = torch.all(torch.equal(predicted_value, sequence), dim=1)
                total += len(num_nodes)
                """
                ## TODO calculate accuracy
                accuracy = predicted_value
                """

            accuracy = match / predicted_value.size(0)
            return accuracy

        logging.info('Test data: {}'.format(len(queue)))
        for epoch in range(1, epochs + 1):
            accuracy = test_step(queue)
            logging.info("epoch %04d accuracy %.6f", epoch, accuracy)


if __name__ == "__main__":
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", type=str, choices=["train", "test"])
    argparser.add_argument("-sample_size_per_layer", type=int, default=4, help="sample size at each layer")
    argparser.add_argument("-sample_layer_size", type=int, default=4, help="sample layer size")
    argparser.add_argument("-epochs", type=int, default=100, help="training epochs")
    argparser.add_argument("-learning_rate", type=float, default=conf.learning_rate, help="learning rate")
    argparser.add_argument("-word_embedding_dim", type=int, default=conf.word_embedding_dim, help="word embedding dim")
    argparser.add_argument("-hidden_layer_dim", type=int, default=conf.hidden_layer_dim)

    config = argparser.parse_args()


    config = conf
    mode = "train"
    conf.sample_layer_size = config.sample_layer_size
    conf.sample_size_per_layer = config.sample_size_per_layer
    conf.epochs = config.epochs
    conf.learning_rate = config.learning_rate
    conf.word_embedding_dim = config.word_embedding_dim
    conf.hidden_layer_dim = config.hidden_layer_dim

    """
    mode = "train"
    data_file_path = "data/train_data.json"
    main(mode, data_file_path)