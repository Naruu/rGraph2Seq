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

import imp
imp.reload(conf)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main(mode):
    mode = "train"
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    logging.info("conf = %s", conf)

    conf.source_length = conf.encoder_length = conf.decoder_length = (conf.nodes + 2) * (conf.nodes - 1) // 2
    epochs = conf.epochs

    model = Graph2Seq(mode=mode, conf=conf)

    # load data
    dataset = utils.ControllerDataset(conf.data_file_path, conf.graph_size)
    queue = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True)

    ## model에 input과 target을 batch 만큼 전달
    ## 그러면 그때마다 decoder output과 target 사이의 차이가 나오겠지
    ## train이면 optimize 시키고
    ## test이면 통계를 내야겠지.


    if mode == "train":

        def train_step(train_queue, optimizer):
            objs = utils.AvgrageMeter()
            nll = utils.AvgrageMeter()
            print("train_queue {}".format(train_queue))
            for step, sample in enumerate(train_queue):
                fw_adjs = sample['fw_adjs'] 
                bw_adjs = sample['bw_adjs'] 
                operations = sample['operations'] 
                sequence = sample['sequence'] 

                optimizer.zero_grad()
                log_prob, predicted_value = model(fw_adjs, bw_adjs, operations, targets=sequence)
                loss = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), sequence.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_bound)
                optimizer.step()
                
                n = sequence.size(0)
                objs.update(loss.data, n)
                nll.update(loss.data, n)

            return objs.avg, nll.avg

        model.train()

        logging.info('Train data: {}'.format(len(queue)))
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.l2_reg)
        
        # for epoch in range(1, epochs + 1):
        for epoch in range(2):
            loss, ce = train_step(queue, optimizer)
            print("epoch %04d train loss %.6f ce %.6f", epoch, loss, ce)
            # logging.info("epoch %04d train loss %.6f ce %.6f", epoch, loss, ce)
        
        ## save trainable parameters
        torch.save(model.state_dict(), conf.model_path)
    
    if mode == "test":
        model.load_state_dict(torch.load(conf.model_path))
        model.eval()

        def test_step(test_queue):
            for step, sample in enumerate(test_queue):
                encoder_input = sample['encoder_input']
                decoder_target = sample['decoder_target']
                log_prob, predicted_value = model(encoder_input, targets=decoder_target)
                
                match = torch.all(torch.eq(predicted_value, decoder_target), dim=1)
                accuracy = match / encoder_input.size(1)
                """
                ## TODO calculate accuracy
                accuracy = predicted_value
                """

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
    """

    config = conf
    mode = "train"
    conf.sample_layer_size = config.sample_layer_size
    conf.sample_size_per_layer = config.sample_size_per_layer
    conf.epochs = config.epochs
    conf.learning_rate = config.learning_rate
    conf.word_embedding_dim = config.word_embedding_dim
    conf.hidden_layer_dim = config.hidden_layer_dim

    main(mode)