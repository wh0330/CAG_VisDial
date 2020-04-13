from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from misc.utils import repackage_hidden, scores_to_ranks
import misc.dataLoader_test as dl
import misc.model as model
from misc.CAGraph import _netE
import datetime
import h5py
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='', help='folder to output images and model checkpoints')
parser.add_argument('--input_img_rcnn', default='../data/detectron_features_faster_rcnn_x101_test.h5')
parser.add_argument('--input_ques_h5', default='../data/visdial_test_data.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='../data/visdial_test_params.json', help='path to dataset, now hdf5 file')
parser.add_argument('--model_path', default='../save/XXXXX.pth', help='folder to output images and model checkpoints')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--num_val',default=0,help='')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--save_iter', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--conv_feat_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--log_interval', type=int, default=50, help='how many iterations show the log info')
parser.add_argument('--decoder', default='D_CAG_T3_add_att_h(0.1,d=0.3,lr_10,NTU)_25', help='what decoder to use.')
parser.add_argument('--save_ranks_path', default='../ranks', help='folder to output images and model checkpoints')

opt = parser.parse_args()

opt.manualSeed = 216#random.randint(1, 10000) # fix seed
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################################################################
# Data Loader
####################################################################################

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    model_path = opt.model_path
    data_dir = opt.data_dir
    #input_img_vgg = opt.input_img_vgg
    input_img_rcnn = opt.input_img_rcnn
    input_ques_h5 = opt.input_ques_h5
    input_json = opt.input_json
    save_ranks_path = opt.save_ranks_path
    decoder = opt.decoder
    opt = checkpoint['opt']
    opt.batchSize = 1
    opt.data_dir = data_dir
    opt.model_path = model_path



dataset_val = dl.validate(input_val_rcnn=input_img_rcnn, input_ques_h5=input_ques_h5,
                input_json=input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'test')


dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
####################################################################################
# Build the Model
####################################################################################

n_words = dataset_val.vocab_size
ques_length = dataset_val.ques_length
ans_length = dataset_val.ans_length + 1
his_length = ques_length+dataset_val.ans_length
itow = dataset_val.itow
img_feat_size = 512

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW = model._netW(n_words, opt.ninp, opt.dropout, dataset_val.pretrained_wemb)
netD = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, n_words, opt.dropout)
#critD = model.nPairLoss(opt.nhid, 2)

if opt.model_path != '': # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    print('Loading model Success!')

if opt.cuda: # ship to cuda, if has GPU
    netW.cuda(), netE.cuda(), netD.cuda()
#    critD.cuda()

n_neg = 100
####################################################################################
# Some Functions
####################################################################################

def eval():
    netW.eval()
    netE.eval()
    netD.eval()

    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)
    opt_hidden = netD.init_hidden(opt.batchSize)

    i = 0
    ranks_json = []

    while i < len(dataloader_val):#len(1000):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answerLen, opt_answerLen, img_id, rounds  = data
        batch_size = question.size(0)
        image = image.view(-1, 36, 2048) #   image : batchx36x2048
        img_input.data.resize_(image.size()).copy_(image)

        #image2 = image2.view(-1, img_feat_size) #   image : 6272(128x7x7) x 512
        #img_input2.data.resize_(image2.size()).copy_(image2) #6272x512
        save_tmp = [[] for j in range(batch_size)]

        rnd = int(rounds-1)
        ques = question[:,rnd,:].t()
        his = history[:,:rnd+1,:].clone().view(-1, his_length).t()
        opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()

        ques_input.data.resize_(ques.size()).copy_(ques)
        his_input.data.resize_(his.size()).copy_(his)

        opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
        opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

        ques_emb = netW(ques_input, format = 'index')
        his_emb = netW(his_input, format = 'index')

        ques_hidden = repackage_hidden(ques_hidden, batch_size)
        hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

        featD, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                                    ques_hidden, hist_hidden, rnd+1)

        opt_ans_emb = netW(opt_ans_input, format = 'index')
        opt_hidden = repackage_hidden(opt_hidden, opt_ans_input.size(1))
        opt_feat = netD(opt_ans_emb, opt_ans_input, opt_hidden, n_words)
        opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

        featD = featD.view(-1, opt.ninp, 1)
        score = torch.bmm(opt_feat, featD)
        score = score.view(-1, 100)

        

        ranks = scores_to_ranks(score)


        ranks_json.append({
                "image_id": int(img_id),
                "round_id": int(rounds),
                "ranks": ranks.view(-1).tolist()
            })

        i += 1
        sys.stdout.write('Evaluating: {:d}/{:d}  \r' \
          .format(i, len(dataloader_val)))
        sys.stdout.flush()
        #pdb.set_trace()
    return ranks_json

####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize)
#img_input2 = torch.FloatTensor(opt.batchSize, 49, 512)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

# answer input
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize)
fake_ans_input = torch.FloatTensor(ques_length, opt.batchSize, n_words)
sample_ans_input = torch.LongTensor(1, opt.batchSize)

# answer index location.
opt_index = torch.LongTensor( opt.batchSize)
fake_index = torch.LongTensor(opt.batchSize)

batch_sample_idx = torch.LongTensor(opt.batchSize)
# answer len
fake_len = torch.LongTensor(opt.batchSize)

# noise
noise_input = torch.FloatTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)


if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    opt_ans_input = opt_ans_input.cuda()
    fake_ans_input, sample_ans_input = fake_ans_input.cuda(), sample_ans_input.cuda()
    opt_index, fake_index =  opt_index.cuda(), fake_index.cuda()
    #img_input2 = img_input2.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    gt_index = gt_index.cuda()


ques_input = Variable(ques_input)
img_input = Variable(img_input)
#img_input2 = Variable(img_input2)
his_input = Variable(his_input)

opt_ans_input = Variable(opt_ans_input)
fake_ans_input = Variable(fake_ans_input)
sample_ans_input = Variable(sample_ans_input)

opt_index = Variable(opt_index)
fake_index = Variable(fake_index)

fake_len = Variable(fake_len)
noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
gt_index = Variable(gt_index)

ranks_json = eval()

print("Writing ranks to {}".format(save_ranks_path))
os.makedirs(os.path.dirname(save_ranks_path), exist_ok=True)
json.dump(ranks_json, open('%s/%s_ranks.json' %(save_ranks_path, decoder), 'w'))
