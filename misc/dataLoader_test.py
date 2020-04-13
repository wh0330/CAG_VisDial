import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt

class validate(data.Dataset): # torch wrapper
    def __init__(self, input_val_rcnn, input_ques_h5, input_json, negative_sample, num_val, data_split, emb_dim=300):

        print('DataLoader loading: %s' %data_split)
        print('Loading image feature from %s' %input_val_rcnn)

        if data_split == 'test':
            split = 'val'
        else:
            split = 'train' # train and val split both corresponding to 'train'

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_'+split]

        # get the data split.
        total_num = len(self.img_info)
        if data_split == 'train':
            s = 0
            e = total_num - num_val
            #self.tmp_index = self.train_idx
        elif data_split == 'val':
            s = total_num - num_val
            e = total_num
            #self.tmp_index = self.train_idx
        else:
            s = 0
            e = total_num
            #self.tmp_index = self.val_idx

        self.img_info = self.img_info[s:e]
        print('%s number of data: %d' %(data_split, e-s))

        # load the data.
        f = h5py.File(input_val_rcnn, 'r')
        self.rcnn = f['features'][...]
        self.rcnn_info = f['image_id'][...]
        f.close()

        print('Loading txt from %s' %input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_'+split][...]
        self.ans = f['ans_'+split][...]
        self.cap = f['cap_'+split][...]

        self.ques_len = f['ques_len_'+split][...]
        self.ans_len = f['ans_len_'+split][...]
        self.cap_len = f['cap_len_'+split][...]

        #self.ans_ids = f['ans_index_'+split][s:e]
        self.opt_ids = f['opt_'+split][...]
        self.opt_list = f['opt_list_'+split][...]
        self.opt_len = f['opt_len_'+split][...]
        self.rounds = f['rounds'][...]
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow)+1

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.rnd = 10
        self.negative_sample = negative_sample

        self.emb_dim = emb_dim
        self.init_pretrained_wemb(emb_dim)

    def init_pretrained_wemb(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open("/data/wanghui/dataset/Glove/glove.6B.300d.txt", "r")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.vocab_size+1, emb_dim), dtype=np.float32)
        for i, word in self.itow.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[int(i)] = embedding_v

        self.pretrained_wemb = embedding_mat

    def __getitem__(self, index):

        # get the image
        img_id1 = self.img_info[index]['imgId']
        img_id = float(self.img_info[index]['imgId'])
        rcnn_info = self.rcnn_info.tolist()
        rcnn_index = rcnn_info.index(img_id)

        img = torch.from_numpy(self.rcnn[rcnn_index])

        rounds = int(self.rounds[index])

        # get the history
        
        his = np.zeros((self.rnd, self.his_length))
        his[0,self.his_length-self.cap_len[index]:] = self.cap[index,:self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length+1))
        ans_target = np.zeros((self.rnd, self.ans_length+1))
        quesL = np.zeros((self.rnd, self.ques_length))

        opt_ans = np.zeros((self.rnd, 100, self.ans_length+1))
        #ans_ids = np.zeros(self.rnd)
        opt_ans_target = np.zeros((self.rnd, 100, self.ans_length+1))

        ans_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, 100))


        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i+1 < self.rnd:
                ques_ans = np.concatenate([self.ques[index, i, :q_len], self.ans[index, i, :a_len]])
                his[i+1, self.his_length-qa_len:] = ques_ans

            ques[i, self.ques_length-q_len:] = self.ques[index, i, :q_len]
            quesL[i, :q_len] = self.ques[index, i, :q_len]
            ans[i, 1:a_len+1] = self.ans[index, i, :a_len]
            ans[i, 0] = self.vocab_size

            ans_target[i, :a_len] = self.ans[index, i, :a_len]
            ans_target[i, a_len] = self.vocab_size

            opt_ids = self.opt_ids[index, i] # since python start from 0
            ans_len[i] = self.ans_len[index, i]

            for j, ids in enumerate(opt_ids):
                opt_len = self.opt_len[ids]
                opt_ans[i, j, 1:opt_len+1] = self.opt_list[ids,:opt_len]
                opt_ans[i, j, 0] = self.vocab_size

                opt_ans_target[i, j,:opt_len] = self.opt_list[ids,:opt_len]
                opt_ans_target[i, j,opt_len] = self.vocab_size
                opt_ans_len[i, j] = opt_len

        opt_ans = torch.from_numpy(opt_ans)
        opt_ans_target = torch.from_numpy(opt_ans_target)

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        ans_target = torch.from_numpy(ans_target)
        quesL = torch.from_numpy(quesL)

        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)
        
        

        return img, his, ques, ans, ans_target, quesL, opt_ans, \
                    opt_ans_target, ans_len, opt_ans_len, img_id1, rounds
        #return img, img2, img_id


    def __len__(self):
        return self.ques.shape[0]
