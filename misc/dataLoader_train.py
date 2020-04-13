import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt

class train(data.Dataset): # torch wrapper
    def __init__(self, input_img_rcnn, input_ques_h5, input_json, negative_sample, num_val, data_split, emb_dim=300):

        print('DataLoader loading: %s' %data_split)
        print('Loading image feature from %s' %input_img_rcnn)

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
        elif data_split == 'val':
            s = total_num - num_val
            e = total_num
        else:
            s = 0
            e = total_num
            
        self.img_info = self.img_info[s:e]

        print('%s number of data: %d' %(data_split, e-s))
        # load the data.
        f = h5py.File(input_img_rcnn, 'r')
        self.rcnn = f['features'][...]
        self.rcnn_info = f['image_id'][...]
        f.close()

        #f = h5py.File(input_vgg_train, 'r')
        #self.vgg_train = f['features'][...]
        #self.vgg_train_info = f['image_ids'][...]
        #f.close()

        #f = h5py.File(input_vgg_val, 'r')
        #self.vgg_val = f['features'][...]
        #self.vgg_val_info = f['image_ids'][...]
        #f.close()

        print('Loading txt from %s' %input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_'+split][s:e]
        self.ans = f['ans_'+split][s:e]
        self.cap = f['cap_'+split][s:e]
        
        self.ques_len = f['ques_len_'+split][s:e]
        self.ans_len = f['ans_len_'+split][s:e]
        self.cap_len = f['cap_len_'+split][s:e]

        self.ans_ids = f['ans_index_'+split][s:e]
        self.opt_ids = f['opt_'+split][s:e]
        self.opt_list = f['opt_list_'+split][:]
        self.opt_len = f['opt_len_'+split][:]
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
        
        img_id = float(self.img_info[index]['imgId'])
        rcnn_info = self.rcnn_info.tolist()
        #vgg_train_info = self.vgg_train_info.tolist()
        #vgg_val_info = self.vgg_val_info.tolist()

        rcnn_index = rcnn_info.index(img_id)

        #if index >= 82783:
        #    vgg_index = vgg_val_info.index(img_id)
        #    img2 = torch.from_numpy(self.vgg_val[index-82783])
        #else:
        #    vgg_index = vgg_train_info.index(img_id)
        #    img2 = torch.from_numpy(self.vgg_train[index])

        #img2 = torch.from_numpy(self.vgg[vgg_index])
        img = torch.from_numpy(self.rcnn[rcnn_index])

        # get the image
        #img_id = self.img_info[index]['imgId']
        #id_index = self.train_idx[str(img_id)]
        #img = torch.from_numpy(self.imgs[id_index])

        #img2 = torch.from_numpy(self.imgs2[index])

        # get the history
        his = np.zeros((self.rnd, self.his_length))
        his[0,self.his_length-self.cap_len[index]:] = self.cap[index,:self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length+1))
        ans_target = np.zeros((self.rnd, self.ans_length+1))
        ques_ori = np.zeros((self.rnd, self.ques_length))

        opt_ans = np.zeros((self.rnd, self.negative_sample, self.ans_length+1))
        ans_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, self.negative_sample))

        ans_idx = np.zeros((self.rnd))
        opt_ans_idx = np.zeros((self.rnd, self.negative_sample))


        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i+1 < self.rnd:
                his[i+1, self.his_length-qa_len:self.his_length-a_len] = self.ques[index, i, :q_len]
                his[i+1, self.his_length-a_len:] = self.ans[index, i, :a_len]

            ques[i, self.ques_length-q_len:] = self.ques[index, i, :q_len]

            ques_ori[i, :q_len] = self.ques[index, i, :q_len]
            ans[i, 1:a_len+1] = self.ans[index, i, :a_len]
            ans[i, 0] = self.vocab_size

            ans_target[i, :a_len] = self.ans[index, i, :a_len]
            ans_target[i, a_len] = self.vocab_size
            ans_len[i] = self.ans_len[index, i]

            opt_ids = self.opt_ids[index, i] # since python start from 0
            # random select the negative samples.
            ans_idx[i] = opt_ids[self.ans_ids[index, i]]
            # exclude the gt index.
            opt_ids = np.delete(opt_ids, ans_idx[i], 0)
            random.shuffle(opt_ids)
            for j in range(self.negative_sample):
                ids = opt_ids[j]
                opt_ans_idx[i,j] = ids

                opt_len = self.opt_len[ids]

                opt_ans_len[i, j] = opt_len
                opt_ans[i, j, :opt_len] = self.opt_list[ids,:opt_len]
                opt_ans[i, j, opt_len] = self.vocab_size

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        ans_target = torch.from_numpy(ans_target)
        ques_ori = torch.from_numpy(ques_ori)
        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)
        opt_ans = torch.from_numpy(opt_ans)
        ans_idx = torch.from_numpy(ans_idx)
        opt_ans_idx = torch.from_numpy(opt_ans_idx)
        return img, his, ques, ans, ans_target, ans_len, ans_idx, ques_ori, \
                opt_ans, opt_ans_len, opt_ans_idx

    def __len__(self):
        return self.ques.shape[0]

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

        #f_idx = json.load(open("/data/wanghui/dataset/COCO_2014_Faster_rcnn/train_val_index_file.json", 'r'))
        #self.train_idx = f_idx['train']
        #self.val_idx = f_idx['val']

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

        #f = h5py.File(input_val_vgg, 'r')
        #self.vgg = f['features'][...]
        #self.vgg_info = f['image_ids'][...]
        #f.close()

        print('Loading txt from %s' %input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_'+split][s:e]
        self.ans = f['ans_'+split][s:e]
        self.cap = f['cap_'+split][s:e]

        self.ques_len = f['ques_len_'+split][s:e]
        self.ans_len = f['ans_len_'+split][s:e]
        self.cap_len = f['cap_len_'+split][s:e]

        self.ans_ids = f['ans_index_'+split][s:e]
        self.opt_ids = f['opt_'+split][s:e]
        self.opt_list = f['opt_list_'+split][:]
        self.opt_len = f['opt_len_'+split][:]
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
        img_id = float(self.img_info[index]['imgId'])
        rcnn_info = self.rcnn_info.tolist()
        #vgg_info = self.vgg_info.tolist()
        rcnn_index = rcnn_info.index(img_id)
        #vgg_index = vgg_info.index(img_id)



        #img2 = torch.from_numpy(self.vgg[vgg_index])
        img = torch.from_numpy(self.rcnn[rcnn_index])

        # get the history
        his = np.zeros((self.rnd, self.his_length))
        his[0,self.his_length-self.cap_len[index]:] = self.cap[index,:self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length+1))
        ans_target = np.zeros((self.rnd, self.ans_length+1))
        quesL = np.zeros((self.rnd, self.ques_length))

        opt_ans = np.zeros((self.rnd, 100, self.ans_length+1))
        ans_ids = np.zeros(self.rnd)
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

            ans_ids[i] = self.ans_ids[index, i] # since python start from 0
            opt_ids = self.opt_ids[index, i] # since python start from 0
            ans_len[i] = self.ans_len[index, i]
            ans_idx = self.ans_ids[index, i]

            for j, ids in enumerate(opt_ids):
                opt_len = self.opt_len[ids]
                opt_ans[i, j, 1:opt_len+1] = self.opt_list[ids,:opt_len]
                opt_ans[i, j, 0] = self.vocab_size

                opt_ans_target[i, j,:opt_len] = self.opt_list[ids,:opt_len]
                opt_ans_target[i, j,opt_len] = self.vocab_size
                opt_ans_len[i, j] = opt_len

        opt_ans = torch.from_numpy(opt_ans)
        opt_ans_target = torch.from_numpy(opt_ans_target)
        ans_ids = torch.from_numpy(ans_ids)

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        ans_target = torch.from_numpy(ans_target)
        quesL = torch.from_numpy(quesL)

        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)

        return img, his, ques, ans, ans_target, quesL, opt_ans, \
                    opt_ans_target, ans_ids, ans_len, opt_ans_len, img_id


    def __len__(self):
        return self.ques.shape[0]
