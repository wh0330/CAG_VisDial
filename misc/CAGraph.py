import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from misc.fc import FCNet


class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.neighbourhood_size = 8
        self.img_embed = nn.Linear(2048, nhid )

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.his_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.Wq_1 = nn.Linear(self.nhid, self.nhid) # attention
        self.Wh_1 = nn.Linear(self.nhid, self.nhid)
        self.Wa_1 = nn.Linear(self.nhid, 1)

        self.ref_att = FCNet([self.nhid, self.nhid])
        self.Wqt = nn.Linear(self.nhid, 1)

        self.ref_att2 = FCNet([self.nhid, self.nhid])
        self.Wqt2 = nn.Linear(self.nhid, 1)

        self.ref_att3 = FCNet([self.nhid, self.nhid])
        self.Wqt3 = nn.Linear(self.nhid, 1)
        
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, self.nhid)

        self.W3 = nn.Linear(self.nhid*2, self.nhid)
        self.W4 = nn.Linear(self.ninp, self.nhid)
        self.W5 = nn.Linear(self.nhid*2, self.nhid)

        self.W6 = nn.Linear(self.nhid*2, self.nhid)
        self.W7 = nn.Linear(self.ninp, self.nhid)
        self.W8 = nn.Linear(self.nhid*2, self.nhid)

        self.W9 = nn.Linear(self.nhid*2, self.nhid)
        self.W10 = nn.Linear(self.nhid, self.nhid)
        self.W11 = nn.Linear(self.nhid, 1)
 
        self.fc1 = nn.Linear(self.nhid*4, self.ninp)

    def forward(self, ques_emb, his_emb, img, ques_hidden, his_hidden, rnd):

        b = img.size(0)
        L = ques_emb.size(0)

        r_feat = img.contiguous().view(-1,36,2048)
        r_feat = r_feat/(r_feat.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(r_feat)
        rcnn_feat = F.tanh(self.img_embed(r_feat))

        ques_feat1, ques_hidden = self.ques_rnn(ques_emb, ques_hidden)
        ques_feat = ques_feat1[-1] 

        his_feat, his_hidden = self.his_rnn(his_emb, his_hidden) 
        his_feat = his_feat[-1]
        
        ###################### history attention ##############################
        ques_emb_1 = self.Wq_1(ques_feat).view(-1, 1, self.nhid)
        his_emb_1 = self.Wh_1(his_feat).view(-1, rnd, self.nhid)
        atten_emb_1 = F.tanh(his_emb_1 + ques_emb_1.expand_as(his_emb_1))
        his_atten_weight = F.softmax(self.Wa_1(F.dropout(atten_emb_1, self.d, training=self.training).view(-1, self.nhid)).view(-1, rnd))
        h_emb = torch.bmm(his_atten_weight.view(-1, 1, rnd), his_feat.view(-1, rnd, self.nhid))

        ####################### graph constrution ############################
        graph = torch.cat((rcnn_feat, h_emb.expand_as(rcnn_feat)), dim=2)

        ####################### T == 1 #######################################
        ####################### question command #############################
        q_norm = F.normalize(self.ref_att(ques_feat1.transpose(0,1)), p=2, dim=-1)
        at = F.softmax(self.Wqt(F.dropout(q_norm, self.d, training=self.training).view(-1, self.nhid)).view(-1, L))
        q_c = torch.bmm(at.view(-1, 1, L), ques_emb.transpose(0,1)).squeeze(1)
        ####################### belief_matrix #############################
        mes_b = self.W3(graph) * self.W4(q_c).unsqueeze(1)
        belief_mat = torch.bmm(self.W5(graph), mes_b.transpose(1, 2))
        belief = F.softmax(belief_mat, dim=2)
        ####################### message passing ###########################
        mes = self.W6(graph) * self.W7(q_c).unsqueeze(1)
        sum_mes = self._create_neighbourhood(mes, belief_mat, self.neighbourhood_size)
        context_1 = self.W8(torch.cat((h_emb.expand_as(rcnn_feat), sum_mes), dim=2))
        graph2 = torch.cat((rcnn_feat, context_1), dim=2)

        ####################### T == 2 #######################################
        ####################### question command #############################
        q_norm2 = F.normalize(self.ref_att2(ques_feat1.transpose(0,1)), p=2, dim=-1)
        at2 = F.softmax(self.Wqt2(F.dropout(q_norm2, self.d, training=self.training).view(-1, self.nhid)).view(-1, L))
        q_c2 = torch.bmm(at2.view(-1, 1, L), ques_emb.transpose(0,1)).squeeze(1)
        ####################### belief_matrix #############################
        mes_b2 = self.W3(graph2) * self.W4(q_c2).unsqueeze(1)
        belief_mat2 = torch.bmm(self.W5(graph2), mes_b2.transpose(1, 2))
        belief2 = F.softmax(belief_mat2, dim=2)
        ####################### message passing ###########################
        mes2 = self.W6(graph2) * self.W7(q_c2).unsqueeze(1)
        sum_mes2 = self._create_neighbourhood(mes2, belief_mat2, self.neighbourhood_size)
        context_2 = self.W8(torch.cat((context_1, sum_mes2), dim=2))
        graph3 = torch.cat((rcnn_feat, context_2), dim=2)

        ####################### T == 3 #######################################
        ####################### question command #############################
        q_norm3 = F.normalize(self.ref_att3(ques_feat1.transpose(0,1)), p=2, dim=-1)
        at3 = F.softmax(self.Wqt3(F.dropout(q_norm3, self.d, training=self.training).view(-1, self.nhid)).view(-1, L))
        q_c3 = torch.bmm(at3.view(-1, 1, L), ques_emb.transpose(0,1)).squeeze(1)
        ####################### belief_matrix #############################
        mes_b3 = self.W3(graph3) * self.W4(q_c3).unsqueeze(1)
        belief_mat3 = torch.bmm(self.W5(graph3), mes_b3.transpose(1, 2))
        belief3 = F.softmax(belief_mat3, dim=2)
        ####################### message passing ###########################
        mes3 = self.W6(graph3) * self.W7(q_c3).unsqueeze(1)
        sum_mes3 = self._create_neighbourhood(mes3, belief_mat3, self.neighbourhood_size)
        context_3 = self.W8(torch.cat((context_2, sum_mes3), dim=2))
        graph4 = torch.cat((rcnn_feat, context_3), dim=2)

        ####################### Graph Attention ##############################
        g2_emb = self.W9(graph4).view(-1, 36, self.nhid)
        q_emb = self.W10(ques_feat).view(-1, 1, self.nhid)
        att_gq_emb = F.tanh(g2_emb + q_emb.expand_as(g2_emb))
        graph_att = F.softmax(self.W11(F.dropout(att_gq_emb, self.d, training=self.training).view(-1, self.nhid)).view(-1, 36)).unsqueeze(1)
        graph_emb = torch.bmm(graph_att, graph4)

        ##################### Multi-modal Fusion ############################
        concat_feat = torch.cat((graph_emb.view(-1, 2*self.nhid), ques_feat.view(-1, self.nhid), h_emb.view(-1, self.nhid)),1)
        final_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))
        #pdb.set_trace()

        return final_feat,ques_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def _create_neighbourhood_mes(self, message, top_ind):
        '''
        ## Inputs:
        - message (batch_size, K, feat_dim)
        - top_ind (batch_size, K, neighbourhood_size)
        ## Returns:
        - neighbourhood_message (batch_size, K, neighbourhood_size, feat_dim)
        '''

        batch_size = message.size(0)
        K = message.size(1)
        feat_dim = message.size(2)
        neighbourhood_size = top_ind.size(-1)
        message = message.unsqueeze(1).expand(batch_size, K, K, feat_dim)
        idx = top_ind.unsqueeze(-1).expand(batch_size,
                                           K, neighbourhood_size, feat_dim)
        return torch.gather(message, dim=2, index=idx)

    def _create_neighbourhood(self, message, belief_matrix, neighbourhood_size):

        '''

        Creates a neighbourhood system for each graph node/image object

        ## Inputs:
        - message (batch_size, K, feat_dim): input message features
        - adjacency_matrix (batch_size, K, K): learned adjacency matrix
        - neighbourhood_size (int)
        - weight (bool): specify if the features should be weighted by the adjacency matrix values

        ## Returns:
        - sum_messages (batch_size, K, neighbourhood_size, feat_dim)
        '''

        # Number of graph nodes
        K = message.size(1)
        #pdb.set_trace()

        # extract top k neighbours for each node and normalise
        top_k, top_ind = torch.topk(belief_matrix, k=neighbourhood_size, dim=-1, sorted=False)
        top_k = torch.stack([F.softmax(top_k[:, k]) for k in range(K)]).transpose(0, 1)  # (batch_size, K, neighbourhood_size)

        # extract top k features
        neighbourhood_mes = self._create_neighbourhood_mes(message, top_ind)

        sum_mes = torch.sum(top_k.unsqueeze(-1)*neighbourhood_mes, dim=2)

        return sum_mes