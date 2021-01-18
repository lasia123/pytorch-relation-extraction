# -*- coding: utf-8 -*-

from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNN_ONE(BasicModule):
    '''
    Zeng 2015 DS PCNN
    '''
    def __init__(self, opt):
        super(PCNN_ONE, self).__init__()
        # 初始的opt在config.py中可以看到数据组成,在main_mil.py中加入了新的数据
        self.opt = opt

        self.model_name = 'PCNN_ONE'
        '''输入是一个下标的列表，输出是对应的词嵌入
        创建矩阵，如：self.opt.vocab_size * self.opt.word_dim大小的，即vocab_size个词，每个词word_dim维
        self.word_embs:存文本的对应向量
        self.pos1_embs:存相对实体1的相对位置，各个词离entity的距离进行编码
        self.pos2_embs:存相对实体2的相对位置，各个词离entity的距离进行编码
        '''
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        
        # 数值大小是由于需要将位置特征和文本特征拼接，每个词文本后要拼接相对entity1，entity2的位置
        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2
        '''设置过滤器的大小
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        in_channels:输入的通道数
        out_channels:由卷积产生的通道数
        kernel_size:卷积核尺寸
        padding=0:(补0)：控制zero-padding的数目。
        self.convs :ModuleList(
                              (0): Conv2d(1, 230, kernel_size=(3, 60), stride=(1, 1), padding=(1, 0))
                                )
        '''
        # for more filter size
        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.opt.filters])

        all_filter_num = self.opt.filters_num * len(self.opt.filters)

        if self.opt.use_pcnn:
            all_filter_num = all_filter_num * 3
            '''
            新建一个tensor([[0., 0., 0.],
                           [1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])
            '''
            masks = torch.FloatTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            if self.opt.use_gpu:
                masks = masks.cuda()
            '''创一个矩阵
            mask_embedding.weight初始值：tensor([[ 0.4245,  1.1851, -0.1532],
                                                [ 0.8619, -0.8674, -1.0419],
                                                [-0.7182, -2.2017,  2.9141],
                                                [-0.2915, -1.6366, -0.8132]], requires_grad=True)
            将masks的值复制到mask_embedding.weight的data上，同时将requires_grad改为False
            '''
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False
        # 把拥有all_filter_num种特征值的那种样本输入转变成拥有self.opt.rel_num种特征值的输出，
        # self.linear：Linear(in_features=230, out_features=53, bias=True)
        self.linear = nn.Linear(all_filter_num, self.opt.rel_num)
        # 在不同的训练过程中随机扔掉一部分神经元,self.opt.drop_out是每个元素被保留下来的概率，初始中设定为0.5
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_model_weight()
        self.init_word_emb()
        
    def init_model_weight(self):
        '''
        use xavier to init
        初始化
        '''
        for conv in self.convs:
            # 是一个服从均匀分布的Glorot初始化器
            nn.init.xavier_uniform_(conv.weight)
            # 用值0.0填充向量conv.bias
            nn.init.constant_(conv.bias, 0.0)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)
   
    # 处理单词对应的向量表、实体1、2对应的随机矩阵表，并赋值到opt中
    def init_word_emb(self):

        def p_2norm(path):
            '''
            从np.load(path)创建一个张量，返回的张量和np.load(path)共享同一内存。对张量的修改将反映在np.load(path)中，反之亦然
            如果opt的norm_emb为真，则v变成v÷(v.norm(2, 1).unsqueeze(1))
            v.norm(2, 1):对每行数据求 2 范数,即每个数平方后相加，再开方，最终得到包含每行的2范数的一维数组
            v.norm(2, 1).unsqueeze(1):增加一维将其变为二维数组
            v[v != v] = 0.0 : v中有数据不同的置为0
            :param path:
            :return:处理后的v
            '''
            v = torch.from_numpy(np.load(path))
            if self.opt.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.opt.w2v_path)
        p1_2v = p_2norm(self.opt.p1_2v_path)
        p2_2v = p_2norm(self.opt.p2_2v_path)

        # 如果使用gpu则对数据进行格式处理再赋值，否则直接赋值
        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
            self.pos1_embs.weight.data.copy_(p1_2v.cuda())
            self.pos2_embs.weight.data.copy_(p2_2v.cuda())
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)

    def mask_piece_pooling(self, x, mask):
        '''
        refer: https://github.com/thunlp/OpenNRE
        A fast piecewise pooling using mask
        '''
        x = x.unsqueeze(-1).permute(0, 2, 1, -1)
        masks = self.mask_embedding(mask).unsqueeze(-2) * 100
        x = masks.float() + x
        x = torch.max(x, 1)[0] - torch.FloatTensor([100]).cuda()
        x = x.view(-1, x.size(1) * x.size(2))
        return x

    def piece_max_pooling(self, x, insPool):
        '''
        old version piecewise
        '''
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []
        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()  # all_filter_num * max_len
            pool = split_pool[i].squeeze().data    # 2
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)          # all_filter_num * 1
            seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)    # 1 * 3all_filter_num
            batch_res.append(piece_max_pool)

        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.opt.filters_num
        return out

    def forward(self, x, train=False):
        '''x是bag的数据处理后的tensor，如对于某一个bag中的数据，x的格式则为 tensor(bag的数据, device='cuda:0')
        bags_feature中的一个bag为
                es:[0, 0]
                num:只针对这行，‘train’例：m.010039	m.01vwm8g	NA	99161,292483
                    对第4个进行操作，如果有逗号则切分；最后统计有多少个这个,就是num
                    (bag 里面一样，不变)
                new_sen:句子的数组,数据不变，数组后面用0填充了，如[[0,2,4,525,6,112,15099,....,0,0,0]]
                new_pos:[相对实体1的位置,相对实体2的位置]的数组,数据不变，数组后面用0填充了，如[[84,83,82,81,80,79,....,0,0,0],
                                                                           [50,49,48,47,46,45,....,0,0,0]]
                new_entPos:实体1和实体2在词表的下标的位置且每个值都加1，升序，[[1,35]]
                new_masks:最后的句子的数组，据不变，数组后面用0填充了,即位置如[[1,2,2,2,2,2,2,2,2,....,0,0,0,0]]
         '''
        insEnt, _, insX, insPFs, insPool, insMasks = x
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]

        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)
        x = x.unsqueeze(1)
        x = self.dropout(x)

        x = [conv(x).squeeze(3) for conv in self.convs]
        if self.opt.use_pcnn:
            x = [self.mask_piece_pooling(i, insMasks) for i in x]
            # x = [self.piece_max_pooling(i, insPool) for i in x]
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1).tanh()
        x = self.dropout(x)
        x = self.linear(x)

        return x
