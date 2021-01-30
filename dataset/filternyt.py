# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np


class FilterNYTData(Dataset):

    def __init__(self, root_path, train=True):
        # 如果是训练集，则到train文件夹下，否则去test文件夹下
        if train:
            path = os.path.join(root_path, 'train/')
            print('loading train data')
        else:
            path = os.path.join(root_path, 'test/')
            print('loading test data')
        # 读取npy文件
        self.labels = np.load(path + 'labels.npy')
        self.x = np.load(path + 'bags_feature.npy',allow_pickle=True)  #转变为Python3
        self.x = zip(self.x, self.labels)
        self.x = list(self.x) # 转变为Python3

        print('loading finish')

    # 返回下标对应的数据
    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    # 数据长度
    def __len__(self):
        return len(self.x)

# FilterNYT文件的读取
class FilterNYTLoad(object):
    '''
    load and preprocess data
    '''
    def __init__(self, root_path, max_len=80, limit=50, pos_dim=5, pad=1):

        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.pos_dim = pos_dim
        self.pad = pad
        # vector.txt每行代表单词所对应的向量，行下标就是单词对应的位置,
        # dict.txt 每行代表单词，行下标就是单词对应的位置, vector.txt和dict.txt通过行下标一一对应
        # train/train.txt训练集数据，test/test测试集数据
        self.w2v_path = os.path.join(root_path, 'vector.txt')
        self.word_path = os.path.join(root_path, 'dict.txt')
        self.train_path = os.path.join(root_path, 'train', 'train.txt')
        self.test_path = os.path.join(root_path, 'test', 'test.txt')

        print('loading start....')
        '''
        self.w2v是所有单词对应的向量数组，[[0,0,0,0,0][...],[....]....],
            不过开头是一个【0,0,0,0,0】数组，结尾是个5位元素的一维数组，数值是【-1.0,1.0】的随机数
        self.word2id是所有单词和下标的字典，{‘单词1’:0，'单词2',1,.....}
        self.id2word是所有下标和单词的字典，{‘0’:单词1，'1',单词2,.....}
        self.p1_2v：一个self.pos_dim的全是0的一维数组，后面接着一个生成(self.limit * 2 + 1) * self.pos_dim的二维数组，数值范围为【-1.0,1.0】
        如：[[ 0.00  0.00 0.00 0.00 0.00]
            [-2.57  -7.90  3.62  1.96 -9.49]
            [-8.45  3.34  -5.14  6.29  9.01]...]
        
        self.p2_2v：同self.p1_2v，两者都是随机矩阵
        '''
        self.w2v, self.word2id, self.id2word = self.load_w2v()
        self.p1_2v, self.p2_2v = self.load_p2v()

        # np.save(os.path.join(self.root_path, 'w2v.npy'), self.w2v)
        # np.save(os.path.join(self.root_path, 'p1_2v.npy'), self.p1_2v)
        # np.save(os.path.join(self.root_path, 'p2_2v.npy'), self.p2_2v)

        print("parsing train text...")
        '''
        self.bags_feature中每个bag中的数据：
            entities：:[实体1，实体2]
            num：该bag中的句子个数
            sentences：该bag中的句子的向量的集合,[[句子1],[句子2],....]
            positions: 该bag中的每行句子的相对两个实体的向量位置,存的位置的集合，[[pf1,pf2],[pf1,pf2],....]
            entitiesPos： 该bag中的每行句子的前两个是实体的位置，存的位置加1后排序的集合，[[24,26],[15,17],....]
            masks：该bag中的每行句子的换算得来的mask集合，[[句子1的mask],[句子2的mask],....]
            positions中的每个[pf1,pf2]，pf1表示相对实体1的向量位置数组，pf2表示相对实体1的向量位置数组
            sentences,positions,masks中的每个句子向量，相对位置向量数组，mask不足self.max_len + 2 * self.pad的都在后面用0补足了
        self.labels：每个bag中的第二行的label数组（前四个数）记为rel,如[bag1的rel,bag2的rel,....]
                    例rel为[1,-1,-1,-1]
        '''
        self.bags_feature, self.labels = self.parse_sen(self.train_path)
        np.save(os.path.join(self.root_path, 'train', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'train', 'labels.npy'), self.labels)

        print("parsing test text...")
        # 同上
        self.bags_feature, self.labels = self.parse_sen(self.test_path)
        np.save(os.path.join(self.root_path, 'test', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'test', 'labels.npy'), self.labels)
        print('save finish!')
    # 生成两个随机数组，里面是-1.0到1.0的随机数
    def load_p2v(self):
        #生成一个(self.limit * 2 + 1) * self.pos_dim的二维数组，数值范围为【-1.0,1.0】如：[[-2.57  -7.90  3.62  1.96 -9.49]
        #                                                                           [-8.45  3.34  -5.14  6.29  9.01]...]
        pos1_vec = np.asarray(np.random.uniform(low=-1.0, high=1.0, size=(self.limit * 2 + 1, self.pos_dim)), dtype=np.float32)
        #将一个1*self.pos_dim的二维数组（里面全是0.00）与pos1_vec 竖直拼接，如[[ 0.00  0.00 0.00 0.00 0.00]
        #                                                               [-2.57 -7.90 3.62 1.96 -9.49]....]
        pos1_vec = np.vstack((np.zeros((1, self.pos_dim)), pos1_vec))
        #pos2_vec同上
        pos2_vec = np.asarray(np.random.uniform(low=-1.0, high=1.0, size=(self.limit * 2 + 1, self.pos_dim)), dtype=np.float32)
        pos2_vec = np.vstack((np.zeros((1, self.pos_dim)), pos2_vec))
        # 返回的是数组，如[[ 0.00  0.00 0.00 0.00 0.00]
        #               [-2.57 -7.90 3.62 1.96 -9.49]....]
        return pos1_vec, pos2_vec

    def load_w2v(self):
        '''
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
            : BLANK for the max len sentence
        '''
        wordlist = []
        vecs = []
        # 存单词，一行一个。第一个是'BLANK'，最后一个是'UNK'
        wordlist.append('BLANK')
        wordlist.extend([word.strip('\n') for word in open(self.word_path)])
        # vecs存单词所对应的向量，1*50数组，wordlist是单词
        for line in open(self.w2v_path):
            line = line.strip('\n').split()
            vec = list(map(float, line))
            vecs.append(vec)
        # 单词所代表的向量的长度
        dim = len(vecs[0])
        # 在vecs中0位置插入一个【0,0,0,0,0】数组
        vecs.insert(0, np.zeros(dim))
        wordlist.append('UNK')
        # 在vecs后面加入一个5位元素的一维数组，数值是【-1.0,1.0】的随机数
        vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))
        # rng = np.random.RandomState(3435)
        # vecs.append(rng.uniform(low=-0.5, high=0.5, size=dim))
        # i是单词的下标，j是单词，word2id是{"单词"：下标, .....},id2word是{"下标"：单词, .....}
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word

    # 对文件进行预处理
    def parse_sen(self, path):
        '''
        parse the records in data
        '''
        all_sens =[]
        all_labels =[]
        f = open(path)
        while 1:
            #每次循环，针对的是一个bag
            line = f.readline()
            if not line:
                break
            # entities :[实体1，实体2]
            entities = list(map(int, line.split(' ')))
            line = f.readline()
            bagLabel = line.split(' ')
            # rel存标签，如：[1 -1 -1 -1]
            rel = list(map(int, bagLabel[0:-1]))
            # 每个bag的第二行的最后一个数，代表这个bag中句子的个数
            num = int(bagLabel[-1])
            positions = []
            sentences = []
            entitiesPos = []
            masks = []
            for i in range(0, num):
                # 每行句子
                sent = f.readline().split(' ')
                # 每行句子的前两个是实体的位置
                positions.append(list(map(int, sent[0:2])))
                # 每行句子的前两个是实体的位置，实体的位置加1，然后排序
                epos = list(map(lambda x: int(x) + 1, sent[0:2]))
                epos.sort()
                # 通过实体的位置确定mask中的值
                mask = [1] * (epos[0] + 1)
                mask += [2] * (epos[1] - epos[0])
                mask += [3] * (len(sent[2:-1]) - epos[1])
                #
                entitiesPos.append(epos)
                # 每行句子的第二位开始到最后才是句子的向量
                sentences.append(list(map(int, sent[2:-1])))
                # 每行句子对应的mask
                masks.append(mask)
            '''
            每个bag中的数据：
            entities：:[实体1，实体2]
            num：该bag中的句子个数
            sentences：该bag中的句子的向量的集合,[[句子1],[句子2],....]
            positions: 该bag中的每行句子的前两个是实体的位置,存的位置的集合，[[25,23],[16,14],....]
            entitiesPos： 该bag中的每行句子的前两个是实体的位置，存的位置加1后排序的集合，[[24,26],[15,17],....]
            masks：该bag中的每行句子的换算得来的mask集合，[[句子1的mask],[句子2的mask],....]
            '''
            bag = [entities, num, sentences, positions, entitiesPos, masks]
            # 每个bag中的第二行的标签label数组是rels
            all_labels.append(rel)
            all_sens += [bag]

        f.close()
        '''
        将每个bag中的数据更新为下面：
        entities：:[实体1，实体2]
        num：该bag中的句子个数
        sentences：该bag中的句子的向量的集合,[[句子1],[句子2],....]
        positions: 该bag中的每行句子的相对两个实体的向量位置,存的位置的集合，[[pf1,pf2],[pf1,pf2],....]
        entitiesPos： 该bag中的每行句子的前两个是实体的位置，存的位置加1后排序的集合，[[24,26],[15,17],....]
        masks：该bag中的每行句子的换算得来的mask集合，[[句子1的mask],[句子2的mask],....]
        positions中的每个[pf1,pf2]，pf1表示相对实体1的向量位置数组，pf2表示相对实体1的向量位置数组
        sentences,positions,masks中的每个句子向量，相对位置向量数组，mask不足self.max_len + 2 * self.pad的都在后面用0补足了
        '''
        bags_feature = self.get_sentence_feature(all_sens)

        return bags_feature, all_labels

    def get_sentence_feature(self, bags):
        '''
        : word embedding
        : postion embedding
        return:
        sen list
        pos_left
        pos_right
        '''
        update_bags = []
        '''
        每个bag中的数据：
        entities：:[实体1，实体2]
        num：该bag中的句子个数
        sentences：该bag中的句子的向量的集合,[[句子1],[句子2],....]
        positions: 该bag中的每行句子的前两个是实体的位置,存的位置的集合，[[25,23],[16,14],....]
        entitiesPos： 该bag中的每行句子的前两个是实体的位置，存的位置加1后排序的集合，[[24,26],[15,17],....]
        masks：该bag中的每行句子的换算得来的mask集合，[[句子1的mask],[句子2的mask],....]
        '''
        for bag in bags:
            es, num, sens, pos, enPos, masks = bag
            new_sen = []
            new_pos = []
            new_masks = []

            for idx, sen in enumerate(sens):
                # 根据每个句子的长度、实体的位置、mask数组计算得到实体1的相对位置向量数组和实体2的相对位置向量数组
                # 同时将pf1, pf2, mask，不足self.max_len + 2 * self.pad的用0补足
                # _pos为【pf1, pf2】，_mask为mask补足后的数组
                _pos, _mask = self.get_pos_feature(len(sen), pos[idx], masks[idx])
                new_pos.append(_pos)
                new_masks.append(_mask)
                # 对每个句子向量进行长度固定，固定在self.max_len + 2 * self.pad
                # 对每个句子的向量数组sen，如果长度不满self.max_len + 2 * self.pad，则用0补齐，否则只取self.max_len + 2 * self.pad长度
                new_sen.append(self.get_pad_sen(sen))
            update_bags.append([es, num, new_sen, new_pos, enPos, new_masks])
        return update_bags

    # 一个sen，就是一个句子的向量数组
    # 如果sen数组长度不满self.max_len + 2 * self.pad，则在后面用[0]补齐
    # 如果sen数组的长度已经超过要求，则只保留
    def get_pad_sen(self, sen):
        '''
        padding the sentences
        '''
        # sen.insert(0, self.word2id['BLANK'])
        if len(sen) < self.max_len + 2 * self.pad:
            # self.word2id['BLANK']为 0
            sen += [self.word2id['BLANK']] * (self.max_len +2 * self.pad - len(sen))
        else:
            sen = sen[: self.max_len + 2 * self.pad]

        return sen

    # 编辑位置范围，将传进来的句子的长度、实体的位置、mask数组计算得到实体1的相对位置向量数组和实体2的相对位置向量数组
    # 同时将[pf1, pf2], mask，不足self.max_len + 2 * self.pad的用0补足
    def get_pos_feature(self, sen_len, ent_pos, mask):
        '''
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 1
        : -50 => 1
        : 50 => 101
        : >50: 102
        '''
        # sen_len:传进来的句子的长度
        # ent_pos:传进来的句子对应的实体的位置
        # mask:传进来的句子对应的mask
        def padding(x):
            if x < 1:
                return 1
            if x > self.limit * 2 + 1:
                return self.limit * 2 + 1
            return x
        # index 是从0开始的数组，如[ 0  1  2 .... sen_len-1]
        # sen_len 大于 self.max_len，则[ 0  1  2 .... self.max_len-1]
        if sen_len < self.max_len:
            index = np.arange(sen_len)
        else:
            index = np.arange(self.max_len)

        pf1 = []
        pf2 = []
        # 将
        pf1 += list(map(padding, index - ent_pos[0] + 2 + self.limit))
        pf2 += list(map(padding, index - ent_pos[1] + 2 + self.limit))

        if len(pf1) < self.max_len + 2 * self.pad:
            pf1 += [0] * (self.max_len + 2 * self.pad - len(pf1))
            pf2 += [0] * (self.max_len + 2 * self.pad - len(pf2))
            mask += [0] * (self.max_len + 2 * self.pad - len(mask))
        return [pf1, pf2], mask


if __name__ == "__main__":
    data = FilterNYTLoad('../dataset/FilterNYT/')

