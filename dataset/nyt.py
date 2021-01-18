# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np


class NYTData(Dataset):

    def __init__(self, root_path, train=True):
        #如果是训练集，则到train文件夹下，否则去test文件夹下
        if train:
            path = os.path.join(root_path, 'train/')
            print('loading train data')
        else:
            path = os.path.join(root_path, 'test/')
            print('loading test data')
        #读取npy文件
        self.labels = np.load(path + 'labels.npy')
        self.x = np.load(path + 'bags_feature.npy')
        self.x = list(zip(self.x, self.labels))

        print('loading finish')
    #返回下标对应的数据
    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]
    #数据长度 
    def __len__(self):
        return len(self.x)

# NYT文件的读取
class NYTLoad(object):
    '''
    load and preprocess data
    '''
    def __init__(self, root_path, max_len=80, limit=50, pos_dim=5, pad=1):

        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.pos_dim = pos_dim
        self.pad = pad
        
        # w2v_path 词语对应表里的, bags_train.txt训练集数据，bags_test测试集数据
        self.w2v_path = os.path.join(root_path, 'vector.txt')
        self.train_path = os.path.join(root_path, 'bags_train.txt')
        self.test_path = os.path.join(root_path, 'bags_test.txt')

        print('loading start....')
        # self.w2v是所有单词对应的向量数组，[[...],[....]....]
        # self.word2id是所有单词和下标的字典，{‘单词1’:0，'单词2',1,.....}
        # self.id2word是所有下标和单词的字典，{‘0’:单词1，'1',单词2,.....}
        self.w2v, self.word2id, self.id2word = self.load_w2v()
        self.p1_2v, self.p2_2v = self.load_p2v()
        
        # self.w2v是所有单词对应的向量数组，[[...],[....]....]
        np.save(os.path.join(self.root_path, 'w2v.npy'), self.w2v)
        # self.p1_2v是（limit * 2 + 1）*pos_dim的二维矩阵，里面是-1.0到1.0的随机数
        np.save(os.path.join(self.root_path, 'p1_2v.npy'), self.p1_2v)
        # self.p2_2v是（limit * 2 + 1）*pos_dim的二维矩阵，里面是-1.0到1.0的随机数
        np.save(os.path.join(self.root_path, 'p2_2v.npy'), self.p2_2v)
        
        # 读取训练集的数据并做好预处理
        print("parsing train text...")
        '''  bags_feature中的一个bag为
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
        #parse_sen: 最终全部的rels的数组，如[[0, -1, -1, -1], [0, -1, -1, -1],....]
        #           rels label的总和不足4个则用-1补足，超过4个则只取前4个,如[0, -1, -1, -1]
        self.bags_feature, self.labels = self.parse_sen(self.train_path, 'train')
        np.save(os.path.join(self.root_path, 'train', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'train', 'labels.npy'), self.labels)
        
        #测试集数据预处理，同上
        print("parsing test text...")
        self.bags_feature, self.labels = self.parse_sen(self.test_path, 'test')
        np.save(os.path.join(self.root_path, 'test', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'test', 'labels.npy'), self.labels)
        print('save finish!')
    #生成两个 （limit * 2 + 1）*pos_dim的二维矩阵，里面是-1.0到1.0的随机数
    def load_p2v(self):
        # [array([0., 0., 0., 0., 0.])]
        pos1_vec = [np.zeros(self.pos_dim)]
        # pos1_vec在后面添加（limit * 2 + 1）个范围在【-1.0，1.0】的pos_dim个的数组，如[array([0., 0., 0., 0., 0.]), array([ 0.856,  0.2353, -0.9926,  0.033, -0.61]), ...]
        pos1_vec.extend([np.random.uniform(low=-1.0, high=1.0, size=self.pos_dim) for _ in range(self.limit * 2 + 1)])
        # pos2_vec同上
        pos2_vec = [np.zeros(self.pos_dim)]
        pos2_vec.extend([np.random.uniform(low=-1.0, high=1.0, size=self.pos_dim) for _ in range(self.limit * 2 + 1)])
        # 返回的是数组，如[[ 0.          0.          0.          0.          0.        ]
        #                    [ 0.503  -0.127 -0.734  0.44 -0.12  ]，....]
        return np.array(pos1_vec, dtype=np.float32), np.array(pos2_vec, dtype=np.float32)
    #读取vector.txt
    def load_w2v(self):
        '''
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
        '''
        wordlist = []

        f = open(self.w2v_path)
        # dim = int(f.readline().split()[1])
        # f = f.readlines()
        # vecs存单词所对应的向量，1*50数组，wordlist是单词
        vecs = []
        for line in f:
            line = line.strip('\n').split()
            vec = list(map(float, line[1].split(',')[:-1]))
            vecs.append(vec)
            wordlist.append(line[0])

        #  wordlist.append('UNK')
        #  vecs.append(np.random.uniform(low=-0.5, high=0.5, size=dim))
        # i是单词的下标，j是单词，word2id是{"单词"：下标, .....},id2word是{"下标"：单词, .....}
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word
    # 对文件进行预处理
    def parse_sen(self, path, flag):
        '''
        parse the records in data
        '''
        all_sens =[]
        all_labels =[]
        # 读取bags_train.txt文件
        f = open(path)
        while 1:
#             每次循环，针对的是一个例子，共文件的6行
            
            # 对每行数据循环操作
            line = f.readline()
            if not line:
                break
            
            #只针对这行，‘train’例：m.010039	m.01vwm8g	NA	99161,292483
            # 对第4个进行操作，如果有逗号则切分；最后统计有多少个这个
            # ‘test’ 中是 m.010039	m.01vwm8g	99161,292483
            #同上
            if flag == 'train':
                line = line.split('\t')
                num = line[3].strip().split(',')
                num = len(num)
            else:
                line = line.split('\t')
                num = line[2].strip().split(',')
                num = len(num)
                
#             看num的数量，例子是只有1次，超过两次的加，如[[84,83,82,81,80,79,....]，[....],....]
                
            #相对实体1的位置的数组，如[[84,83,82,81,80,79,....]]
            ldists = []
            #相对实体2的位置的数组，如[[50,49,48,47,46,45,....]]
            rdists = []
            #句子的数组，如[[0,2,4,525,6,112,15099,....]]
            sentences = []
            #实体1.2在词表中的位置的下标数值且每个值都加1，升序，[[1,35]]
            entitiesPos = []
            #实体1.2在词表中的位置的下标数值且每个值都加1，[[35,1],[36,11],...]
            pos = []
            #最后的句子的数组，即位置如[[1,2,2,2,2,2,2,2,2,....]]
            masks = []
            #label数，循环后，label的总和不足4个则用-1补足，超过4个则只取前4个的值，如[0,-1,-1,-1]
            rels = []
            
            for i in range(num):
                #针对每个例子的第二行，如：denton,daisy_hill,34,0,0,44
                #ent_pair_line是 ['denton', 'daisy_hill', '34', '0', '0', '44']
                ent_pair_line = f.readline().strip().split(',')
                #  entities = ent_pair_line[:2]
                # ignore the entities index in vocab
                entities = [0, 0]
                #取每个label的前两个并加1（即实体1.2在词表中的位置的下标数值），如得到 [35, 1]
                epos = list(map(lambda x: int(x) + 1, ent_pair_line[2:4]))
                pos.append(epos)
                #升序
                epos.sort()
                entitiesPos.append(epos)
                #label
                rel = int(ent_pair_line[4])
                rels.append(rel)
                #针对每个例子的第三行，即句子，['0', '2', '4', '525', '6', '112', '15099',.....]
                sent = f.readline().strip().split(',')
                #转成int，如[0, 2, 4, 525, 6, 112, 15099, .....]
                sentences.append(list(map(lambda x: int(x), sent)))
                #针对每个例子的第四行，即相对实体1的位置的向量
                ldist = f.readline().strip().split(',')
                #针对每个例子的第五行，即相对实体2的位置的向量
                rdist = f.readline().strip().split(',')
                #针对每个例子的第六行，即最后的位置
                mask = f.readline().strip().split(",")
                #同上功能
                ldists.append(list(map(int, ldist)))
                rdists.append(list(map(int, rdist)))
                masks.append(list(map(int, mask)))
            
            #循环后，label的总和不足4个则用-1补足，超过4个则只取前4个
            rels = list(set(rels))
            if len(rels) < 4:
                rels.extend([-1] * (4 - len(rels)))
            else:
                rels = rels[:4]
                
             '''
            entities:[0, 0]
            num:只针对这行，‘train’例：m.010039	m.01vwm8g	NA	99161,292483
                对第4个进行操作，如果有逗号则切分；最后统计有多少个这个,就是num
            sentences:句子的数组，如[[0,2,4,525,6,112,15099,....]]
            ldists:相对实体1的位置的数组，如[[84,83,82,81,80,79,....]]
            rdists:相对实体2的位置的数组，如[[50,49,48,47,46,45,....]]
            pos:前两个label数组且每个值都加1，[[35,1]]
            entitiesPos:实体1和实体2在词表的下标的位置且每个值都加1，升序，[[1,35]]
            masks:最后的句子的数组，即位置如[[1,2,2,2,2,2,2,2,2,....]]
            '''
            bag = [entities, num, sentences, ldists, rdists, pos, entitiesPos, masks]
#           这两个是读文件外的变量
            #最终全部的rels，如[[0, -1, -1, -1], [0, -1, -1, -1],....]
            all_labels.append(rels)
            #全部bag放在一起，如[[bag1],[bag2],...]
            all_sens += [bag]

        f.close()
        bags_feature = self.get_sentence_feature(all_sens)
        '''  bags_feature
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

        for bag in bags:
            '''  bag里的数据，按顺序排的
            entities:[0, 0]
            num:只针对这行，‘train’例：m.010039	m.01vwm8g	NA	99161,292483
                对第4个进行操作，如果有逗号则切分；最后统计有多少个这个,就是num
            sentences:句子的数组，如[[0,2,4,525,6,112,15099,....]]
            ldists:相对实体1的位置的数组，如[[84,83,82,81,80,79,....]]
            rdists:相对实体2的位置的数组，如[[50,49,48,47,46,45,....]]
            pos:前两个label数组且每个值都加1，[[35,1]]
            entitiesPos:实体1和实体2在词表的下标的位置且每个值都加1，升序，[[1,35]]
            masks:最后的句子的数组，即位置如[[1,2,2,2,2,2,2,2,2,....]]
            '''
            es, num, sens, ldists, rdists, pos, enPos, masks = bag
            new_sen = []
            new_pos = []
            new_entPos = []
            new_masks= []
            #idx下标，sen第一个句子的数组
            for idx, sen in enumerate(sens):
                '''  
                sen:数据不变，数组后面用0填充了
                pf1:每个数值加1，数组后面用0填充了
                pf2:每个数值加1，数组后面用0填充了
                pos:不变
                mask:不变，再放到数组里
                '''
                sen, pf1, pf2, pos, mask = self.get_pad_sen_pos(sen, ldists[idx], rdists[idx], enPos[idx], masks[idx])
                new_sen.append(sen)
                new_pos.append([pf1, pf2])
                new_entPos.append(pos)
                new_masks.append(mask)
            update_bags.append([es, num, new_sen, new_pos, new_entPos, new_masks])
             '''  update_bags
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
        return update_bags

    def get_pad_sen_pos(self, sen, ldist, rdist, pos, mask):
        '''
        refer: github.com/SharmisthaJat/RE-DS-Word-Attention-Models
        '''
        #句子
        x = []
        #相对实体1的位置
        pf1 = []
        #相对实体2的位置
        pf2 = []
        #位置
        masks = []

        # shorter than max_len
        # 句子不变，剩下相对实体1、2的位置每个数值加1，位置变量不变，全部放到新数组里       
        if len(sen) <= self.max_len:
            for i, ind in enumerate(sen):
                x.append(ind)
                pf1.append(ldist[i] + 1)
                pf2.append(rdist[i] + 1)
                masks.append(mask[i])
        # longer than max_len, expand between two entities
        
        else:
            #在pos的两个数中展开，如pos为[1,35],则idx为[1,2,3,....,35]
            idx = [i for i in range(pos[0], pos[1] + 1)]
            #如果idx大于设定的最大长度则只算到最大长度处。句子不变，剩下相对实体1、2的位置每个数值加1，位置变量不变，全部放到新数组里 
            #pos的值改为[1,self.max_len-1]
            if len(idx) > self.max_len:
                idx = idx[:self.max_len]
                for i in idx:
                    x.append(sen[i])
                    pf1.append(ldist[i] + 1)
                    pf2.append(rdist[i] + 1)
                    masks.append(mask[i])
                pos[0] = 1
                pos[1] = len(idx) - 1
            #如果idx不大于设定的最大长度，不改变idx。第一个句子不变，剩下两个每个数值加1，位置变量不变，全部放到新数组里 
            #pos的值改为[1,self.max_len-1]
            else:
                for i in idx:
                    x.append(sen[i])
                    pf1.append(ldist[i] + 1)
                    pf2.append(rdist[i] + 1)
                    masks.append(mask[i])
                #before记录最开始在txt文本里的相对实体1或2的位置的值（pos经过排序后，无法确定是第一个还是第二个实体）
                before = pos[0] - 1
                #after是pos的第二个数（相对实体1、2的位置的最大值）再加1
                after = pos[1] + 1
                pos[0] = 1
                pos[1] = len(idx) - 1
                numAdded = 0
                while True:
                    added = 0
                    if before >= 0 and len(x) + 1 <= self.max_len + self.pad:
                        x.append(sen[before])
                        pf1.append(ldist[before] + 1)
                        pf2.append(rdist[before] + 1)
                        masks.append(mask[before])
                        added = 1
                        numAdded += 1

                    if after < len(sen) and len(x) + 1 <= self.max_len + self.pad:
                        x.append(sen[after])
                        pf1.append(ldist[after] + 1)
                        pf2.append(rdist[after] + 1)
                        masks.append(mask[after])
                        added = 1

                    if added == 0:
                        break

                    before -= 1
                    after += 1

                pos[0] = pos[0] + numAdded
                pos[1] = pos[1] + numAdded

        while len(x) < self.max_len + 2 * self.pad:
            x.append(0)
            pf1.append(0)
            pf2.append(0)
            masks.append(0)

        if pos[0] == pos[1]:
            if pos[1] + 1 < len(sen):
                pos[1] += 1
            else:
                if pos[0] - 1 >= 1:
                    pos[0] = pos[0] - 1
                else:
                    raise Exception('pos= {},{}'.format(pos[0], pos[1]))

        return [x, pf1, pf2, pos, masks]

    def get_pad_sen(self, sen):
        '''
        padding the sentences
        '''
        sen.insert(0, self.word2id['BLANK'])
        if len(sen) < self.max_len + 2 * self.pad:
            sen += [self.word2id['BLANK']] * (self.max_len +2 * self.pad - len(sen))
        else:
            sen = sen[: self.max_len + 2 * self.pad]

        return sen

    def get_pos_feature(self, sen_len, ent_pos):
        '''
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 1
        : -50 => 1
        : 50 => 101
        : >50: 102
        '''

        def padding(x):
            if x < 1:
                return 1
            if x > self.limit * 2 + 1:
                return self.limit * 2 + 1
            return x

        if sen_len < self.max_len:
            index = np.arange(sen_len)
        else:
            index = np.arange(self.max_len)

        pf1 = [0]
        pf2 = [0]
        pf1 += list(map(padding, index - ent_pos[0] + 2 + self.limit))
        pf2 += list(map(padding, index - ent_pos[1] + 2 + self.limit))

        if len(pf1) < self.max_len + 2 * self.pad:
            pf1 += [0] * (self.max_len + 2 * self.pad - len(pf1))
            pf2 += [0] * (self.max_len + 2 * self.pad - len(pf2))
        return [pf1, pf2]


if __name__ == "__main__":
    data = NYTLoad('./dataset/NYT/')
