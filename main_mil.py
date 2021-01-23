# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
sys.path.append('/home/iiip/.local/lib/python3.6/site-packages')
import fire


from config import opt
import models
import dataset
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils import save_pr, now, eval_metric


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def test(**kwargs):
    pass

# 设置随机数种子
def setup_seed(seed):
    # 为CPU设置种子用于生成随机数
    torch.manual_seed(seed)
    #为所有的GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    # 按顺序产生一组固定的数组，如果使用相同的seed值，则每次生成的随机数都相同
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(**kwargs):
    # 设置随机数种子
    setup_seed(opt.seed)
    #调用config.py里的parse函数，可对opt更改默认参数，增加缺少的数值
    kwargs.update({'model': 'PCNN_ONE'})
    opt.parse(kwargs)
    
    # 如果使用gpu，设置使用指定的gpu
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # torch.manual_seed(opt.seed)
    '''将PCNN_ONE中的各种设定赋到model里
    model： PCNN_ONE(
                      (word_embs): Embedding(114043, 50)
                      (pos1_embs): Embedding(102, 5)
                      (pos2_embs): Embedding(102, 5)
                      (convs): ModuleList(
                        (0): Conv2d(1, 230, kernel_size=(3, 60), stride=(1, 1), padding=(1, 0))
                      )
                      (mask_embedding): Embedding(4, 3)
                      (linear): Linear(in_features=690, out_features=53, bias=True)
                      (dropout): Dropout(p=0.5, inplace=False)
                    )
    '''
    model = getattr(models, 'PCNN_ONE')(opt)
    #如果使用gpu，则对model里的数值进行处理
    if opt.use_gpu:
        # torch.cuda.manual_seed_all(opt.seed)
        model.cuda()
        # parallel
        #  model = nn.DataParallel(model)

    # loading data
    # DataModel ： dataset.nyt.NYTData  将dataset中的opt.data + 'Data'（即NTYData）对应的值给dataModel
    DataModel = getattr(dataset, opt.data + 'Data')
    # 将NTY中的train，test数据拿到加载
    '''以train数据为例，拿到dataset/NYT/train/  下的两个npy文件
       每个数据以二元组（bag，rel）组成，bag为bags_feature.npy中的一个bag，
       rel为labels.npy中的一个数据、记录的是原始bags_train.txt中每个bag中句子标签，
       两者间是一一对应的
            bags_feature中的一个bag为
                es:[0, 0]
                num:只针对这行，‘train’例：m.010039	m.01vwm8g	NA	99161,292483
                    对第4个进行操作，如果有逗号则切分；最后统计有多少个这个,就是num
                    (bag 里面一样，不变)
                new_sen:句子的数组,数据不变，数组后面用0填充了，如[[0,2,4,525,6,112,15099,....,0,0,0]]
                new_pos:[相对实体1的位置,相对实体2的位置]的数组,数据不变，数组后面用0填充了，如[[84,83,82,81,80,79,....,0,0,0],
                                                                           [50,49,48,47,46,45,....,0,0,0]]
                new_entPos:实体1和实体2在词表的下标的位置且每个值都加1，升序，[[1,35]]
                new_masks:最后的句子的数组，数据不变，数组后面用0填充了,即位置如[[1,2,2,2,2,2,2,2,2,....,0,0,0,0]]
        labels中的一个rel为：[0, -1, -1, -1]，一个bag中的label的总和不足4个则用-1补足，超过4个则只取前4个,如[0, -1, -1, -1]

    '''
    train_data = DataModel(opt.data_root, train=True)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    #同上
    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))
    
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    '''
    model.parameters() ：  <bound method Module.parameters of PCNN_ONE(
                                          (word_embs): Embedding(114043, 50)
                                          (pos1_embs): Embedding(102, 5)
                                          (pos2_embs): Embedding(102, 5)
                                          (convs): ModuleList(
                                            (0): Conv2d(1, 230, kernel_size=(3, 60), stride=(1, 1), padding=(1, 0))
                                          )
                                          (mask_embedding): Embedding(4, 3)
                                          (linear): Linear(in_features=690, out_features=53, bias=True)
                                          (dropout): Dropout(p=0.5, inplace=False)
                                        )>
    优化算法的设定 optim.Adadelta(net.parameters(), rho=0.9,eps=1e-6, weight_decay=opt.weight_decay)
                 params(iterable):待优化参数的iterable或者是定义了参数组的dict
                 rho:用于计算平方梯度的运行平均值的系数（默认： 0.9）
                 eps:为了增加数值计算的稳定性二加到分母里的项（默认：1e-6）
                 weight_decay:权重衰减（L2惩罚）（默认：0）
    '''
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    # optimizer = optim.Adadelta(model.parameters(), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)
    # train
    print("start training...")
    max_pre = -1.0
    max_rec = -1.0
    for epoch in range(opt.num_epochs):

        total_loss = 0
        for idx, (data, label_set) in enumerate(train_data_loader):
            '''
            data 元组: （bag1, bag2,bag3,.....）bag的数据内容如上
            label_set 元组:([bag1的rels]，[],.....)
            label: 将每个bag的第一个句子中的label取出作为一个数组
            '''
            label = [l[0] for l in label_set]

            if opt.use_gpu:
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)
            '''
            data:  [select_ent, select_num, select_sen, select_pf, select_pool, select_mask],里面都是tensor格式
            select_ent, select_num：存的是，每个bag里的es和num [[bag1的对应数据],[bag2],....]
            select_sen, select_pf, select_pool, select_mask：每个bag里的对应的数据中的一条的数组集合
            如select_sen里的是每个bag里sen中的一个句子数组（怎么选择这个句子则是由select_instance里的max_ins_id下标决定）
            [[bag1的对应数据],[bag2],....]
            '''
            data = select_instance(model, data, label)
            model.batch_size = opt.batch_size
            # 将梯度初始化为零
            optimizer.zero_grad()
            
            '''
            model(data, train=True)等价于调用了 model.forward(data, train=True)
            只是隐藏了
            '''
            #向前传播，求出预测的值
            out = model(data, train=True)
            #求loss
            loss = criterion(out, label)
            #反向传播求梯度
            loss.backward()
            #更新所有参数
            optimizer.step()

            total_loss += loss.item()

        if epoch < -1:
            continue
        '''
        true_y:tesr_data_loader中的lanbels放到新的数组里，[[bag1的rel]，[],.....,]labels中的一个rel为：[0, -1, -1, -1]，一个bag中的label的总和不足4个则用-1补足，超过4个则只取前4个,如[0, -1, -1, -1]
        pred_y:把每个bag经过向前传播后得到out。当out中第i行的最大值不在第一个数且out中第i行的最大值大于 -1.0，pred_label为out中第i行最大值的下标，否则为0，[bag1的预测最大值的下标，bag2....,....]
        pred_p:把每个bag经过向前传播后得到out。out中前i行的每行最大值的最大的那个数的值，如果大于-1.0则为tmp_prob或tmp_NA_prob，否则为-1.0，最后将tmp_prob或tmp_NA_prob加入pred_p数组中
                以上的i的范围均为[0, bag中句子的个数]
        '''
        true_y, pred_y, pred_p = predict(model, test_data_loader)
        '''调用utils.py里的eval_metric函数得到pr，re fp_re的数组
        all_pre：每次循环计算得到的pr值，且保证下一个值不会和前一个重复。循环次数由true_y的个数决定（即test_data_loader中bag的个数）
        all_rec：每次循环计算得到的recall值，且保证下一个值不会和前一个重复。循环次数由true_y的个数决定（即test_data_loader中bag的个数）
        fp_res： 第idx个最大值的下标（即这个最大值处于第几个bag） 和  对应的bag的经过向前传播后得到out。out中前i行的每行最大值的最大的那个数的值
            有关fp_res的补充：
                 idx表示第几个bag，i：第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag的rel数组
                 第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag的第一个句子的label
        如果 第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag的第一个句子的label为0，
            且j(第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag，处理后得到out中最大值的下标)大于0，即该最大值位置不是第一个
               才算入fp_res中

        '''
        all_pre, all_rec, fp_res = eval_metric(true_y, pred_y, pred_p)
        #得到最新算到的pre和recall
        last_pre, last_rec = all_pre[-1], all_rec[-1]
        if last_pre > 0.24 and last_rec > 0.24:
            #将数据all_pre, all_rec, fp_res写入相关文件中
            save_pr(opt.result_dir, model.model_name, epoch, all_pre, all_rec, fp_res, opt=opt.print_opt)
            print('{} Epoch {} save pr'.format(now(), epoch + 1))
            #记录峰值，将该model相关数据写入文件，同时更新峰值
            if last_pre > max_pre and last_rec > max_rec:
                print("save model")
                max_pre = last_pre
                max_rec = last_rec
                model.save(opt.print_opt)

        print('{} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, last_pre, last_rec))


def select_instance(model, batch_data, labels):
    '''
        batch_data 元组: （bag1, bag2,bag3,.....）bag的数据内容如上方
        labels :将每个bag的第一个句子中的label取出作为一个数组
    '''
    model.eval()
    ''' out经过模型之后得到预测的数据值，然后拿到max_ins_id
    select_ent :将各bag的es继续放进去，[[bag1中es],[bag2],....]
    select_num :将各bag的num继续放进去，[[bag1中num],[bag2],....]
    select_sen :预测的对应的数据，即对应max_ins_id下标的句子的数组，[[bag1中的new_sen[max_ins_id]],[bag2],....]
    select_pf :预测的对应的数据，即对应max_ins_id下标的[相对实体1的位置,相对实体2的位置]的数组，[[bag1中的new_pos[max_ins_id]],[bag2],....]
    select_pool :预测的对应的数据，即对应max_ins_id下标的实体1和实体2在词表的下标的位置且每个值都加1，[[bag1中的new_entPos[max_ins_id]],[bag2],....]
    select_mask :预测的对应的数据，即对应max_ins_id下标的最后的句子的数组，[[bag1中的new_masks[max_ins_id]],[bag2],....]
    '''
    select_ent = []
    select_num = []
    select_sen = []
    select_pf = []
    select_pool = []
    select_mask = []
    for idx, bag in enumerate(batch_data):
        '''
        idx为每个bag的下标，bag为
                        es:[0, 0]
                        num:只针对这行，‘train’例：m.010039	m.01vwm8g	NA	99161,292483
                            对第4个进行操作，如果有逗号则切分；最后统计有多少个这个,就是num
                            (bag 里面一样，不变)
                        new_sen:句子的数组,数据不变，数组后面用0填充了，如[[0,2,4,525,6,112,15099,....,0,0,0]]
                        new_pos:[相对实体1的位置,相对实体2的位置]的数组,数据不变，数组后面用0填充了，如[[84,83,82,81,80,79,....,0,0,0],
                                                                                   [50,49,48,47,46,45,....,0,0,0]]
                        new_entPos:实体1和实体2在词表的下标的位置且每个值都加1，升序，[[1,35]]
                        new_masks:最后的句子的数组，数据不变，数组后面用0填充了,即位置如[[1,2,2,2,2,2,2,2,2,....,0,0,0,0]]
        insNum: 每个bag中句子的数量
        label: 对应bag中的第一个句子中的label，如：tensor(0, device='cuda:0')
        '''
        insNum = bag[1]
        label = labels[idx]
        max_ins_id = 0
        if insNum > 1:
            model.batch_size = insNum
            # 将bag里的每个数据变成tensor(原本数据, device='cuda:0')
            # data 里的数据也跟bag相同，只是类型不一样了
            if opt.use_gpu:
                data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            else:
                data = map(lambda x: torch.LongTensor(x), bag)
            #调用向前传播，PCNN_ONE.py里的forward函数，out的大小变为 torch.Size([？, 27])，行有所不同
            out = model(data)

            #  max_ins_id = torch.max(torch.max(out, 1)[0], 0)[1]
            '''
            out[:, label] ：将out中的第label列的数据（label: 对应bag中的第一个句子中的label）
            torch.max(out[:, label], 0)：从拿到的列数据取其中最大值，torch.max(out[:, label], 0)[1]：最大值的下标
            例 若 out[:, label]为 tensor([0.0000e+00, 0.0000e+00, 6.3505e-14])
               则 torch.max(out[:, label], 0)： torch.return_types.max( values=tensor(0.), indices=tensor(0))
                  torch.max(out[:, label], 0)[1]：tensor(0)
            然后判断opt是否使用gpu，然后用对应的方法拿出其中的值
            '''
            max_ins_id = torch.max(out[:, label], 0)[1]

            if opt.use_gpu:
                #  max_ins_id = max_ins_id.data.cpu().numpy()[0]
                max_ins_id = max_ins_id.item()
            else:
                max_ins_id = max_ins_id.data.numpy()[0]
        #取的是对应max_ins_id下标的句子向量数组、[相对实体1的位置,相对实体2的位置]的数组、实体1和实体2在词表的下标的位置且每个值都加1、最后的句子的数组
        max_sen = bag[2][max_ins_id]
        max_pf = bag[3][max_ins_id]
        max_pool = bag[4][max_ins_id]
        max_mask = bag[5][max_ins_id]
        #前两个不变，其他取的都是每个包里的对应max_ins_id下标的句子信息
        select_ent.append(bag[0])
        select_num.append(bag[1])
        select_sen.append(max_sen)
        select_pf.append(max_pf)
        select_pool.append(max_pool)
        select_mask.append(max_mask)
    #将select_ent, select_num, select_sen, select_pf, select_pool, select_mask里的数据改成tensor，然后都放在数组里赋值给data
    if opt.use_gpu:
        data = map(lambda x: torch.LongTensor(x).cuda(), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])
    else:
        data = map(lambda x: torch.LongTensor(x), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])

    model.train()
    return data


def predict(model, test_data_loader):

    model.eval()

    pred_y = []
    true_y = []
    pred_p = []
    for idx, (data, labels) in enumerate(test_data_loader):
        '''
        data 元组: （bag1, bag2,bag3,.....）bag的数据内容如上
        labels 元组:([bag1的rels]，[],.....)
        '''
        true_y.extend(labels)
        for bag in data:
             '''
                bag为
                    es:[0, 0]
                    num:只针对这行，‘train’例：m.010039	m.01vwm8g	NA	99161,292483
                        对第4个进行操作，如果有逗号则切分；最后统计有多少个这个,就是num(bag 里面一样，不变)
                    new_sen:句子的数组,数据不变，数组后面用0填充了，如[[0,2,4,525,6,112,15099,....,0,0,0]]
                    new_pos:[相对实体1的位置,相对实体2的位置]的数组,数据不变，数组后面用0填充了，如[[84,83,82,81,80,79,....,0,0,0],
                                                                                   [50,49,48,47,46,45,....,0,0,0]]
                    new_entPos:实体1和实体2在词表的下标的位置且每个值都加1，升序，[[1,35]]
                    new_masks:最后的句子的数组，数据不变，数组后面用0填充了,即位置如[[1,2,2,2,2,2,2,2,2,....,0,0,0,0]]
        insNum: 每个bag中句子的数量
        '''
            insNum = bag[1]
            model.batch_size = insNum
            #将bag里的每个数据变为tensor格式
            if opt.use_gpu:
                data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            else:
                data = map(lambda x: torch.LongTensor(x), bag)
            #向前传播
            out = model(data)
            #将out一行一行归一化处理
            out = F.softmax(out, 1)
            '''torch.max(out, 1)：从每行中取最大值组成tensor格式，
            如对于tensor([[0.0000e+00, 0.0000e+00, 1.4013e-45],
                        [0.0000e+00, 0.0000e+00, 0.0000e+00],
                        [2.8906e-01, 4.5632e-41, 2.1860e+01]])
                经过取每行最大值为
            torch.return_types.max(
                values=tensor([1.4013e-45, 0.0000e+00, 2.1860e+01]),
                indices=tensor([2, 2, 2]))
            通过下方的处理后 max_ins_prob记录的是每行最大值的数组，如[1.4012985e-45 0.0000000e+00 2.1860031e+01]
                          max_ins_label记录的是每行最大值的下标，如 [2 2 2]
            max_ins_prob, max_ins_label 为普通数组格式，不是tensor
            '''
            max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
            tmp_prob = -1.0
            tmp_NA_prob = -1.0
            pred_label = 0
            pos_flag = False
            
            for i in range(insNum):
                #当pos_flag为真且 out数据中第i行最大值的下标要小于1， 即out中第i行的最大值在第一个数
                if pos_flag and max_ins_label[i] < 1:
                    continue
                else:
                    #当out中i行的最大值的下标不是0时（即out中第i行的最大值不在第一个数），令当out中i行的最大值的下标pos_flag = True
                    #        同时，out中第i行的最大值大于 -1.0 则令pred_label等于out中第i行的最大值的下标，tmp_prob 等于中第i行的最大值
                    #当out中i行的最大值的下标是0时（即out中第i行的最大值在第一个数），pos_flag不改变，则tmp_NA_prob取 -1.0和out中第i行的最大值 两者中的最大值
                    if max_ins_label[i] > 0:
                        pos_flag = True
                        if max_ins_prob[i] > tmp_prob:
                            pred_label = max_ins_label[i]
                            tmp_prob = max_ins_prob[i]
                    else:
                        if max_ins_prob[i] > tmp_NA_prob:
                            tmp_NA_prob = max_ins_prob[i]

            if pos_flag:
                pred_p.append(tmp_prob)
            else:
                pred_p.append(tmp_NA_prob)
            #当out中第i行的最大值不在第一个数且out中第i行的最大值大于 -1.0，pred_label为out中第i行最大值的下标，否则为0
            pred_y.append(pred_label)

    size = len(test_data_loader.dataset)
    #断言函数assert  后面必须为真否则就触发异常，保证pred_y、true_y的大小与数据集大小相同
    assert len(pred_y) == size and len(true_y) == size

    model.train()
    return true_y, pred_y, pred_p


if __name__ == "__main__":
    import fire
    fire.Fire()
