# -*- coding: utf-8 -*-

import numpy as np
import time


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def save_pr(out_dir, name, epoch, pre, rec, fp_res=None, opt=None):
    if opt is None:
        out = open('{}/{}_{}_PR.txt'.format(out_dir, name, epoch + 1), 'w')
    else:
        out = open('{}/{}_{}_{}_PR.txt'.format(out_dir, name, opt, epoch + 1), 'w')

    if fp_res is not None:
        fp_out = open('{}/{}_{}_FP.txt'.format(out_dir, name, epoch + 1), 'w')
        for idx, r, p in fp_res:
            fp_out.write('{} {} {}\n'.format(idx, r, p))
        fp_out.close()

    for p, r in zip(pre, rec):
        out.write('{} {}\n'.format(p, r))

    out.close()


def eval_metric(true_y, pred_y, pred_p):
    '''
    calculate the precision and recall for p-r curve
    reglect the NA relation
    '''
    '''
    true_y:tesr_data_loader中的labels放到新的数组里，[[bag1的rel]，[],.....,]labels中的一个rel为：[0, -1, -1, -1]，一个bag中的label的总和不足4个则用-1补足，超过4个则只取前4个,如[0, -1, -1, -1]
    pred_y:把每个bag经过向前传播后得到out。当out中第i行的最大值不在第一个数且out中第i行的最大值大于 -1.0，pred_label为out中第i行最大值的下标，否则为0，[bag1的预测最大值的下标，bag2....,....]
    pred_p:把每个bag经过向前传播后得到out。out中前i行的每行最大值的最大的那个数的值，如果大于-1.0则为tmp_prob或tmp_NA_prob，否则为-1.0，最后将tmp_prob或tmp_NA_prob加入pred_p数组中
    以上的i的范围均为[0, bag中句子的个数]
    '''
    assert len(true_y) == len(pred_y)
    #将每个bag中第一个句子的label为正的则计数
    positive_num = len([i for i in true_y if i[0] > 0])
    #将pred_p从小到大排序然后返回数值原本的索引下标，然后的[::-1]，将索引小标的数组变成倒序
    #即得到pred_p从大到小排序然后返回数值原本的索引下标的数组
    index = np.argsort(pred_p)[::-1]

    tp = 0
    fp = 0
    fn = 0
    #记录每个pr值和recall值，两者都不与上一次循环得到的值相同
    all_pre = [0]
    all_rec = [0]
    fp_res = []

    for idx in range(len(true_y)):
        #idx表示第几个bag，i：第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag的rel数组
        #j：第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag，处理后得到out中最大值的下标
        i = true_y[index[idx]]
        j = pred_y[index[idx]]

        '''
        i[0]: 第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag的第一个句子的label
        如果 第一个句子的label为0，且j(第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag，处理后得到out中最大值的下标)大于0，即该最大值位置不是第一个
                    fp_res记录 第idx个最大值的下标（即这个最大值处于第几个bag） 和  对应的bag的经过向前传播后得到out。out中前i行的每行最大值的最大的那个数的值
                    fp的数量加1
        如果 第一个句子的label不为0
                    若j(第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag，处理后得到out中最大值的下标)大于0，即该最大值位置是第一个
                        fn的数量加1
                    若j(第idx个最大值的下标（即这个最大值处于第几个bag），所对应的bag，处理后得到out中最大值的下标)大于0，即该最大值位置不是第一个
                        遍历i中的label，如果label是-1 则停止遍历
                                      如果label等于j，tp的数量加1，停止遍历
        '''
        if i[0] == 0:  # NA relation
            if j > 0:
                fp_res.append((index[idx], j, pred_p[index[idx]]))
                fp += 1
        else:
            if j == 0:
                fn += 1
            else:
                for k in i:
                    if k == -1:
                        break
                    if k == j:
                        tp += 1
                        break

        '''
        如果fp和tp都为0（两者最少都为0），pr值为1.0，否则计算pr值
        计算召回率recall
        如果pr值不等于上次得到的pr值 或recall不等于上次得到的recall值
            all_pre、all_rec添加其值
        '''
        if fp + tp == 0:
            precision = 1.0
        else:
            precision = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
    #打印总的tp，fp，fn，和positive_num数
    print("tp={}; fp={}; fn={}; positive_num={}".format(tp, fp, fn, positive_num))
    #得到的all_pre[1:], all_rec[1:], fp_res（all_pre[1:], all_rec[1:]是为了除去第一个的0）
    return all_pre[1:], all_rec[1:], fp_res
