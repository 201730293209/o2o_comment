import jieba
import csv
import pandas as pd

def textProcess():
    #切分数据集，将label和comment分开
    trainSet=pd.read_csv('train.csv',sep='\t')
    trainSet['comment']
    trainSet=['label']

    #对comment做分割
    word_cut=jieba.cut(comment,cut_all=False)
    word_list=list(word_cut)
    data_list=[]
    data_list.append(word_list)
    print(data_list)

if __name__=='__main__':
    textProcess()
