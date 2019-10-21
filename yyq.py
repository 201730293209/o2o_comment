import jieba
import csv
import pandas as pd

def textProcess():
    fR = open('train.txt', 'r', encoding='UTF-8')
    label = []
    df=pd.read_csv('train.txt',sep='\t')
    print(df)
    df['comment']
    print(df['comment'])

if __name__=='__main__':
    textProcess()
