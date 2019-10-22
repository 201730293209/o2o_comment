import numpy as np
import jieba
import pandas as pd


def loadcomment_set():
    '''
    函数说明：加载数据
    :return: 
    '''
    fr = open("train.csv", 'r', encoding='UTF-8')
    comment_set = []
    comment_split = []
    label = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        label.append(lineArr[0])
        comment_set.append(lineArr[1])
    fr.close()
    del (label[0])
    del (comment_set[0])
    for line in comment_set:
        string = "/".join(jieba.cut(line, cut_all=False))
        comment_split.append(string.split('/'))
    return comment_split, label


def split_word(text, stopwords):
    word_list = jieba.cut(text)
    start = True
    result = ''
    for word in word_list:
        word = word.strip()
        if word not in stopwords:
            if start:
                result = word
                start = False
            else:
                result += ' ' + word
    return result.encode('utf-8')


if __name__ == '__main__':
    comment_split, label = loadcomment_set()
    print(comment_split[0])
    print("comment_split:\n", comment_split)
    print("label:\n", label)