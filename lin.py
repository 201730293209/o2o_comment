import numpy as np
import jieba
import pandas as pd

def loadcomment_list():
    '''
    函数说明：加载数据
    :return: 
    '''
    fr = open("C:\\Users\\lin78\\Desktop\\CCF\\train.txt", 'r', encoding = 'UTF-8')
    stopwords = open("C:\\Users\\lin78\\Desktop\\CCF\\stopwords_cn.txt", 'r', encoding = 'UTF-8')
    comment_list = []
    label = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        label.append(lineArr[0])
        string = "/".join(jieba.cut(lineArr[1], cut_all = False))
        comment_list.append(string.split("/"))
    fr.close()
    del(label[0])
    del(comment_list[0])
    return comment_list, label

def sort_by_frequency(comment_list):
    '''
    函数说明：根据词频降序排列每个词
    :param comment_list: 评论列表
    :return: all_words_list: 降序排列的所有词
    '''
    all_words_dic = {} #所有词的字典
    for comment in comment_list:  #将每个词对应出现的频率加到字典中
        for word in comment:
            if word in all_words_dic.keys():
                all_words_dic[word] += 1
            else:
                all_words_dic[word] = 1
    all_words_tuple_list = sorted(all_words_dic.items(), key = lambda s: s[1],
                                  reverse = True) #用key指定按字典的值降序排列
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  #解压，返回两个元组
    all_words_list = list(all_words_list)
    return all_words_list


def delete_words(all_words_list, delete_num = 100):
    '''
    函数说明：文本清洗，去除高频词，数字，停用词
    :param all_words_list: 所有词的列表
    :param delete_num: 删除的高频词数目，默认100，需要通过观察确定最好的数目
    :return: feature_words: 特征词，即没有被清洗的词
    '''
    #----------------加载停用词文件数据----------------------
    fr = open("C:\\Users\\lin78\\Desktop\\CCF\\stopwords_cn.txt", 'r', encoding= 'UTF-8')
    stopwords_set = set() #使用set去重，虽然没有必要
    for line in fr.readlines():
        stopword = line.strip()
        if len(stopword) > 0:
            stopwords_set.add(stopword)
    #-------------------------------------------------------
    feature_words = []  #特征词，即有效的词
    for t in range(delete_num, len(all_words_list), 1):
        # 如果这个词不是数字，并且不是停用词，且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
    return feature_words

if __name__ == '__main__':
    comment_list, label = loadcomment_list()
    all_words_list = sort_by_frequency(comment_list)
    feature_words = delete_words(all_words_list)
    print("feature_words:\n", feature_words)
    print("comment_list:\n", comment_list)
    print("label:\n", label)