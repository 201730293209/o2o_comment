import numpy as np
import jieba
import pandas as pd
import sklearn
from sklearn.naive_bayes import MultinomialNB


def train_loadcomment_list():
    '''
    函数说明：加载训练数据
    :return: 切分后的comment以及对应的label
    '''
    fr = open("train.csv", 'r', encoding='UTF-8')
    stopwords = open("stopwords_cn.txt", 'r', encoding='UTF-8')
    comment_list = []
    label = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        label.append(lineArr[0])
        string = "/".join(jieba.cut(lineArr[1], cut_all=False))
        comment_list.append(string.split("/"))
    fr.close()
    del (label[0])
    del (comment_list[0])
    return comment_list, label


def test_loadcomment_list():
    '''
    函数说明：加载测试数据
    :return: 切分后的测试comment，以及对应的id号
    '''
    fr = open("test_new.csv", 'r', encoding='UTF-8')
    stopwords = open("stopwords_cn.txt", 'r', encoding='UTF-8')
    comment_list = []
    id = []
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        id.append(lineArr[0])
        string = "/".join(jieba.cut(lineArr[1], cut_all=False))
        comment_list.append(string.split("/"))
    fr.close()
    del (id[0])
    del (comment_list[0])
    return comment_list, id


def sort_by_frequency(comment_list):
    '''
    函数说明：根据词频降序排列每个词
    :param comment_list: 评论列表
    :return: all_words_list: 降序排列的所有词
    '''
    all_words_dic = {}  # 所有词的字典
    for comment in comment_list:  # 将每个词对应出现的频率加到字典中
        for word in comment:
            if word in all_words_dic.keys():
                all_words_dic[word] += 1
            else:
                all_words_dic[word] = 1
    all_words_tuple_list = sorted(all_words_dic.items(), key=lambda s: s[1],
                                  reverse=True)  # 用key指定按字典的值降序排列
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压，返回两个元组
    all_words_list = list(all_words_list)
    return all_words_list


def delete_words(all_words_list, delete_num=100):
    '''
    函数说明：文本清洗，去除高频词，数字，停用词
    :param all_words_list: 所有词的列表
    :param delete_num: 删除的高频词数目，默认100，需要通过观察确定最好的数目
    :return: feature_words: 特征词，即没有被清洗的词
    '''
    # ----------------将停用词存入stopwords_set中----------------------
    fr = open("stopwords_cn.txt", 'r', encoding='UTF-8')
    stopwords_set = set()  # 使用set去重，虽然没有必要
    for line in fr.readlines():
        stopword = line.strip()
        if len(stopword) > 0:
            stopwords_set.add(stopword)
    # -------------------------------------------------------
    feature_words = []  # 特征词，即有效的词
    for t in range(delete_num, len(all_words_list), 1):
        # 如果这个词不是数字，并且不是停用词，且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    """
    函数说明:根据feature_words将文本向量化 
    Parameters:
        train_data_list - 训练集
        test_data_list - 测试集
        feature_words - 特征集
    Returns:
        train_feature_list - 训练集向量化列表
        test_feature_list - 测试集向量化列表
    """
    def text_features(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_list, test_list, train_label, test_label):
    '''
    函数说明：文本分类器，计算精确度
    :param train_list: 将向量化之后的训练集的已切分comment传入
    :param test_list: 将向量化后的测试集的已切分的comment传入
    :return: test_label: 用多重贝叶斯预测出的测试集的label值
    '''
    classifier = MultinomialNB().partial_fit(train_list, train_label, classes = np.array([0, 1]))
    test_label = classifier.predict(test_list)
    # test_accuracy = classifier.score(test_list, test_label)
    return test_label





if __name__ == '__main__':
    train_comment_list, train_label_list = train_loadcomment_list()
    test_comment_list, test_id_list = test_loadcomment_list()
    train_comment_list = train_comment_list[0 : 5000]
    train_label_list = train_label_list[0 : 5000]

    #----------词频排序与特征词选择------------#
    all_words_list = sort_by_frequency(train_comment_list)
    feature_words = delete_words(all_words_list)
    
    # ----------将训练集和测试集向量化------------#
    train_feature_list, test_feature_list = TextFeatures(train_comment_list, test_comment_list, feature_words)

    # ----------运用贝叶斯将测试集的label值预测出来------------#
    test_label = TextClassifier(train_feature_list, test_feature_list,  train_label_list, train_label_list)

    # ----------将预测的到的label与对应的ID打入新的csv文件------------#
    res=pd.DataFrame({'id':test_id_list,'label':test_label})
    res.to_csv('result1.csv',index=0)

