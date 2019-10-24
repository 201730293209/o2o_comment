import jieba
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import sklearn

def loadcomment_list(filename):
    '''
    函数说明：加载数据
    :return: 
    '''
    fr = open(filename, 'r', encoding = 'UTF-8')
    comment_list = []
    label = []
    delimiter = '\t'
    if filename == 'test_new.csv':
        delimiter = ','
    for line in fr.readlines():
        lineArr = line.strip().split(delimiter)
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
    fr = open("stopwords_cn.txt", 'r', encoding= 'UTF-8')
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


def create_words_vec(comment_list, feature_words):
    '''
    函数说明：生成词条（句子）向量
    :param comment_list: 评论列表
    :param feature_words: 特征词
    :return: words_vec: 词条向量，采用one-hot热编码方式
    '''
    words_vec = []   #词条向量
    for comment in comment_list:  #取出每条评论
        temp_vec = [0] * len(feature_words)    #生成和feature_words相同长度的词向量
        for word in comment:  #取出评论中的每个词
            if word in feature_words:   #如果该词在features_words（词汇表）中出现
                temp_vec[feature_words.index(word)] = 1    #则在对应位置记1
        words_vec.append(temp_vec)
    return words_vec



def test_or_predict(classifier, train_words_vec, test_words_vec, train_label, test_label = None):
    '''
    函数说明：传入分类器进行测试或预测
    :param classifier: sklearn的分类器
    :param train_words_vec: 训练集词条向量
    :param test_words_vec: 测试集词条向量
    :param train_label: 训练集标签向量
    :param test_label: 测试集标签向量，不传入时为预测
    :param: is_train: 是否为测试，否则是预测
    :return:  返回两个其中之一 test_accuracy：测试的准确率
                               test_label: 预测的标签
    '''
    classifier.fit(train_words_vec, train_label)
    if(test_label != None): #测试准确率
        test_accuracy = classifier.score(test_words_vec, test_label)
        return test_accuracy
    else: #预测
        test_label = classifier.predict(test_words_vec)
        return test_label


if __name__ == '__main__':
    #-----------------数据预处理----------------
    #训练集
    comment_list, label = loadcomment_list("train.csv")  #加载数据集
    all_words_list = sort_by_frequency(comment_list)  #生成词汇表
    feature_words = delete_words(all_words_list, delete_num= 100) #清洗词汇表
    words_vec = create_words_vec(comment_list, feature_words)  #词条向量
    #预测验证集
    predict_comment_list, id = loadcomment_list("test_new.csv")  #加载需要预测的评论
    predict_words_vec = create_words_vec(predict_comment_list, feature_words)

    #--------------------各种分类器------------------------------
    #classifier = MultinomialNB()   #朴素贝叶斯多项式分类器
    #AdaBoost集成
    #classifier = AdaBoostClassifier(base_estimator = MultinomialNB(),
    #                                         n_estimators = 50, learning_rate = 1.0)
    classifier = sklearn.svm.LinearSVC(C = 1.0, max_iter = 1000)

    #--------------------------------------------------------
    test_flag = False   #True时进行测试，Flase时进行预测
    if(test_flag):
        # ----------------测试准确率------------------------
        split_num = 8000
        train_words_vec = words_vec[0: split_num]
        train_label = label[0: split_num]
        test_words_vec = words_vec[split_num: len(words_vec)]
        test_label = label[split_num: len(words_vec)]
        test_accuracy = test_or_predict(classifier, train_words_vec, test_words_vec, train_label, test_label)
        print("test_accuracy:\n", test_accuracy)
    else:
        # ---------------------进行预测并输出文件-------------------------
        predict_label = test_or_predict(classifier, words_vec, predict_words_vec, label)
        result = pd.DataFrame({'id': id,
                               'label': predict_label})
        result.to_csv('reuslt_lin_svm_linearSVC.csv', index = 0)
        print("ok!")








