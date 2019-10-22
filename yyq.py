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
        lineArr = line.strip().split()#对每一行通过空格和回车进行划分
        label.append(lineArr[0])#将0,1值打入label数组中
        comment_set.append(lineArr[1])#将评论语句打入数组中
    fr.close()
    del (label[0])#将第一行的单词‘label’删除
    del (comment_set[0])#将第一行的单词‘comment’删除

    '''
    用jieba将comment分词，放入到comment_split数组中
    '''
    for line in comment_set:
        string = "/".join(jieba.cut(line, cut_all=False))
        comment_split.append(string.split('/'))

    '''
    对切分好之后的词组进行排序
    '''
    all_words_dict = {}  # 用于统计训练集词频
    for word_list in comment_split:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表

    return comment_split, label, all_words_list


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
    comment_split, label,all_words_list = loadcomment_set()
    for line in range(0,10000):
        print(label[line],' ',comment_split[line],'\n')
        #print(comment_split[line])

    print("按词频降序展示：",'\n',all_words_list)
    #print(comment_split[0])
    #print("comment_split_sorted:\n", all_words_list)
    #print("label:\n", label)