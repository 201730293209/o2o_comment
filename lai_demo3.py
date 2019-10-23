#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
import pandas as pd

#读入训练集
df=pd.read_csv('train.csv',sep='\t')

#分词
words=[jieba.lcut(i) for i in df['comment']]


# In[2]:


#读入stopwords_cn.txt
stop_words=pd.read_csv('stopwords_cn.txt',header=None)


# In[3]:


all_words_dict = {}                                            #统计训练集词频
for word_list in words:
    for word in word_list:
        if word in all_words_dict.keys():
            all_words_dict[word] += 1
        else:
            all_words_dict[word] = 1

# In[4]:


#去掉标点符号、stop_words中的词语
words_clean=[]
words_set=set()
for l in words:
    item=[]
    for i in l:
        if i not in list(stop_words[0]) and i!=' ' and all_words_dict[i]>1:
            item.append(i)
            words_set.add(i)
    words_clean.append(item)


# In[5]:


vocabulary=list(words_set)


# In[6]:


words_vec = []   #词条向量

for row in words_clean:  #取出每条评论

    temp_vec = [0] * len(vocabulary)    #生成和feature_words相同长度的词向量

    for word in row:  #取出评论中的每个词

        if word in vocabulary:   #如果该词在features_words（词汇表）中出现

            temp_vec[vocabulary.index(word)] = 1    #则在对应位置记1

    words_vec.append(temp_vec)


# In[7]:


df=pd.DataFrame(columns=vocabulary,data=words_vec)


# In[102]:


df.to_csv('vec4.csv',header=None,index=0)


# In[8]:





# In[ ]:




