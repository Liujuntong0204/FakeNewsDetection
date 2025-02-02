import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import csv


train_df=pd.read_csv('train.news.csv',encoding='utf-8')
train_df=train_df.dropna()#如果有空数据则删除行
train_df=shuffle(train_df)#打乱
#print(train_df)

test_df=pd.read_csv('test.feature.csv',encoding='utf-8')


#分词+去除停用词
# 获取停用词集合
def get_stopwords():
    stopwords = [line.strip() for line in open('stop_words.txt','r',encoding='utf-8').readline()]
    return stopwords

import jieba
def cutsentences(sentences):
    stop_words=get_stopwords()
    cutwords=list(jieba.lcut_for_search(sentences))#分词
    lastwords=''

    for word in cutwords:
        if word not in stop_words:
            lastwords+=word+' '
    return lastwords

#train_df['Report Content']=train_df['Report Content'].apply(lambda x:x.split('##'))
t=pd.DataFrame(train_df.astype(str))
#train_df['alldata']=t['Ofiicial Account Name']+t['Title']+t['Report Content']
train_df['alldata']=t['Title']+t['Ofiicial Account Name']
t=pd.DataFrame(train_df.astype(str))
train_df['alldata']=t['alldata'].apply(cutsentences)
train_alldata=train_df['alldata']
x_train=train_alldata
y_train=np.asarray(train_df['label'])
# print(x_train)
#print(y_train)

#test_df['Report Content']=test_df['Report Content'].apply(lambda x:x.split('##'))
t=pd.DataFrame(test_df.astype(str))
#test_df['alldata']=t['Ofiicial Account Name']+t['Title']+t['Report Content']
test_df['alldata']=t['Title']+t['Ofiicial Account Name']
t=pd.DataFrame(test_df.astype(str))
test_df['alldata']=t['alldata'].apply(cutsentences)
test_alldata=test_df['alldata']
x_test=test_alldata
#print(x_test)





from sklearn.feature_extraction.text import TfidfVectorizer
#向量化
transfer=TfidfVectorizer()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)
print(x_train)
# print(x_test)









# #高斯贝叶斯模型0.7223
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# x_train=x_train.toarray()
# model.fit(x_train, y_train)
# x_test=x_test.toarray()
# y_test=model.predict(x_test)

# #伯努利贝叶斯模型0.7348
# from sklearn.naive_bayes import BernoulliNB
# clf = BernoulliNB()
# x_train=x_train.toarray()
# clf.fit(x_train, y_train)
# BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
# x_test=x_test.toarray()
# y_test=clf.predict(x_test)

# #随机森林分类模型0.7359
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# model = RandomForestClassifier(n_estimators=91, random_state=123,n_jobs=-1,max_features=6)
# x_train=x_train.toarray()
# model.fit(x_train, y_train)
# x_test=x_test.toarray()
# y_test=model.predict(x_test)


# #随机树参数调优
# estimator=RandomForestClassifier(oob_score=True,random_state=1)
# estimator.fit(x_train,y_train)
# print(estimator.oob_score)
# param_test1={"n_estimators":range(1,101,10)}
# gird_search=GridSearchCV(estimator=RandomForestClassifier(random_state=1),param_grid=param_test1,scoring='roc_auc',cv=10)
# gird_search.fit(x_train,y_train)
# print(gird_search.best_params_)
# print(gird_search.best_score_)
#
# param_test2={'max_features':range(1,21,1)}
# gird_search1=GridSearchCV(estimator=RandomForestClassifier(n_estimators=91,random_state=1),param_grid=param_test2,scoring='roc_auc',cv=10)
# gird_search1.fit(x_train,y_train)
# print(gird_search1.best_params_)
# print(gird_search1.best_score_)


# #随机森林回归模型 0.7657（100）0.7676（10） 0.7699（50） 0.7638（30）
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=50,random_state=1,n_jobs=-1,oob_score=True)
x_train=x_train.toarray()
model.fit(x_train, y_train)
x_test=x_test.toarray()
y_test=model.predict(x_test)



# # #SVM
# x_train=x_train.toarray()
# x_test=x_test.toarray()
# from sklearn import svm
# clf=svm.LinearSVC(dual=True)
# clf.fit(x_train,y_train)
# y_test=clf.predict(x_test)
# for i in range(len(y_test)):
#     if y_test[i]>=0.5:
#         y_test[i]=1
#     else:
#         y_test[i]=0


# #XGBoost
# x_train=x_train.toarray()
# x_test=x_test.toarray()
# import xgboost as xgb
# model=xgb.XGBClassifier()
# model.fit(x_train,y_train)
# y_test=model.predict(x_test)
# for i in range(len(y_test)):
#     if y_test[i]>=0.5:
#         y_test[i]=1
#     else:
#         y_test[i]=0


# 生成csv文件
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])  # 写入表头
    for i, prediction in enumerate(y_test):
        writer.writerow([i+1, prediction])  # 写入预测结果，假设每个样本有一个唯一的ID

