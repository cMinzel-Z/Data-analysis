import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import os
import pydotplus
# 这里的path是Graphviz安装位置
os.environ["PATH"]+=os.pathsep+"C:/Program Files (x86)/Graphviz2.38/bin/"

# sl:satisfaction_level---False:MinMaxScaler;True:StandardScaler
# le:last_evaluation---False:MinMaxScaler;True:StandardScaler
# npr:number_project---False:MinMaxScaler;True:StandardScaler
# amh:average_monthly_hours--False:MinMaxScaler;True:StandardScaler
# tsc:time_spend_company--False:MinMaxScaler;True:StandardScaler
# wa:Work_accident--False:MinMaxScaler;True:StandardScaler
# pl5:promotion_last_5years--False:MinMaxScaler;True:StandardScaler
# dp:department--False:LabelEncoding;True:OneHotEncoding
# slr:salary--False:LabelEncoding;True:OneHotEncoding
def hr_preprocessing(sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,dp=False,slr=False,lower_d=False,ld_n=1):
    df=pd.read_csv("./data/HR.csv")

    # 1、清洗数据
    df=df.dropna(subset=["satisfaction_level","last_evaluation"])
    df=df[df["satisfaction_level"]<=1][df["salary"]!="nme"]
    # 2、得到标注
    label = df["left"]
    df = df.drop("left", axis=1)
    # 3、特征选择
    # 4、特征处理
    scaler_lst=[sl,le,npr,amh,tsc,wa,pl5]
    column_lst=["satisfaction_level","last_evaluation","number_project",\
                "average_monthly_hours","time_spend_company","Work_accident",\
                "promotion_last_5years"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]]=\
                MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
        else:
            df[column_lst[i]]=\
                StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
    scaler_lst=[slr,dp]
    column_lst=["salary","department"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i]=="salary":
                df[column_lst[i]]=[map_salary(s) for s in df["salary"].values]
            else:
                df[column_lst[i]]=LabelEncoder().fit_transform(df[column_lst[i]])
            df[column_lst[i]]=MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]

        else:
            df=pd.get_dummies(df,columns=[column_lst[i]])
    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values),label
    return df,label


d=dict([("low",0),("medium",1),("high",2)])
def map_salary(s):
    return d.get(s,0)

def hr_modeling(features,label):
    from sklearn.model_selection import train_test_split
    f_v = features.values
    f_names=features.columns.values
    l_v = label.values
    # 切分验证集
    X_tt,X_validation,Y_tt,Y_validation = train_test_split(f_v,l_v,test_size=0.2)
    # 切分训练集
    X_train,X_test,Y_train,Y_test = train_test_split(X_tt,Y_tt,test_size=0.25)
    print(len(X_train),len(X_validation),len(X_test))

    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    from sklearn.tree import DecisionTreeClassifier,export_graphviz
    from sklearn.externals.six import StringIO
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    
    models=[]
    #models.append(('KNN',KNeighborsClassifier(n_neighbors=3)))
    # 当数据是离散值的时候考虑Naive Bayes方法
    #models.append(('GaussianNB',GaussianNB()))
    #models.append(('BernoulliNB',BernoulliNB()))
    models.append(("DecisionTreeGini",DecisionTreeClassifier()))
    #models.append(("DecisionTreeEntropy",DecisionTreeClassifier(criterion="entropy")))
    # SVM中的参数C控制精度 C=100000
    #models.append(('SVM Classifier',SVC()))
    models.append(('RandomForest',RandomForestClassifier()))
    models.append(('Adaboost',AdaBoostClassifier()))
    
    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst=[(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            # 0是训练集，1是验证集，2是测试集
            print(i)
            print(clf_name,'-ACC:',accuracy_score(Y_part,Y_pred))
            print(clf_name,'-REC:',recall_score(Y_part,Y_pred))
            print(clf_name,'-F1:',f1_score(Y_part,Y_pred))
            '''
            dot_data=StringIO()
            export_graphviz(clf,
                                     out_file=dot_data,
                                     feature_names=f_names,
                                     class_names=['NL','L'],
                                     filled=True,
                                     rounded=True,
                                     special_characters=True)
            graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_pdf('dt_tree.pdf')
            '''
            print('-'*20)
  
'''
    # 模型存储
    from sklearn.externals import joblib
    joblib.dump(knn_clf, 'knn_clf')
    # 模型读取
    knn_clf = joblib.load('knn_clf')
    # 测试集
    print('Test:')
    Y_pred=knn_clf.predict(X_test)
    print('ACC:', accuracy_score(Y_test, Y_pred))
    print('REC:', recall_score(Y_test, Y_pred))
    print('F-Score:', f1_score(Y_test, Y_pred))
 '''   
 
def regr_test(features,label):
    print('X',features)
    print('Y',label)
    from sklearn.linear_model import LinearRegression,Ridge,Lasso
    #regr=LinearRegression()
    #regr=Ridge(alpha=1)
    # 调试alpha的值
    regr=Lasso(alpha=0.002)
    regr.fit(features.values,label.values)
    Y_pred = regr.predict(features.values)
    print('Coef:',regr.coef_)
    from sklearn.metrics import mean_squared_error
    # 均方差越小，alpha的值就越适合
    print('MSE:',mean_squared_error(Y_pred,label.values))
     
     
def main():
    features,label=hr_preprocessing()
    regr_test(features[['number_project','average_monthly_hours']],features['last_evaluation'])
    #hr_modeling(features,label)
    
if __name__=="__main__":
    main()

