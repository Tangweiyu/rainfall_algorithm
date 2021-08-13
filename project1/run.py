import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.svm as svm
from sklearn.metrics import precision_recall_fscore_support
import copy
# 导入采样库
from imblearn.combine import SMOTETomek,SMOTEENN
# 导入模型保存库
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# 将DisasterTy属性由字符串变成数值类型
def del_y(data):
    if "滑坡" in str(data):
        return 1
    # elif "滑坡" in str(data):
    #     return 2
    else:
        return 0
# 以下4个函数是用来将岩性名称,地形名称,地形坡度,地貌名称属性转化为数值类型
def del_yanxingname(data):
    if "极软岩" in str(data):
        return 1
    elif "较软岩" in str(data):
        return 2
    elif "坚硬岩" in str(data):
        return 3
    elif "软岩" in str(data):
        return 4
    elif "较硬岩" in str(data):
        return 5
    else:
        return 0

def del_dixingname(data):
    if "平地" in str(data):
        return 1
    elif "缓坡" in str(data):
        return 2
    else:
        return 0

def del_podu(data):
    if "＜10°" in str(data):
        return 1
    elif "10°~45°" in str(data):
        return 2
    else:
        return 0

def del_dimaoname(data):
    if "小起伏山地" in str(data):
        return 1
    elif "低海拔平原" in str(data):
        return 2
    elif "中起伏山地" in str(data):
        return 3
    elif "大起伏山地" in str(data):
        return 4
    elif "丘陵" in str(data):
        return 5
    elif "冲积台地" in str(data):
        return 6
    elif "剥蚀台地" in str(data):
        return 7
    elif "湿地" in str(data):
        return 8
    else:
        return 0

# 处理数据集，去掉不相关的样本，只保留DisasterTy属性=滑坡的样本
def del_data(df):
    df.drop(df[df[df.columns[5]] == "地面沉降"].index,inplace=True)
    df.drop(df[df[df.columns[5]] == "泥石流"].index, inplace=True)
    df.drop(df[df[df.columns[5]] == "地裂缝"].index, inplace=True)
    df.drop(df[df[df.columns[5]] == "不稳定斜坡"].index, inplace=True)
    df.drop(df[df[df.columns[5]] == "地面塌陷"].index, inplace=True)
    df.drop(df[df[df.columns[5]] == "崩塌"].index, inplace=True)
    return df

# 读取文件并且处理数据集
df = pd.read_csv("data.csv")
df = del_data(df)
# 以下4个函数用来处理输入属性Rainfall01,Rainfall03,Rainfall06,Rainfall24的0值，使用所有样本的平均值来替代
x1 = df[df.columns[11]].mean()
def del_zero1(data):
    if data == 0:
        return x1
    else:
        return data

x2 = df[df.columns[12]].mean()
def del_zero2(data):
    if data == 0:
        return x2
    else:
        return data

x3 = df[df.columns[13]].mean()
def del_zero3(data):
    if data == 0:
        return x3
    else:
        return data

x4 = df[df.columns[14]].mean()
def del_zero4(data):
    # print(x)
    if data == 0:
        return x4
    else:
        return data

# 生成负样本，对Rainfall01,Rainfall03,Rainfall06,Rainfall24这4个属性取阈值为0.5来生成负样本
# 具体方法是取正样本中Rainfall01,Rainfall03,Rainfall06,Rainfall24所对应的值的0.5倍来作为负样本，正负样本比例为1：1
def get_negaitive(df):
    new_df = copy.deepcopy(df)
    for i in new_df.columns[10:26]:
        for j in range(len(df[i])):
            if new_df[i].iloc[j] > new_df["Rainfall"].iloc[j]:
                new_df[i].iloc[j] = new_df["Rainfall"].iloc[j]
        # print(i)
        new_df[i] = new_df[i] * 0.5
    new_df["DisasterTy"] = "negaitive"
    # print(new_df)
    return new_df

# 如果样本降水量都为0，则样本有误，应删掉该样本
def del_error(df):
    for i in df.index:
        if df.loc[i,'Rainfall']==df.loc[i,'Rainfall01']== df.loc[i, 'Rainfall02']==df.loc[i,'Rainfall03']==df.loc[i,'Rainfall04']==df.loc[i,'Rainfall05']==df.loc[i,'Rainfall06']==df.loc[i,'Rainfall07']==df.loc[i,'Rainfall08']==df.loc[i,'Rainfall09']==df.loc[i,'Rainfall10']==df.loc[i,'Rainfall11']==df.loc[i,'Rainfall11']==df.loc[i,'Rainfall12']==df.loc[i,'Rainfall13']==df.loc[i,'Rainfall14']==df.loc[i,'Rainfall15']==0:
        #if all(df.loc[i,t]==0  for t in df.columns[0:15]):
            df.drop(index=i, inplace=True)
    return df

# 主函数
def main():
    # 读取文件，处理数据集
    df = pd.read_csv("new_data.csv")
    df = del_data(df)
    # print(len(df))
    # 合并正负样本
    new_df = get_negaitive(df)
    df = pd.concat([df,new_df])
    # 将岩性名称,地形名称,地形坡度,地貌名称属性转化为数值类型
    df["岩性名称"] = df["岩性名称"].apply(del_yanxingname)
    df["地形名称"] = df["地形名称"].apply(del_dixingname)
    df["地形坡度"] = df["地形坡度"].apply(del_podu)
    df["地貌名称"] = df["地貌名称"].apply(del_dimaoname)
    # 预测目标为样本DisasterTy属性的数值化类型
    df["DisasterTy"] = df["DisasterTy"].apply(del_y)
    # 处理模型输入的属性（Rainfall01,Rainfall03,Rainfall06,Rainfall24）中所存在的0值问题
    #df[df.columns[11]] = df[df.columns[11]].apply(del_zero1)
    #df[df.columns[12]] = df[df.columns[12]].apply(del_zero2)
    #df[df.columns[13]] = df[df.columns[13]].apply(del_zero3)
    #df[df.columns[14]] = df[df.columns[14]].apply(del_zero4)
    # df.to_excel("处理好的样本数据.xlsx",encoding="utf-8")
    # 构造模型输入属性列表（Rainfall01,Rainfall03,Rainfall06,Rainfall24）
    feature_li = []
    for i in df.columns[10:26]:
        feature_li.append(i)
    # 构造额外的模型输入属性列表（岩性名称,地形名称,地形坡度,地貌名称）
    for i in df.columns[-4:]:
        feature_li.append(i)
    # 生成模型输入、输出数据
    x = df[feature_li]
    y = df["DisasterTy"]
    col = pd.concat([x, y], axis=1)
    col.reset_index(drop=True,inplace=True)
    #删除无效样本
    col = del_error(col)
    print(col)
    #col.to_csv("训练的数据集.csv")
    #划分训练集和验证集，训练集：验证集 = 8：2
    x = col.iloc[:, 0:-1]
    y = col.iloc[:, -1:]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
    # 使用SMOTETomek方法进行样本采样,缓解样本不平衡问题
    #smote_tomek = SMOTETomek(random_state=2020)
    #xtrain, ytrain = smote_tomek.fit_resample(xtrain, ytrain)
    # 使用svm模型
    #model = svm.SVC(kernel="rbf",gamma="auto", decision_function_shape="ovo")
    #model.fit(xtrain,ytrain)
    #使用逻辑回归
    model = LogisticRegression()
    model.fit(xtrain,ytrain)
    #使用高斯朴素贝叶斯
    #model = GaussianNB()
    #model.fit(xtrain, ytrain)
    # 保存模型
    joblib.dump(model, 'model.pkl')
    # 保存模型输入列表
    model_columns = list(x.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    # 模型预测概率与分类结果
    #y_pred_proba = model.predict_proba(xtest)
    y_pred = model.predict(xtest)
    print(y_pred)
    yt = ytest.values.tolist()
    # 保存测试与预测的值，以便分析
    col_test = pd.concat([xtest, ytest], axis=1)
    col_test["PRED"] = y_pred
    #col_test.to_csv('测试与预测数据.csv')
    # 预测结果分析
    # f = open("re.txt","w",encoding="utf-8")
    # for num in range(len(yt)):
    #     if yt[num] == 2:
    #         # if yt[num] != y_pred[num]:
    #         f.write(str(yt[num]) + "\t" + str(y_pred[num]) + "\n")
    # f.close()
    # 模型预测结果指标：Precision，Recall，F1值
    p, r, f, s = precision_recall_fscore_support(ytest, y_pred, labels=[0, 1])
    # print(ytest.values.tolist()[:50])
    # print(y_pred[:50])
    # 打印输出预测结果
    print("事故不发生准确率：",p[0])
    print("事故不发生召回率：",r[0])
    print("事故不发生f1值：",f[0])
    # print("不使用负样本")
    print("事故发生准确率：", p[1])
    print("事故发生召回率：", r[1])
    print("事故发生f1值：", f[1])
    # print("滑坡准确率：", p[2])
    # print("滑坡召回率：", r[2])
    # print("滑坡f1值：", f[2])
    # print(s)

# 用于调试代码，没有实际意义
def test():
    df = pd.read_csv("data.csv")
    df = del_data(df)
    new_li = []
    for i in range(len(df["Rainfall24"])):
        if df["Rainfall24"].iloc[i] > df["Rainfall"].iloc[i]:
            df["Rainfall24"].iloc[i] = df["Rainfall"].iloc[i]
    print(df["Rainfall24"])

    # print(df["Rainfall24"])
    # feature_li = []
    # for i in df.columns[11:15]:
    #     feature_li.append(i)
    # print(feature_li)
    # for i in df.columns[-4:]:
    #     feature_li.append(i)
    # print(feature_li)
    # print(df[feature_li])
    # print(df["地貌名称"].unique())
    # new_df = get_negaitive(df)
    # print(new_df["DisasterTy"])

if __name__ == '__main__':
    main()
    # test()