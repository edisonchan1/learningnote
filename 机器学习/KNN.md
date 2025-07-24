---
title: K近邻法
date: 2025-07-24
updated: 2025-07-24
description:
---

## K近邻法

### 回归问题的K近邻法

考虑回归问题。假设响应变量 $y$ 连续，而 $X$ 为 $p$ 维特征向量。根据线性回归，能使均方误差最小化的函数为条件期望函数 $E(y|X)$ 。问题是：在实践中应如何估计 $E(y|X)$ 。

如果对于任意给定 $X$，均有很多不同的 $y$ 观测值，则可对这些 $y$ 值进行简单算术平均，如图所示：

![fix](/images/imageknn.png)

现实数据大多比较稀疏，对于给定 $X$，可能只有很少 $y$ 观测值，甚至连一个 $y$ 观测值都没有：可以理解为，在 $X$ 轴上并不是每个点都有样本点，也就是说，在没有样本点的地方的 $y$ 值，使用 KNN 来替代。

![fix](/images/image-1-2.png)

一个解决方法是考虑离 $X$ 最近的 $K$ 个邻居。

记 $N_K(X)$ 为最靠近 $X$ 的 $K$ 个观测值 $X_i$ 的集合。

$K$ 近邻估计法以离 $X$ 最近的 $K$ 个邻居的 $y$ 观测值的平均作为预测值：

$$
\hat{f}_{KNN}(\mathbf{x}) \equiv \frac{1}{K} \sum_{\mathbf{x}_i \in N_K(\mathbf{x})} y_i
$$

当 $K= 1$ ，则为“最近邻法”，即以离 $X$ 最近邻居的 $y$ 标签作为预测值。而此时，因 $X_i$ 是训练集中离 $X$ 最近的样本，理论上 $X_i$ 与 $X$ 的真实分布高度相近

为找到最近的 $K$ 个邻居，首先需要在特征空间中，定义一个距离函数。通常以欧式距离（即二范数：参考惩罚回归中岭回归二范数）作为此距离函数，即 $\|\mathbf{x}_i - \mathbf{x}\|_2$ 。

使用 KNN 法的一个前提是，所有特征变量均为实值性；否则无法计算欧式距离。

为避免某些变量由于距离函数的影响过大，一般建议先将所有变量标准化：减去该变量的均值，再除以其标准差，使得所有变量的均值为0而标准差为1。

### 如何选择K

再进行 KNN 估计时，一个重要选择为如何确定 K。这依然要考虑偏差与方差的权衡。

如果 $K = 1$ (最近邻法)，则偏差较小(仅使用最近邻的信息)，但方差较大(未将近邻的 y观测值进行平均)。所估计的回归函数很不规则，导致“过拟合”(overfit)，使得模型的泛化能力(generalization)较差：即在训练集表现极佳（偏差小），但测试集表现差。

如果 $K$ 很大，则偏差较大（用到更远邻居的信息），而方差较小（用更多观测值进行平均，更接近估计值的期望）。如果 $K = n$，则使用样本均值进行所有预测，导致偏差很大，出现“前拟合”。如果 $K$ 太大，回归函数过于光滑，未能充分捕捉数据中的信号，使得算法的泛化能力下降。

最优的邻居数 $K$ 应在偏差与方差之间保持较好的权衡。

在实践中，一般可用交叉验证选择最优的 $K$。

比如，对于 $K = 1,\dots , 50,$，分别计算相应的交叉验证误差，然后选择使交叉验证误差最小的 $K$ 值。

接下来展示摩托车撞击实验数据的案例：

```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
```

导入数据并查看：

```python
mcycle = pd.read_csv('mcycle.csv')
sns.scatterplot(x = 'times', y = 'accel' , data = mcycle)
plt.title('Simulated Motorcycle Accident')
```

结果如下：

![fix](/images/image-2-2.png)

加速度(accel)是关于时间(times)的一个高度非线性的函数。由于此原因，统计学常使用 mcycle 数据演示非参数回归。

```python
X_raw = np.array(mcycle.times)
X = np.array(mcycle.times).reshape(-1,1)
y = mcycle.accel
```

接下来将 $K = 1,10,25,50$ 分别进行 KNN 回归，结果在 $2 \times 2$ 的画布上展示。

```python
fig, ax = plt.subplots(2, 2, figsize=(9,6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
for i,k in zip([1,2,3,4],[1,10,25,50]):
    model = KNeighborsRegressor(n_neighbors = k)
    model.fit(X,y)
    pred = model.predict(np.arange(60).reshape(-1,1))
    plt.subplot(2,2,i)
    sns.scatterplot(x = 'times',y = 'accel' , s = 20,data = mcycle , facecolot = 'none',edgecolor = 'k') #画散点图，散点为空心点，其余代码在之前各章节都有讲解，这里不多赘述。
    plt.plot(np.arange(60) , pred , 'b')
    plt.text(0,55,f'k = {k}')
plt.tight_layout()
```

先设定 $2\times 2$ 的画布，再用 `for` 循环进行4次 KNN 估计，并分别作出图像。

![fix](/images/image-3-2.png)

左上角为 $K = 1$ 的拟合效果。回归函数非常不光滑，呈锯齿状跳跃。此时，训练误差很小，但显然存在过拟合，模型的泛化能力较差

右上角为 $K = 10$ 的拟合效果。回归函数虽然仍不太光滑，但较好抓住数据特征，拟合效果比较合适。

左下角为 $K = 25$ 的拟合效果。此时，回归函数更为光滑，但已经不能充分捕捉数据的趋势特征，存在欠拟合。

右下角为 $K = 50$ 的拟合效果，回归函数已不能捕捉数据的主要恶政，存在严重的欠拟合。

### 分类问题的K近邻法

对于分类问题，同样可食用 KNN 法。

特征变量 $X$ 依然为数值型，而响应变量 $y$ 则为离散型，仅取有限的几个值，比如 $y \in {1,2,\dots,j}$ ，共分为 $J$ 类。

给定 $X$，在进行 KNN估计时，首先确定离 $X$ 最近的 $K$ 个邻居之集合 $N_K(X)$。

然后采取**多数票规则**，即以 $K$ 个近邻中最常见的类别作为预测。

如果 $K$ 近邻中最常见的类别有两个或多个并列，则可随机选一个最常见类别作为预测结果。

对于分类问题进行 KNN 估计时，如何选择 $K$ 同样重要。

下面以Hastie et al. (2009)的模拟数据 mixture.example，直观地展示不同 $K$ 值，对于分类问题的决策边界的影响。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
```

导入数据，这里是导入python库的数据集

```python
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['diagnosis'] = cancer.target
d = {0:'malignant', 1:'benign'}
df['diagnosis'] = df['diagnosis'].map(d)
```

查看数据的信息

```python
df.iloc[:,:3].describe()
```

![fix](/images/image-4-2.png)

这里介绍两行代码。

```python
df.diagnosis.value_counts()#统计每个类别的数量
df.diagnosis.value_counts(normalize = True)#统计每个类别的比例
```

箱型图

```python
sns.boxplot(x = 'diagnosis',y = 'mean radius',data = df)
```

![fix](/images/image-5-2.png)

接下来训练模型：

```python
X,y = load_brest_cancer(return_X_y = True)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y,test_size =100,random_state = 1)
```

数据标准化处理：

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transfor(X_test)
```

这里是否有遗憾：为什么明明是对`X_train`进行训练，而`X_test`却也可以使用结果呢？

这是因为`scaler.fit()`的作用是获取参数的均值与方差，而训练集和测试集的标准化应当以同一个方差和均值为标准进行。

```python
np.mean(X_test_scalerd,axis = 0) # 查看测试集的均值
np.std(X_train_scaled, axis=0) # 查看训练集的标准差
```

以 $K = 5$ 进行 KNN 分类

```python
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train_scaled , y_train)
model.score(X_test_scaled , y_test)

0.97
```

```python
pred = model.predict(X_test_scaled)
pd.crosstab(y_test,pred,rownames = ['Actual'],colnames = ['predictd'])
```

![fix](/images/image-6-2.png)

接下来看看遍历寻找最佳的 $K$ 值

```python
scores = []
ks = range(1,51)
for k in ks:
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train_scaled,y_train)
    score = model.score(X_test_scaled,t_test)
    scores.append(score)

index_max = np.argmax(scores) # 最大准确率对应的序号，反推出K值
print(f'最佳k值为：{ks[index_max]}，对应的准确率为：{scores[index_max]}')
```

结果如下：最佳k值为：3，对应的准确率为：0.97

接下来看看遍历过程的图片：

```python
plt.plot(ks,scores,'o-')
plt.xlabel('K')
plt.axvline(ks[index_max],linewidth =1,linestyle = '--') # 画出最大准确率到X轴的垂线的虚线
plt.ylabel('Accuracy')
plt.title('KNN')
plt.tight_layout
```

![fix](/images/image-7-2.png)

网格搜索寻找最佳参数：

```python
param_grid = {'n_neighbors':range(1,51)}
kfold = StratifiedKFold(n_splits = 10,shuffle = True , random_state = 1)
model = GridSearchCV(KNeighborsClassifier(),param_grid , cv= Kfold)
model.fit(X_train_scaled,y_train)

print(model.best_params)
print(model.score(X_test_scaked,y_test))

{'n_neighbors': 12}
0.96
```

### K近邻法的优缺点

KNN 的优点之一是算法简单易懂。

作为非参数估计，不依赖于具体的函数形式，故较为稳健。

但 KNN 所顾及的回归函数或决策边界一般较不规则，因为当 $X$ 在特征空间变动，其 $K$ 个邻居之集合 $N_K(X)$ 可能发生不连续的变化，即加入新邻居而去掉旧邻居。

如果真实的回归函数或决策边界也较不规则，则 KNN 效果较好。

KNN 直到预测时才去找邻居，导致预测较慢，不适用于“在线学习”

(online learning)，即需要实时预测的场景。
在高维空间中可能很难找到邻居，会遇到所谓“维度灾难”(curse of dimensionality)，此时 KNN 算法的效果可能较差。

KNN 对于“噪音变量”(noise variable)也不稳健。
