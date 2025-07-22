可导入(a,b)两个特征，使用degree = 2的二次多项式则为$（1，a, a^2, ab, b ,b^2)$。
PolynomialFeatures主要有以下几个参数：

degree：度数，决定多项式的次数

interaction_only： 默认为False，字面意思就是只能交叉相乘，不能有a^2这种.

include_bias: 默认为True, 这个bias指的是多项式会自动包含1，设为False就没这个1了.

order：有"C" 和"F" 两个选项。官方写的是在密集情况（dense case）下的输出array的顺序，F可以加快操作但可能使得subsequent estimators变慢。
利用代码试验一下：

```python
from sklearn.preprocessing import PolynomialFeatures
a=[[2,3]]
pf=PolynomialFeatures(degree=2)
print(pf.fit_transform(a))
pf=PolynomialFeatures(degree=2,include_bias=False)
print(pf.fit_transform(a))
pf=PolynomialFeatures(degree=2,interaction_only=True)
print(pf.fit_transform(a))
```

结果如下

```python
[[1. 2. 3. 4. 6. 9.]]
[[2. 3. 4. 6. 9.]]
[[1. 2. 3. 6.]]
```

如果是c=`[[a],[b]]`这种形式，生成的多项式就没有ab交叉项了，只有`[[1,a,a^2], [1,b,b^2]]` 。

```python
c=[[2],[3]]
print(pf.fit_transform(c))
```

结果如下

```python
[[2. 4.]
 [3. 9.]]
```
