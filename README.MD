---
title: TODOLIST
date: 2025-07-23
updated: 2025-07-23
description: 在这里采用时间倒序安排每日任务
---

## TODOLIST

### 2025.07.24

- [ ] 修改过去文档，先从机器学习开始，保证显示正常，满足下列要求
  - [ ] 行内公式要前后如果有文本要有一个空格隔开如 $x x^2$， ，独行公式 (`$$ $$`) 公式内容单独成行且前后都要空格隔开
  - [ ] 图片显示前后都要空一行，多个图片也要用空格隔开
  - [x] 标题内容不能重复，将重复的修改，
- [ ] github pr 学习 内容很少
- [ ] 线性回归.md 第 69 行出现图片问题修正
- [ ] 代码块中如果表达意义相同不要空行，这是 obsidian 复制出现的问题，修正

<!-- 每次完成后，用 x 替换掉空格，或者直接在预览中点击就行。 你先按照这个弄，不懂的问我 -->

示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例 $x x_2$   文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本

$$
P(y_i = k \mid \mathbf{x} = \mathbf{x}\_i) = \frac{P(\mathbf{x} = \mathbf{x}\_i \mid y_i = k) P(y_i = k)}{P(\mathbf{x} = \mathbf{x}\_i)} \equiv \frac{f_k(\mathbf{x}\_i) \pi_k}{P(\mathbf{x} = \mathbf{x}\_i)}
$$

示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本示例文本

```python
model2 = QuadraticDiscriminantAnalysis()#二次判别分析

model2.fit(X2, y)

model2.score(X2, y) #模型评分

plot_decision_regions(np.array(X2), y, model2)

plt.xlabel('petal_length')

plt.ylabel('petal_width')

plt.title('Decision Boundary for QDA')
```

修正为

```python
# 二次判别分析
model2 = QuadraticDiscriminantAnalysis()
model2.fit(X2, y)
model2.score(X2, y) # 模型评分

plot_decision_regions(np.array(X2), y, model2)

plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for QDA')
```

### 2025.07.23

- [x] create new folder, classify your note, and format it
- [ ] learn how to git, and upload to github
- [ ] learn markdown language
  - [ ] how to insert latex in it
  - [x] how to display /images/image, the uri to load
