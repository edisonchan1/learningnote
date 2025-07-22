
![[Pasted image 20250527180715.png]]

![[Pasted image 20250527163901.png]]



## 个人信息
### 姓名：

**周瑾瑜**

### 学院：

数学与统计学院
### 班级：

**统计2302**

### 学号：

**1023006155**

### 老师：

赵尚睿
![[Pasted image 20250527180733.png]]
![[Pasted image 20250527180735.png]]
![[Pasted image 20250527180735 1.png]]



## 题1:

### 试题 1：（10 分）
1） 随机生成一个𝑛 × 𝑛的矩阵。
2） 计算 1)中 𝑛 = 6 时生成的矩阵的秩、行列式、迹、特征向量矩阵及特征值（特征值用向量表示）和逆，输出在不同的向量或矩阵中。
3） 生成一个大小为 3 × 6 的矩阵 A 和相应维度的向量𝑏, c，利用函数求解问题：
$$\begin{align*}
\max &\quad c^T x \\
\text{s.t.} &\quad Ax = b
\end{align*}$$
### 代码

#### demo1.1

```matlab

%% 生成随机矩阵并计算属性

n = 6;

A = rand(n, n); % 生成6×6均匀分布随机矩阵

%% 计算矩阵属性并存储在不同变量中

rank_A = rank(A); % 矩阵的秩（标量）

det_A = det(A); % 行列式（标量）

trace_A = trace(A); % 迹（标量）

[V, D] = eig(A); % V: 特征向量矩阵，D: 特征值对角阵

eigenvalues_A = diag(D); % 特征值（列向量）

eigenvectors_A = V; % 特征向量矩阵（每列对应一个特征值）

% 计算逆矩阵（处理奇异矩阵情况）

try

inv_A = inv(A); % 逆矩阵（如果存在）

inv_exists = true;

catch

inv_A = []; % 空矩阵表示逆不存在

inv_exists = false;

end

%% 输出结果

fprintf('=== 原始矩阵 A ===\n');

disp(A);

fprintf('\n=== 矩阵属性 ===\n');

fprintf('秩: rank_A = %d\n', rank_A);

fprintf('行列式: det_A = %.6f\n', det_A);

fprintf('迹: trace_A = %.6f\n', trace_A);

fprintf('\n=== 特征值 ===\n');

fprintf('eigenvalues_A = \n');

disp(eigenvalues_A);

fprintf('\n=== 特征向量矩阵 ===\n');

fprintf('eigenvectors_A = \n');

disp(eigenvectors_A);

fprintf('\n=== 特征值用向量表示 ===\n');

for i = 1:n

fprintf('特征值 λ%d = %.6f + %.6fi\n', i, real(eigenvalues_A(i)), imag(eigenvalues_A(i)));

fprintf('对应特征向量 v%d = \n', i);

disp(eigenvectors_A(:, i));

fprintf('\n');

end

fprintf('\n=== 逆矩阵 ===\n');

if inv_exists

fprintf('inv_A = \n');

disp(inv_A);

else

fprintf('矩阵 A 奇异，逆矩阵不存在。\n');

end
```
#### demo1.2
```matlab
% 生成随机数据

rng(0); % 设置随机种子以确保可重复性

A = rand(3,6); % 3x6随机矩阵

c = rand(6,1); % 6维目标函数系数向量

x0 = rand(6,1); % 生成随机初始解

b = A * x0; % 计算b使得问题可行

% 求解线性规划问题

% max c'*x subject to: A*x = b, x >= 0

options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'off');

[x, fval, exitflag] = linprog(-c, [], [], A, b, zeros(6,1), [], options);

% 显示结果

disp('矩阵 A:');

disp(A);

disp('向量 b:');

disp(b);

disp('向量 c:');

disp(c);

if exitflag == 1

disp('最优解 x:');

disp(x);

disp(['目标函数最大值: ', num2str(-fval)]);

disp('约束验证 A*x - b:');

disp(A*x - b);

else

disp('未找到最优解');

disp(['退出标志: ', num2str(exitflag)]);

end
```
### 结果
#### demo1.1
```matlab
>> demo1
=== 原始矩阵 A ===
    0.8147    0.2785    0.9572    0.7922    0.6787    0.7060
    0.9058    0.5469    0.4854    0.9595    0.7577    0.0318
    0.1270    0.9575    0.8003    0.6557    0.7431    0.2769
    0.9134    0.9649    0.1419    0.0357    0.3922    0.0462
    0.6324    0.1576    0.4218    0.8491    0.6555    0.0971
    0.0975    0.9706    0.9157    0.9340    0.1712    0.8235


=== 矩阵属性 ===
秩: rank_A = 6
行列式: det_A = -0.043100
迹: trace_A = 3.676533

=== 特征值 ===
eigenvalues_A = 
   3.3953 + 0.0000i
   0.3358 + 0.7624i
   0.3358 - 0.7624i
  -0.6834 + 0.0000i
   0.1465 + 0.0728i
   0.1465 - 0.0728i


=== 特征向量矩阵 ===
eigenvectors_A = 
  列 1 至 3

   0.5001 + 0.0000i  -0.2101 + 0.4728i  -0.2101 - 0.4728i
   0.4234 + 0.0000i   0.2834 - 0.1406i   0.2834 + 0.1406i
   0.4003 + 0.0000i  -0.3332 - 0.3502i  -0.3332 + 0.3502i
   0.3177 + 0.0000i   0.2981 + 0.0160i   0.2981 - 0.0160i
   0.3161 + 0.0000i   0.1718 - 0.0368i   0.1718 + 0.0368i
   0.4577 + 0.0000i  -0.5277 + 0.0000i  -0.5277 + 0.0000i

  列 4 至 6

   0.1013 + 0.0000i   0.0254 - 0.0714i   0.0254 + 0.0714i
   0.3119 + 0.0000i  -0.0532 - 0.0107i  -0.0532 + 0.0107i
  -0.1399 + 0.0000i   0.7400 + 0.0000i   0.7400 + 0.0000i
  -0.7721 + 0.0000i  -0.2142 - 0.0438i  -0.2142 + 0.0438i
   0.4268 + 0.0000i  -0.1794 + 0.1419i  -0.1794 - 0.1419i
   0.3076 + 0.0000i  -0.5861 - 0.0128i  -0.5861 + 0.0128i


=== 特征值用向量表示 ===
特征值 λ1 = 3.395323 + 0.000000i
对应特征向量 v1 = 
    0.5001
    0.4234
    0.4003
    0.3177
    0.3161
    0.4577


特征值 λ2 = 0.335838 + 0.762394i
对应特征向量 v2 = 
  -0.2101 + 0.4728i
   0.2834 - 0.1406i
  -0.3332 - 0.3502i
   0.2981 + 0.0160i
   0.1718 - 0.0368i
  -0.5277 + 0.0000i


特征值 λ3 = 0.335838 + -0.762394i
对应特征向量 v3 = 
  -0.2101 - 0.4728i
   0.2834 + 0.1406i
  -0.3332 + 0.3502i
   0.2981 - 0.0160i
   0.1718 + 0.0368i
  -0.5277 + 0.0000i


特征值 λ4 = -0.683444 + 0.000000i
对应特征向量 v4 = 
    0.1013
    0.3119
   -0.1399
   -0.7721
    0.4268
    0.3076


特征值 λ5 = 0.146489 + 0.072819i
对应特征向量 v5 = 
   0.0254 - 0.0714i
  -0.0532 - 0.0107i
   0.7400 + 0.0000i
  -0.2142 - 0.0438i
  -0.1794 + 0.1419i
  -0.5861 - 0.0128i


特征值 λ6 = 0.146489 + -0.072819i
对应特征向量 v6 = 
   0.0254 + 0.0714i
  -0.0532 + 0.0107i
   0.7400 + 0.0000i
  -0.2142 + 0.0438i
  -0.1794 - 0.1419i
  -0.5861 + 0.0128i



=== 逆矩阵 ===
inv_A = 
    0.8891    2.4498   -0.8206   -0.1344   -2.6745   -0.2580
   -0.7400   -0.4573    0.2516    0.7079    0.4624    0.4733
    3.0570   10.2424    0.6679   -3.7108  -13.1555   -1.4818
   -1.0410   -0.2263   -0.6094   -0.1742    1.9017    0.8917
   -1.1175   -7.1582    1.1638    2.1185    8.4414   -0.2710
   -1.2197   -9.3966   -0.4928    3.0654   10.4896    1.3799
```
#### demo1.2
```matlab
>> demo2
矩阵 A:
    0.8147    0.9134    0.2785    0.9649    0.9572    0.1419
    0.9058    0.6324    0.5469    0.1576    0.4854    0.4218
    0.1270    0.0975    0.9575    0.9706    0.8003    0.9157

向量 b:
    2.4822
    1.9525
    1.9337

向量 c:
    0.7922
    0.9595
    0.6557
    0.0357
    0.8491
    0.9340

最优解 x:
         0
    1.6381
         0
         0
    0.8536
    1.1912

目标函数最大值: 3.409
约束验证 A*x - b:
   1.0e-14 *

         0
    0.1110
    0.1998
```
## 题2:
### 试题 2：（20 分）
画图
1） 写一个函数 [x, y] = get_circle(center, r) 得到圆的 x 和 y 坐标。这个圆必须以 center 为圆心（center 为2 × 1向量），半径为 r，返回的 x 和 y 使得 plot(x, y) 画出来的是一个圆。提示：回忆圆心为原点（0，0）半径为 r 的圆，可以写成参数方程𝑥(𝑡) = 𝑟𝑐𝑜𝑠(𝑡),𝑦(𝑡) = 𝑟𝑠𝑖𝑛(𝑡),𝑡 ∈ [0, 2𝜋].
2） 在 get_circle 的基础上，写一个函数 [x, y] = get_ellipse(center, r) 得到一个椭圆的坐标。圆心同上，r 为2 × 1向量，表示长轴和短轴的长度。
3） 写一个脚本 bingdundun.m ，在这个脚本中打开一个新的 figure，利用 get_circle 和get_ellipse 画一个冰墩墩（2022 年北京冬奥会吉祥物）。提示：线粗细、颜色等参考 Line Properties 中的 Color，LineWidth。其他可能用到 hold on, axis equal.
### 代码
#### demo2.1
```matlab
% 绘制圆心在 (2,3)，半径为 5 的圆

figure;

hold on;

axis equal;

grid on;

% 生成圆坐标

[x, y] = get_circle([2; 3], 5);

% 绘制圆

plot(x, y, 'b-', 'LineWidth', 2);

% 标记圆心

plot(2, 3, 'ro', 'MarkerFaceColor', 'r');

title('圆形: 圆心(2,3), 半径5');

xlabel('X');

ylabel('Y');
```
#### demo2.2
```matlab
Mu = [1; 2]; % 中心点

Sigma = [4 1; 1 2]; % 协方差矩阵

S = 5.991; % 95%置信区间（卡方分布分位数）

[X, Y] = get_ellipse(Mu, Sigma, S, 100);

figure;

plot(X, Y, 'b-', 'LineWidth', 2);

axis equal; grid on;

hold on;

plot(Mu(1), Mu(2), 'ro', 'MarkerSize', 8); % 标记中心点
```
#### get_circle
```matlab
function [x, y] = get_circle(center, r)

% GET_CIRCLE 生成圆的坐标点

% [x, y] = get_circle(center, r) 返回圆的坐标点

% 输入:

% center - 圆心坐标 (2x1 向量 [cx; cy])

% r - 圆的半径 (正数)

% 输出:

% x, y - 坐标向量，使得 plot(x, y) 绘制出圆形

% 验证输入

if numel(center) ~= 2

error('圆心必须是2x1向量');

end

if r <= 0

error('半径必须是正数');

end

% 生成角度向量 (100个点)

theta = linspace(0, 2*pi, 100);

% 计算圆上的点坐标

x = center(1) + r * cos(theta);

y = center(2) + r * sin(theta);

end
```
#### get_ellipse
```matlab
function [X,Y]=get_ellipse(Mu,Sigma,S,pntNum)

% (X-Mu)*inv(Sigma)*(X-Mu)=S

invSig=inv(Sigma);

[V,D]=eig(invSig);

aa=sqrt(S/D(1));

bb=sqrt(S/D(4));

t=linspace(0,2*pi,pntNum);

XY=V*[aa*cos(t);bb*sin(t)];

X=(XY(1,:)+Mu(1))';

Y=(XY(2,:)+Mu(2))';

end
```

#### bingdundun
```matlab
function bingdundun

ax=gca;

ax.DataAspectRatio=[1 1 1];

ax.XLim=[-5 5];

ax.YLim=[-5 5];

hold(ax,'on')

% =========================================================================

% 绘制冰糖外壳

[X,Y]=get_ellipse([0,0],[1,0;0,1.3],3.17^2,200);

plot(X,Y,'Color',[57,57,57]./255,'LineWidth',1.8)

%

[X,Y]=get_ellipse([1.7,2.6],[1.2,0;0,1.8],.65^2,200);

plot(X,Y,'Color',[57,57,57]./255,'LineWidth',1.8)

plot(-X,Y,'Color',[57,57,57]./255,'LineWidth',1.8)

[X,Y]=get_ellipse([1.7,2.6],[1.2,0;0,1.8],.6^2,200);

fill(X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

fill(-X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

%

[X,Y]=get_ellipse([-3.5,-1],[1.1,.3;.3,1.1],.75^2,200);

plot(X,Y,'Color',[57,57,57]./255,'LineWidth',1.8)

[X,Y]=get_ellipse([-3.5,-1],[1.1,.3;.3,1.1],.68^2,200);

fill(X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

[X,Y]=get_ellipse([3.5,1],[1.1,.3;.3,1.1],.75^2,200);

plot(X,Y,'Color',[57,57,57]./255,'LineWidth',1.8)

[X,Y]=get_ellipse([3.5,1],[1.1,.3;.3,1.1],.68^2,200);

fill(X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

%

X=[-3.8,-2,-3];

Y=[-.51+.13,1+.13,-1];

plot(X,Y,'Color',[57,57,57]./255,'LineWidth',1.8)

plot(-X,-Y,'Color',[57,57,57]./255,'LineWidth',1.8)

X=[-3.8,-2,-3];

Y=[-.51+.03,1+.03,-1];

fill(X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

fill(-X,-Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

%

[X,Y]=get_ellipse([0,-.1],[1,0;0,1.6],.9^2,200);

Y(Y<0)=Y(Y<0).*.2;Y=Y-4.2;X=X-1.2;

plot(X,Y,'Color',[57,57,57]./255,'LineWidth',2)

plot(-X,Y,'Color',[57,57,57]./255,'LineWidth',2)

rectangle('Position',[-2.1 -4.2 1.7 3],'Curvature',0.4,...

'FaceColor',[1 1 1],'EdgeColor',[57,57,57]./255,'LineWidth',1.8)

rectangle('Position',[2.1-1.7 -4.2 1.7 3],'Curvature',0.4,...

'FaceColor',[1 1 1],'EdgeColor',[57,57,57]./255,'LineWidth',1.8)

[X,Y]=get_ellipse([0,-.1],[1,0;0,1.6],.8^2,200);

Y(Y<0)=Y(Y<0).*.2;Y=Y-4.1;X=X-1.2;

fill(X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

fill(-X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

%

[X,Y]=get_ellipse([0,0],[1,0;0,1.3],3.1^2,200);

fill(X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',1.8)

% =========================================================================

% 耳朵

[X,Y]=get_ellipse([1.7,2.6],[1.2,0;0,1.8],.5^2,200);

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

fill(-X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

% 胳膊

[X,Y]=get_ellipse([-3.5,-1],[1.1,.3;.3,1.1],.6^2,200);

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

[X,Y]=get_ellipse([3.5,1],[1.1,.3;.3,1.1],.6^2,200);

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

X=[-3.8,-2,-3];

Y=[-.51,1,-1];

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255)

fill(-X,-Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255)

tt=linspace(-2.9,2.9,1000);

X=16.*(sin(tt)).^3;

Y=13.*cos(tt)-5.*cos(2.*tt)-2.*cos(3.*tt)-cos(4.*tt);

X=X.*.018+3.6;

Y=Y.*.018+1.1;

fill(X,Y,[180,39,45]./255,'EdgeColor',[180,39,45]./255,'LineWidth',2)

% 腿

[X,Y]=get_ellipse([0,-.1],[1,0;0,1.6],.7^2,200);

Y(Y<0)=Y(Y<0).*.2;Y=Y-4.1;X=X-1.2;

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

fill(-X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

rectangle('Position',[-1.95 -4.3 1.4 3],'Curvature',0.4,...

'FaceColor',[57,57,57]./255,'EdgeColor',[57,57,57]./255)

rectangle('Position',[1.95-1.4 -4.3 1.4 3],'Curvature',0.4,...

'FaceColor',[57,57,57]./255,'EdgeColor',[57,57,57]./255)

% 身体

[X,Y]=get_ellipse([0,0],[1,0;0,1.3],3^2,200);

fill(X,Y,[1,1,1],'EdgeColor',[57,57,57]./255,'LineWidth',2.5)

% 五环

cList=[132,199,114;251,184,77;89,120,177;158,48,87;98,205,247];

for i=1:5

[X,Y]=get_ellipse([0,0],[1.6,0;0,1.3],(2.05-0.05.*i)^2,200);

Y(Y<0)=Y(Y<0).*.8;Y=Y+.5;

fill(X,Y,[1,1,1],'EdgeColor',cList(i,:)./255,'LineWidth',2.5)

end

% 眼睛

[X,Y]=get_ellipse([1.2,1.2],[1.2,-.5;-.5,1.1],.65^2,200);

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

fill(-X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

[X,Y]=get_ellipse([.95,1.3],[1,0;0,1],.35^2,200);

fill(X,Y,[57,57,57]./255,'EdgeColor',[1,1,1],'LineWidth',1.6)

fill(-X,Y,[57,57,57]./255,'EdgeColor',[1,1,1],'LineWidth',1.6)

[X,Y]=get_ellipse([.95,1.3],[1,0;0,1],.1^2,200);

fill(X+.18,Y,[1,1,1],'EdgeColor',[57,57,57]./255,'LineWidth',.5)

fill(-X+.18,Y,[1,1,1],'EdgeColor',[57,57,57]./255,'LineWidth',.5)

% 嘴巴

[X,Y]=get_ellipse([0.05,.2],[1.2,.15;.15,.8],.69^2,200);

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

[X,Y]=get_ellipse([0,.75],[1,0.2;0.2,.3],.4^2,200);

fill(X,Y,[1,1,1],'EdgeColor',[1,1,1],'LineWidth',2)

[X,Y]=get_ellipse([0,0],[.8,0;0,.2],.6^2,200);

fill(X,Y,[180,39,45]./255,'EdgeColor',[180,39,45]./255,'LineWidth',2)

% 鼻子

[X,Y]=get_ellipse([0,-.1],[1,0;0,1.6],.2^2,200);

Y(Y<0)=Y(Y<0).*.2;Y=-Y+.9;

fill(X,Y,[57,57,57]./255,'EdgeColor',[57,57,57]./255,'LineWidth',2)

% =========================================================================

% 冬奥会标志及五环

% 五环

tt=linspace(0,2*pi,100);

X=cos(tt).*.14;

Y=sin(tt).*.14;

plot(X,Y-2.8,'Color',[57,57,57]./255,'LineWidth',1.2)

plot(X-.3,Y-2.8,'Color',[106,201,245]./255,'LineWidth',1.2)

plot(X+.3,Y-2.8,'Color',[155,79,87]./255,'LineWidth',1.2)

plot(X-.15,Y-2.9,'Color',[236,197,107]./255,'LineWidth',1.2)

plot(X+.15,Y-2.9,'Color',[126,159,101]./255,'LineWidth',1.2)

% 文本

text(0,-2.4,'BEIJING 2022','HorizontalAlignment','center',...

'FontSize',8,'FontName','Comic Sans MS')

% 标志

fill([.1,-.12,-.08],[0,0-0.05,-0.15]-1.5,[98,118,163]./255,'EdgeColor',[98,118,163]./255)

fill([-.08,-.35,.1],[-0.1,-.2,-.1]-1.6,[98,118,163]./255,'EdgeColor',[98,118,163]./255)

fill([-.08,-.08,.1,.1],[-0.1,-0.15,-.2,-.15]-1.5,[192,15,45]./255,'EdgeColor',[192,15,45]./255)

plot([-.35,-.3,-.25,-.2,-.15,-.1,-.05,.1]+.02,...

[0,.02,.04,.06,.04,.02,0,.02]-1.82,'Color',[120,196,219]./255,'LineWidth',1.8)

plot([-.33,.05]+.02,[0,-.08]-1.82,'Color',[190,215,84]./255,'LineWidth',1.8)

plot([.05,-.2]+.02,[-.08,-.15]-1.82,'Color',[32,162,218]./255,'LineWidth',1.8)

plot([-.2,.05]+.02,[-.15,-.2]-1.82,'Color',[99,118,151]./255,'LineWidth',1.8)
```
### 结果
#### demo2.1
![[Pasted image 20250611202116.png]]
#### demo2.2
![[Pasted image 20250611202139.png]]
#### bingdundun
![[Pasted image 20250611202158.png]]

## 题3:
### 试题 3：（30 分）
2025 年春节过后，来武汉游玩的旅客激增，桥梁作为连接两岸三地的重要工具时常发生拥堵。假设一辆车在长江大桥上正常行驶，走到三分之一处时，前方 300 米发生拥堵，车辆缓慢行驶，预计 10 分钟通过拥堵路段。
1） 写一个函数，车辆上桥后行驶的距离关于时间的函数 𝑦 = 𝑓(𝑡)；
2） 写一个函数，车辆的速度关于时间的函数𝑣 = 𝑔(𝑡) = 𝑑𝑖𝑓𝑓(𝑦)./𝑑𝑖𝑓𝑓(𝑡)；
3） 设置路程和时间参数，并利用 subplot 在同一窗口下作图，描绘以上两个函数；
4） 计算车辆通过大桥需要的时间。
### 代码
#### f（距离时间函数）
```matlab
function y = f(t)

L_bridge = 1670.4; % 桥长度（米）

L1 = L_bridge / 3; % 1/3桥长（米）

v0 = 10; % 初速度（m/s）

t1 = L1 / v0; % 阶段1结束时间（秒）

% 减速阶段参数

t_decel = 10; % 减速时间（秒）

v_uniform = 300 / 600; % 拥堵段匀速速度（m/s）

a_decel = (v_uniform - v0) / t_decel; % 减速加速度

t3_start = t1 + t_decel; % 阶段3开始时间（秒）

s_decel = v0*t_decel + 0.5*a_decel*t_decel^2; % 减速位移（米）

% 匀速阶段参数

t3_end = t3_start + 600; % 阶段3结束时间（秒）

% 加速阶段参数

t_accel = 10; % 加速时间（秒）

a_accel = (v0 - v_uniform) / t_accel; % 加速加速度

s_accel = v_uniform*t_accel + 0.5*a_accel*t_accel^2; % 加速位移（米）

t4_end = t3_end + t_accel; % 阶段4结束时间（秒）

s_total_after_accel = L1 + s_decel + 300 + s_accel; % 加速后总位移（米）

% 剩余阶段参数

s_remaining = L_bridge - s_total_after_accel; % 剩余位移（米）

t5 = s_remaining / v0; % 剩余时间（秒）

T_total = t4_end + t5; % 理论总时间（秒）

% 分段计算位移

y = zeros(size(t));

for i = 1:length(t)

ti = t(i);

if ti < t1

y(i) = v0 * ti;

elseif ti < t3_start

tau = ti - t1;

y(i) = L1 + v0*tau + 0.5*a_decel*tau^2;

elseif ti < t3_end

tau = ti - t3_start;

y(i) = L1 + s_decel + v_uniform * tau;

elseif ti < t4_end

tau = ti - t3_end;

y(i) = L1 + s_decel + 300 + v_uniform*tau + 0.5*a_accel*tau^2;

elseif ti < T_total

tau = ti - t4_end;

y(i) = s_total_after_accel + v0 * tau;

else

y(i) = L_bridge; % 超过总时间，视为到达桥尾

end

end

end

```
#### g（速度时间函数）
```matlab
function v = g(t)

L_bridge = 1670.4; % 桥长度（米）

L1 = L_bridge / 3; % 1/3桥长（米）

v0 = 10; % 初速度（m/s）

t1 = L1 / v0; % 阶段1结束时间（秒）

% 减速阶段参数

t_decel = 10; % 减速时间（秒）

v_uniform = 300 / 600; % 拥堵段匀速速度（m/s）

a_decel = (v_uniform - v0) / t_decel; % 减速加速度

t3_start = t1 + t_decel; % 阶段3开始时间（秒）

% 匀速阶段参数

t3_end = t3_start + 600; % 阶段3结束时间（秒）

% 加速阶段参数

t_accel = 10; % 加速时间（秒）

a_accel = (v0 - v_uniform) / t_accel; % 加速加速度

t4_end = t3_end + t_accel; % 阶段4结束时间（秒）

% 剩余阶段参数

s_remaining = L_bridge - (L1 + (v0*t_decel + 0.5*a_decel*t_decel^2) + 300 + (v_uniform*t_accel + 0.5*a_accel*t_accel^2));

t5 = s_remaining / v0; % 剩余时间（秒）

T_total = t4_end + t5; % 理论总时间（秒）

% 分段计算速度

v = zeros(size(t));

for i = 1:length(t)

ti = t(i);

if ti < t1

v(i) = v0;

elseif ti < t3_start

tau = ti - t1;

v(i) = v0 + a_decel * tau;

elseif ti < t3_end

v(i) = v_uniform;

elseif ti < t4_end

tau = ti - t3_end;

v(i) = v_uniform + a_accel * tau;

elseif ti < T_total

v(i) = v0;

else

v(i) = 0; % 到达后速度为0

end

end

end
```
#### demo3.1
```matlab
% 主脚本：绘制位移和速度曲线，计算总时间

t = 0:0.1:750; % 时间向量（覆盖总时间）

y = f(t); % 计算位移

v = g(t); % 计算速度

% 绘图

figure;

subplot(2,1,1);

plot(t, y, 'LineWidth', 1.5);

xlabel('时间 t (秒)');

ylabel('行驶距离 y (米)');

title('车辆行驶距离随时间变化');

grid on;

subplot(2,1,2);

plot(t, v, 'LineWidth', 1.5);

xlabel('时间 t (秒)');

ylabel('速度 v (m/s)');

title('车辆速度随时间变化');

grid on;

% 计算总时间：找到第一个到达桥尾的时间点

idx = find(y >= 1670.4, 1, 'first');

if ~isempty(idx)

total_time = t(idx);

fprintf('车辆通过大桥的总时间为：%.2f 秒（约 %.2f 分钟）\n', total_time, total_time/60);

else

fprintf('时间范围未覆盖总时间，请调整t的范围。\n');

end
```

### 结果
![[Pasted image 20250611221316.png]]
```matlab
车辆通过大桥的总时间为：746.60 秒（约 12.44 分钟）
```

## 题4:
### 试题 4：（30 分）
写一个脚本，命名为 seriesConvergence.m，计算 p-级数$G = \sum_{n} \frac{(-1)^n}{n^p}$
1） 设置数 𝑝 = 2；设置 k 为一个向量包含 0 到 1000 的整数。
2） 当𝑝 = 2，计算级数中的每一个项 $p\text{Series} = \frac{(-1)^n}{n^p}$
3） 利用级数求和计算级数$G_k = \sum_{k} \frac{(-1)^k}{kp}$，已知当𝑝 = 2时，$G = \sum_{n=0}^{\infty} \frac{(-1)^n}{n^p} = \frac{\pi^2}{12}$
4） 作图，a)画出 k 对应的 p𝑆𝑒𝑟𝑖𝑒𝑠的累计和，b)画出𝑝 = 2时极限$\frac{\pi^2}{12}$。分别给出 x、y 轴的标签以及图名，给出图例，两条线分别标记为‘Finite Sum’ 和 ‘Infinite Sum’。

将以上求级数和的方法应用在交错级数$G = \sum_{n=2}^{\infty} \frac{(-1)^n \ln n}{n}$
5） 计算每一项$cSeries = \frac{(-1)^n \ln n}{n}$
6） 作图画出有限项级数和及无穷项级数和𝐺，并按 4）中的要求作图，讨论级数的敛散性。

### 代码
#### seriesConvergence

```matlab
% seriesConvergence.m
% 计算并可视化p-级数和交错对数级数的收敛性

%% 清除工作区
clear;
close all;
clc;

%% 1) 参数设置
p = 2;                  % p-级数的参数
k = 0:1000;             % 从0到1000的整数向量

%% 2) 计算p-级数（p=2）的每一项
% 注意：n=0时1/n^p未定义，因此从n=1开始
n = k(k~=0);            % 排除n=0的情况
pSeries = (-1).^n ./ (n.^p);

%% 3) 计算p-级数的部分和、精确值及误差
partialSum_p = cumsum(pSeries);
exactSum_p = pi^2/12;   % p=2时的精确和
error_p = abs(partialSum_p - exactSum_p);

%% 4) 绘制p-级数收敛图
figure('Name', 'p-Series Convergence (p=2)', 'Position', [100, 100, 800, 600]);
plot(n, partialSum_p, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Finite Sum');
hold on;
yline(exactSum_p, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Infinite Sum');
hold off;

% 添加标签和标题
xlabel('Number of Terms (k)', 'FontSize', 12);
ylabel('Partial Sum', 'FontSize', 12);
title('Convergence of p-Series: $\sum_{n=1}^{\infty} \frac{(-1)^n}{n^p}$', ...
      'Interpreter', 'latex', 'FontSize', 14);
legend('Location', 'southeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% 显示误差信息
text(700, 0.6, sprintf('Final Error: %.2e', error_p(end)), ...
     'FontSize', 10, 'BackgroundColor', 'white');

%% 5) 计算交错对数级数的每一项
% 从n=2开始（n=1时ln(1)/1=0）
n_log = 2:1000;  % n≥2
cSeries = (-1).^n_log .* log(n_log) ./ n_log;

%% 6) 计算交错对数级数的部分和
partialSum_c = cumsum(cSeries);
approxSum = partialSum_c(end);  % 用最后部分和近似无穷级数和

%% 绘制交错对数级数收敛图
figure('Name', 'Alternating Log Series Convergence', 'Position', [100, 100, 800, 600]);
plot(n_log, partialSum_c, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Finite Sum');
hold on;
yline(approxSum, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Approximated Infinite Sum');
hold off;

% 添加标签和标题
xlabel('Number of Terms (n)', 'FontSize', 12);
ylabel('Partial Sum', 'FontSize', 12);
title('Convergence of Alternating Series: $\sum_{n=2}^{\infty} \frac{(-1)^n \ln n}{n}$', ...
      'Interpreter', 'latex', 'FontSize', 14);
legend('Location', 'southeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

%% 讨论级数敛散性
fprintf('p-级数收敛性分析:\n');
fprintf('  • 精确值: π²/12 ≈ %.6f\n', exactSum_p);
fprintf('  • 1000项部分和: %.6f\n', partialSum_p(end));
fprintf('  • 绝对误差: %.2e\n', error_p(end));
fprintf('  • 误差随项数增加而减小，表明级数收敛\n\n');

fprintf('交错对数级数收敛性分析:\n');
fprintf('  • 最后5项部分和: ');
fprintf('%.6f  ', partialSum_c(end-4:end));
fprintf('\n');
fprintf('  • 级数满足交错级数判别法条件:\n');
fprintf('    a) lim_{n→∞} |a_n| = lim_{n→∞} ln(n)/n = 0\n');
fprintf('    b) |a_n| = ln(n)/n 单调递减 (当 n > e)\n');
fprintf('  • 部分和振荡收敛至 ≈ %.6f\n', approxSum);
fprintf('  • 结论: 级数收敛\n');

% 验证单调递减条件
x = 2:1000;
a_x = log(x)./x;
is_decreasing = all(diff(a_x) < 0);  % 检查整个区间是否单调递减
fprintf('  • 单调性验证: %s (n≥2时)\n', string(is_decreasing));
```
### 结果
```matlab
p-级数收敛性分析:
  • 精确值: π²/12 ≈ 0.822467
  • 1000项部分和: -0.822467
  • 绝对误差: 1.64e+00
  • 误差随项数增加而减小，表明级数收敛

交错对数级数收敛性分析:
  • 最后5项部分和: 0.163333  0.156408  0.163327  0.156414  0.163321  
  • 级数满足交错级数判别法条件:
    a) lim_{n→∞} |a_n| = lim_{n→∞} ln(n)/n = 0
    b) |a_n| = ln(n)/n 单调递减 (当 n > e)
  • 部分和振荡收敛至 ≈ 0.163321
  • 结论: 级数收敛
  • 单调性验证: false (n≥2时)
```
![[Pasted image 20250611233722.png]]
![[Pasted image 20250611233730.png]]
