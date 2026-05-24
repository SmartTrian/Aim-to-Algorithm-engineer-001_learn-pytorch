# pytorch_深度学习框架_线性模型

## 1.1.1监督学习模型步骤及问题

	1.Dataset：    准备数据集   
	2.Model：      模型选择设计(决策树、朴素贝叶斯...根据需求不同选择不同的模型)  
	3.Training：   训练  
	                            ps:有些模型不需要训练：KNN找最近邻
	4.inferring：  应用测试集 评估性能   
	
**问题**：过拟合？ 需要泛化能力！

1.在训练集上训练得很好：对测试集 露头就秒。称之为__过拟合__

2.为了模型能够在一般真实场景具有很好的能力。称之为__泛化能力__

**解决办法**

增加开发集：|训练集|开发集|测试集|=数据集

## 1.2.1模型设计

什么是模型？
	
	能够实现y=f(x) y对应输出 x对应输入
	
	以线性模型举例：y_hat(预测的值)=x*w

常见的模型？

	y=e^x
	y=ax^2+bx+c....

怎么选择模型？

![线性模型为例](https://github.com/SmartTrian/Aim-to-Algorithm-engineer-001_learn-pytorch/blob/main/001.png?raw=true)
	
	优先使用线性模型:y=w*x+b (y输出、w权重、x输入、b偏置):训练w与b
	
## 1.2.2模型设计_线性模型举例

### 因为建模更加复杂，往往更真实的应用场景会有***更多参数***和权重要去***挨个***迭代训练
#### 下面线性模型仅有一个权重相当于就是训练了一个参数

##### 对测试集的数据进行线性模型里代入：
**1.随机猜测该参数的值：**
我们猜测唯一的 参数 $w$为a
在数据点x处 挨个代入$\hat{y}$-$y$
**2.***评估***每个数据点的损失**
![评估当前训练某个参数的准确与否：损失函数](https://github.com/SmartTrian/Aim-to-Algorithm-engineer-001_learn-pytorch/blob/main/images/002.png?raw=true)
以上的计算 是一个样本的损失
***因为你不可能只有一个样本，故最终看的是所有样本的平均平方误差MSE***
![图片描述](https://github.com/SmartTrian/Aim-to-Algorithm-engineer-001_learn-pytorch/blob/main/images/003.png?raw=true)
**3.确定一个有MSE最小值对应输入的范围**
然后进行穷举法：
下面是一个损失函数的可视化实现##其实MSE足矣：
```js 
import numpy as np
import matplotlib.pyplot as plt

x_data=[1,2,3]
y_data=[2,4,6]

def foward(x):
  return x*w//这里是定义了模型

def loss(x,y):
  y_pred=foward(x)
  return (y_pred-y)**2//这里是用来计算损失函数

w_list=[]//存放权重
mse_list=[]//存放损失函数
for w in range(0,4.1,0.1)
  print('w=',w)
  l_sum=0
  for x_val y_val in zip(x_data,y_data):
    y_pred_val=forward(x_val)
    loss_val=(x_val,y_val)
    l_sum+=loss_val
    print('\t',x_val,y_val,y_pred_val,loss_val)
    print('MSE=',l_sum/3)
  w_list.append(w)##将里面的变量出现的值依次写进去
  mse_list.append(l_sum/3)
```
**4.真实的应用场景LOSS是横坐标是轮数，纵坐标是MSE**
通过你的轮数来判断第几轮的参数更加贴近
通过***visdom***调用库外部服务，实时画图查看loss
**5.记得存盘模型**
往往会遇到模型在训练得时候崩溃了，你之前的数据都没有了。
需要定时保存模型数据，每一轮存一个数据

## 本讲的作业是
![这一讲的作业](https://github.com/SmartTrian/Aim-to-Algorithm-engineer-001_learn-pytorch/blob/main/images/004.png?raw=true)
1.建立y=kx+b的模型
2.绘制损失函数

```js
#part1 建立模型
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x_data=np.array([1,2,3])
y_data=np.array([3,5,7])

def forward(x,w,b):
  return x*w+b
def loss(x,b,w,y):
  y_pred=forward(x,w,b)
  return np.mean((y_pred-y)**2)#这里省去了用zip遍历xy的所有取值，直接用np.mean就能够遍历所有的值

#part2 绘制损失函数的图
#**计算损失函数
w_list=[]
mse_list=[]
w=np.linspace(-5,5,100)##-5开始5结束中间存100个数
b=np.linspace(-5,5,100)
W,B=np.meshgrid(w,b)##将w的100种可能*b的100种可能并且W,B都是lens(w)*lens(b)的矩阵

MSE=np.zeros_like(W)##创造一个全0的矩阵
for i in range(len(w)):
  for j in range(len(b)):#挨个遍历w与b所有可能性
    MSE[j,i]=loss(x_data,B[j,i],W[j,i],y_data)#loss的输入得是B与W的二维矩阵，得包括可能性，并且给MSE里面反馈的是计算好的平方差
fig=plt.figure(figsize=(10,6))#拿画布告诉他这个图有多大
ax=fig.add_subplot(111,projection='3d')#1 行 1 列第 1 个子图且是3d的，ax是3D的意思
surf=ax.plot_surface(W,B,MSE,cmap='coolwarm')#把所有 (w,b,MSE) 点连起来，画出一张立体曲面
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE loss')
ax.set_title('loss surface of linear model y= x*w+b')
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.show()
```