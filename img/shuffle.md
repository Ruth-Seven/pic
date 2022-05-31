# 机器学习\深度学习笔记



# 数据处理

### [Shuffle](https://blog.csdn.net/qq_19672707/article/details/88864207)

### 什么是Shuffle？

shuffle（中文意思：洗牌，混乱）。shuffle在机器学习与深度学习中代表的意思是，将训练模型的数据集进行打乱的操作。
原始的数据，在样本均衡的情况下可能是按照某种顺序进行排列，如前半部分为某一类别的数据，后半部分为另一类别的数据。但经过打乱之后数据的排列就会拥有一定的随机性，在顺序读取的时候下一次得到的样本为任何一类型的数据的可能性相同。

模型训练过程中需要Shuffle么？

Shuffle是一种训练的技巧，因为机器学习其假设和对数据的要求就是要满足独立同分布。所以任何样本的出现都需要满足“随机性”。所以在数据有较强的**“人为”**次序特征的情况下，Shuffle显得至关重要。

但是模型本身就为序列模型，则数据集的次序特征为数据的主要特征，并且模型需要学到这种次序规律时，则不可以使用Shuffle。否则会将数据集中的特征破坏。

### Shuffle为什么重要？

1. Shuffle可以防止训练过程中的模型抖动，有利于模型的健壮性
   假设训练数据分为两类，在未经过Shuffle的训练时，首先模型的参数会去拟合第一类数据，当大量的连续数据（第一类）输入训练时，会造成模型在第一类数据上的过拟合。当第一类数据学习结束后模型又开始对大量的第二类数据进行学习，这样会使模型尽力去逼近第二类数据，造成新的过拟合现象。这样反复的训练模型会在两种过拟合之间徘徊，造成模型的抖动，也不利于模型的收敛和训练的快速收敛
2. Shuffle可以防止过拟合，并且使得模型学到更加正确的特征
   NN网络的学习能力很强，如果数据未经过打乱，则模型反复依次序学习数据的特征，很快就会达到过拟合状态，并且有可能学会的只是数据的次序特征。模型的缺乏泛化能力。
   **如**：100条数据中前50条为A类剩余50条为B类，模型在很短的学习过程中就学会了50位分界点，且前半部分为A后半部分为B。则并没有学会真正的类别特征。
3. 为使得训练集，验证集，测试集中数据分布类似
   question：不同类别的data是在一起做shuffle，然后划分数据集；还是分开类别分别做对应的操作？
   有知道的小伙伴可以在下面留言

### 小结

其实Shuffle的作用归结起来就是两点，在针对随机性敏感的数据集上

+ 提升模型质量
+ 提升预测表现





# 学习





## **What is Covariate shift?**

An example：If you are building a neural network (or any other classifier for that matter), and if you train your classifier by showing examples of all black cats, then the performance of that classifier will not be so great when it is presented with pictures of non-black cats.

The reason is that the **distribution of the pixel intensity vector has shifted considerably**. And this will be true even if the original nonlinear decision boundary remains unchanged between the positive and negative examples.

用更数学化的方式表示，在

 ![公式](../../../%E9%87%8D%E8%A6%81%E6%96%87%E4%BB%B6/%E6%8A%80%E6%9C%AF%E7%AC%94%E8%AE%B0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%20Andrew/Deep-Learning-in-deeplearning.ai/3%23Structuring%20Machine%20Learning%20Projects/Structuring%20Machine%20Learning%20Projects.assets/equation.svg)



这个过程中，Covariate shift 问题就是 $P_{train}(y|x) = P_{test}(y|x))$， 但是 	$P_{train}(X) \neq P_{test}(x)$ 

这篇[文章](Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift)中提到了 Batch  Normalization 可以缓解Covariate shift 问题，从而加速深度学习网络的问题。（2019-12）







