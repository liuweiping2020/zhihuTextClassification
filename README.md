# 知乎看山杯文本分类竞赛
# zhihuTextClassificaion

Link:https://biendata.com/competition/zhihu/

大约7月10日左右，学校开始放暑假，我也闲了起来。
正好实验室有几个同学在参加知乎的文本分类比赛，我看任务不太复杂，想着做个比赛学习一下。
室友给我发了[覃秉丰老师讲的关于本次比赛的框架](https://github.com/Qinbf/Tensorflow/tree/master/Tensorflow%E5%9F%BA%E7%A1%80%E4%BD%BF%E7%94%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%BA%94%E7%94%A8)
参照着这个框架，结合自己的一些想法进行实验。
做了大概三个星期，到8月5号放假回家，当时做到30名，线上评测F1得分0.40108。
由于水平有限，途中遇到了不少问题，只做了[textCNN](https://github.com/buptchan/zhihuTextClassification/tree/master/textCNN)和[fastText](https://github.com/buptchan/zhihuTextClassification/tree/master/fastText)两个模型，没有做融合，其中大部分时间在做textCNN。

## 1.数据预处理
此次提供的数据为知乎上标题（title）和描述(description)的文本信息，标题和描述均提供了word（词）和char（字）的数据。
其中词语/字经过脱敏处理，文本为词语/字的编号组成的序列。
主办方提供了word和char经过预训练后的Embedding矩阵，选手可直接使用其进行初始化。

以标题（Title）的word数据为例，原数据为word的编号组成的序列，则首先需要将每个word的编号映射为一个int类型的id，例如将 ‘w2565’ 映射为19234。char数据同理。

本次任务的类别有1999类，提供的标签信息是该问题所属类别的编号，但在TensorFlow中我们需要提供的label为编号的One-hot coding。如果在预处理中进行编号到One-hot coding的转化，那么会是label占据大量的存储空间。因此这个步骤在每次读取batch时进行。

因为我未使用char的数据，经过预处理将得到title和description的word id 序列，以及对应的label。

| question id | title word id | desc word id | topic label |
| :--: | :--- |:---|:---|
| 	0  | 305,13549,22752,11,7225,2565,1106,16,...   |231,54,1681,54,11506,5714,7,...   |1277304323405234 ,3738968195649774859  |
| 1  | 377,54,285,57,349,54,108215,...   |12508,1380,72,27045,276,111   |                    -3149765934180654494  |
| 2  | 875,15450,42394,15863,6,95421,...  |140340,54,48398,54,140341,54,12856,... |-760432988437306018|   

接着，对数据进行打乱并以9:1的比例划分为训练集与验证集。（10 fold CV）

## 2. textCNN
在本次比赛中，textCNN以其稳定的表现受到众多选手的欢迎。textCNN的结构很简单，经过一次卷积之后取max pooling(最大激活值)，送入分类网络即可。
虽然简单，但却是最好用的模型，精髓就在于卷积与max pooling操作。其中卷积相当于实现了N-GRAM的效果，max pooling则起到了去掉句子不等长对提取特征的影响。
为了提取到更复杂的非线性特征，我尝试在textCNN中增加了一层卷积层，然而适得其反。或许文本相较于图像，本身就已经包含了抽象的语义信息，并不需要过多地进行变换。

在QinBF老师提供的模型的基础上，增加了一个全连接层，并将title序列长度截短到40，description长度截短到120。分别送入一个具有窗口大小为3，4，5的卷积网络中，接着把max pooling输出的向量进行拼接，放入全连接网络进行分类，以BCE loss(Binary Cross Entropy loss)作为多分类的损失函数。

在实验过程中，还发现了两个可以提升性能的点：
- **使用[Batch Normalization](https://arxiv.org/abs/1502.03167)** : 使用Batch Normalization（以下简称BN）可以加快网络收敛，并且提高收敛时的准确率。
- **通过预训练的Embedding向量对网络进行初始化**： 对网络进行一定程度训练后，保存此时的Embedding向量，并作为之后训练网络的初始化向量。该方法一定程度上可以提升效果，但有造成过拟合的风险。
