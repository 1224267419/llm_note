# [1_Tokenizer](code\1.1 Tokenization\1.1 Tokenization.md) 

word -> token ,转换过程中保证token有相对独立完整的语义,用于后续任务.

- **Word词粒度**:英语天生按词分离,中文可以用**分词工具实现(如jieba**)
  - 优点:**保留了词语含义和边界**
  - 缺点:
    1. OOV(out of vocabulary)问题，对于**词表之外的词(<\>unk)无能为力**
    2. 无法处理**单词的形态关系和词缀**关系(dog和dogs,don’t和do not等对应关系)
- **char 字粒度**:
  a b c … z ! @ …  字母+标点
  - 优点:词表小,
  - 缺点:序列长,**单个token语义含量低**,**中文的词表也很大**
- **subword子词粒度**
  - 优点:**介于上述二者之间,**
  - 主流Subword分词算法，分别是**BPE, WordPiece和 Unigram Language Model**

一个可以可视化分词结果的[网站](https://tiktokenizer.vercel.app/)

### BPE(Byte-Pair Encoding) (GPT)

论文：[Neural Machine Translation of Rare Words with Subword Units](https:/arxiv.org/pdf/1508.07909)

核心思想：词频统计+词表合并(合并词表次数t or 期望词表大小V)
**从一个基础小词表开始，通过不断合并最高频的连续token对来产生新的token**,直至达到期望词表大小V

具体做法：输入训练语料和期望词表大小V

- a.**准备基础词表**：比如英文中26个字母加上各种符号，并初始化ID
- b,基于基础词表将准备的语料拆分为最小单元
- C.在语料上**统计单词内相邻单元对的频率，选择频率最高的单元对进行合并**
- d.重复第3步**直到达到预先设定的subword词表大小或下一个最高频率为1**

优点：可以有效地**平衡词汇表大小和编码步数**（编码句子所需的token数量，与词表大小和粒度有关)
缺点：**基于贪婪和确定的符号替换**，**不能提供**带概率的**多个**分词**结果**（这是相对于ULM而言的）；解码的时候面临**歧义问题**（比如对于同一个句子"Hello World"分词结果可能不同"Hell/o/world"或者"He/llo/world")

 [1_BPE.py](code\1_BPE.py) 为BPE的代码,建议结合代码理解

### **Byte-level BPE(BBPE)** (GPT2)

[**论文：*Neural Machine Translation with Byte-Level Subwords***](https://arxiv.org/pdf/1909.03341)

BPE的改进,将**字节(byte)视为基本token**,础词表使用**256的字节集，UTF-8编码**
**两个字节合并即可以表示Unicode**,比如中文、日文、阿拉伯文、表情符号等等

* **优点：**

  1. 效果与BPE相当，但**词表大为减小**
  2. 可以在**多语言之间通过字节级别的子词实现更好的共享**
  3. **高效的压缩效果：**&#x42;BPE 可以根据文本中的重复模式和常见片段来动态地生成词汇表，从而实现高效的文本压缩，尤其适用于包含大量重复内容的文本数据。
  4. **适用于多种类型的数据：**&#x42;BPE 可以应用于各种类型的数据，包括文本数据、图像数据等，因为它是基于字节级别的编码方法
* **缺点：**

  1. 编码序列时，长度可能会略长于BPE，**计算成本更高**
  2. 由**byte解码时可能会遇到歧义**，需要通过**上下文信息和动态规划来进行解码，**&#x4FDD;证输出有效的句子


### WordPiece

[论文：Fast WordPiece Tokenization](https://arxiv.org/pdf/2012.15524)

**核心思想：**&#x4E0E;BPE类似，也是从一个基础小词表出发，通过不断合并来产生最终的词表。主要的差别在于，BPE按频率来选择合并的token对，而**wordpiece按token间的互信息score来进行合并**

$$\text{score}=\frac{P(t_z)}{P(t_x)P(t_y)}$$其中z是token x ,y合并后token的token,$P(t_z)$是z的频率具体计算原因看论文,计算得分最高者即**合并后可以最大程度地增加训练数据概率的token**

* **优点：**可以**较好的平衡词表大小和OOV问题**

* **缺点：**可能会产生一些**不太合理的子词或者说错误的切分**；对**拼写错误非常敏感**；**前缀的支持不够好；**&#x4E00;种解决方案是：**将复合词拆开、将前缀也拆开**

### Unigram Language Model

和上述思想不同,词表**从大到小**。

![image-20251109224242253](./note.assets/image-20251109224242253.png)

**核心思想：**&#x521D;始化一个大词表，然后通过 unigram 语言模型计算**删除**不同**subword的损失**来代表subword的重要性(负对数似然)，**保留loss较大或者说重要性较高的subword**，&#x55;LM会倾向于保留那些以**较高频率出现在很多句子的分词结果中的子词**，因为这些子词如果被删除，其损失会很大 

* **优点：**

  1. 使用的训练算法可以**利用所有可能的分词结果，这是通过data sampling算法实现的**

  2. 提出一种基于语言模型的分词算法，这种语言模型可以**给多种分词结果赋予概率，从而可以学到其中的噪声**

  3. 使用时**也可以给出带概率的多个分词结果**
* **缺点：**

  1. **效果与初始词表息息相关，初始的大词表要足够好，比如可以通过BPE来初始化**
  2. 计算复杂(可以采用**viterbi 算法**)

## 总结

**BPE、VordPiece、Unigram**的缺点

- 假设输入文本使用空格来分隔单词，但并非所有语言都使用空格来分隔单词（如中文、韩文、日文、阿拉伯语)
- 可以使角特定语言的pre-tokenizer分词，但不太通用

为解决这个问题，**SentencePiece**

- 将**输入视为**输入**字节流**，包括空格
- 然后使用**Byte-level BPE或unigram算法来构建适当的词汇表**

Unigram算法经常在SentencePiece中使用，是AIBERT、TS、
mBART、Big Bird和XLNet等模型使用的tokenization算法

| Tokenization method          | Advantages                                                   | Disadvantages                                                |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| BPE                          | Allows for a large vocabulary size Handles rare and out-of-vocabulary words well Efficiently tokenizes subwords | Mayresultinmanysubwordsforasingleword Notidealforlanguageswithoutclearwordboundaries |
| WordPiece                    | Allowsfor alarge vocabulary size Handlesrare andout-of-vocabularywordswell Efficiently tokenizes subwords | Mayresultin many subwords fora singleword Notideal forlanguageswithoutclearwordboundaries |
| Unigram                      | Efficiently tokenizes subwords Scales well forlarge vocabularies | May produce many subwords for a single word Notidealforlanguageswithoutclearwordboundaries |
| SentencePiece with Unigram   | Can handle languages without clear word boundaries Efficientlytokenizessubwords | May produce many subwords for a single word                  |
| SentencePiece with BPE       | Can handle languages without clear word boundaries Efficiently tokenizes subwords | May result in many subwords fora single word                 |
| Byte-level BPE               | Canhandle anycharacter withoutout-of-vocabulary tokens Efficientlytokenizessubwords | May resultin many subwords fora singleword                   |
| Word-level Tokenization      | Simple to implement Fewer subwords per word than other methods | May not handle rare or out-of-vocabulary words well Not efficientforlarge vocabularies |
| Character-level Tokenization | Handles anywordorcharacter Efficientforsmallvocabularies     | Mayproducetoomanysubwordsforasingleword Notefficientforlargevocabularies |

todo:四种tokenizer的codeing

# [2_Embedding](code/1.2%20Embedding/1.2%20Embedding.md)

定义:**将高维数据转换为较低维度的向量**以**onehot为输入，稠密向量为输出**,常用的实现为**`nn.Embedding(vocab_size, embed_dim)`，**

## 2_1 One-Hot

定义:各个元素占据一个维度,值只有0,1两种

优点:简单容易实现

缺点:

1. **维度很大 (可以考虑PCA降维)**
2. **向量间相互垂直,不能表现出向量间的关系**



## 2_2 Word2Vec

![image](note.assets/image.png)

**Distributed representation**通过训练，将原来One-Hot编码的每个词都映射到一个**较短的词向量**上来，而这个较短的词向量的维度可以由自己在训练时根据任务需要来指定

**Word2Vec**的训练模型本质上是**有一个隐含层的神经元网络，最后取隐藏层的表示作为词向量。主要分为两种任务类型 CBOW 和 Skip-gram**

###  [CBOW](code\2_CBOW.py) 

根据上下文语境预测当前词,具体结构如图所示

![image-4](note.assets/image-4.png)

###  [Skip-gram.py](code\2_Skip-gram.py) 

如果**两个不同的单词有着非常相似的“上下文”**（也就是窗口单词很相似，比如“Kitty climbed the tree”和“Cat climbed the tree”），那么**通过我们的模型训练，这两个单词的嵌入向量将非常相似**。

for example:对于同义词“intelligent”和“smart”，我们觉得这两个单词应该拥有相同的“上下文”。而如”engine“和”transmission“这样相关的词语，可能也拥有着相似的上下文。

![image-1](note.assets/image-1.png)



### Word2Vec speed up

如果vocab_size很大,那么在softmax时会消耗大量时间,因此下面给出两种加快训练的方法



#### 霍夫曼树 Hierarchical softmax

根据词频来建立哈夫曼树。**将多分类转为二分类问题**。哈夫曼树的所有内部节点就类似之前神经网络隐藏层的神经元，其中，根节点的词向量对应我们的投影后的词向量，而**所有叶子节点就类似于之前神经网络softmax输出层的神经元**，叶子节点的个数就是词汇表的大小。

**优点**：

- **计算效率高**：由于是二叉树，使得计算量从$$V
  $$变为了$$log_2V
   $$
- **符合贪心优化思想**：高频词更加接近树根，能更快速得被检索，符合贪心优化的思想

**缺点：**

- 如果我们的训练样本里的**中心词词频很低**，要**沿着书向下走很久**了。负采样可以解决该问题

##### 负采样

目标函数$  g(w) = \prod_{u \in \{w\} \cup NEG(w)} p(u|\text{Context}(w)) $

- 输入是“词 \( w \) 的上下文 \( \text{Context}(w) \)”，我们需要让模型学习“上下文与 \( w \) 高度相关（正样本），与其他随机选的词（负样本）不相关”。
- \( \{w\} \) 表示**正样本**（即当前要训练的目标词 \( w \)）；\( NEG(w) \) 表示**负样本集合**（从词汇表中随机选的少量不相关词）。
- 目标是**最大化这个乘积**：让正样本的“相关性概率”尽可能高，同时让负样本的“相关性概率”尽可能低（因为负样本本就不该和上下文相关）。

2. 概率函数$  p(u|x_w, \theta^u) = \begin{cases} \sigma(x_w^T \theta^u), & L^w(u) = 1 \\ 1 - \sigma(x_w^T \theta^u), & L^w(u) = 0 \end{cases} $

- $ x_w $：上下文 $\text{Context}(w)$ 对应的词向量；$\theta^u$：词 \( u \) 对应的输出层参数向量。
- **指示函数**——当 $ u $ 是正样本（即 u=w）时，$L^w(u)=1$；当 $u$ 是负样本（即 $( u \in NEG(w) )$）时，$ L^w(u)=0$ 。

传统Softmax需要对**整个词汇表**（规模为 \( V \)）计算概率，而负采样通过“只选1个正样本 + 少量负样本（比如5~20个）”，将计算量从“与 \( V \) 线性相关”降到“与负样本数量 \( k \) 线性相关”$( k \ll V )$

通过“局部更新正/负样本的权重”，避免了对所有词汇的无效计算，从而大幅提升了训练效率。



##### 采样方式

概率采样，可以根**据词频进行随机抽样**，我们倾向于选择**词频比较大的负样本**，比如“的”，这种词语其实是**对我们的目标单词没有很大贡献的**。**Word2vec则在词频基础上取了0.75次幂，减小词频之间差异过大所带来的影响**，使得词频比较小的负样本也有机会被采到





### FastText

![image-3](note.assets/image-3.png)

fastText模型也只有三层：输入层、隐含层、输出层，

- **输入都是多个经向量表示的单词，输出都是一个特定的target**
- 隐含层是对多个词向量求平均**；**不同的是，**&#x43;BOW的输入是目标单词的上下文，fastText的输入是多个单词及其**n-gram特征**，这些特征用来表示单个文档；CBOW的输入单词被onehot编码过，fastText的输入特征是被embedding过；CBOW的输出是目标词汇，fastText的输出是**文档对应的类标**；
- 输出层，根据任务不同分为两种：
  1. **文本分类任务**：输出层是**Softmax 层**（或分层 Softmax），输出文本属于每个类别的概率。
  2. **词向量学习任务**：输出层与 Word2Vec 类似，通过 **负采样** 或 **分层 Softmax** 来预测上下文词，从而优化嵌入向量。

- **算法细节**

1. **损失函数：**&#x4EA4;叉熵损失

2. **Hierarchical Softmax：**根据**类别的频率构造霍夫曼树**来代替标准softmax，通过分层softmax可以将复杂度从N降低到logN，下图给出分层softmax示例：

   ![](note.assets/image-2.png)

   具体来说，这棵哈夫曼树除了根结点以外的所有非叶节点中都含有一个由参数`θ`确定的sigmoid函数，不同节点中的`θ`不一样。训练时隐藏层的向量与这个sigmoid函数进行运算，根据结果进行分类，若分类为负类则沿左子树向下传递，编码为0；若分类为正类则沿右子树向下传递，编码为1。

3. **N-gram feature：**&#x5C06;**文本内容按照子节顺序进行大小为N的窗口滑动操作**，最终形成窗口为N的字节片段序列。而且需要额外注意一点是n-gram可以根据粒度不同有不同的含义，有**字粒度的n-gram和词粒度的n-gram**。fasttext**针对n-gram额外优化的点为：过滤掉出现次数少的单词、使用hash存储、由采用字粒度变化为采用词粒度**

优点:

1. 训练速度快
   - 简化了隐藏层计算（求和平均代替复杂的矩阵乘法）。
   - 采用分层 Softmax 或负采样优化输出层。
2. 处理未登录词
   - 字符级 n-gram 嵌入让模型能处理训练集中未出现过的词。
3. 捕捉形态特征
   - 能学到词的前缀、后缀等形态规律，适合处理形态丰富的语言（比如德语、法语）。
4. 参数高效
   - 字符 n-gram 的数量远小于词的数量，嵌入矩阵的参数规模更小。



# 3_Attention

AI总结 :讨论了注意力机制在自然语言处理中的应用，包括其核心思想、在Transformer中的应用、计算复杂度优化、键值缓存技术、不同注意力机制的对比及相关代码实现等内容。关键要点包括：

## Attention核心思想

![self-attention](note.assets/image-13.png)

![image-14](note.assets/image-14.png)**网络应关注输入重要部分**，通过学习权重显式加权重要部分，Scaled Dot-Product Attention用于计算，Attention除以特定值$\sqrt{d_k}$使得计算前后的矩阵服从(0,1)分布),数值存在于softmax梯度较大的范围,易于模型收敛到更优值

## Transformer中的Attention

![attention](note.assets/image-1-1762745855112.png)![transformer](note.assets/image-1762745844954.png)

1. Self-Attention捕捉输入序列内部依赖关系，

   1. Encoder中当前token与所有token计算，

      ```
      获取输入的原始数据 X = input
      转换为Embedding并加上位置编码 X = emb(X) + pe(X)
      然后从X获取Q、K、V向量 Q, K, V = Qlinear(X) , Klinear(X) , Vlinear(X)
      计算Attention输出 Attention = softmax(Q@K^T / sqrt(d)) * V
      ```

   2. Decoder中只与之前token计算(mask模拟真实情况,防止信息泄露)；

      ```
      获取输入的原始数据 X = input
      转换为Embedding并加上位置编码 X = emb(X) + pe(X)
      # KV：encoder_output，Q：X
      Q= Qlinear(X)，KV则是同一层的Encoder的输出
      计算Attention输出 Attention = softmax(Q@K^T / sqrt(d)) @ V
      ```

      

2. Cross-Attention允许**解码器关注编码器输出**，主要用于解码器。

   ## Attention计算复杂度优化

Self Attention的时间复杂度是$$O(N^2)
$$的，它要对序列中的**任意两个向量都要计算相关度**，得到一个$$N^2$$大小的相关度矩阵，图示如下：

![attention](note.assets/image-2-1762746646246.png)

### **Sparse Attention**

从注意力矩阵上看**除了相对距离不超过k的、相对距离为k,2k,3k,…的注意力都设为0**，这样一来Attention就具有“**局部紧密相关和远程稀疏相关**”

Sparse Attention具有“局部紧密相关和远程稀疏相关”特性，但选择保留区域需人工决定且不易推广；

> 但是很明显，这种思路有两个**不足之处**:
>
> 1. 如何选择要保留的注意力区域，这是人工主观决定的，带有很大的不智能性
> 2. 它需要从编程上进行特定的设计优化，才能得到一个高效的实现，所以它不容易推广

![](note.assets/image-3-1762746743504.png)

完整的Sparse Attention实现可以参考 https://github.com/openai/sparse_attention/blob/master/attention.py







### [Linear Attention](https://arxiv.org/abs/2006.16236)

![img](note.assets/v2-ca4954b4c0f4b72fb2ced11b82e527ff_r.jpg)

图1中提到了多项式注意力多项式注意力和[RBF核注意力](https://zhida.zhihu.com/search?content_id=256552727&content_type=Article&match_order=1&q=RBF核注意力&zhida_source=entity)，这里也介绍下：
1）多项式注意力：是一种通过对输入向量进行多项式变换来计算注意力权重的方法。这种机制一般会将输入向量经过一个多项式函数（如二次函数*f*(*x*)=*ax^*2+*bx*+*c*或三次函数）来增强或抑制特定的特征，优点在于它能够捕捉输入数据中更复杂的非线性关系。注意：多项式函数的参数a/b/c会在训练中不断更新。
2）RBF核注意力：一种基于径向基函数（Radial Basis Function）的注意力机制。RBF 核是一种常用的核函数，可以用于度量向量之间的相似性。RBF 核的一个典型形式是高斯核：$K(x,y) = exp⁡(−||x−y||^2 / σ^2)$，其中，σ是一个可调的参数，控制核函数的“宽度”。

**主要思想就是将softmax拿掉，然后先算K转置V，这样算法复杂度从$$O(N^2d)
$$变为$$O(Nd^2)
$$,接近线性**(d是超参数,与seq_len无关)

Softmax是为了从 $$QK^T$$得到非负的权重值，如果有其他方法也可以得到，那么就可以去掉softmax，从而根据矩阵运算规律去先算 $$K^TV$$

> 注意 $$QKV$$向量的矩阵都是 $$N\times d$$的，矩阵 $$A∈R^{m×n}, B∈R^{n×k}$$，矩阵乘法 $$A \times B$$的复杂度为 $$O(mnk)$$

可以使用**核函数形式**的线性attention，使用两个非负的激活函数处理：

$$Attention(Q, K, V)_i = \frac{\sum_{j = 1}^{n} \text{sim}(q_i, k_j)v_j}{\sum_{j = 1}^{n} \text{sim}(q_i, k_j)}$$

$$sim(q_i, k_j) = \phi(q_i)^{\top} \phi(k_j)$$

如果“**Q在d那一维是归一化的、并且K在n那一维是归一化的”，那么 $$QK^T$$就是自动满足归一化了**：

$$Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})=softmax_{2}\left(\boldsymbol{Q}\right)softmax_{1}(\boldsymbol{K})^{\top}\boldsymbol{V}$$

**其中`softmax1`、`softmax2`分别指在第一个 $$n$$、第二个维度 $$d$$进行Softmax运算。也就是说，这时候我们是各自给 $$QK$$加`Softmax`，而不是算完 $$QK^T$$之后才加Softmax**

如果直接取 $$\phi(\boldsymbol{q}_i)=softmax(\boldsymbol{q}_i),\varphi(\boldsymbol{k}_j)=softmax(\boldsymbol{k}_j)$$。另外这个设计在CV中出现过不止一次，比如[A2-Nets](https://papers.nips.cc/paper/7318-a2-nets-double-attention-networks.pdf)也包含了同样的做法

![](note.assets/image-4-1762747112126.png)

完整代码在：https://github.com/lucidrains/linear-attention-transformer

```python
def linear_attn(q, k, v, kv_mask = None):
    dim = q.shape[-1]
#  mask attention
    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1) # query 在最后一个维度（特征维度）做 softmax
    k = k.softmax(dim=-2) # key 在倒数第二个维度（序列长度维度）做 softmax
# 对 query 进行缩放，避免 softmax 后梯度消失
    q = q * dim ** -0.5
	#计算 k 和 v 的加权聚合（O(N * dim) 复杂度）
    context = einsum('bhnd,bhne->bhde', k, v)
    
    # 第二步：用 q 与聚合后的 context 计算最终注意力（O(N * dim) 复杂度）
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)
```









## KV Cache键值缓存

> **一句话总结**：`kv cache`只出现在`transformer`结构的自回归的`decoder`中，为了避免`scaled dot-product attention`过程中的重复计算，将之前序列token计算过的KV缓存下来用。

### **原理**

用于自回归的decoder，避免scaled dot-product attention重复计算，可通过共用、窗口优化、量化与稀疏、存储与计算优化等方法优化。

1. 计算公式为：

$$Att_1(Q,K,V) = softmax(Q_1, K_1^T)V_1$$

$$Att_2(Q,K,V) = softmax(Q_2, K_1^T,)V_1+softmax(Q_2, K_2^T)V_2$$

会发现，由于 $$Q_1K_2^T$$这个值会被mask掉，$$Q_1$$在第二步参与的计算与第一步是一样的，而且第二步生成的 $$V_1$$也仅仅依赖于 $$Q_1$$。 $$V_2$$的计算也仅仅会依赖于 $$Q_2$$，与 $$Q_1$$无关。

所以第二步可简化为下图，后面的token以此类推

![](note.assets/diagram-2.png)

- **Hugging face的KV Cache实现**

if layer_past is not None:
        past_key, past_value = layer_past
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
    
```python
if use_cache is True:
    present = (key, value)
else:
    present = None

if self.reorder_and_upcast_attn:
    attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
else:
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
```


> 3. - - 对每个chunk的KV，Q和他们进行之前一样的FlashAttention获取这个chunk的结果
>      - 对每个chunk的结果进行reduce

![](note.assets/image-6.png)

**窗口优化示意图**

PPL:困惑度,	低 → 模型对预测下一个词“很有把握”

#### (a) Dense Attention（密集注意力）

- 传统 Transformer 的全注意力，当前 token（红色）与所有历史缓存的`T`个 token（蓝色）逐一计算注意力。

- **复杂度**：*O*($T^2$)（计算量随历史 token 数量平方增长，效率极差）。

#### (b) Window Attention（固定窗口注意力） 

代表作Longformer

- **机制**：仅在固定窗口`L`内计算注意力，超出窗口的`T-L`个历史 token 被 “丢弃（evicted）”。
- **复杂度**：*O*(TL)（效率比密集注意力好，但窗口外信息完全丢失）。

- **缺点**：丢弃早期 token 会导致模型 **“忘记” 前文**，**长文本连贯性严重受损**。
- 效率高,精度低

#### (c) Sliding Window w/ Re-computation（带重计算的滑动窗口）

- **机制**：滑动窗口时截断历史 token，对最近`L`个 token**重新计算缓存**。

- **复杂度**：*O*($TL^2$)（每次新 token 输入都要重算缓存，计算量陡增，效率差）。
- 精度高,效率低

#### (d) StreamingLLM 

- **机制**：引入**Attention Sink（注意力汇点，黄色 token）**，保留关键的历史 token 作为 “锚点”，同时缓存`L`个近期 token；超出的 token 被 “丢弃（evicted）”。
- **复杂度**：*O*($TL$)（效率与窗口注意力相当）。
- 精度高效率高





## **缓存与效果的取舍**：

MHA每个头关注不同子空间特征；MQA所有query head共享KV，影响学习效率和效果；GQA分组共享KV，平衡效果和缓存；MLA优化MQA，减少KV Cache但推理计算量增加。

## **DCA (Dual Chunk Attention)**：

将长文本分块，分别进行块内和块间注意力计算，能有效处理长文本，保持计算效率和低内存占用。   

## **S2-Attention (Shifted Sparse Attention)**：

将上下文长度分组，半注意力头中移位token保证相邻信息流动，可通过微调注意力掩码避免信息泄漏。



















