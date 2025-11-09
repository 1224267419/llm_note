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
