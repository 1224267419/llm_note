# 1_Tokenizer

word -> token ,转换过程中保证token有相对独立完整的语义,用于后续任务.

- **Word词粒度**:英语天生按词分离,中文可以用**分词工具实现(如jieba**)
  - 优点:**保留了词语含义和边界**
  - 缺点:
    1. OOV(out of vocabulary)问题，对于**词表之外的词无能为力**
    2. 无法处理**单词的形态关系和词缀**关系
- **char 字粒度**:
  - 优点:词表小,
  - 缺点:序列长,**语义含量低**
- **subword子词粒度**
  - 优点:**介于上述二者之间,**
  - 主流Subword分词算法，分别是**BPE, WordPiec e和 Unigram Language Model**

一个可以可视化分词结果的[网站](https://tiktokenizer.vercel.app/)

## BPE

论文：[Neural Machine Translation of Rare Words with Subword Units](https:/arxiv.org/pdf/1508.07909)

核心思想：**从一个基础小词表开始，通过不断合并最高频的连续token对来产生新的token**

具体做法：输入训练语料和期望词表大小V

- a.**准备基础词表**：比如英文中26个字母加上各种符号，并初始化ID
- b,基于基础词表将准备的语料拆分为最小单元
- C.在语料上**统计单词内相邻单元对的频率，选择频率最高的单元对进行合并**
- d.重复第3步**直到达到预先设定的subword词表大小或下一个最高频率为1**

优点：可以有效地**平衡词汇表大小和编码步数**（编码句子所需的token数量，与词表大小和粒度有关)
缺点：**基于贪婪和确定的符号替换**，**不能提供**带概率的**多个**分词**结果**（这是相对于ULM而言的）；解码的时候面临**歧义问题**（比如对于同一个句子"Hello World"分词结果可能不同"Hell/o/world"或者"He/llo/world")

 [1_BPE.py](code\1_BPE.py) 为BPE的代码,建议结合代码理解





, WordPiec e和 Unigram Language Model