import re, collections

# text = "The aims for this subject is for students to develop an understanding of the main algorithms used in naturallanguage processing, for use in a diverse range of applications including text classification, machine translation, and question answering. Topics to be covered include part-of-speech tagging, n-gram language modelling, syntactic parsing and deep learning. The programming language used is Python, see for more information on its use in the workshops, assignments and installation at home."
text = 'low '*5 +'lower '*2+'newest '*6 +'widest '*3

text = 'low '*3 +'lower '*2+'newest '*6 +'widest '*3+'low '*2
'''
先统计词频
'''


def get_vocab(text):
    # 初始化为 0
    vocab = collections.defaultdict(int)
    # 去头去尾再根据空格split
    for word in text.strip().split():
        # note: we use the special token </w> (instead of underscore in the lecture) to denote the end of a word
        # 给list中每个元素增加空格，并在最后增加结束符号，即文中提到的_,用于表示文章结尾,
        # 同时统计单词出现次数
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab


print(get_vocab(text))

def get_state(vocab):
    # 在访问不存在的键时，可以自动创建并返回一个默认值,用来方便地处理各种没有预设值的情况。
    pairs=collections.defaultdict(int)

    # items() 方法用于返回一个字典的可遍历的键值对（元组）数组。
    for word,freq in vocab.items():
        # 便利重复的词对
        for i in range(len(word)-1):
            pairs[word[i],word[i+1]]+=freq
    return  pairs

def merge_pair(pairs,text):
    # 把pair拆开，然后用空格合并起来，然后用\把空格转义
    sorted_by_value_asc = sorted(pairs.items(), key=lambda item: item[1],reverse=True)
    new_pair=sorted_by_value_asc[0]