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

def merge_pair(pair,v_in):
    """
        word = 'T h e <\w>'
        pair = ('e', '<\w>')
        word_after_merge = 'T h e<\w>'
    输入:
        pair: Tuple[str, str] # 需要合并的字符对
        v_in: Dict[str, int]  # 合并前的vocab
    输出:
        v_out: Dict[str, int] # 合并后的vocab
    注意:
        当合并word 'Th e<\w>'中的字符对 ('h', 'e')时，'Th'和'e<\w>'字符对不能被合并。
    """


    v_out={}

    # 将pair中的两个字符用空格连接，
    # 然后使用re.escape进行转义处理
    # 这样做是为了确保字符中的特殊符号不会被正则表达式引擎误解释
    bigram=re.escape(' '.join(pair))

    # 创建一个匹配特定二元组的正则表达式模式, 只有前面、后面不是非空白字符
    # 才匹配h,e，这样就可以把Th, e<\w>排除在外
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for i in v_in:
        # 遍历当前的vocabulary，找到匹配正则的v时，才用合并的pair去替换变成新的pair new，如果没有匹配上，那就保持原来的。
        new=p.sub(" ".join(pair),i)
        # 统计替换后的单词出现的次数
        v_out[new]=v_in[i]
    return v_out