[链接](https://github.com/datawhalechina/happy-llm/tree/main)借助happy-llm中的第5,6章内容,实现llm从搭建到实践应用


## [build_LLaMA2](code/happy_LLM/build_LLaMA2.ipynb)
注意,这里搭建的是[ LLaMA2 大模型](https://datawhalechina.github.io/happy-llm/#/./chapter5/第五章 动手搭建大模型?id=_51-动手实现一个-llama2-大模型),LLaMa也是一个经典模型,辅助理解LLama系列

LLaMA2 模型结构如下图所示：

![alt text](./happy_llm.assets/LLama2.png)

### RMSNorm

$\mathrm{RMSNorm}(\mathbf{x})=\gamma\cdot\frac{\mathrm{x}}{\sqrt{\mathrm{RMS}(\mathbf{x})^2+\epsilon}}+\beta $,相较于传统的LayerNorm,不减去均值,仅仅除以均方$\mathrm{RMS}(\mathbf{x})^{2}=\frac{1}{d}\sum_{i=1}^{d}x_{i}^{2}$

简化归一化计算的同时,保留关键信息

在 [build_LLaMA2.ipynb](code\happy_LLM\build_LLaMA2.ipynb) ,Transformer类具有forward和generate两个方法

| **特性**     | **forward (前向传播)**                                       | **generate (生成/推理)**                                     |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **核心任务** | **预测概率**。给定一串词，算出词表中**每一个词**作为下一个词的概率。 | **创造文本**。利用 `forward` 的预测结果，实际决定下一个词是谁，并不断重复这个过程。 |
| **执行次数** | 通常只执行**一次**（在训练时）或每生成一个词执行一次。       | 包含一个 `for` 循环，**多次**调用 `forward`。                |
| **输出结果** | Logits（一堆原始数值/概率）和 Loss（损失）。                 | 最终人类可读的 Token 序列（完整的句子）。                    |
| **使用场景** | **训练阶段**为主，或者作为推理的基础计算单元。               | **推理/应用阶段**（如聊天机器人回复用户）。                  |

https://github.com/1224267419/happy-llm/blob/main/docs/chapter7/%E7%AC%AC%E4%B8%83%E7%AB%A0%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8.md

TODO
