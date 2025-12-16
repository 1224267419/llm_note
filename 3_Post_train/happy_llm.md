[链接](https://github.com/datawhalechina/happy-llm/tree/main)借助happy-llm中的第5,6章内容,实现llm从搭建到实践应用


## [build_LLaMA2](code/happy_LLM/build_LLaMA2.ipynb)
注意,这里搭建的是[ LLaMA2 大模型](https://datawhalechina.github.io/happy-llm/#/./chapter5/第五章 动手搭建大模型?id=_51-动手实现一个-llama2-大模型),LLaMa也是一个经典模型,辅助理解LLama系列

LLaMA2 模型结构如下图所示：

![alt text](./happy_llm.assets/LLama2.png)

### RMSNorm

$\mathrm{RMSNorm}(\mathbf{x})=\gamma\cdot\frac{\mathrm{x}}{\sqrt{\mathrm{RMS}(\mathbf{x})^2+\epsilon}}+\beta $,相较于传统的LayerNorm,不减去均值,仅仅除以均方$\mathrm{RMS}(\mathbf{x})^{2}=\frac{1}{d}\sum_{i=1}^{d}x_{i}^{2}$

简化归一化计算的同时,保留关键信息

TODO: https://datawhalechina.github.io/happy-llm/#/./chapter5/%E7%AC%AC%E4%BA%94%E7%AB%A0%20%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B5.1.5 
LLaMA2 Decoder Layer

