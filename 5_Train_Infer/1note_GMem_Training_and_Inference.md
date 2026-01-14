# 1.显存占用

模型在训练时,首要的部分就是**计算显存占用情况**,否则模型会直接跑不起来

显存占用分为两部分:**框架**和**系统** , 系统部分和我们这里讨论的无关,

下面的例子默认为稠密LLM,不考虑MoE等架构

工具:[VRAM 计算器: NVIDIA GPU 与 Apple Silicon](https://apxml.com/zh/tools/vram-calculator)

## 训练

显存消耗的内容包括：模型参数（parameter）、优化器状态值（optimizer\_state）、激活值（activation）、梯度值（gradient）、输出数据（input）、临时变量（temporary）、自动梯度（autograd\_detail）、未知数量（unknown)。从用户侧可以将这些数据进行一个分类：

- **可估算值：**模型参数（parameter）、优化器状态值（optimizer\_state）、激活值（activation）、梯度值（gradient）、输出数据（input）
- **未命名数据：**临时时变量（temporary）、未知数据（unknown)
- **其他（框架）：**自动梯度（autograd\_detail）



### 静态值

#### 模型显存

$$\begin{aligned}fp32=4*params/(1024*1024*1024)\\fp16/bf16=2*params/(1024*1024*1024)\\fp8/int8=1*params/(1024*1024*1024)\end{aligned}$$

存储`checkpoint`时仅考虑模型本身，只要将显存上模型内容存储到磁盘中。举例：以1B模型为例，若采用fp32类型将其存储在磁盘上，其大小为：$$=4*1*10^9/(1024*1024*1024)\approx3.725GB\approx4GB$$ ,因此可以也可以跟据checkpoint大小反推模型参数量 : 如LLama13b，大约需要52GB存储空间 ; 
注意:**混合精度计算最后也使用fp32存储** , 静态值的计算公式不变

#### optimizer state

以Adam为例:

优化器中每个参数需要一个Momentum和一个Variance状态参数，在混合精度训练(16;32)中Adam还有一份模型参数副本:Adam参数器状态值计算
公式 : $$OptMem=(4+4+4)*Params/(1024*1024*1024)$$

- 模型副本 4 Bytes
- Momentum 参数 4 Bytes
- Variance 参数 4 Bytes



如果是8位优化器，则

$$8BitOptMem=(4+1+1)*Params/(1024*1024*1024)$$

- 模型副本 4 Bytes
- Momentum参数 1Byte
- Variance参数 1Byte



<h3> 动态值分析</h3>
<table>
<tr>
<th style="background-color: #E6E6FA; width: 50%;">激活值</th>
<th style="background-color: #FFFACD; width: 50%;">梯度值</th>
</tr>
<tr style="vertical-align: top;">
<td>
<p>激活值的大小跟模型参数、重计算、并行策略等相关，这里我们参考 Megtron 论文里面给的计算公式，来求解激活值所占用的显存大小。</p>
<p><strong>激活值显存消耗：</strong></p>
<p>s * b * h * (34 + 5 * a * s / h) * L * \gamma</p>
<p>(单位GB)，参数说明：</p>
<ul>
<li><strong>s</strong>: 序列长度（sequence length）, tokens的量</li>
<li><strong>b</strong>: 微批量大小（microbatch size）</li>
<li><strong>h</strong>: 隐藏层大小（hidden dimension size）</li>
<li><strong>a</strong>: attention的头数（number of attention heads）</li>
<li><strong>L</strong>: transformer模型的层数</li>
<li><strong>\gamma</strong>: 比例系数，当为fp16时 值等于 1 / (1024 * 1024 * 1024)</li>
</ul>
</td>
<td>
<p>梯度值与模型数据类型保持一致，计算如下（单位 GB）：</p>
<ul>
<li><strong>fp32的模型梯度值：</strong></li>
</ul>
<p>Mem_{fp32} = 4 * params / (1024 * 1024 * 1024)</p>
<ul>
<li><strong>fp16或者bf16模型：</strong></li>
</ul>
<p>Mem_{fp16/bf16} = 2 * params / (1024 * 1024 * 1024)</p>
</td>
</tr>
</table>

## 推理

显存占用： $$Infer \ Memory\approx1.2*Model\ Memory$$

具体的推导可以参考： [推理场景显存](https://kipp.ly/transformer-inference-arithmetic/)



## 优化

由于显存物理大小一定，我们获得额外空间的方式不外乎两种：

- 时间换空间；如，重计算
- 空间转移；如，多卡并行 / offload

其中，时间换空间通常会消耗算力、带宽；空间转移主要是消耗I/O带宽，有一定的时延，可能会降低吞吐

显存优化的过程一般是从模型算法本身到底层，可以参考的优化路径：
**多卡并行 -> 算子/数据类型 -> 消除框架副本 -> 显存管理 -> 底层API**

1. **多卡并行：**该手段相对来说是使用频率最高，且一般不会影响运算的精度，可以用2节中的计算公式为参考去设计新的TP/PP/DP/Zero/重计算的相关参数来降低显存消耗。缺点：这些方式可能会增加额外的带宽消耗
2. **算子优化：**选取精度相同但显存消耗更低的算子/方案。缺点：一般情况下，算子优化的过程耗时较长
3. **数据类型修改：**用低精度替换高精度数据。比如用fp16代替fp32，或者用更低的int8/int4。缺点：该方式可能影响训练收敛性/推理性能
4. **消除框架副本：**在AI框架（如pytorch）中有些数据是一些由框架产生的中间副本，可以进行优化消除；缺点：游湖成本较大
5. **显存管理：**通过显存管理的知识可知，框架的显存管理会产生显存碎片，通过优化显存管理来优化碎片；缺点：目前可用的手段较少
6. **底层API：** 在GPU的驱动库中/CUDA算子库中，不同API显存消耗不一样，我们可以用**显存消耗更小算子去替换大显存消耗算子**，比如FlashAttention；有些默认的操作会产生额外系统显存，也可以考虑替换更高版本优化后的API