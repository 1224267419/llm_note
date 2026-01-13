https://academy.langchain.com/courses
本文档为上述课程的学习笔记

目标:使用langchain构建一个能用的agent

```python
#使用简单chat模型实现,使用镜像站记得改地址

from langchain.chat_models import init_chat_model
import os
os.environ["OPENAI_API_BASE"] = "https://sg.uiuiapi.com/v1"
model = init_chat_model(model="gpt-5-nano",
                           # Kwargs passed to the model:
                        #在init_model时即可客制化model
    						temperature=1.0)
#使用其它模型也只需要修改模型名字和API_BASE即可使用
response = model.invoke("What's the capital of the Moon?")
print(response.content)
```

```python
#使用agent实现多轮对话,从而使用In-context

from langchain.agents import create_agent
os.environ["OPENAI_API_BASE"] = "https://sg.uiuiapi.com/v1"
agent = create_agent(model=model)

from langchain.messages import HumanMessage,AIMessage
#多轮对话实现上下文in-context构建输出
response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?"),
    AIMessage(content="The capital of the Moon is Luna City."),
    HumanMessage(content="Interesting, tell me more about Luna City")]}
)
```

```python
#流式输出
for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Tell me all about Luna City, the capital of the Moon")]},
    stream_mode="messages"
):

    # token is a message chunk with token content
    # metadata contains which node produced the token
    
    if token.content:  # Check if there's actual content
        print(token.content, end="", flush=True)  # Print token
```

使用system_prompt优化回复(更符合要求)

```python
system_prompt = "You are a science fiction writer, create a capital city at the users request."

scifi_agent = create_agent(    model="gpt-5-nano",    system_prompt=system_prompt)

response = scifi_agent.invoke(    {"messages": [question]})

print(response['messages'][1].content)
```

使用 LangChain 创建一个能够返回结构化数据的 AI 代理

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel
#通过指定
class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

agent = create_agent(
    model='gpt-5-nano',
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
    response_format=CapitalInfo
)

question = HumanMessage(content="What is the capital of The Moon?")

response = agent.invoke(
    {"messages": [question]}
)

response["structured_response"]
```

自定义工具tool

```python
from langchain.tools import tool
#通过给函数加上tool解释器即可,而在装饰器上的名字即agent调用时使用的的名字
#""" 部分为描述,
@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

#手动调用
square_root.invoke({"x": 467})

```

实际上,上面的代码块和下面的代码块是等价的

```python
@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5
```



在不使用工具的情况下,多花了8s才得到答案,而model是通过tool的描述来了解tool的功能的

而且如果使用了tools , response会记录tools的调用过程,保存在message.tool_calls中

```python
print(response["messages"][1].tool_calls)
```



搜索:利用带有搜索功能的tools,增强  训练数据集没涉及到部分的性能(实时新闻等)

```python
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

web_search.invoke("Who is the current mayor of San Francisco?")

#同样,在agent上附带上tools即可
agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search]
)

question = HumanMessage(content="Who is the current mayor of San Francisco?")

response = agent.invoke(
    {"messages": [question]}
)
```

memory

```python
from langgraph.checkpoint.memory import InMemorySaver  


agent = create_agent(
    "gpt-5-nano",
    checkpointer=InMemorySaver(),  
)
from langchain.messages import HumanMessage

question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [question]},
    config,  
)
question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]},
    config,  
)

pprint(response)

```

`InMemorySaver` 是 LangGraph 内置的「内存级检查点存储器」，负责将 Agent 的会话状态（包括历史消息、交互上下文）以「键值对」的形式存储在**内存中**，是 Agent 记忆的 “物理载体”。

两次 `agent.invoke` 都传入了相同的 `config`（`thread_id: "1"`），这是实现记忆的关键：意味着这两次invoke是同一段对话





多模态图片,需要将图片以base64编码后,在content中一起发送 , 音视频处理同理(需要音频模型)

```python
import base64

# Get the first (and only) uploaded file dict
uploaded_file = uploader.value[0]

# This is a memoryview
content_mv = uploaded_file["content"]

# Convert memoryview -> bytes
img_bytes = bytes(content_mv)  # or content_mv.tobytes()

# Now base64 encode
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Please describe the given cultural relic in detail"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = agent.invoke(
    {"messages": [multimodal_question]}
)

print(response['messages'][-1].content)
```

