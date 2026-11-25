# LangChain AgentState 状态管理

> 来源: https://www.runoob.com/langchain/langchain-agent-state.html

---

# LangChain AgentState 状态管理

Agent 在运行过程中需要维护状态，例如对话消息、工具执行结果、学习进度以及结构化输出。

在 LangChain 1.x 中，AgentState 负责定义 Agent 的状态结构，而 LangGraph 负责状态的更新、合并和管理。

本章节介绍 AgentState 的默认字段、Reducer 合并机制、Middleware 流程控制、自定义状态，以及如何通过 Command 和 Checkpointer 管理状态。

---

## 一、AgentState 基本结构

AgentState 基于 Python 的 `TypedDict`，主要包含以下三个字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `messages` | 是 | 消息历史，使用 add\_messages 合并 |
| `jump_to` | 否 | Middleware 的瞬态流程控制字段 |
| `structured_response` | 否 | 结构化输出结果，不作为普通输入字段 |

可以通过 Python 查看当前版本的定义：

## 实例

from langchain.agents import AgentState

for name, annotation in AgentState.\_\_annotations\_\_.items():
    print(name, annotation)

print("必填字段：", AgentState.\_\_required\_keys\_\_)
print("可选字段：", AgentState.\_\_optional\_keys\_\_)

其中最常用的是 `messages`，保存用户消息、AI 回复以及工具执行结果。

**注意：**AgentState 只是状态结构定义，并不意味着数据已经保存到数据库。是否能够跨调用恢复，需要配置 Checkpointer。

---

## 二、messages：消息历史与 Reducer

`messages` 使用 add\_messages Reducer，负责合并新旧消息。

它支持三种主要操作：

- **追加：**新消息 ID 不存在时，添加到消息列表。
- **替换：**新消息 ID 与已有消息相同时，替换对应消息。
- **删除：**通过 RemoveMessage 删除指定 ID 的消息。

### 1、追加消息

## 实例

from langchain.messages import HumanMessage, AIMessage
from langgraph.graph.message import add\_messages

existing = [
    HumanMessage(content="你好", id="1"),
    AIMessage(content="你好！", id="2")
]

new = [HumanMessage(content="介绍 Python", id="3")]
result = add\_messages(existing, new)

print("合并前：", len(existing))
print("合并后：", len(result))
for msg in result:
    print(msg.id, msg.type, msg.content)

输出：

```
合并前： 2

合并后： 3

1 human 你好

2 ai 你好！

3 human 介绍 Python
```

### 2、相同 ID 替换消息

## 实例

from langchain.messages import AIMessage
from langgraph.graph.message import add\_messages

old = [AIMessage(content="原来的回答", id="msg\_1")]
new = [AIMessage(content="修改后的回答", id="msg\_1")]

result = add\_messages(old, new)
print(len(result), result[0].content)

输出：

```
1 修改后的回答
```

这里比较的是消息 `id`，而不是消息内容或消息类型。

### 3、RemoveMessage 删除消息

## 实例

from langchain.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add\_messages

messages = [
    HumanMessage(content="你好", id="1"),
    AIMessage(content="你好！", id="2"),
    HumanMessage(content="介绍 Python", id="3")
]

result = add\_messages(messages, [RemoveMessage(id="2")])
for msg in result:
    print(msg.id, msg.content)

输出：

```
1 你好

3 介绍 Python
```

**注意：**删除 Tool Calling 消息时，应同时考虑 AIMessage 和 ToolMessage 的关联关系，避免产生不符合模型 API 要求的消息历史。

---

## 三、jump\_to：Middleware 流程控制

`jump_to` 用于让 Middleware 在特定阶段控制 Agent 的下一步执行方向。

| 值 | 说明 |
| --- | --- |
| `"model"` | 转向模型执行流程 |
| `"tools"` | 转向工具节点，需要合法工具调用上下文 |
| `"end"` | 结束当前主循环或进入相应的结束钩子 |

`jump_to` 属于瞬态状态，用于控制执行流程，不适合保存长期业务数据。

### 实例：提前结束 Agent

下面使用 `@before_model` 拦截一个演示关键词，展示如何跳过模型调用。

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import before\_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

load\_dotenv()

@before\_model(can\_jump\_to=["end"])
def check\_question(state: AgentState, runtime: Runtime) -> dict | None:
    """拦截演示关键词。"""
    users = [m for m in state["messages"] if m.type == "human"]
    if users and "BLOCK\_DEMO" in str(users[-1].content):
        return {
            "messages": [AIMessage(content="请求已被演示规则拦截。")],
            "jump\_to": "end"
        }
    return None

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[check\_question],
    system\_prompt="你是一名中文编程助手。"
)

result = agent.invoke({"messages": [{"role": "user", "content": "BLOCK\_DEMO"}]})
print(result["messages"][-1].content)

输出：

```
请求已被演示规则拦截。
```

这里的中间件在模型执行前检查消息，如果符合条件，就添加一条 AIMessage 并跳转到结束流程。

**注意：**`jump_to="tools"` 不能凭空执行任意 Python 函数；工具节点仍然需要有效的 Tool Calling 请求。

---

## 四、structured\_response：结构化输出

如果通过 `create_agent()` 的 `response_format` 参数设置结构化输出，Agent 会将解析结果存入 `structured_response`。

这个字段通常不需要用户在输入 State 时提供，但可以从执行完成后的返回值中获取。

### 实例：推荐课程

## 实例

from pydantic import BaseModel, Field
from langchain.agents import create\_agent

class CourseRecommendation(BaseModel):
    """课程推荐。"""
    course: str = Field(description="课程名称")
    reason: str = Field(description="推荐理由")
    level: str = Field(description="学习难度")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    response\_format=CourseRecommendation,
    system\_prompt="你是编程学习顾问，请根据用户需求推荐学习方向。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我是零基础，推荐一个编程学习方向。"}]
})

rec = result["structured\_response"]
print("课程：", rec.course)
print("理由：", rec.reason)
print("难度：", rec.level)

如果模型正确完成结构化输出，可能得到：

```
课程： Python 基础教程

理由： Python 语法简洁，适合零基础学习

难度： 入门
```

返回的是符合定义的 Pydantic 数据结构，而不是必须自行从自然语言中提取字段。

具体的结构化输出机制由模型及 LangChain 集成决定。模型不支持、Schema 不兼容或解析失败时，可能抛出异常。

---

## 五、自定义 AgentState

除了默认字段，还可以通过继承 AgentState 添加业务状态。

例如，学习助手可以保存学习等级、已完成主题和当前目标。

## 实例

from typing\_extensions import NotRequired
from langchain.agents import AgentState

class LearningAgentState(AgentState):
    user\_level: NotRequired[str]
    completed\_topics: NotRequired[list[str]]
    current\_goal: NotRequired[str]

其中 `NotRequired` 表示初始状态可以省略该字段，但不会自动设置默认值。

读取时建议使用：

```
level = state.get("user_level", "入门")

topics = state.get("completed_topics", [])
```

创建 Agent 时指定：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[],

    state_schema=LearningAgentState

)
```

注意，自定义 State 仅负责描述字段，工具需要通过正式的状态更新机制才能修改这些字段。

---

## 六、ToolRuntime：读取 Agent 状态

LangChain 1.x 推荐通过 `ToolRuntime` 访问当前 Agent State。

例如，可以在工具中读取学习等级和已完成主题：

## 实例

from langchain.tools import tool, ToolRuntime

@tool
def view\_progress(runtime: ToolRuntime) -> str:
    """查看当前学习进度。"""
    level = runtime.state.get("user\_level", "入门")
    topics = runtime.state.get("completed\_topics", [])
    return f"等级：{level}；已完成：{'、'.join(topics) if topics else '无'}"

`runtime` 由 Agent 执行框架注入，模型不需要提供该参数。

因此，不应假设直接执行 `view_progress.invoke({})` 就能获得完整 Agent State。

如果需要修改 State，应使用下一节介绍的 `Command`。

---

## 七、Command：更新 AgentState

直接修改 `runtime.state` 中的字典或列表，并不是可靠的状态更新方式。

推荐使用 Command(update=...) 提交修改。

### 实例：记录学习主题

## 实例

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def complete\_topic(topic: str, runtime: ToolRuntime) -> Command:
    """将指定学习主题标记为已完成。"""
    topics = list(runtime.state.get("completed\_topics", []))
    if topic not in topics:
        topics.append(topic)

    return Command(update={
        "completed\_topics": topics,
        "messages": [ToolMessage(
            content=f"已记录完成主题：{topic}",
            tool\_call\_id=runtime.tool\_call\_id
        )]
    })

这里通过 `Command` 同时更新两个字段：

- `completed_topics`：保存新的学习主题列表。
- `messages`：追加一条与当前调用 ID 关联的工具结果。

在 Agent Tool Calling 场景中，返回 Command 时应保持对应的 ToolMessage，避免破坏消息协议。

**注意：**这里采用的是列表整体更新。如果多个工具并行修改同一字段，需要额外设计 Reducer 或顺序执行机制。

---

## 八、完整实例：学习进度 Agent

下面将自定义 State、ToolRuntime、Command 和 Checkpointer 结合起来，创建可以记录与查询学习进度的 Agent。

文件名：`agent_state_demo.py`

安装依赖：

```
pip install -U langchain langchain-deepseek langgraph python-dotenv
```

配置 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from typing\_extensions import NotRequired
from dotenv import load\_dotenv
from langchain.agents import create\_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

load\_dotenv()

class LearningState(AgentState):
    user\_level: NotRequired[str]
    completed\_topics: NotRequired[list[str]]

@tool
def complete\_topic(topic: str, runtime: ToolRuntime) -> Command:
    """记录一个已完成的学习主题。"""
    topics = list(runtime.state.get("completed\_topics", []))
    if topic not in topics:
        topics.append(topic)
    return Command(update={
        "completed\_topics": topics,
        "messages": [ToolMessage(content=f"已记录：{topic}", tool\_call\_id=runtime.tool\_call\_id)]
    })

@tool
def view\_progress(runtime: ToolRuntime) -> str:
    """查询当前学习进度。"""
    level = runtime.state.get("user\_level", "入门")
    topics = runtime.state.get("completed\_topics", [])
    return f"等级：{level}；已完成：{'、'.join(topics) if topics else '无'}"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[complete\_topic, view\_progress],
    state\_schema=LearningState,
    checkpointer=InMemorySaver(),
    system\_prompt="你是学习助手。用户要求记录主题时调用 complete\_topic，查询进度时调用 view\_progress。"
)

config = {"configurable": {"thread\_id": "learning\_demo\_001"}}

# 第一轮：记录学习主题
result = agent.invoke({
    "messages": [{"role": "user", "content": "我学完 Python 基础了，请记录。"}],
    "user\_level": "入门",
    "completed\_topics": ["HTML 基础"]
}, config=config)
print("第一轮状态：", result.get("completed\_topics"))

# 第二轮：读取此前保存的状态
result = agent.invoke({
    "messages": [{"role": "user", "content": "查询我已完成的主题。"}]
}, config=config)
print("第二轮状态：", result.get("completed\_topics"))
print("AI：", result["messages"][-1].content)

# 读取 Checkpointer 保存的状态
snapshot = agent.get\_state(config)
print("已保存：", snapshot.values.get("completed\_topics", []))

如果模型正确调用工具，第一轮状态应包含：

```
['HTML 基础', 'Python 基础']
```

第二轮使用相同的 `thread_id`，可以读取此前保存的状态。

这里的 `InMemorySaver` 只保存进程内存。程序退出后，状态通常会丢失。

**注意：**模型是否调用工具并不保证。需要强制记录业务数据时，应由确定性的应用逻辑执行，而不是只依赖模型选择工具。

---

## 九、购物车 Agent：状态管理实战

购物车是另一个常见的自定义 State 场景。

例如，需要保存购物车商品列表，并根据商品目录计算总价。

下面使用模拟商品数据，不创建真实订单，也不进行支付。

为了防止模型随意修改价格，商品价格由程序内部目录决定，而不是由模型传入。

## 实例

from typing\_extensions import NotRequired
from langchain.agents import create\_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

PRODUCTS = {"Python 教程": 49.90, "Java 教程": 39.90, "HTML 教程": 29.90}

class ShoppingState(AgentState):
    cart: NotRequired[list[str]]

@tool
def add\_to\_cart(item: str, runtime: ToolRuntime) -> Command:
    """将模拟商品加入购物车。"""
    cart = list(runtime.state.get("cart", []))
    if item in PRODUCTS:
        cart.append(item)
        message = f"已加入：{item}"
    else:
        message = f"商品不存在：{item}"

    return Command(update={
        "cart": cart,
        "messages": [ToolMessage(content=message, tool\_call\_id=runtime.tool\_call\_id)]
    })

@tool
def view\_cart(runtime: ToolRuntime) -> str:
    """查看购物车及总价。"""
    cart = runtime.state.get("cart", [])
    total = sum(PRODUCTS[item] for item in cart)
    return f"购物车：{'、'.join(cart) or '空'}；总价：¥{total:.2f}"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[add\_to\_cart, view\_cart],
    state\_schema=ShoppingState,
    system\_prompt="你是模拟购物车助手。商品价格以工具目录为准，不创建真实订单。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "把 Python 教程加入模拟购物车，并查看购物车。"}],
    "cart": []
})

print("购物车：", result.get("cart", []))
print("AI：", result["messages"][-1].content)

如果 Agent 成功执行相关工具，购物车状态为：

```
['Python 教程']
```

工具计算出的金额为：

```
购物车：Python 教程；总价：¥49.90
```

这里没有让模型提供 `price` 参数，而是由程序读取预先定义的模拟商品目录。

**注意：**如果模型并行调用多个修改购物车的工具，可能产生并发状态冲突。生产级购物车应结合数据库事务或专门的状态更新机制管理。

---

## 十、自定义 Reducer

AgentState 的自定义字段默认通常采用覆盖式更新。

如果需要合并多个更新，可以通过 `Annotated` 指定 Reducer。

### 实例：追加学习日志

## 实例

import operator
from typing import Annotated
from typing\_extensions import NotRequired
from langchain.agents import AgentState

class LearningState(AgentState):
    learning\_logs: NotRequired[Annotated[list[str], operator.add]]

当原有状态是：

```
["学习 HTML"]
```

新更新为：

```
["学习 Python"]
```

Reducer 合并后可以得到：

```
["学习 HTML", "学习 Python"]
```

需要注意，`operator.add` 只是列表拼接，不会自动去重。

如果需要唯一记录、严格顺序或防止重复执行，应设计专门的 Reducer 或在业务层处理。

---

## 十一、state\_schema 与 Middleware state\_schema

除了通过 `create_agent(state_schema=...)` 定义全局业务状态，Middleware 也可以声明自己的扩展字段。

| 方式 | 适用场景 |
| --- | --- |
| `create_agent(state_schema=...)` | Agent 共用的业务状态 |
| `Middleware state_schema` | 模型调用计数、内部流程标记等中间件状态 |

不能假设 create\_agent 中声明的同名字段总能覆盖 Middleware 中的字段定义。两个 Schema 的类型和 Reducer 必须保持兼容。

### 实例：中间件统计模型调用次数

## 实例

from typing\_extensions import NotRequired
from langchain.agents import create\_agent
from langchain.agents.middleware import AgentState, after\_model
from langgraph.runtime import Runtime

class TrackingState(AgentState):
    model\_call\_count: NotRequired[int]

@after\_model(state\_schema=TrackingState)
def count\_calls(state: TrackingState, runtime: Runtime) -> dict:
    """累计模型调用次数。"""
    return {"model\_call\_count": state.get("model\_call\_count", 0) + 1}

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[count\_calls]
)

result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print("模型调用次数：", result.get("model\_call\_count"))

对于只调用一次模型的简单任务，通常输出：

```
模型调用次数： 1
```

如果 Agent 需要多轮模型调用，该计数也会增加。

---

## 十二、Checkpointer：保存 AgentState

默认情况下，Agent 每次调用不一定保留上一次的会话状态。

如果需要跨多次调用恢复消息和自定义字段，可以配置 Checkpointer，并通过 `thread_id` 区分会话。

### 实例：保存多轮对话

## 实例

from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread\_id": "session\_001"}}

agent.invoke({
    "messages": [{"role": "user", "content": "我正在学习 Python。"}]
}, config=config)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我刚才说在学习什么？"}]
}, config=config)

print(result["messages"][-1].content)

snapshot = agent.get\_state(config)
print("消息数量：", len(snapshot.values["messages"]))

这里使用相同的 `thread_id`，因此第二轮可以读取此前保存的会话消息。

但 `InMemorySaver` 不是磁盘持久化方案。

需要在服务重启后继续保留数据时，应使用数据库支持的 Checkpointer。

---

## 十三、AgentState 与 Store 的区别

| 维度 | AgentState | Store |
| --- | --- | --- |
| 主要用途 | 当前线程的消息与任务状态 | 用户偏好、长期记忆和共享记录 |
| 跨调用 | 通过 Checkpointer 恢复 | 通过 Store 读写 |
| 数据组织 | 状态字段 | Namespace 与 Key |
| 持久化 | 取决于 Checkpointer | 取决于 Store 实现 |

例如，一个学习助手可以这样设计：

```
AgentState：

  当前对话历史

  当前学习章节

  本轮任务进度



Store：

  用户已完成的全部课程

  长期学习偏好

  跨会话学习记录
```

对于需要长期保存的真实业务数据，通常应使用持久化 Store 或数据库，而不是仅依赖 AgentState。

---

## 十四、常见问题

### 1、AgentState 是普通字典吗？

AgentState 使用 TypedDict 描述状态，运行时通常以字典形式传递，但实际合并行为还受到 Reducer 和 LangGraph 执行机制控制。

### 2、add\_messages 是否永远只追加消息？

不是。它还可以根据相同 ID 替换消息，以及使用 RemoveMessage 删除消息。

### 3、jump\_to="tools" 能直接执行某个 Python 函数吗？

不能。它只负责流程路由，工具节点仍需要合法的工具调用请求。

### 4、structured\_response 为什么不存在？

通常是没有配置 `response_format`，或结构化输出没有成功完成。

### 5、为什么直接修改 runtime.state 没有效果？

因为这不是可靠的图状态更新方式。工具应使用 `Command(update=...)` 提交状态变更。

### 6、Command 需要 ToolMessage 吗？

在 Agent 工具调用场景中，应提供与当前 Tool Call ID 对应的 ToolMessage，保持工具请求与工具响应正确关联。

### 7、自定义 State 可以省略字段吗？

可以使用 `NotRequired` 定义可选字段，但读取时需要自行处理默认值。

### 8、如何解决多个工具同时更新状态的冲突？

可以使用合适的 Reducer、限制并行执行或通过确定性的业务流程处理。

### 9、State 会自动保存到数据库吗？

不会。需要配置 Checkpointer，并选择适合的持久化实现。

### 10、AgentState 能保存用户长期学习进度吗？

可以配合 Checkpointer 保存线程状态。但对于需要跨多个会话共享的学习进度，更适合使用 Store 或业务数据库。

---

## 总结

AgentState 是 LangChain Agent 状态管理的核心。

本章节主要介绍了：

- `messages`：保存消息历史。
- `add_messages`：追加、替换和删除消息。
- `jump_to`：控制 Middleware 流程跳转。
- `structured_response`：保存结构化响应。
- `state_schema`：扩展自定义状态字段。
- `ToolRuntime`：在工具中访问 State。
- `Command`：提交状态更新。
- `Reducer`：定义状态合并规则。
- `Checkpointer`：保存和恢复会话状态。

可以简单记住：

```
读取状态：runtime.state

更新状态：Command(update={...})

保存会话：checkpointer + thread_id

长期数据：Store
```
