# LangChain Agent API 参考

> 来源: https://www.runoob.com/langchain/langchain-agent-api.html

---

# LangChain Agent API 参考

LangChain 1.x 提供了统一的 Agent API，可以将大语言模型、工具调用、Middleware、对话记忆和结构化输出组合为一个可运行的智能体。

其中最核心的函数是 create\_agent()。

与旧版 AgentExecutor 不同，LangChain 1.x 的 create\_agent() 基于 LangGraph 实现，返回一个编译后的 CompiledStateGraph，支持同步、异步、流式输出、状态持久化和人工介入。

本章节系统整理 create\_agent() 的完整参数、CompiledStateGraph 常用方法、AgentState 结构、运行时上下文和各种配置的实际用法。

---

## 一、create\_agent() 是什么？

`create_agent()` 是 LangChain 1.x 创建 Agent 的核心入口。

一个典型的 Agent 执行流程如下：

```
用户消息

    ↓

   Agent

    ↓

 调用 Chat Model

    ↓

 是否需要工具？

    |

    ├── 否 → 生成最终回复 → 结束

    |

    └── 是 → 调用 Tool

                 ↓

              工具结果

                 ↓

             再次调用模型

                 ↓

               循环判断
```

开发者只需要提供模型、工具和必要的配置，LangChain 就可以自动组织模型与工具之间的执行循环。

### 最简单的 Agent

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    system\_prompt="你是一个 Python 编程助手。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "什么是 Python？"}]
})

print(result["messages"][-1].content)

这里没有注册工具，因此 Agent 主要执行模型调用，不会进入普通的工具调用循环。

如果提供工具，模型可以根据问题选择调用对应工具。

---

## 二、create\_agent() 完整函数签名

LangChain 当前 API 的主要函数签名如下：

```
create_agent(

    model,

    tools=None,

    *,

    system_prompt=None,

    middleware=(),

    response_format=None,

    state_schema=None,

    context_schema=None,

    checkpointer=None,

    store=None,

    interrupt_before=None,

    interrupt_after=None,

    debug=False,

    name=None,

    cache=None,

    transformers=None

)
```

其中 `model` 是必填参数，其余参数都有默认值。

### 参数速查表

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model` | str | BaseChatModel | 必填 | 聊天模型或模型标识 |
| `tools` | Sequence | None | None | 可用工具列表 |
| `system_prompt` | str | SystemMessage | None | None | 系统提示词 |
| `middleware` | Sequence[AgentMiddleware] | () | Agent 中间件 |
| `response_format` | ResponseFormat | type | dict | None | None | 结构化输出配置 |
| `state_schema` | type[AgentState] | None | None | 自定义 Agent 状态结构 |
| `context_schema` | type | None | None | 运行时上下文类型 |
| `checkpointer` | Checkpointer | None | None | 对话和执行状态持久化 |
| `store` | BaseStore | None | None | 跨会话长期数据存储 |
| `interrupt_before` | list[str] | None | None | 指定节点执行前暂停 |
| `interrupt_after` | list[str] | None | None | 指定节点执行后暂停 |
| `debug` | bool | False | 启用图执行调试日志 |
| `name` | str | None | None | Agent 图的名称 |
| `cache` | BaseCache | None | None | 图执行缓存后端 |
| `transformers` | Sequence[TransformerFactory] | None | None | 注册额外的流事件转换器 |

参数看起来比较多，但实际开发时通常只需要使用 model、tools 和 system\_prompt。

随着功能增加，再逐步加入 Middleware、Checkpointer、Store 等配置。

---

## 三、model：语言模型

`model` 指定 Agent 使用的大语言模型。

支持两种方式：

- 模型名称字符串。
- BaseChatModel 实例。

### 方式一：字符串指定模型

## 实例

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[]
)

字符串一般采用：

```
provider:model_name
```

需要提前安装对应模型服务商的集成包。

例如 DeepSeek：

```
pip install -U langchain-deepseek
```

### 方式二：传入模型实例

## 实例

from langchain\_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    timeout=30,
    max\_retries=2
)

agent = create\_agent(model=model, tools=[])

这种方式适合需要配置超时、重试以及其他模型参数的场景。

**注意：**模型必须具备对应的功能。例如，如果 Agent 需要调用工具，底层模型和集成接口就必须支持工具调用。

---

## 四、tools：注册工具

`tools` 指定 Agent 可以使用的工具集合。

常见工具形式包括：

- 使用 `@tool` 装饰的函数。
- BaseTool 实例。
- 符合要求的 Python Callable。
- 受支持的工具定义字典。

Pydantic BaseModel 通常用于定义工具参数 Schema，而不是直接替代可执行的 Python 工具函数。

### 实例：注册多个工具

## 实例

from langchain.agents import create\_agent
from langchain.tools import tool

@tool
def get\_weather(city: str) -> str:
    """查询城市模拟天气。"""
    return f"{city}：模拟晴天，25°C"

@tool
def calculate\_sum(a: float, b: float) -> float:
    """计算两个数字之和。"""
    return a + b

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[get\_weather, calculate\_sum],
    system\_prompt="你是工具助手。根据用户问题选择合适的工具。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请使用工具计算 25 加 17。"}]
})

print(result["messages"][-1].content)

模型可以根据工具名称、说明和参数 Schema 选择需要调用的工具。

**注意：**工具注册并不意味着模型一定会调用工具。如果业务必须执行某个操作，应使用确定性工作流或进一步约束执行流程。

---

## 五、system\_prompt：系统提示词

`system_prompt` 用于定义 Agent 的角色、目标和行为规则。

支持普通字符串或 SystemMessage。

### 实例

## 实例

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    system\_prompt="""你是菜鸟教程 RUNOOB 的 Python 学习助手。

回答要求：
1. 优先使用中文。
2. 给出简洁的技术解释。
3. 必要时提供代码示例。
4. 不确定的信息不要编造。"""
)

system\_prompt 在模型调用时作为系统消息使用。

如果需要根据时间、用户身份或其他上下文动态生成提示词，可以使用 `@dynamic_prompt` Middleware。

**注意：**system\_prompt 不能替代程序中的身份验证、数据库权限或工具执行安全检查。

---

## 六、middleware：Agent 中间件

`middleware` 用于在 Agent 执行过程中加入自定义逻辑。

例如：

- 模型调用之前检查消息。
- 动态修改系统提示词。
- 限制工具调用次数。
- 模型失败时自动重试。
- 在执行敏感工具前请求人工审批。

### 常用 Middleware Hook

| Hook | 执行时机 |
| --- | --- |
| `before_agent` | Agent 开始执行时 |
| `before_model` | 每次模型调用前 |
| `after_model` | 每次模型调用后 |
| `after_agent` | Agent 正常完成后 |
| `wrap_model_call` | 包裹模型调用，允许修改请求及处理响应 |
| `wrap_tool_call` | 包裹工具执行，支持异常处理和重试 |
| `dynamic_prompt` | 动态生成模型系统提示词 |

### 实例：使用动态提示词

## 实例

from datetime import datetime
from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def inject\_date(request: ModelRequest) -> str:
    today = datetime.now().date().isoformat()
    return f"你是 Python 学习助手。当前服务器日期：{today}。"

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    middleware=[inject\_date]
)

如果需要用户当地日期，应使用可信用户时区，而不是直接依赖服务器默认时间。

多个 Middleware 可以组合使用。对于包裹式 Hook，通常是列表中靠前的 Middleware 位于外层。

---

## 七、response\_format：结构化输出

`response_format` 用于让 Agent 返回符合指定 Schema 的结构化数据。

支持直接传入 Pydantic 模型，也可以显式指定结构化输出策略。

### 主要策略

| 策略 | 说明 |
| --- | --- |
| `ToolStrategy` | 使用模型工具调用能力生成结构化输出 |
| `ProviderStrategy` | 使用服务商原生结构化输出能力 |
| `AutoStrategy` | 根据模型能力选择适用策略 |

直接传入 Schema 时，LangChain 可以尝试根据模型能力选择适当的策略。

### 实例：Pydantic 结构化输出

## 实例

from pydantic import BaseModel, Field
from langchain.agents.structured\_output import ToolStrategy

class CourseInfo(BaseModel):
    name: str = Field(description="课程名称")
    level: str = Field(description="课程难度")
    language: str = Field(description="编程语言")

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    response\_format=ToolStrategy(CourseInfo)
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "提取：Python3 基础教程，入门级，Python 语言。"
    }]
})

print(result["structured\_response"])

结构化输出通常保存在：

```
result["structured_response"]
```

它与普通的：

```
result["messages"][-1].content
```

是不同的数据读取方式。

**注意：**不是每个模型都支持所有结构化输出策略。结构化数据通过 Schema 校验，也不意味着其中的业务事实已经得到验证。

---

## 八、state\_schema：自定义 Agent 状态

`state_schema` 用于扩展 Agent 的状态结构。

默认状态主要保存消息、流程控制和可选的结构化响应。

如果需要加入业务字段，可以定义继承 AgentState 的 TypedDict。

### 实例：添加任务类别

## 实例

from typing import NotRequired
from langchain.agents import AgentState, create\_agent

class CustomState(AgentState):
    task\_category: NotRequired[str]

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    state\_schema=CustomState
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "你好"}],
    "task\_category": "general"
})

print("任务类别：", result.get("task\_category"))

这里扩展了一个普通状态字段：

```
task_category
```

需要注意，定义状态字段并不会自动让模型更新它。

如果希望在执行过程中修改该字段，需要在相应的 Middleware、工具或自定义节点中显式返回状态更新。

**建议：**如果某个状态字段只服务于一个 Middleware，可以将其状态 Schema 放在对应 Middleware 中，避免全局 AgentState 不断膨胀。

---

## 九、context\_schema：运行时上下文

`context_schema` 用于定义执行时传入的上下文信息。

例如：

- 当前用户 ID。
- 用户所在时区。
- 应用所属租户。
- 当前请求的业务权限信息。

运行时上下文和 AgentState 的区别是：

| 对比 | Agent State | Runtime Context |
| --- | --- | --- |
| 主要用途 | 保存执行过程中的状态 | 提供当前请求的外部上下文 |
| 是否适合被节点更新 | 可以通过状态更新修改 | 通常作为当前执行的只读配置使用 |
| 是否由 Checkpointer 保存 | 可以 | 不会自动作为状态持久化 |
| 典型内容 | messages、任务进度 | user\_id、timezone |

### 实例：向工具注入用户身份

## 实例

from dataclasses import dataclass
from langchain.agents import create\_agent
from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
    user\_id: str

@tool
def get\_my\_profile(runtime: ToolRuntime[UserContext]) -> str:
    """查看当前用户的模拟账户标识。"""
    return f"当前用户 ID：{runtime.context.user\_id}"

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[get\_my\_profile],
    context\_schema=UserContext
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "请调用工具查看我的账户标识。"}]},
    context=UserContext(user\_id="student-001")
)

print(result["messages"][-1].content)

这里的 `runtime.context.user_id` 不是由模型生成的工具调用参数，而是由应用调用方提供。

**安全提醒：**在 Web 服务中，user\_id 必须由可信后端根据登录状态确定，不能允许浏览器传入任意用户 ID。

---

## 十、checkpointer：对话持久化

`checkpointer` 用于保存同一线程中的 Agent 执行状态。

常见应用包括：

- 多轮对话记忆。
- 恢复中断的执行。
- 查看历史状态。
- 调试某个对话线程。

### 实例：InMemorySaver

## 实例

from langgraph.checkpoint.memory import InMemorySaver

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread\_id": "conversation-001"}}

agent.invoke({
    "messages": [{"role": "user", "content": "我正在学习 Python。"}]
}, config=config)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我刚才说自己在学什么？"}]
}, config=config)

print(result["messages"][-1].content)

相同 thread\_id 下，Agent 可以继续使用之前保存的消息历史。

InMemorySaver 适合测试，进程退出后内存中的状态会丢失。

正式系统可以选择 SQLite、PostgreSQL 等持久化 Checkpointer。

**注意：**thread\_id 是会话标识，不是安全认证凭证。业务系统必须校验会话属于哪个用户。

---

## 十一、store：跨会话长期存储

`store` 与 checkpointer 的用途不同。

Checkpointer 主要保存单个线程中的执行状态，而 Store 可以保存跨线程共享的长期数据。

例如：

```
用户 student-001

       |

       ├── 会话 A

       ├── 会话 B

       └── 会话 C

              |

              ↓

           共享 Store

              |

              ↓

       Python 学习偏好
```

### 实例：配置 InMemoryStore

## 实例

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

store.put(
    ("users", "student-001"),
    "learning\_preference",
    {"language": "Python", "level": "beginner"}
)

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    store=store
)

item = store.get(
    ("users", "student-001"),
    "learning\_preference"
)

print(item.value)

这里只是演示 Store 的创建和读取。

**注意：**将 store 传给 Agent，并不会自动把 Store 中的全部数据注入模型上下文。

需要通过工具、Middleware 或其他业务逻辑显式读取所需数据。

---

## 十二、interrupt\_before 与 interrupt\_after

这两个参数用于设置静态中断点。

常见节点名称包括：

```
model

tools
```

| 参数 | 暂停时机 |
| --- | --- |
| `interrupt_before=["tools"]` | 工具节点执行前 |
| `interrupt_after=["tools"]` | 工具节点执行后 |
| `interrupt_before=["model"]` | 模型节点执行前 |
| `interrupt_after=["model"]` | 模型节点执行后 |

### 实例：工具执行前暂停

## 实例

from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

@tool
def delete\_demo\_item(item\_id: str) -> str:
    """模拟删除一条数据。"""
    return f"模拟删除：{item\_id}"

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[delete\_demo\_item],
    checkpointer=InMemorySaver(),
    interrupt\_before=["tools"],
    system\_prompt="你是管理助手。用户要求模拟删除时使用 delete\_demo\_item。"
)

config = {"configurable": {"thread\_id": "approval-demo-001"}}

result = agent.invoke({
    "messages": [{"role": "user", "content": "请使用工具模拟删除 item-001。"}]
}, config=config)

state = agent.get\_state(config)

print("下一步节点：", state.next)

如果模型决定调用工具，Agent 可以在 tools 节点执行之前暂停。

需要注意：

- 中断点需要配合 Checkpointer 使用，才能可靠恢复执行。
- 在 tools 节点之前暂停，并不意味着只暂停某个特定工具。
- 静态中断本身不会自动实现完整的审批和权限校验。

如果需要针对单个工具进行批准、拒绝或修改参数，更适合使用 `interrupt()` 或 HumanInTheLoopMiddleware。

---

## 十三、debug：调试模式

`debug=True` 可以在执行时输出较详细的图运行信息。

### 实例

## 实例

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    debug=True
)

agent.invoke({
    "messages": [{"role": "user", "content": "你好"}]
})

这种方式适合本地开发时排查执行过程。

但不应依赖某种固定的 Debug 文本格式，因为实际输出可能随 LangChain 和 LangGraph 版本变化。

生产环境建议使用结构化日志和 LangSmith 等可观测性工具，而不是长期输出完整调试数据。

---

## 十四、name：Agent 名称

`name` 用于为编译后的 Agent 图指定名称。

例如：

## 实例

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    name="python\_learning\_agent"
)

在多 Agent 项目中，可以使用：

```
coordinator

course_expert

math_expert

code_reviewer
```

这些名称有助于图组合与追踪。

当 Agent 作为子图嵌入其他 LangGraph 图时，name 也可以作为对应的图标识。

但不能假设 name 会自动成为所有 AIMessage 的 name 字段。

---

## 十五、cache：图执行缓存

`cache` 用于配置 LangGraph 图执行缓存后端。

例如：

## 实例

from langgraph.cache.memory import InMemoryCache

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    cache=InMemoryCache()
)

这里为 Agent 图提供了缓存后端。

**注意：**设置 cache 并不意味着所有模型调用和工具结果都会自动缓存。

是否命中缓存，还需要对应节点具有适当的缓存策略。

如果希望缓存特定模型响应或检索结果，应根据具体的缓存层和业务需求单独设计。

---

## 十六、transformers：流事件转换器

`transformers` 用于注册额外的 StreamTransformer 工厂。

它属于比较高级的图流处理能力，可以用于调整或扩展流事件的处理方式。

一般的 Agent 开发通常不需要设置。

如果只是希望输出模型文本或工具执行状态，应优先使用：

```
agent.stream(..., stream_mode="updates")

agent.stream(..., stream_mode="messages")
```

只有确实需要自定义流事件转换逻辑时，才考虑 transformers。

因此，对于普通教程和业务 Agent，保留默认值 None 即可。

---

## 十七、CompiledStateGraph 常用方法

`create_agent()` 返回的是一个 `CompiledStateGraph`。

这意味着创建好的 Agent 不仅可以调用，还可以进行流式执行、状态查询和历史检查点管理。

| 方法 | 用途 | 典型返回值 |
| --- | --- | --- |
| `invoke()` | 同步执行 Agent | 最终状态 |
| `ainvoke()` | 异步执行 Agent | 最终状态 |
| `stream()` | 同步流式执行 | 事件迭代器 |
| `astream()` | 异步流式执行 | 异步事件迭代器 |
| `get_state()` | 获取线程状态快照 | StateSnapshot |
| `aget_state()` | 异步读取状态 | StateSnapshot |
| `get_state_history()` | 获取历史检查点 | StateSnapshot 迭代器 |
| `update_state()` | 更新线程状态 | 检查点配置 |
| `get_graph()` | 获取图结构表示 | Graph 对象 |

这些方法来自 LangGraph 和 Runnable 接口，实际上还支持更多配置参数及其他异步、批量操作。

---

## 十八、invoke()：同步运行 Agent

`invoke()` 是最常用的 Agent 执行方法。

### 实例

## 实例

result = agent.invoke({
    "messages": [{"role": "user", "content": "介绍 Python 列表。"}]
})

print(result["messages"][-1].content)

返回值通常是包含消息等字段的最终 Agent State。

如果配置 response\_format，还可能包含：

```
structured_response
```

如果执行过程中发生 interrupt，则可能返回中断信息，而不是普通的最终回答。

---

## 十九、ainvoke()：异步运行

在 FastAPI、异步 Web 服务或 asyncio 程序中，可以使用 ainvoke()。

### 实例

## 实例

import asyncio
from langchain.agents import create\_agent

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[]
)

async def main():
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "你好"}]
    })
    print(result["messages"][-1].content)

asyncio.run(main())

如果 Agent 中包含自定义工具，还需要考虑工具本身是否支持异步执行。

---

## 二十、stream()：流式运行 Agent

Agent 的流式执行与直接调用 Chat Model 略有不同。

因为 Agent 除了生成文本，还可能调用工具和执行多个图节点。

### 常用 stream\_mode

| 模式 | 主要用途 |
| --- | --- |
| `updates` | 查看节点执行后的状态更新 |
| `values` | 查看执行过程中的完整状态值 |
| `messages` | 获取模型消息增量和相关元数据 |
| `custom` | 获取自定义流事件 |
| `debug` | 获取更详细的图调试事件 |

### 实例：查看节点更新

## 实例

for event in agent.stream(
    {"messages": [{"role": "user", "content": "你好"}]},
    stream\_mode="updates"
):
    print(event)

### 实例：模型文本流式输出

## 实例

for chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "介绍 Python 的特点。"}]},
    stream\_mode="messages"
):
    if metadata.get("langgraph\_node") == "model":
        if isinstance(chunk.content, str):
            print(chunk.content, end="", flush=True)

print()

这里逐步输出模型生成的文本。

如果模型还生成了工具调用或中间分析性文本，应根据事件类型和应用需求进一步过滤。

**注意：**流式事件中可见的是模型接口和 Agent 运行时公开的内容，不代表可以读取模型内部隐藏的推理过程。

---

## 二十一、get\_state()：读取当前状态

如果配置了 Checkpointer，可以使用 get\_state() 读取指定线程的当前状态快照。

### 实例

## 实例

from langgraph.checkpoint.memory import InMemorySaver

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread\_id": "state-demo-001"}}

agent.invoke({
    "messages": [{"role": "user", "content": "你好"}]
}, config=config)

state = agent.get\_state(config)

print("状态字段：", state.values.keys())
print("消息数量：", len(state.values.get("messages", [])))
print("下一步节点：", state.next)
print("检查点配置：", state.config)

这里的 state 是 StateSnapshot，不是普通的 Agent 返回字典。

常见字段包括：

| 字段 | 说明 |
| --- | --- |
| `values` | 当前状态数据 |
| `next` | 待执行的下一步节点 |
| `config` | 当前检查点配置 |
| `metadata` | 检查点元数据 |
| `tasks` | 当前任务及可能的中断信息 |

如果 state.next 为空，通常表示当前没有待执行的普通后续节点。

---

## 二十二、get\_state\_history()：查看历史状态

Checkpointer 可以保存同一线程的多个历史检查点。

使用 `get_state_history()` 可以查看此前的状态快照。

### 实例

## 实例

config = {"configurable": {"thread\_id": "state-demo-001"}}

for snapshot in agent.get\_state\_history(config):
    print("Checkpoint：", snapshot.config)
    print("下一节点：", snapshot.next)

这种方式适合调试：

- Agent 执行到哪一步。
- 某轮对话产生了什么状态。
- 是否存在待处理的中断。
- 此前的检查点信息。

但它不意味着所有 Checkpointer 都永久保存无限历史。数据保留能力和清理策略取决于存储实现与部署配置。

---

## 二十三、update\_state()：手动更新状态

`update_state()` 允许向指定线程写入状态更新。

例如，为 Agent 添加一条新的用户消息。

### 实例

## 实例

from langchain.messages import HumanMessage

config = {"configurable": {"thread\_id": "state-demo-001"}}

new\_config = agent.update\_state(
    config,
    {"messages": [HumanMessage(content="补充说明：我正在学习 Python。")]}
)

print("新检查点配置：", new\_config)

state = agent.get\_state(config)
print("当前消息数量：", len(state.values["messages"]))

**注意：**update\_state() 不是直接覆盖某个历史 Checkpoint。

它会根据状态字段对应的 Reducer 合并更新，并生成新的检查点状态。

对于 messages，默认使用 add\_messages Reducer，相同消息 ID 可以替换已有消息，不同 ID 的消息则会追加。

如果要真正清空旧消息，应使用符合消息 Reducer 语义的删除方式，而不是简单返回一个较短的消息列表。

---

## 二十四、AgentState：默认状态结构

LangChain 1.x 默认的 AgentState 是一个 TypedDict。

它包含消息历史、可选的路由控制字段和结构化响应字段。

| 字段 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `messages` | list[AnyMessage] | 是 | 消息历史，使用 add\_messages Reducer |
| `jump_to` | "tools" / "model" / "end" / None | 否 | 内部临时路由指令 |
| `structured_response` | ResponseT | 否 | 结构化输出结果，不作为普通输入字段 |

### messages

这是最常用的状态字段。

例如：

```
{

    "messages": [

        {"role": "user", "content": "你好"}

    ]

}
```

执行完成后，Agent 通常会返回用户消息和模型回复。

### jump\_to

用于特定 Middleware Hook 控制后续执行路径。

允许的目标包括：

```
tools

model

end
```

该字段属于临时的内部路由控制，不适合作为普通业务数据保存。

如果 Middleware 希望通过 jump\_to 跳转，还需要在对应 Hook 声明允许的跳转目标。

### structured\_response

当配置了 response\_format 并成功生成结构化输出时，可以通过：

```
result["structured_response"]
```

读取结果。

如果没有配置结构化输出，通常不会出现该字段。

---

## 二十五、AgentState 与 ToolRuntime 的区别

在实际开发中，AgentState 和 ToolRuntime 很容易混淆。

两者作用不同：

| 对象 | 主要用途 |
| --- | --- |
| AgentState | 保存当前执行流程中的状态 |
| ToolRuntime | 向工具提供状态、上下文、Store 等运行时能力 |
| Runtime Context | 提供可信用户信息和请求配置 |
| Checkpointer | 保存与恢复线程执行状态 |
| Store | 保存跨会话长期数据 |

例如：

## 实例

from langchain.tools import tool, ToolRuntime

@tool
def inspect\_runtime(runtime: ToolRuntime) -> str:
    """查看当前会话中的消息数量。"""
    messages = runtime.state.get("messages", [])
    return f"当前状态中共有 {len(messages)} 条消息。"

模型调用这个工具时，Runtime 会自动注入，模型不需要生成 runtime 参数。

这可以避免把内部状态对象直接暴露为模型可控制的工具参数。

---

## 二十六、完整实例：Agent API 综合使用

下面将模型、工具、Middleware、结构化输出和 Checkpointer 组合起来。

文件名：`agent_api_demo.py`

安装依赖：

```
pip install -U langchain langchain-deepseek langgraph pydantic python-dotenv
```

配置 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dotenv import load\_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.middleware import dynamic\_prompt, ModelRequest
from langchain.agents.structured\_output import ToolStrategy
from langchain.tools import tool
from langchain\_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

load\_dotenv()

# ========== 1. 工具 ==========

@tool
def search\_course(keyword: str) -> str:
    """在模拟课程目录中查找课程。"""
    data = {
        "Python": "Python 基础、Python 函数",
        "Java": "Java 基础、Java 面向对象"
    }
    return data.get(keyword, "没有找到匹配的模拟课程")

# ========== 2. 结构化输出 ==========

class CourseResult(BaseModel):
    keyword: str = Field(description="用户查询的关键词")
    courses: list[str] = Field(description="检索到的课程")
    found: bool = Field(description="是否找到匹配课程")

# ========== 3. Middleware ==========

@dynamic\_prompt
def build\_prompt(request: ModelRequest) -> str:
    return (
        "你是编程课程搜索助手。"
        "回答课程查询必须先使用 search\_course 工具。"
        "只能根据工具返回的模拟课程目录生成结果。"
        "不得编造不存在的课程。"
    )

# ========== 4. 创建 Agent ==========

agent = create\_agent(
    model=ChatDeepSeek(model="deepseek-chat", temperature=0),
    tools=[search\_course],
    middleware=[build\_prompt],
    response\_format=ToolStrategy(CourseResult),
    checkpointer=InMemorySaver(),
    name="course\_search\_agent"
)

# ========== 5. 执行 ==========

config = {"configurable": {"thread\_id": "agent-api-demo"}}

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "使用工具查询 Python 有哪些课程，并返回结构化结果。"
    }]
}, config=config)

print("结构化输出：", result.get("structured\_response"))

# ========== 6. 查看状态 ==========

state = agent.get\_state(config)

print("消息数量：", len(state.values.get("messages", [])))
print("下一步节点：", state.next)

# ========== 7. 查看历史 ==========

for snapshot in agent.get\_state\_history(config):
    print("Checkpoint：", snapshot.config)

这个示例展示了：

- 通过 create\_agent() 创建 Agent。
- 使用 @tool 注册业务工具。
- 使用 dynamic\_prompt 定义模型规则。
- 使用 ToolStrategy 获取结构化输出。
- 使用 InMemorySaver 保存线程状态。
- 通过 get\_state() 获取当前状态。
- 通过 get\_state\_history() 查看历史检查点。

需要注意，模型是否按要求调用工具及返回有效结构化数据，取决于其实际能力和输出行为。

对于必须进行真实数据库查询的业务，不能仅依靠 Prompt 保证调用顺序，应使用确定性的检索或工具执行流程。

---

## 二十七、常见问题

### 1、create\_agent() 返回的是什么对象？

返回 CompiledStateGraph，可以通过 invoke、stream 等方法执行。

### 2、model 必须使用字符串吗？

不是，也可以传入 ChatDeepSeek 等 BaseChatModel 实例。

### 3、tools 可以不填写吗？

可以。tools 默认为 None，此时不会构建普通工具调用循环。

### 4、Pydantic BaseModel 可以直接作为可执行工具吗？

通常不是。Pydantic 主要用于定义工具输入 Schema 或结构化输出；执行工具应使用 Callable、BaseTool 或受支持的工具定义。

### 5、response\_format 可以直接传入 Pydantic 类吗？

可以。LangChain 会根据支持情况选择适当的结构化输出策略。

### 6、AgentState 与 context\_schema 有什么区别？

AgentState 用于执行状态；context\_schema 用于定义运行时上下文。

### 7、为什么配置了 Store，Agent 却没有读取用户偏好？

因为 Store 数据不会自动注入模型，需要使用工具或 Middleware 显式读取。

### 8、为什么多轮对话不生效？

需要检查 Checkpointer 是否配置，以及多轮调用是否使用相同的 thread\_id。

### 9、interrupt\_before=["tools"] 是否只暂停敏感工具？

不是。它对指定的 tools 节点设置中断点，不会自动区分敏感与普通工具。

### 10、如何针对单个敏感工具审批？

可以使用 interrupt() 或 HumanInTheLoopMiddleware。

### 11、debug=True 是否适合线上长期使用？

通常不建议，可能输出大量消息和状态数据。

### 12、name 参数会修改所有 AIMessage 的名称吗？

不会。它主要用于标识 Agent 图。

### 13、配置 cache 后是否自动缓存所有模型回答？

不会。还需要实际节点的缓存策略或单独的模型缓存逻辑。

### 14、update\_state() 会覆盖原来的历史 Checkpoint 吗？

通常不会，它会创建新的状态版本，并按照字段的 Reducer 应用更新。

### 15、messages 为什么不能直接用较短列表替换？

因为默认使用 add\_messages Reducer，普通列表更新通常会合并消息而不是删除旧消息。

### 16、stream\_mode="updates" 和 "messages" 有什么区别？

updates 用于观察节点状态更新；messages 用于获取模型生成的消息增量。

### 17、什么情况下需要自定义 state\_schema？

当 Agent 需要保存额外任务状态、业务步骤或其他执行数据时，可以扩展 AgentState。

### 18、什么情况下需要 transformers？

当需要自定义图流事件的转换和呈现时可以使用。普通业务通常不需要。

---

## 总结

LangChain 1.x 的 `create_agent()` 提供了统一、可扩展的 Agent 创建接口。

本章节介绍了：

- `model`：模型配置。
- `tools`：工具注册。
- `system_prompt`：系统提示词。
- `middleware`：Agent 执行过程控制。
- `response_format`：结构化输出。
- `state_schema`：自定义状态。
- `context_schema`：运行时上下文。
- `checkpointer`：单线程对话状态持久化。
- `store`：跨会话长期数据存储。
- `interrupt_before` 和 `interrupt_after`：静态中断点。
- `debug`：调试日志。
- `name`：Agent 名称。
- `cache`：图缓存。
- `transformers`：流事件转换器。
- `CompiledStateGraph`：Agent 的执行和状态管理接口。
- `AgentState`：Agent 默认状态结构。

最常用的 Agent 写法：

```
from langchain.agents import create_agent



agent = create_agent(

    model="deepseek:deepseek-chat",

    tools=[my_tool],

    system_prompt="你是一个智能助手。"

)



result = agent.invoke({

    "messages": [{"role": "user", "content": "你好"}]

})
```

**核心原则：**create\_agent() 负责将模型、工具和执行流程组织起来；Middleware 负责扩展行为；AgentState 和 Checkpointer 负责执行状态；Runtime Context 和 Store 则提供外部上下文与长期数据能力。

掌握这些参数后，就可以根据业务需要逐步构建具备多工具协作、对话记忆、结构化输出、人工审批和长期存储能力的 Agent。

## 参考文档

- [create\_agent API Reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
- [AgentState API Reference](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentState)
- [LangChain Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Tools 官方文档](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain Middleware 官方文档](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [LangChain Structured Output 官方文档](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain Streaming 官方文档](https://docs.langchain.com/oss/python/langchain/streaming)
