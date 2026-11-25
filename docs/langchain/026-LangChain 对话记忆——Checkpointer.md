# LangChain 对话记忆——Checkpointer

> 来源: https://www.runoob.com/langchain/langchain-checkpointer.html

---

# LangChain 对话记忆——Checkpointer

默认情况下，Agent 每次独立调用 `invoke()` 时，并不会自动记住上一次对话的内容。

例如，第一轮告诉 AI"我叫小明"，第二轮再问"我叫什么名字"，如果没有传入历史消息，Agent 通常无法回答。

LangChain 提供 Checkpointer（检查点保存器），通过保存和恢复 Agent State，让同一个对话线程具备多轮记忆能力。

本章节介绍 `InMemorySaver`、`SqliteSaver`、`PostgresSaver`、`thread_id`、状态管理、历史检查点以及对话记忆在实际项目中的应用。

---

## 一、没有 Checkpointer 会怎样？

先看一个不使用 Checkpointer 的示例。

每次调用 Agent，只发送当前用户问题，没有主动传入前面的消息。

### 实例：两次独立对话

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的中文编程助手。"
)

result1 = agent.invoke({"messages": [{"role": "user", "content": "我叫小明，正在学习 Python。"}]})
print("第一轮：", result1["messages"][-1].content)

result2 = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]})
print("第二轮：", result2["messages"][-1].content)

可能得到：

```
第一轮：你好，小明！Python 是很适合入门的编程语言。



第二轮：你还没有告诉我你的名字。
```

原因在于，第二次调用时只传入了新的用户问题。

对于当前这次模型请求，Agent 无法自动获取上一轮独立调用的历史消息。

**注意：**没有 Checkpointer 并不代表完全无法实现多轮对话。你也可以在应用程序中手动维护消息历史，并在每次调用时一并传入。

Checkpointer 的优势是由 LangGraph 统一管理状态持久化，减少手动拼接历史消息的工作。

---

## 二、使用 InMemorySaver 记住对话

最简单的 Checkpointer 是 `InMemorySaver`，它将 Agent State 保存在 Python 进程内存中。

只需要在创建 Agent 时传入 `checkpointer`：

```
checkpointer=InMemorySaver()
```

然后通过 `thread_id` 标识一个对话线程。

### 完整实例：两轮对话记忆

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    checkpointer=InMemorySaver(),
    system\_prompt="你是菜鸟教程 RUNOOB 的中文编程助手。"
)

config = {"configurable": {"thread\_id": "learning-001"}}

result1 = agent.invoke({
    "messages": [{"role": "user", "content": "我叫小明，正在学习 Python。"}]
}, config=config)
print("第一轮：", result1["messages"][-1].content)

result2 = agent.invoke({
    "messages": [{"role": "user", "content": "我叫什么名字？正在学什么？"}]
}, config=config)
print("第二轮：", result2["messages"][-1].content)

第二轮可能得到：

```
你叫小明，正在学习 Python。
```

原因是两次调用都使用了相同的：

```
thread_id = "learning-001"
```

LangGraph 会在下一次调用时恢复该线程已经保存的状态，然后合并本轮新增消息。

**注意：**模型的实际回复不是固定的。即使保存了历史，也不能保证模型每次都能准确回忆所有细节。

---

## 三、thread\_id：对话线程标识

`thread_id` 是 Checkpointer 管理对话状态的关键参数。

同一个 thread\_id 代表同一个逻辑对话线程，不同 thread\_id 则用于区分不同会话。

| 参数 | 作用 | 示例 |
| --- | --- | --- |
| `thread_id` | 标识一个对话线程 | chat-001 |
| `checkpoint_id` | 标识线程中的具体检查点 | 由 Checkpointer 生成 |
| `checkpoint_ns` | 区分检查点命名空间 | 通常由框架管理 |

一个用户可以有多个 thread\_id：

```
用户 user-001

    ├── chat-001：Python 学习

    ├── chat-002：Java 学习

    └── chat-003：项目调试
```

不同用户也可以使用同一个 Agent 实例，但必须为各自的会话分配独立线程。

### 实例：创建两个独立会话

## 实例

alice = {"configurable": {"thread\_id": "alice-chat-001"}}
bob = {"configurable": {"thread\_id": "bob-chat-001"}}

agent.invoke({"messages": [{"role": "user", "content": "我是 Alice。"}]}, config=alice)
agent.invoke({"messages": [{"role": "user", "content": "我是 Bob。"}]}, config=bob)

alice\_state = agent.get\_state(alice)
bob\_state = agent.get\_state(bob)

print("Alice：", alice\_state.values["messages"][0].content)
print("Bob：", bob\_state.values["messages"][0].content)

输出：

```
Alice： 我是 Alice。

Bob： 我是 Bob。
```

这说明两个线程保存了不同的状态。

**重要：**thread\_id 不是身份认证令牌，也不是安全隔离机制。Web 应用必须在后端检查当前用户是否有权限读取或继续某个 thread\_id。

不能允许用户随意修改 URL 或请求参数，就能访问其他用户的对话。

---

## 四、Checkpointer 的工作原理

Checkpointer 不只是一个"聊天记录数据库"，它保存的是 LangGraph 执行过程中的状态检查点。

一个简化的执行过程：

```
第一次 invoke()

    ↓

识别 thread_id

    ↓

读取已有 Checkpoint（如果存在）

    ↓

合并用户输入

    ↓

执行 Agent 节点

    ↓

保存状态检查点

    ↓

返回结果



第二次 invoke()（相同 thread_id）

    ↓

恢复该线程最新状态

    ↓

合并新的用户消息

    ↓

继续执行 Agent

    ↓

保存新的检查点
```

检查点可以记录：

- 消息历史 `messages`。
- 自定义 AgentState 字段。
- 当前执行状态。
- 后续待执行节点。
- 与执行恢复相关的检查点信息。

LangGraph 可以在执行过程中按步骤创建检查点，而不只是任务全部结束后才保存。

这也是 Checkpointer 能支持中断恢复和状态历史查看的重要原因。

---

## 五、查看当前对话状态 get\_state()

使用 `agent.get_state(config)` 可以读取指定线程当前保存的 State。

返回的是 `StateSnapshot`，其中包含状态值以及执行相关信息。

### 实例：查看消息和执行状态

## 实例

state = agent.get\_state(config)

print("下一步节点：", state.next)
print("消息数量：", len(state.values.get("messages", [])))
print("检查点配置：", state.config)

for msg in state.values.get("messages", []):
    print(f"[{msg.type}] {str(msg.content)[:80]}")

对于已经正常完成的简单 Agent，`state.next` 通常是：

```
()
```

表示当前检查点没有等待执行的后续节点。

如果 Agent 处于中断或等待状态，`state.next` 可能包含待执行节点名称。

### StateSnapshot 常用属性

| 属性 | 说明 |
| --- | --- |
| `values` | 当前 State 字段和值 |
| `next` | 接下来计划执行的节点 |
| `config` | 当前检查点的配置信息 |
| `metadata` | 检查点元数据 |
| `created_at` | 检查点创建时间 |
| `tasks` | 当前相关任务信息 |

**注意：**必须先有对应线程的检查点，才能读取有意义的历史状态。

---

## 六、查看多轮对话的状态变化

下面通过三次连续调用，观察消息历史如何增加。

### 实例：三轮对话

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread\_id": "demo-003"}}
questions = ["我叫小明。", "我正在学习 Python。", "总结一下刚才介绍过的我的信息。"]

for i, question in enumerate(questions, 1):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    state = agent.get\_state(config)
    print(f"第{i}轮：消息数={len(state.values['messages'])}，next={state.next}")
    print("回复：", str(result["messages"][-1].content)[:80])

如果每轮只有一次模型回答，可能得到：

```
第1轮：消息数=2，next=()

第2轮：消息数=4，next=()

第3轮：消息数=6，next=()
```

但如果模型使用工具或 Middleware 添加了消息，实际消息数量可能更多。

因此，不能简单假设"每轮必定增加两条消息"。

---

## 七、Checkpointer 的类型

LangGraph 提供多种检查点存储实现，可以根据应用规模选择。

| 类型 | 存储位置 | 进程重启后保留 | 适用场景 |
| --- | --- | --- | --- |
| `InMemorySaver` | Python 进程内存 | 否 | 开发、调试、测试 |
| `SqliteSaver` | SQLite 数据库文件 | 是 | 单机、小型应用 |
| `PostgresSaver` | PostgreSQL 数据库 | 是 | 生产应用、多实例部署 |

对应安装方式：

```
# 基础 Checkpointer

pip install -U langgraph



# SQLite

pip install -U langgraph-checkpoint-sqlite



# PostgreSQL

pip install -U langgraph-checkpoint-postgres "psycopg[binary]"
```

其中，InMemorySaver 随 LangGraph 基础依赖提供，SQLite 与 PostgreSQL 的检查点实现需要额外安装。

生产部署还应考虑数据库访问权限、连接池、备份、状态清理以及敏感信息保护。

---

## 八、SqliteSaver：将对话保存到数据库

`SqliteSaver` 可以把 Agent State 持久化保存到 SQLite 文件。

即使 Python 程序退出，只要数据库文件没有丢失，之后仍可以通过原来的 thread\_id 恢复会话。

SQLite 很适合本地工具、小型学习项目和单机应用。

### 完整实例：SQLite 对话记忆

文件名：`sqlite_memory.py`

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load\_dotenv()

with SqliteSaver.from\_conn\_string("conversations.db") as checkpointer:
    agent = create\_agent(
        model="deepseek:deepseek-flash",
        tools=[],
        checkpointer=checkpointer,
        system\_prompt="你是菜鸟教程 RUNOOB 的学习助手。"
    )

    config = {"configurable": {"thread\_id": "student-001"}}
    result = agent.invoke({
        "messages": [{"role": "user", "content": "我正在学习 Python 装饰器。"}]
    }, config=config)

    print("AI：", result["messages"][-1].content)

运行后，程序会在当前工作目录创建或使用：

```
conversations.db
```

数据库会保存该线程的状态检查点。

### 重新运行程序恢复历史

第二次启动程序时，仍然使用相同的数据库文件与 thread\_id：

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load\_dotenv()

with SqliteSaver.from\_conn\_string("conversations.db") as checkpointer:
    agent = create\_agent(model="deepseek:deepseek-flash", tools=[], checkpointer=checkpointer)
    config = {"configurable": {"thread\_id": "student-001"}}

    result = agent.invoke({
        "messages": [{"role": "user", "content": "我之前正在学习什么知识点？"}]
    }, config=config)
    print(result["messages"][-1].content)

如果之前的状态已经成功保存，模型可以根据历史消息回答，例如：

```
你之前正在学习 Python 装饰器。
```

需要注意：

- 两个示例必须使用相同的数据库文件。
- 必须使用相同的 thread\_id。
- 数据库文件必须具有正确的读取和写入权限。
- `with` 代码块结束后，数据库连接会关闭。

对于长期运行的 Web 服务，Checkpointer 的生命周期应该由应用统一管理，不应在每次 HTTP 请求中无意义地重复创建和关闭。

---

## 九、PostgresSaver：生产环境持久化

当 Agent 需要服务大量用户，或者部署在多个应用实例中时，可以使用 PostgreSQL 保存检查点。

PostgreSQL 支持更完整的数据库管理能力，更适合生产环境。

### 安装依赖

```
pip install -U langgraph-checkpoint-postgres "psycopg[binary]"
```

### 实例：使用 PostgreSQL 保存记忆

## 实例

import os
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.postgres import PostgresSaver

load\_dotenv()
DB\_URI = os.environ["POSTGRES\_URI"]

with PostgresSaver.from\_conn\_string(DB\_URI) as checkpointer:
    checkpointer.setup()
    agent = create\_agent(
        model="deepseek:deepseek-flash",
        tools=[],
        checkpointer=checkpointer
    )

    config = {"configurable": {"thread\_id": "student-001"}}
    result = agent.invoke({
        "messages": [{"role": "user", "content": "请介绍 Python 列表。"}]
    }, config=config)

    print(result["messages"][-1].content)

对应的 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key

POSTGRES_URI=postgresql://user:password@localhost:5432/langchain_db
```

这里的 `checkpointer.setup()` 会初始化所需的数据库表，首次使用时必须执行。

通常可以在应用初始化阶段完成数据库结构设置，而不是每次普通请求都执行 setup。

**注意：**生产数据库连接应使用可靠的凭据管理、访问权限控制及适当的加密连接配置。不要将真实数据库密码提交到公开代码仓库。

---

## 十、异步 Checkpointer

如果 Agent 通过 `ainvoke()` 或 `astream()` 异步运行，可以使用对应的异步数据库检查点实现。

| 同步版本 | 异步版本 |
| --- | --- |
| SqliteSaver | AsyncSqliteSaver |
| PostgresSaver | AsyncPostgresSaver |

### 实例：AsyncSqliteSaver

## 实例

import asyncio
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load\_dotenv()

async def main():
    async with AsyncSqliteSaver.from\_conn\_string("async\_chat.db") as checkpointer:
        agent = create\_agent(model="deepseek:deepseek-flash", tools=[], checkpointer=checkpointer)
        config = {"configurable": {"thread\_id": "async-001"}}
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "你好，我正在学习 Python。"}]
        }, config=config)
        print(result["messages"][-1].content)

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(main())

异步调用使用：

```
await agent.ainvoke(...)
```

如果需要读取异步状态，可以使用：

```
state = await agent.aget_state(config)
```

对于 PostgreSQL，异步实现可以通过 `AsyncPostgresSaver` 及 `await checkpointer.setup()` 完成初始化。

---

## 十一、get\_state\_history()：查看历史检查点

Checkpointer 不仅可以保存最新状态，还能记录一个线程中产生过的多个检查点。

使用 `get_state_history()` 可以遍历这些历史状态快照。

### 实例：列出历史检查点

## 实例

config = {"configurable": {"thread\_id": "demo-003"}}

for i, snapshot in enumerate(agent.get\_state\_history(config), 1):
    checkpoint\_id = snapshot.config["configurable"].get("checkpoint\_id")
    count = len(snapshot.values.get("messages", []))
    print(f"{i}. checkpoint={checkpoint\_id}，消息数={count}")
    if i >= 5:
        break

可能输出：

```
1. checkpoint=...，消息数=6

2. checkpoint=...，消息数=5

3. checkpoint=...，消息数=4

4. checkpoint=...，消息数=3

5. checkpoint=...，消息数=2
```

这里只是示意，实际检查点数量和保存状态取决于图的执行步骤。

一个 Agent 调用可能产生多个检查点，因此不能简单认为"一个 Checkpoint 等于一轮对话"。

**注意：**get\_state\_history() 主要用于状态追踪和历史检查点管理，不应直接将其视为普通聊天记录分页接口。

---

## 十二、checkpoint\_id：读取指定历史版本

如果需要查看过去某个时间点的 Agent State，可以根据历史检查点 ID 读取对应状态。

### 实例：读取之前的检查点

## 实例

config = {"configurable": {"thread\_id": "demo-003"}}
history = list(agent.get\_state\_history(config))

if len(history) >= 2:
    old\_config = history[1].config
    old\_state = agent.get\_state(old\_config)
    print("历史消息数：", len(old\_state.values.get("messages", [])))
    print("历史检查点：", old\_config["configurable"].get("checkpoint\_id"))

这里通过历史快照中的 `config` 获取具体 checkpoint\_id，而不是自行构造检查点 ID。

这种方式适合调试历史状态及了解 Agent 执行过程。

LangGraph 还支持基于历史检查点的时间旅行、状态分支及执行重放等高级功能。

但执行重放涉及图节点和外部工具的行为，不代表可以无风险地重复真实业务操作。

---

## 十三、update\_state()：手动修改对话状态

有些场景需要在 Agent 执行结束后修改保存的状态。

例如，修改当前学习进度或补充自定义信息。

可以使用：

```
agent.update_state(config, values)
```

需要注意，`update_state()` 是向图状态写入更新，并创建新的检查点，而不是直接修改数据库中所有旧版本的状态记录。

### 实例：更新学习进度

首先定义一个包含自定义字段的 State。

## 实例

from typing\_extensions import NotRequired
from langchain.agents import create\_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class LearningState(AgentState):
    current\_topic: NotRequired[str]
    user\_level: NotRequired[str]

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    state\_schema=LearningState,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread\_id": "learning-state-001"}}
agent.invoke({
    "messages": [{"role": "user", "content": "我正在学习 Python。"}],
    "current\_topic": "变量",
    "user\_level": "入门"
}, config=config)

agent.update\_state(config, {"current\_topic": "装饰器"})
print(agent.get\_state(config).values["current\_topic"])

输出：

```
装饰器
```

这里更新的是自定义业务状态字段，而不是通过伪造对话消息表达学习进度。

**注意：**实际用户进度如果需要跨多个会话共享，通常更适合持久化 Store 或业务数据库。

---

## 十四、修改或删除消息历史

AgentState 中的 `messages` 使用 `add_messages` Reducer。

因此，直接提交：

```
agent.update_state(config, {"messages": []})
```

并不能保证清空所有历史消息。

如果需要删除指定消息，应使用 `RemoveMessage`。

### 删除指定消息

## 实例

from langchain.messages import RemoveMessage

state = agent.get\_state(config)
messages = state.values.get("messages", [])

if messages:
    agent.update\_state(config, {
        "messages": [RemoveMessage(id=messages[0].id)]
    })

这会从最新 State 的消息列表中删除对应 ID 的消息。

### 清空全部消息

如果希望清空当前线程的消息列表，可以使用 `REMOVE_ALL_MESSAGES`：

## 实例

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE\_ALL\_MESSAGES

agent.update\_state(config, {
    "messages": [RemoveMessage(id=REMOVE\_ALL\_MESSAGES)]
})

print("剩余消息：", len(agent.get\_state(config).values.get("messages", [])))

正常情况下，最新状态中的消息列表将被清空。

但需要特别注意：

- 清空 messages 不等于删除整个线程。
- 清空最新 State 不等于物理删除全部旧检查点。
- 如果还有自定义状态字段，这些字段不会自动清空。
- 删除部分历史时，必须保证 AI 工具请求和 ToolMessage 的关系仍然合法。

对于"开始一段全新对话"，通常最简单的做法是创建新的 thread\_id。

对于"彻底删除用户数据"的需求，则需要结合具体 Checkpointer 的线程删除能力、数据库清理和数据保留策略。

---

## 十五、Checkpointer 与 Store 的区别

Checkpointer 解决的是同一线程中的状态持久化，而 Store 用于存储跨会话的长期信息。

| 对比项 | Checkpointer | Store |
| --- | --- | --- |
| 主要用途 | 保存 Agent 执行状态 | 保存长期记忆和共享数据 |
| 数据范围 | 通常按 thread\_id 区分 | 按 Namespace、Key 等组织 |
| 典型数据 | 消息历史、当前任务状态 | 用户偏好、长期学习记录 |
| 跨会话共享 | 不自动共享不同线程 | 可以根据业务设计跨线程读取 |
| 持久化能力 | 取决于 Checkpointer 实现 | 取决于 Store 实现 |

例如，编程学习网站可以这样设计：

```
Checkpointer：

    当前聊天消息

    当前任务进度

    Agent 执行检查点



Store / 数据库：

    用户完成的课程

    长期学习偏好

    学习记录

    会员和权限信息
```

其中，会员身份和权限应以可信业务数据库为准，而不是让模型根据对话消息自行判断。

---

## 十六、生产环境的注意事项

### 1、不要直接信任客户端传入的 thread\_id

应用后端必须检查当前用户是否有权访问指定会话。

不能因为用户知道某个 thread\_id，就允许读取其他用户的聊天历史。

### 2、控制同一线程的并发调用

如果多个请求同时修改同一个对话线程，可能产生状态竞争或执行顺序不符合预期的问题。

建议按照应用需求串行处理同一会话的写入，或者采用明确的并发控制策略。

### 3、控制对话历史长度

Checkpointer 会保存消息，但不会自动保证模型上下文永远不会超出限制。

长对话可以结合 `SummarizationMiddleware`、消息裁剪或其他上下文管理方案。

### 4、保护持久化数据

数据库检查点可能包含用户输入、模型回复、工具结果及自定义状态。

应考虑数据加密、最小权限、备份、日志脱敏与数据保留期限。

### 5、不要把 Checkpointer 当作长期用户资料库

对话历史适合按线程保存，而用户长期偏好、订单和权限等信息应放在专门的业务存储中。

### 6、处理数据库连接生命周期

SQLite 和 PostgreSQL 的连接应由应用合理管理。

对于 Web 服务，建议使用适合部署环境的连接管理与异步实现，而不是每个请求都重新创建数据库连接。

---

## 十七、常见问题

### 1、没有 Checkpointer 就不能实现多轮对话吗？

不是。可以自行维护消息历史并传给 Agent，但 Checkpointer 提供了统一的状态保存和恢复机制。

### 2、thread\_id 是用户 ID 吗？

不是。thread\_id 是会话标识，一个用户可以有多个对话线程。

### 3、不同 thread\_id 的数据会自动共享吗？

不会自动共享线程状态。如果需要共享长期信息，应使用 Store 或业务数据库。

### 4、InMemorySaver 重启后数据还在吗？

通常不在。它使用进程内存保存检查点。

### 5、SqliteSaver 是否支持重新启动后恢复？

支持。只要保存的数据库文件仍然存在，并且使用相同 thread\_id。

### 6、PostgresSaver 为什么需要 setup()？

因为首次使用时需要初始化检查点相关的数据表和结构。

### 7、一个 invoke() 会产生几个检查点？

不一定。LangGraph 可以在执行过程中按步骤保存检查点，不只是最终状态。

### 8、get\_state() 和 get\_state\_history() 有什么区别？

get\_state() 用于读取指定检查点或当前线程状态；get\_state\_history() 用于查看多个历史状态快照。

### 9、update\_state() 会覆盖全部历史检查点吗？

不会。它会根据 Reducer 更新状态并创建新的检查点，不是直接修改全部旧版本。

### 10、如何清空对话历史？

可以创建新 thread\_id，或者通过 RemoveMessage 更新已有线程的消息列表。

### 11、清空消息等于删除用户数据吗？

不等于。旧检查点、备份和其他业务存储可能仍然保存相关数据，需要按照数据保留与删除策略统一处理。

### 12、Checkpointer 能保证模型准确记住所有内容吗？

不能。它负责保存状态，但模型仍然受到上下文窗口、信息干扰和生成准确性的影响。

---

## 总结

Checkpointer 是 LangChain Agent 实现短期记忆和状态持久化的重要机制。

本章节介绍了：

- `InMemorySaver`：进程内检查点。
- `SqliteSaver`：SQLite 数据库持久化。
- `PostgresSaver`：PostgreSQL 持久化。
- `thread_id`：区分对话线程。
- `get_state()`：读取当前或指定检查点状态。
- `get_state_history()`：查看历史检查点。
- `update_state()`：手动提交状态更新。
- `RemoveMessage`：删除指定历史消息。
- `REMOVE_ALL_MESSAGES`：清空最新 State 的消息列表。
- `Store`：保存跨会话的长期记忆。

最常见的写法：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    checkpointer=InMemorySaver()

)



config = {"configurable": {"thread_id": "chat-001"}}



agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)

agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config)
```

> **核心原则：**Checkpointer 负责保存 Agent 线程状态，thread\_id 负责区分会话，Store 或业务数据库负责长期数据。对于线上应用，还必须做好会话授权、数据库持久化、上下文管理和敏感信息保护。
