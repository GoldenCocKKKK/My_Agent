# LangChain 配置与错误类 API 参考

> 来源: https://www.runoob.com/langchain/langchain-config-errors.html

---

# LangChain 配置与错误类 API 参考

在前面的教程中，我们已经学习了 LangChain 的 Chat Model、Agent、Tools、Messages、Middleware 和 RAG。

本章节继续介绍实际开发中经常使用的 RunnableConfig、Checkpointer、Store 以及常见异常类。

这些接口不直接决定 AI 的回答内容，却影响 Agent 的运行方式、并发执行、对话记忆、数据持久化、日志追踪和错误处理。

例如：

- 使用 RunnableConfig 控制执行参数。
- 使用 thread\_id 识别对话线程。
- 使用 Checkpointer 保存和恢复 Agent 状态。
- 使用 Store 保存用户长期偏好。
- 通过异常类型判断是否应该重试、终止或提示用户。

本章节基于 LangChain 1.x 与 LangGraph 1.x，提供常用 API 速查表、Python 示例、异常处理方法及运行环境排查清单。

---

## 一、RunnableConfig 是什么？

`RunnableConfig` 是 LangChain 用于配置 Runnable 执行行为的数据结构。

Chat Model、Agent、Chain 以及其他 Runnable 通常都支持通过 `config` 参数传入运行时配置。

例如：

```
result = agent.invoke(

    {"messages": [{"role": "user", "content": "你好"}]},

    config={

        "tags": ["demo"],

        "metadata": {"source": "web"}

    }

)
```

RunnableConfig 可以在不修改 Agent 定义的情况下，为不同请求设置不同的执行标签、追踪信息以及其他参数。

主要作用包括：

- 控制运行时配置。
- 设置并发限制。
- 限制图执行步骤。
- 添加追踪标签和元数据。
- 配置回调处理器。
- 指定线程和运行标识。

---

## 二、RunnableConfig 完整字段

RunnableConfig 是 TypedDict，配置字段通常都是可选的。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `tags` | list[str] | 执行标签，用于筛选和追踪 |
| `metadata` | dict[str, Any] | 附加元数据，通常要求可 JSON 序列化 |
| `callbacks` | Callbacks | 回调处理器或 CallbackManager |
| `run_name` | str | 本次运行的追踪名称 |
| `max_concurrency` | int | None | 并行调用的最大并发数量 |
| `recursion_limit` | int | 最大递归或图执行步骤限制，默认 25 |
| `configurable` | dict[str, Any] | 动态配置和线程相关参数 |
| `run_id` | UUID | None | 本次追踪运行的标识 |

其中使用频率最高的是：

```
configurable

tags

metadata

recursion_limit

max_concurrency
```

**注意：**RunnableConfig 中的 recursion\_limit 与 Python 解释器的 `sys.setrecursionlimit()` 不是同一回事。

---

## 三、configurable：运行时动态配置

`configurable` 用于传入运行时配置。

在 LangGraph Agent 中，最常见的用途是设置：

```
thread_id
```

例如：

## 实例：配置对话线程

config = {
    "configurable": {
        "thread\_id": "conversation-001"
    }
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config=config
)

当 Agent 配置了 Checkpointer 后，同一个 thread\_id 可以用于读取和继续此前保存的对话状态。

例如：

```
thread_id = conversation-001



第一轮：你好

第二轮：我正在学习 Python

第三轮：我刚才说在学什么？
```

这些调用可以共享同一个线程历史。

如果换成新的 thread\_id，则会使用另一个线程。

**注意：**thread\_id 只是线程标识，不是用户身份认证凭证。多用户系统必须验证当前用户是否有权访问指定线程。

---

## 四、configurable 不只用于 thread\_id

除了 Checkpointer，configurable 还可以修改已经明确声明为可配置的 Runnable 参数。

例如：

## 实例：动态修改模型参数

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    temperature=0,
    configurable\_fields=("temperature",)
)

result = model.invoke(
    "介绍 Python 语言。",
    config={
        "configurable": {
            "temperature": 0.5
        }
    }
)

print(result.content)

这个例子允许调用时修改 temperature。

如果没有通过 configurable\_fields 或其他可配置机制声明某个参数，不能假定在 configurable 中添加同名字段就会修改模型行为。

**安全提醒：**不要直接允许普通用户通过 configurable 修改 api\_key、base\_url 等敏感配置。

---

## 五、recursion\_limit：执行步骤限制

LangChain Agent 基于 LangGraph 执行。

Agent 可能在模型和工具之间多次循环：

```
模型

 ↓

工具

 ↓

模型

 ↓

工具

 ↓

模型

 ...
```

如果流程始终没有结束，就可能超过最大执行步骤限制。

RunnableConfig 的默认递归限制为：

```
recursion_limit = 25
```

可以在运行时调整：

## 实例

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "请查询 Python 课程。"}
        ]
    },
    config={
        "recursion\_limit": 20
    }
)

这里将本次运行的限制设为 20。

需要特别注意：**recursion\_limit 不等于最多调用 20 次模型。**

因为一次 Agent 工具调用循环通常包含多个图执行步骤。

### 超过限制会怎样？

如果执行步骤超过限制，通常会抛出：

```
GraphRecursionError
```

例如：

## 实例：捕获递归限制异常

from langgraph.errors import GraphRecursionError

try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查询 Python 课程"}]},
        config={"recursion\_limit": 10}
    )

except GraphRecursionError:
    print("Agent 执行步骤超过限制，请检查是否重复调用工具。")

**建议：**遇到 GraphRecursionError 时，首先检查 Agent 是否重复调用工具、工具返回内容是否完整，以及模型能否判断任务已经完成。

不要仅通过不断提高 recursion\_limit 掩盖无限循环问题。

---

## 六、max\_concurrency：并发控制

RunnableConfig 可以通过 `max_concurrency` 控制 Runnable 并行调用的数量。

例如使用 batch() 批量向模型发送问题：

## 实例：批量调用

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model("deepseek:deepseek-chat")

questions = [
    "什么是 Python？",
    "什么是 Java？",
    "什么是 Go？",
    "什么是 Rust？"
]

results = model.batch(
    questions,
    config={"max\_concurrency": 2}
)

for question, response in zip(questions, results):
    print(question)
    print(response.content)

这里最多允许两个并发调用任务。

需要注意，max\_concurrency 控制的是相关 Runnable 的并发执行，不等于服务商 API 的每秒请求数限制。

如果需要控制请求频率，还应使用 Rate Limiter 或服务端限流机制。

---

## 七、tags：追踪标签

`tags` 用于给本次运行添加标签。

例如：

```
development

production

rag

customer-service

course-search
```

这些标签可以在 LangSmith 等追踪系统中用于筛选和分组。

### 实例

## 实例

config = {
    "tags": [
        "development",
        "course-search"
    ]
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "查询 Python 课程"}]},
    config=config
)

tags 可以随父级 Runnable 配置向部分子调用传播，方便关联整个执行流程。

---

## 八、metadata：附加元数据

`metadata` 适合保存运行环境和业务分类信息。

例如：

## 实例

config = {
    "metadata": {
        "environment": "development",
        "source": "web",
        "app\_version": "1.0.0",
        "feature": "course\_search"
    }
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "查询 Java 课程"}]},
    config=config
)

与 tags 的区别：

| 字段 | 适用情况 |
| --- | --- |
| `tags` | 简单的分类标签 |
| `metadata` | 带字段名称的结构化数据 |

**注意：**metadata 应尽量使用可以 JSON 序列化的数据，不要存放 API Key、密码、身份凭据等敏感信息。

---

## 九、callbacks：回调处理器

`callbacks` 可以指定 Runnable 执行时使用的回调处理器。

例如，自定义回调可以记录：

- 模型调用开始。
- 模型调用结束。
- 模型发生错误。
- 工具调用。

可以继承 `BaseCallbackHandler` 创建处理器。

### 实例：记录模型执行

## 实例

from langchain\_core.callbacks import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on\_llm\_start(self, serialized, prompts, \*\*kwargs):
        print("[Callback] 模型开始执行")

    def on\_llm\_end(self, response, \*\*kwargs):
        print("[Callback] 模型执行完成")

    def on\_llm\_error(self, error, \*\*kwargs):
        print("[Callback] 模型错误：", type(error).\_\_name\_\_)

config = {
    "callbacks": [DebugCallback()]
}

这个示例定义了模型回调处理器。

具体触发的回调事件取决于模型集成和执行方式。聊天模型通常还会使用专门的 Chat Model 启动事件，因此不要假定所有模型都只通过 on\_llm\_start 触发。

对于基于 LangChain 1.x 的 Agent 执行控制，更推荐使用 Middleware；对于模型和工具的运行追踪，可以结合 callbacks 与 LangSmith。

---

## 十、run\_name 与 run\_id

这两个字段主要用于执行追踪。

| 字段 | 作用 |
| --- | --- |
| `run_name` | 指定本次运行的名称 |
| `run_id` | 指定本次追踪运行的 UUID |

### 实例

## 实例

import uuid

config = {
    "run\_name": "python\_course\_query",
    "run\_id": uuid.uuid4(),
    "tags": ["course-search"]
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "搜索 Python 课程"}]},
    config=config
)

其中 run\_id 是追踪运行的标识，不应与 Checkpointer 的 thread\_id 混淆。

同一个 thread\_id 可以包含多次不同的运行，每次运行都可以有独立的 run\_id。

---

## 十一、RunnableConfig 完整使用示例

将常用字段组合起来：

## 实例

import uuid

config = {
    "configurable": {
        "thread\_id": "conversation-001"
    },
    "recursion\_limit": 20,
    "max\_concurrency": 2,
    "tags": [
        "development",
        "chat"
    ],
    "metadata": {
        "source": "web",
        "app\_version": "1.0.0"
    },
    "run\_name": "runoob\_agent\_request",
    "run\_id": uuid.uuid4()
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config=config
)

**注意：**如果 Agent 没有配置 Checkpointer，那么 thread\_id 本身不会自动提供持久化对话记忆。

---

## 十二、Checkpointer 是什么？

Checkpointer 用于保存 LangGraph 的执行状态。

它可以支持：

- 多轮对话。
- 恢复执行状态。
- Human-in-the-Loop 人工中断。
- 历史检查点查看。
- 调试执行过程。

典型流程：

```
Agent.invoke()

      ↓

   执行节点

      ↓

   保存 Checkpoint

      ↓

   继续执行

      ↓

   保存 Checkpoint
```

Checkpointer 保存的是 Agent 的状态，而不只是聊天文本。

---

## 十三、Checkpointer 实现类速查

| 实现类 | 安装包 | 持久化 | 适用场景 |
| --- | --- | --- | --- |
| `InMemorySaver` | langgraph-checkpoint | 否 | 本地测试 |
| `SqliteSaver` | langgraph-checkpoint-sqlite | 是 | 本地轻量应用 |
| `AsyncSqliteSaver` | langgraph-checkpoint-sqlite | 是 | 异步 SQLite 应用 |
| `PostgresSaver` | langgraph-checkpoint-postgres | 是 | 服务器持久化 |
| `AsyncPostgresSaver` | langgraph-checkpoint-postgres | 是 | 异步服务器应用 |

常用导入方式：

```
from langgraph.checkpoint.memory import InMemorySaver



from langgraph.checkpoint.sqlite import SqliteSaver



from langgraph.checkpoint.postgres import PostgresSaver
```

同步和异步实现不能随意混用。使用异步 Checkpointer 时，通常应该与 ainvoke()、astream() 等异步执行方式配合。

---

## 十四、InMemorySaver：内存检查点

`InMemorySaver` 适合本地开发和功能测试。

它会将状态保存在当前 Python 进程的内存中。

### 实例：多轮对话

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    checkpointer=InMemorySaver()
)

config = {
    "configurable": {
        "thread\_id": "memory-demo-001"
    }
}

agent.invoke(
    {"messages": [{"role": "user", "content": "我正在学习 Python。"}]},
    config=config
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "我刚才说正在学什么？"}]},
    config=config
)

print(result["messages"][-1].content)

相同 thread\_id 下，第二轮可以访问此前的消息历史。

**注意：**程序退出后，InMemorySaver 中的数据会丢失，不适合作为生产环境的持久化方案。

---

## 十五、SqliteSaver：SQLite 持久化

如果希望重启程序后继续保留对话历史，可以使用 SQLite。

### 安装

```
pip install -U langgraph-checkpoint-sqlite
```

### 正确的使用方式

需要注意，`SqliteSaver.from_conn_string()` 返回的是上下文管理器。

因此，应该通过 `with` 使用：

## 实例

from langchain.agents import create\_agent
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from\_conn\_string("checkpoints.db") as checkpointer:

    agent = create\_agent(
        model="deepseek:deepseek-chat",
        tools=[],
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "thread\_id": "sqlite-demo-001"
        }
    }

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        config=config
    )

    print(result["messages"][-1].content)

程序运行后，会在当前目录创建：

```
checkpoints.db
```

之后可以使用相同数据库和 thread\_id 恢复历史状态。

**重要：**不要这样写：

```
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

然后直接将 checkpointer 传给 Agent，因为此时得到的不是已经进入连接上下文的 Saver 实例。

SQLite 适合轻量本地应用，不建议直接用于大量并发写入的多实例 Agent 服务。

---

## 十六、PostgresSaver：PostgreSQL 持久化

对于需要长期运行的多用户 Agent，可以考虑使用 PostgreSQL。

### 安装依赖

```
pip install -U langgraph-checkpoint-postgres "psycopg[binary,pool]"
```

PostgreSQL Checkpointer 支持更可靠的数据库持久化管理。

### 实例：PostgresSaver

## 实例

import os
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.checkpoint.postgres import PostgresSaver

load\_dotenv()

DB\_URI = os.environ["POSTGRES\_URI"]

with PostgresSaver.from\_conn\_string(DB\_URI) as checkpointer:

    # 首次使用时初始化所需数据库结构
    checkpointer.setup()

    agent = create\_agent(
        model="deepseek:deepseek-chat",
        tools=[],
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "thread\_id": "postgres-demo-001"
        }
    }

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        config=config
    )

    print(result["messages"][-1].content)

在 `.env` 中配置：

```
POSTGRES_URI=postgresql://db_user:db_password@localhost:5432/langchain_db
```

上面的连接字符串仅用于格式演示。

`setup()` 用于创建或迁移相关数据库结构，通常在首次部署或受控迁移阶段执行，不必在每个用户请求中调用。

**建议：**生产环境应为数据库使用独立账号，并通过环境变量或密钥管理服务保存密码。

---

## 十七、Checkpointer 常用方法

Checkpointer 本身负责底层状态保存，而 Agent 则通过 CompiledStateGraph 提供更方便的状态访问接口。

对于使用 Checkpointer 的 Agent，常见方法包括：

| 方法 | 说明 |
| --- | --- |
| `agent.get_state(config)` | 读取当前状态快照 |
| `agent.get_state_history(config)` | 查看历史检查点 |
| `agent.update_state(config, values)` | 写入新的状态更新 |
| `agent.aget_state(config)` | 异步读取状态 |

### 实例：读取当前状态

## 实例

config = {
    "configurable": {
        "thread\_id": "memory-demo-001"
    }
}

state = agent.get\_state(config)

print("消息数量：", len(state.values.get("messages", [])))
print("下一步节点：", state.next)
print("检查点配置：", state.config)

这里读取的是 StateSnapshot，而不是普通的对话消息字典。

需要保证 agent 是已经使用 Checkpointer 创建、并且当前 thread\_id 存在相关状态的 Agent。

---

## 十八、Store：长期数据存储

Checkpointer 主要保存单个线程中的执行状态。

如果需要保存跨线程共享的数据，可以使用 Store。

例如，用户有三个会话：

```
用户 user-001

      |

      ├── thread-a

      ├── thread-b

      └── thread-c

             |

             ↓

          共享 Store

             |

             ↓

      Python 学习偏好
```

Store 可以保存：

- 用户学习偏好。
- 应用配置。
- 用户长期资料。
- 跨会话共享的业务数据。

但 Store 不会自动读取和总结用户的聊天内容，长期记忆的提取与更新仍需要业务代码实现。

---

## 十九、Store 实现类速查

| 类 | 导入路径 | 持久化 |
| --- | --- | --- |
| `InMemoryStore` | langgraph.store.memory | 否 |
| `PostgresStore` | langgraph.store.postgres | 是 |
| `AsyncPostgresStore` | langgraph.store.postgres | 是 |

与 Checkpointer 类似，Store 也分为内存实现和持久化实现。

多用户系统应根据并发要求、访问权限和备份策略选择合适的存储方案。

---

## 二十、InMemoryStore 常用 API

InMemoryStore 适合演示 Store 的基本操作。

### 常用方法

| 方法 | 作用 |
| --- | --- |
| `put(namespace, key, value)` | 保存数据 |
| `get(namespace, key)` | 获取数据 |
| `search(namespace_prefix)` | 搜索命名空间下的数据 |
| `delete(namespace, key)` | 删除数据 |

### 完整实例：Store CRUD

## 实例

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

namespace = ("users", "user-001")

# 1. 保存
store.put(
    namespace,
    "learning\_preference",
    {
        "language": "Python",
        "level": "beginner"
    }
)

# 2. 获取
item = store.get(namespace, "learning\_preference")

if item:
    print("用户偏好：", item.value)

# 3. 搜索
items = store.search(namespace)

for item in items:
    print("键：", item.key)
    print("值：", item.value)

# 4. 删除
store.delete(namespace, "learning\_preference")

print("删除后：", store.get(namespace, "learning\_preference"))

示意输出：

```
用户偏好： {'language': 'Python', 'level': 'beginner'}

键： learning_preference

值： {'language': 'Python', 'level': 'beginner'}

删除后： None
```

InMemoryStore 数据不会跨进程重启持久化。

---

## 二十一、PostgresStore：长期持久化存储

如果需要将 Store 数据持久化到 PostgreSQL，可以使用 PostgresStore。

### 安装

```
pip install -U langgraph-checkpoint-postgres "psycopg[binary,pool]"
```

### 实例

## 实例

import os
from dotenv import load\_dotenv
from langgraph.store.postgres import PostgresStore

load\_dotenv()

DB\_URI = os.environ["POSTGRES\_URI"]

with PostgresStore.from\_conn\_string(DB\_URI) as store:

    # 首次使用或迁移数据库结构时执行
    store.setup()

    namespace = ("users", "user-001")

    store.put(
        namespace,
        "language",
        {"value": "Python"}
    )

    item = store.get(namespace, "language")

    print(item.value if item else "未找到数据")

PostgresStore 适合保存长期数据。

它与 PostgresSaver 不同：

| 对象 | 保存内容 |
| --- | --- |
| `PostgresSaver` | Agent 的线程执行状态和检查点 |
| `PostgresStore` | 用户偏好及其他跨会话数据 |

**注意：**即使两者使用同一个 PostgreSQL 服务，也应根据应用需求明确数据生命周期、权限和备份策略。

---

## 二十二、Store 与 Agent 集成

在创建 Agent 时，可以通过 `store` 参数传入长期存储。

## 实例

from langchain.agents import create\_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    store=store
)

但这里存在一个非常重要的区别：

**将 Store 传入 Agent，不代表模型自动读取 Store 中的数据。**

需要通过 ToolRuntime、Middleware 或其他业务逻辑显式读取。

例如，用户询问自己的学习偏好时，工具根据可信用户身份从 Store 中获取数据，再将结果提供给模型。

---

## 二十三、LangChain 常见异常类

LangChain 和 LangGraph 在模型调用、工具执行、结构化输出及图状态管理中，都可能产生异常。

实际开发时，最好根据异常来源分类处理。

| 异常 | 来源 | 说明 |
| --- | --- | --- |
| `ImportError` | Python | 依赖缺失或导入错误 |
| `ValueError` | Python | 参数值错误 |
| `TypeError` | Python | 参数类型或调用方式错误 |
| `NotImplementedError` | Python | 当前接口或实现不支持指定操作 |
| `TimeoutError` | Python / SDK | 部分请求超时场景 |
| `ToolException` | langchain\_core.tools | 工具主动报告执行错误 |
| `StructuredOutputError` | langchain.agents.structured\_output | 结构化输出相关错误的基类 |
| `StructuredOutputValidationError` | langchain.agents.structured\_output | 结构化工具参数无法按 Schema 解析 |
| `MultipleStructuredOutputsError` | langchain.agents.structured\_output | 模型返回多个结构化输出调用 |
| `GraphRecursionError` | langgraph.errors | 图执行超过步骤限制 |
| `InvalidUpdateError` | langgraph.errors | 图状态更新不合法 |
| `EmptyInputError` | langgraph.errors | 图收到空输入 |
| `GraphInterrupt` | langgraph.errors | 图运行时内部中断信号 |

**注意：**模型提供商 SDK 也会定义自己的错误类型，例如认证失败、限流、API 连接异常等，不会全部转换为 Python 内置 TimeoutError。

---

## 二十四、ToolException：工具执行异常

ToolException 用于表示工具执行错误。

例如，用户查询的课程编号不合法：

## 实例

from langchain.tools import tool, ToolException

@tool
def get\_course(course\_id: int) -> str:
    """根据 ID 查询模拟课程。"""
    if course\_id <= 0:
        raise ToolException("课程编号必须大于 0")

    return f"模拟课程 {course\_id}：Python 基础"

get\_course.handle\_tool\_error = True

print(get\_course.invoke({"course\_id": -1}))

当配置 handle\_tool\_error=True 时，工具可以将 ToolException 转换成错误结果。

但需要注意：

- ToolException 不会自动重试。
- 不代表任何异常都会转换成 ToolMessage。
- 真实的网络和数据库故障应使用相应的异常分类。

在 Agent 中，也可以使用 ToolErrorMiddleware 等机制统一处理工具异常。

---

## 二十五、StructuredOutputError：结构化输出异常

LangChain Agent 可以通过 response\_format 返回结构化结果。

例如：

```
response_format=ToolStrategy(MySchema)
```

如果模型生成的结果不符合要求，就可能出现结构化输出相关错误。

### 异常层次

```
StructuredOutputError

       |

       ├── StructuredOutputValidationError

       |

       └── MultipleStructuredOutputsError
```

其中：

| 异常 | 触发条件 |
| --- | --- |
| `StructuredOutputValidationError` | 结构化工具调用参数不能通过 Schema 解析 |
| `MultipleStructuredOutputsError` | 期望单个结构化输出，却收到多个相关工具调用 |

### 实例：捕获结构化输出错误

## 实例

from langchain.agents.structured\_output import (
    StructuredOutputError,
    StructuredOutputValidationError,
    MultipleStructuredOutputsError
)

try:
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "提取课程结构化信息。"}
        ]
    })

except StructuredOutputValidationError as e:
    print("结构化输出校验失败：", type(e).\_\_name\_\_)

except MultipleStructuredOutputsError:
    print("模型生成了多个结构化输出。")

except StructuredOutputError:
    print("结构化输出处理失败。")

这里假设 agent 已配置了适当的 response\_format。

**注意：**使用 ToolStrategy 时，LangChain 可能根据 handle\_errors 配置将部分异常反馈给模型并请求重新生成，因此并不是每次校验失败都会直接向应用抛出异常。

另外，这些结构化输出异常并不覆盖所有提供商原生 JSON 输出或 Pydantic 错误。

---

## 二十六、ToolStrategy.handle\_errors

如果希望模型在结构化输出错误时尝试修正，可以配置：

```
ToolStrategy(

    MySchema,

    handle_errors=True

)
```

主要配置：

| 配置 | 说明 |
| --- | --- |
| `True` | 使用默认错误反馈机制 |
| `False` | 不通过该策略自动处理错误 |
| `str` | 使用指定错误提示文本 |
| `异常类型` | 只处理指定类型的异常 |
| `Callable` | 自定义错误反馈函数 |

### 实例：严格结构化输出

## 实例

from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

class CourseInfo(BaseModel):
    name: str = Field(description="课程名称")
    level: Literal["beginner", "intermediate", "advanced"]

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    response\_format=ToolStrategy(
        CourseInfo,
        handle\_errors=True
    )
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "返回课程信息：Python 入门课程，beginner 难度。"
    }]
})

print(result["structured\_response"])

使用 Pydantic Schema 可以在解析结果时执行字段校验。

需要注意，如果直接使用原始 JSON Schema 字典，相关 ToolStrategy 路径并不一定执行同等的 Schema 校验。需要可靠字段验证时，建议使用 Pydantic 模型。

---

## 二十七、GraphRecursionError：图执行循环

GraphRecursionError 表示图执行超过设定的步骤限制。

常见原因：

- Agent 重复调用工具。
- 工具结果未能让模型判断任务完成。
- 自定义图存在循环路由错误。
- 执行步骤预算过小。

可以捕获：

```
from langgraph.errors import GraphRecursionError
```

但不建议只通过增加 recursion\_limit 解决问题。

对于普通 Agent，还可以考虑 ModelCallLimitMiddleware 和 ToolCallLimitMiddleware，进一步限制模型与工具调用次数。

---

## 二十八、InvalidUpdateError：状态更新错误

LangGraph 使用 State 和 Reducer 管理状态。

如果节点返回不符合状态要求的更新，或多个并行节点向不支持并发写入的字段写入值，就可能出现：

```
InvalidUpdateError
```

例如：

```
节点 A ──→ 写入 status

节点 B ──→ 写入 status



同一执行步骤中发生冲突
```

解决方式通常包括：

- 检查状态字段的类型。
- 检查节点返回值是否符合 State Schema。
- 为确实需要合并的字段定义 Reducer。
- 调整并行执行或状态更新设计。

**注意：**不要只使用 try/except 忽略 InvalidUpdateError，它通常表示程序状态设计存在问题。

---

## 二十九、超时与网络异常

模型调用可能产生各种网络异常。

例如：

- 连接超时。
- 读取超时。
- 连接中断。
- 服务端限流。
- 模型服务不可用。

可以为模型设置请求超时：

## 实例

from langchain\_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-chat",
    timeout=30,
    max\_retries=2
)

这里设置 30 秒请求超时和模型客户端重试配置。

但不同提供商 SDK 可能抛出不同的异常，不能假定所有超时都继承 Python 的内置 TimeoutError。

**建议：**在生产环境中，根据当前模型 SDK 的异常类，分别处理认证、限流、连接错误和超时。

---

## 三十、异常处理综合实例

下面创建一个简单的 Agent，并为执行过程添加基础异常处理。

文件名：`agent_error_demo.py`

## 完整实例

import logging
import os

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain\_deepseek import ChatDeepSeek
from langgraph.errors import (
    GraphRecursionError,
    InvalidUpdateError
)
from langchain.agents.structured\_output import StructuredOutputError

load\_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runoob.agent")

if not os.getenv("DEEPSEEK\_API\_KEY"):
    raise RuntimeError("未配置 DEEPSEEK\_API\_KEY")

agent = create\_agent(
    model=ChatDeepSeek(
        model="deepseek-chat",
        timeout=30,
        max\_retries=1
    ),
    tools=[],
    system\_prompt="你是 Python 编程助手。"
)

def ask(question: str):
    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config={
                "recursion\_limit": 20,
                "tags": ["error-demo"]
            }
        )

        print(result["messages"][-1].content)

    except GraphRecursionError:
        logger.error("图执行步骤超过限制")
        print("请求未完成：执行步骤过多。")

    except InvalidUpdateError:
        logger.exception("Agent 状态更新失败")
        print("请求处理失败：状态更新错误。")

    except StructuredOutputError:
        logger.exception("结构化输出失败")
        print("请求处理失败：模型输出格式不正确。")

    except Exception:
        logger.exception("Agent 执行发生未知错误")
        print("请求处理失败，请检查程序日志。")

if \_\_name\_\_ == "\_\_main\_\_":
    ask("介绍 Python 的主要特点。")

这里的异常处理主要用于演示分类思路。

生产环境还应该根据所用模型 SDK 增加认证、连接、限流和超时异常的处理。

不要直接向用户返回完整异常堆栈，也不要在日志中输出 API Key。

---

## 三十一、LangChain 环境配置检查清单

如果程序无法运行，可以按下面的顺序检查。

| 检查项目 | 命令或方法 |
| --- | --- |
| Python 版本 | `python --version` |
| Python 环境路径 | `python -c "import sys; print(sys.executable)"` |
| LangChain 版本 | `python -m pip show langchain` |
| LangGraph 版本 | `python -m pip show langgraph` |
| DeepSeek 集成 | `python -m pip show langchain-deepseek` |
| SQLite Checkpointer | `python -m pip show langgraph-checkpoint-sqlite` |
| PostgreSQL Checkpointer | `python -m pip show langgraph-checkpoint-postgres` |
| 依赖兼容性 | `python -m pip check` |
| API Key 是否配置 | 通过 Python 检查环境变量是否存在 |
| 模型连通性 | 发送简单请求，并检查实际错误 |
| Checkpointer | 检查 thread\_id、数据库权限和连接生命周期 |

对于 LangChain 1.x，建议使用 Python 3.10 或更新的兼容版本，并以实际安装的包元数据确认运行环境要求。

### 安全检查 API Key

## 实例

import os
from dotenv import load\_dotenv

load\_dotenv()

key = os.getenv("DEEPSEEK\_API\_KEY")

print("DEEPSEEK\_API\_KEY 已配置：", bool(key))

不建议打印 API Key 的前几位或完整内容。

### 检查依赖版本

## 实例

from importlib.metadata import version, PackageNotFoundError

packages = [
    "langchain",
    "langchain-core",
    "langgraph",
    "langchain-deepseek",
    "langgraph-checkpoint-sqlite"
]

for package in packages:
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, "未安装")

这个示例不会连接模型，也不需要 API Key。

---

## 三十二、完整实例：配置与持久化综合演示

最后，将 RunnableConfig、Checkpointer、Store 和状态查询组合起来。

这个例子包含：

- 创建 Agent。
- 保存多轮对话。
- 使用 RunnableConfig。
- 创建长期 Store。
- 读取历史状态。

文件名：`config_api_demo.py`

### 安装依赖

```
pip install -U langchain langchain-deepseek langgraph python-dotenv
```

## 完整实例

from dotenv import load\_dotenv

from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

load\_dotenv()

# ========== 1. Checkpointer ==========

checkpointer = InMemorySaver()

# ========== 2. Store ==========

store = InMemoryStore()

store.put(
    ("users", "student-001"),
    "preference",
    {
        "language": "Python",
        "level": "beginner"
    }
)

# ========== 3. 创建 Agent ==========

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    system\_prompt="你是 Python 编程学习助手。",
    checkpointer=checkpointer,
    store=store
)

# ========== 4. RunnableConfig ==========

config = {
    "configurable": {
        "thread\_id": "config-demo-001"
    },
    "recursion\_limit": 20,
    "tags": ["tutorial", "memory"],
    "metadata": {
        "source": "runoob",
        "environment": "development"
    }
}

# ========== 5. 第一轮对话 ==========

agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "我正在学习 Python。"
        }]
    },
    config=config
)

# ========== 6. 第二轮对话 ==========

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "我刚才说正在学习什么？"
        }]
    },
    config=config
)

print("Agent 回答：", result["messages"][-1].content)

# ========== 7. 读取 Checkpointer 状态 ==========

state = agent.get\_state(config)

print("消息数量：", len(state.values.get("messages", [])))
print("下一步节点：", state.next)

# ========== 8. 读取 Store 数据 ==========

item = store.get(
    ("users", "student-001"),
    "preference"
)

print("长期偏好：", item.value if item else None)

这个示例展示了 Checkpointer 和 Store 同时使用的方式。

需要特别注意：

- Checkpointer 保存的是 thread\_id 对应的执行状态。
- Store 保存的是指定 namespace 和 key 对应的长期数据。
- Agent 不会自动读取 Store 中保存的偏好。
- InMemorySaver 和 InMemoryStore 都不会跨进程重启持久化。

---

## 三十三、常见问题

### 1、RunnableConfig 是必填参数吗？

不是。大多数 Runnable 可以在不传 config 的情况下执行。但如果 Agent 使用需要线程标识的 Checkpointer，就需要提供合适的 thread\_id。

### 2、recursion\_limit 默认是多少？

当前 Python RunnableConfig 文档中的默认值是 25，不是 9999。

### 3、recursion\_limit=20 表示最多调用模型 20 次吗？

不是，它限制的是底层图的执行步骤数量。

### 4、max\_concurrency 可以防止所有 API 限流吗？

不能。它控制并发执行数量，模型服务商的请求速率和配额还需要独立管理。

### 5、run\_id 和 thread\_id 有什么区别？

run\_id 标识一次运行追踪；thread\_id 标识一个可持续保存状态的对话线程。

### 6、Checkpointer 和 Store 有什么区别？

Checkpointer 保存线程执行状态；Store 保存跨线程长期数据。

### 7、InMemorySaver 能跨程序重启保存历史吗？

不能。它只保存当前进程中的状态。

### 8、SqliteSaver.from\_conn\_string() 可以直接赋值使用吗？

它返回上下文管理器，推荐在 with 代码块中使用，以确保连接正常打开和关闭。

### 9、PostgresSaver 为什么需要 setup()？

因为首次使用时需要初始化相关的数据库结构。一般在部署或迁移阶段执行。

### 10、Store 中保存的内容会自动进入模型上下文吗？

不会，需要通过工具、Middleware 或其他业务逻辑读取。

### 11、ToolException 会自动让 Agent 重新调用工具吗？

不一定。需要根据工具错误处理配置和 Agent 策略决定。

### 12、StructuredOutputValidationError 何时发生？

通常是在工具策略产生的结构化输出参数无法通过指定 Schema 解析时。

### 13、MultipleStructuredOutputsError 是什么？

模型在只应生成一个结构化结果的场景中返回了多个结构化工具调用。

### 14、GraphRecursionError 应该怎么处理？

先排查循环调用和流程设计，再根据实际需求调整执行步骤限制。

### 15、InvalidUpdateError 可以忽略吗？

不建议。它通常说明状态更新或 Reducer 配置存在问题。

### 16、Python TimeoutError 能捕获所有模型超时吗？

不能。不同模型提供商、HTTP 客户端和 LangGraph 节点可能使用不同的异常类。

### 17、RunnableConfig.metadata 可以存放 API Key 吗？

不建议。metadata 可能被传递给追踪系统或回调处理器，不应该存放敏感凭据。

### 18、LangChain 运行环境应该使用哪个 Python 版本？

建议使用 Python 3.10 或更新的兼容版本，同时检查实际安装包声明的 Python 版本要求。

---

## 总结

本章节整理了 LangChain Agent 开发中常见的配置、存储与异常处理 API。

主要内容包括：

- `RunnableConfig`：统一的执行配置。
- `configurable`：动态配置与 thread\_id。
- `recursion_limit`：图执行步骤限制。
- `max_concurrency`：并发控制。
- `tags`、`metadata`：追踪信息。
- `callbacks`：执行回调。
- `run_name`、`run_id`：运行标识。
- `InMemorySaver`：内存检查点。
- `SqliteSaver`：SQLite 持久化。
- `PostgresSaver`：PostgreSQL 状态持久化。
- `InMemoryStore`：内存长期数据存储。
- `PostgresStore`：PostgreSQL 长期存储。
- `ToolException`：工具执行错误。
- `StructuredOutputError`：结构化输出错误。
- `GraphRecursionError`：图执行超过步骤限制。
- `InvalidUpdateError`：状态更新错误。

最常用的 RunnableConfig 写法：

```
config = {

    "configurable": {

        "thread_id": "conversation-001"

    },

    "recursion_limit": 20,

    "tags": ["production"],

    "metadata": {

        "source": "web"

    }

}
```

然后执行：

```
result = agent.invoke(

    {"messages": [{"role": "user", "content": "你好"}]},

    config=config

)
```

**核心原则：**RunnableConfig 控制执行方式，Checkpointer 保存线程状态，Store 保存跨会话数据，异常类帮助程序识别失败原因。合理组合这些能力，才能构建可恢复、可观测且容易维护的 LangChain Agent。

## 参考文档

- [RunnableConfig API Reference](https://reference.langchain.com/python/langchain-core/runnables/config/RunnableConfig)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangChain 短期记忆](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangChain 长期记忆](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [Structured Output API Reference](https://reference.langchain.com/python/langchain/agents/structured_output)
- [LangGraph Errors API Reference](https://reference.langchain.com/python/langgraph/errors)
- [LangChain 内置 Middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
