# LangChain 跨会话存储——Store

> 来源: https://www.runoob.com/langchain/langchain-store.html

---

# LangChain 跨会话存储——Store

上一章节我们介绍了 Checkpointer，它可以保存 Agent 的对话历史，让同一个线程具备多轮记忆能力。

但如果用户开启新的对话，使用了不同的 thread\_id，之前的聊天记录通常不会自动出现在新会话中。

如果希望 Agent 在不同会话之间共享用户偏好、学习进度或长期信息，就可以使用 Store（长期存储）。

LangChain 1.x 基于 LangGraph Store 提供跨会话的数据存储能力，支持按命名空间管理数据，并通过 ToolRuntime 在工具中读写。

本章节介绍 Store 的基本操作、在 Agent 中使用 Store、跨会话记忆、与 Checkpointer 配合、PostgreSQL 持久化以及数据安全。

---

## 一、Checkpointer 与 Store 的区别

两者都能保存数据，但负责的内容不同。

| 对比项 | Checkpointer | Store |
| --- | --- | --- |
| 主要用途 | 保存 Agent 执行状态 | 保存长期信息 |
| 数据范围 | 按 thread\_id 管理 | 按 namespace 和 key 管理 |
| 保存内容 | 消息、状态、检查点 | JSON 文档 |
| 跨会话共享 | 不同线程不自动共享 | 可按业务设计跨线程访问 |
| 读写方式 | Agent 执行时自动保存状态 | 由程序或工具显式读写 |
| 常见用途 | 多轮对话、中断恢复 | 用户偏好、学习记录、知识资料 |

可以这样理解：

```
Checkpointer

    ├── 对话 A 的消息历史

    ├── 对话 B 的消息历史

    └── 当前 Agent 执行状态



Store

    ├── 用户学习偏好

    ├── 已完成的课程

    └── 长期学习目标
```

同一个用户即使开启不同对话，也可以通过 Store 读取自己的长期信息。

**注意：**Store 并不会自动从聊天中提取用户偏好。需要程序或 Agent 工具明确执行保存和读取操作。

---

## 二、Store 的数据组织结构

Store 采用 namespace + key + value 的方式组织数据。

可以将它理解为：

```
namespace：数据所属的命名空间

key：命名空间中的唯一键

value：需要保存的 JSON 数据
```

例如：

```
namespace = ("users", "user_001")

key = "preferences"



value = {

    "language": "zh-CN",

    "theme": "dark"

}
```

其中，namespace 是字符串元组，可以用于区分用户、项目或业务模块。

### 推荐的命名空间设计

| 数据类型 | namespace | key |
| --- | --- | --- |
| 用户偏好 | `("users", user_id)` | preferences |
| 学习进度 | `("users", user_id)` | progress |
| 个人记忆 | `("memories", user_id)` | memory\_id |
| 公共教程 | `("runoob", "courses")` | course\_id |
| 项目配置 | `("projects", project_id)` | settings |

建议将数据按访问范围划分：公共数据使用公共命名空间，用户私人数据使用包含可信用户 ID 的命名空间。

---

## 三、InMemoryStore 基本操作

`InMemoryStore` 是最简单的 Store 实现，数据保存在当前 Python 进程内存中。

主要提供四种常见操作：

| 方法 | 说明 |
| --- | --- |
| `put()` | 写入或更新数据 |
| `get()` | 读取指定数据 |
| `search()` | 检索命名空间中的数据 |
| `delete()` | 删除指定数据 |

### 实例：读写用户偏好和学习进度

## 实例

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = ("users", "user\_001")

# 写入
store.put(namespace, "preferences", {"theme": "dark", "language": "zh-CN", "level": "入门"})
store.put(namespace, "progress", {"completed\_courses": ["HTML 基础", "Python 基础"], "total\_hours": 35})

# 读取
prefs = store.get(namespace, "preferences")
print("用户偏好：", prefs.value if prefs else None)

progress = store.get(namespace, "progress")
print("学习进度：", progress.value if progress else None)

# 搜索
for item in store.search(namespace):
    print(item.key, item.value)

# 删除
store.delete(namespace, "preferences")
print("删除后：", store.get(namespace, "preferences"))

运行结果：

```
用户偏好： {'theme': 'dark', 'language': 'zh-CN', 'level': '入门'}

学习进度： {'completed_courses': ['HTML 基础', 'Python 基础'], 'total_hours': 35}

preferences {'theme': 'dark', 'language': 'zh-CN', 'level': '入门'}

progress {'completed_courses': ['HTML 基础', 'Python 基础'], 'total_hours': 35}

删除后： None
```

需要注意：

- `get()` 返回 Store Item，实际数据在 `item.value` 中。
- 如果没有找到对应数据，通常返回 `None`。
- `put()` 使用相同 namespace 和 key 时，会更新对应数据。
- `delete()` 删除的是指定数据项，不代表删除整个用户的其他数据。

**注意：**InMemoryStore 只适合开发和测试。程序退出后数据会丢失，不应该直接作为长期生产存储。

---

## 四、理解 Store Item

通过 `store.get()` 读取数据时，返回的不是简单字典，而是包含数据及元信息的 Item。

### 实例

## 实例

item = store.get(namespace, "progress")

if item:
    print("命名空间：", item.namespace)
    print("键：", item.key)
    print("数据：", item.value)
    print("创建时间：", item.created\_at)
    print("更新时间：", item.updated\_at)

这些属性可以帮助开发者了解数据的内容和保存时间。

但需要注意，更新时间不等于数据一定是最新的业务信息。

例如，Store 中保存的课程目录可能已经过时，需要由业务程序定期同步。

---

## 五、在 Agent 中访问 Store

创建 Agent 时，通过 `store` 参数传入 Store 实例：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[...],

    store=store

)
```

LangChain 1.x 推荐使用 `ToolRuntime` 在工具中访问 Store。

在工具函数中增加：

```
runtime: ToolRuntime
```

然后通过：

```
runtime.store
```

读取或写入数据。

与旧版注入写法相比，这种方式更简洁，也可以统一访问 State、Context 和工具调用 ID。

### 实例：课程查询工具

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

load\_dotenv()

store = InMemoryStore()
store.put(("runoob", "courses"), "python3", {
    "name": "Python3 基础教程", "level": "入门", "price": "免费"
})

@tool
def query\_course\_info(course\_id: str, runtime: ToolRuntime) -> str:
    """根据课程 ID 查询模拟课程信息。"""
    if runtime.store is None:
        return "课程存储不可用"

    item = runtime.store.get(("runoob", "courses"), course\_id)
    return str(item.value) if item else "未找到相关课程"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[query\_course\_info],
    store=store,
    system\_prompt="你是课程顾问。查询课程时使用工具，课程数据为教学模拟数据。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请用工具查询 python3 课程。"}]
})
print(result["messages"][-1].content)

如果模型调用工具，可能回答：

```
Python3 基础教程是一门免费的入门课程。
```

这里的课程资料来自 InMemoryStore，不会连接菜鸟教程网站的真实课程数据库。

**注意：**模型不一定主动调用工具。如果某项查询必须执行，应该由应用程序或确定性工作流负责。

---

## 六、ToolRuntime：读取运行时信息

`ToolRuntime` 是 LangChain 1.x 中工具获取运行时信息的主要方式。

| 属性 | 说明 |
| --- | --- |
| `runtime.store` | 当前配置的 Store |
| `runtime.context` | 当前调用的运行时 Context |
| `runtime.state` | 当前 Agent State |
| `runtime.tool_call_id` | 当前工具调用 ID |
| `runtime.config` | 当前执行配置 |
| `runtime.stream_writer` | 自定义流式事件写入器 |

工具参数中的 `runtime` 由 LangChain 自动注入，不会作为普通工具参数发送给大语言模型。

例如：

```
@tool

def my_tool(keyword: str, runtime: ToolRuntime) -> str:

    ...
```

模型只需要生成 `keyword` 参数。

**注意：**只有创建 Agent 时传入了 Store，才能保证 `runtime.store` 可用于相应的存储操作。

---

## 七、根据可信用户 ID 读取个人数据

如果 Store 中保存的是用户私人信息，不应该让模型自己提供任意 user\_id 查询。

例如，下面这种工具设计存在越权风险：

```
@tool

def get_user_info(user_id: str):

    ...
```

因为模型可能生成其他用户的 ID。

更合理的方式是通过 `context_schema` 接收应用后端已经认证的用户信息。

### 实例：读取当前用户学习偏好

## 实例

from dataclasses import dataclass
from langchain.agents import create\_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class UserContext:
    user\_id: str

store = InMemoryStore()
store.put(("users", "user\_001"), "preferences", {
    "language": "zh-CN", "level": "入门", "favorite": "Python"
})

@tool
def get\_my\_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """读取当前用户的学习偏好。"""
    if runtime.store is None:
        return "存储服务不可用"

    user\_id = runtime.context.user\_id
    item = runtime.store.get(("users", user\_id), "preferences")
    return str(item.value) if item else "尚未保存学习偏好"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[get\_my\_preferences],
    store=store,
    context\_schema=UserContext,
    system\_prompt="你是学习助手。查询用户偏好时使用 get\_my\_preferences。"
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "请查询我的学习偏好。"}]},
    context=UserContext(user\_id="user\_001")
)
print(result["messages"][-1].content)

如果工具正确执行，Agent 可以获取当前用户的偏好：

```
学习语言：Python

学习级别：入门

回答语言：中文
```

这里没有将 user\_id 暴露为模型可填写的工具参数，而是从 Runtime Context 获取。

**重要：**Context 中的 user\_id 应由可信后端认证后注入。不能直接把用户浏览器提交的任意 user\_id 当作已认证身份。

---

## 八、通过工具保存用户长期记忆

除了读取 Store，工具也可以写入 Store。

例如，用户告诉学习助手：

```
以后给我讲 Python 时，尽量使用简单的例子。
```

Agent 可以调用工具，保存这一偏好。

### 实例：保存学习偏好

## 实例

from langchain.tools import tool, ToolRuntime

@tool
def save\_learning\_preference(preference: str, runtime: ToolRuntime[UserContext]) -> str:
    """保存当前用户明确要求记住的学习偏好。"""
    if runtime.store is None:
        return "存储服务不可用"

    user\_id = runtime.context.user\_id
    namespace = ("users", user\_id)
    item = runtime.store.get(namespace, "preferences")
    data = dict(item.value) if item else {}

    data["learning\_style"] = preference
    runtime.store.put(namespace, "preferences", data)
    return "学习偏好已保存"

这里先读取原有的 preferences，再更新学习风格，避免直接覆盖之前保存的语言和学习等级等字段。

存储后的数据可能为：

```
{

    "language": "zh-CN",

    "level": "入门",

    "favorite": "Python",

    "learning_style": "尽量使用简单的例子"

}
```

如果使用 `InMemoryStore`，数据只会在进程内保留。

**注意：**长期偏好应尽量由用户明确提出或经过确认，避免把模型的推测当作用户真实偏好保存。

---

## 九、完整实例：跨两个会话共享学习偏好

这是 Store 最重要的应用场景。

下面将 `Checkpointer` 与 `Store` 一起使用。

两个会话使用不同的 thread\_id，但使用同一个 user\_id。

这样可以实现：

- 不同会话拥有独立的聊天记录。
- 不同会话共享同一个用户的长期学习偏好。
- 学习偏好通过 Store 保存，不需要从之前的聊天历史中重新提取。

文件名：`store_memory_demo.py`

安装依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

配置 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dataclasses import dataclass
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver

load\_dotenv()

@dataclass
class UserContext:
    user\_id: str

store = InMemoryStore()
checkpointer = InMemorySaver()

@tool
def save\_preference(preference: str, runtime: ToolRuntime[UserContext]) -> str:
    """根据用户要求保存学习偏好。"""
    if runtime.store is None:
        return "存储不可用"
    namespace = ("users", runtime.context.user\_id)
    runtime.store.put(namespace, "learning\_style", {"text": preference})
    return "偏好已保存"

@tool
def read\_preference(runtime: ToolRuntime[UserContext]) -> str:
    """读取当前用户保存的学习偏好。"""
    if runtime.store is None:
        return "存储不可用"
    item = runtime.store.get(("users", runtime.context.user\_id), "learning\_style")
    return item.value["text"] if item else "没有保存的学习偏好"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[save\_preference, read\_preference],
    store=store,
    checkpointer=checkpointer,
    context\_schema=UserContext,
    system\_prompt="你是编程学习助手。用户要求保存偏好时调用 save\_preference；查询偏好时调用 read\_preference。"
)

context = UserContext(user\_id="student-001")
chat\_a = {"configurable": {"thread\_id": "chat-a"}}
chat\_b = {"configurable": {"thread\_id": "chat-b"}}

result1 = agent.invoke({
    "messages": [{"role": "user", "content": "请记住：给我讲 Python 时，多用简单的代码示例。"}]
}, config=chat\_a, context=context)
print("会话 A：", result1["messages"][-1].content)

result2 = agent.invoke({
    "messages": [{"role": "user", "content": "请读取我保存的学习偏好。"}]
}, config=chat\_b, context=context)
print("会话 B：", result2["messages"][-1].content)

如果模型正确调用保存和读取工具，可能得到：

```
会话 A：学习偏好已经保存。



会话 B：你希望学习 Python 时，多使用简单的代码示例。
```

两个 thread\_id 不同：

```
chat-a

chat-b
```

但两个会话使用相同的：

```
user_id = "student-001"
```

因此，虽然聊天历史不同，但工具可以通过相同的 namespace 访问长期偏好。

可以理解为：

```
用户 student-001

        |

        +---- 会话 chat-a ---- Checkpointer

        |

        +---- 会话 chat-b ---- Checkpointer

        |

        +---- 学习偏好 ----- Store
```

**注意：**这个示例的工具调用由模型决定。生产环境如果必须保证某项偏好被保存，应由应用程序显式执行保存操作，并检查结果。

---

## 十、让 Agent 自动读取长期偏好

前面的例子需要模型主动调用工具才能读取学习偏好。

但有时希望用户每次提问时，Agent 都能自动加载已保存的偏好。

可以使用 `@dynamic_prompt`，在模型请求前读取 Store，然后动态生成 System Prompt。

### 实例：将偏好注入系统提示词

## 实例

from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def personalized\_prompt(request: ModelRequest[UserContext]) -> str:
    """读取当前用户的学习偏好。"""
    base = "你是菜鸟教程 RUNOOB 的编程助手。"
    store = request.runtime.store
    if store is None:
        return base

    namespace = ("users", request.runtime.context.user\_id)
    item = store.get(namespace, "learning\_style")
    if not item:
        return base

    preference = item.value.get("text", "")
    return f"{base}\n用户保存的学习偏好：{preference}\n在不影响准确性的前提下参考此偏好。"

创建 Agent 时加入：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[],

    store=store,

    context_schema=UserContext,

    middleware=[personalized_prompt]

)
```

这样即使用户没有主动询问偏好，Agent 也可以在每次模型调用前读取已保存的数据。

**注意：**Store 中的文本应被视为数据，而不是具有最高优先级的系统指令。不能让未经验证的存储内容覆盖权限控制或其他关键应用规则。

---

## 十一、search()：搜索 Store 中的数据

除了按 key 精确读取，还可以使用 `search()` 在命名空间中查询数据。

### 实例：搜索用户记忆

## 实例

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = ("memories", "student-001")

store.put(namespace, "memory-001", {"type": "language", "text": "正在学习 Python"})
store.put(namespace, "memory-002", {"type": "goal", "text": "希望掌握 Web 开发"})
store.put(namespace, "memory-003", {"type": "language", "text": "已经学习 HTML"})

items = store.search(namespace, filter={"type": "language"}, limit=10)
for item in items:
    print(item.key, item.value["text"])

运行结果可能为：

```
memory-001 正在学习 Python

memory-003 已经学习 HTML
```

这里使用 `filter` 根据字段值进行检索。

但要特别注意，普通的 `search()` 不等于语义搜索。

如果希望根据"我最近在学什么编程语言"之类的自然语言查询相似记忆，需要额外配置 Embedding 和相应索引。

---

## 十二、Store 的语义搜索

LangGraph Store 支持基于向量的语义检索，但需要配置 Embedding 模型。

普通 InMemoryStore 默认不会自动为所有内容建立语义向量索引。

例如，可以使用：

```
InMemoryStore(index={

    "embed": embedding_model,

    "dims": embedding_dimensions

})
```

之后在搜索时提供：

```
store.search(namespace, query="Python 学习目标", limit=5)
```

Store 可以按照向量相似度返回相关内容。

需要注意：

- Embedding 模型需要与索引维度一致。
- 不同模型的向量不能随意混用。
- 长期知识库应考虑数据更新和重新索引。
- 大型知识库通常需要持久化向量索引或专门的检索系统。

如果只是保存用户语言、学习等级、主题设置等简单结构化信息，使用普通 `get()` 和 `put()` 通常更合适。

---

## 十三、PostgresStore：持久化长期记忆

InMemoryStore 只能保存在当前进程中。

生产环境需要长期保存数据时，可以使用 PostgreSQL 后端的 `PostgresStore`。

安装依赖：

```
pip install -U langgraph-checkpoint-postgres "psycopg[binary]"
```

### 完整实例：PostgreSQL 存储

在 `.env` 中配置：

```
POSTGRES_URI=postgresql://user:password@localhost:5432/langchain_db
```

## 实例

import os
from dotenv import load\_dotenv
from langgraph.store.postgres import PostgresStore

load\_dotenv()
DB\_URI = os.environ["POSTGRES\_URI"]

with PostgresStore.from\_conn\_string(DB\_URI) as store:
    store.setup()

    namespace = ("users", "student-001")
    store.put(namespace, "preferences", {"language": "zh-CN", "level": "入门"})

    item = store.get(namespace, "preferences")
    print(item.value if item else "没有数据")

`store.setup()` 用于初始化所需的数据库结构，首次使用时需要执行。

程序重新启动后，只要连接到同一个数据库，就可以继续读取已经保存的数据。

使用时要注意：

- 数据库需要提前创建，并具有相应的访问权限。
- `with` 代码块结束后连接会关闭。
- 生产环境应在应用生命周期内管理连接或连接池。
- 数据库密码不应直接写入公开源代码。
- 使用语义搜索时，还需满足向量索引及相关扩展的要求。

---

## 十四、在 Agent 中使用 PostgresStore

使用 PostgresStore 的方法与 InMemoryStore 基本相同。

只需要在创建 Agent 时传入数据库 Store 对象：

## 实例

import os
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langgraph.store.postgres import PostgresStore

load\_dotenv()

with PostgresStore.from\_conn\_string(os.environ["POSTGRES\_URI"]) as store:
    store.setup()

    agent = create\_agent(
        model="deepseek:deepseek-flash",
        tools=[get\_my\_preferences],
        store=store,
        context\_schema=UserContext,
        system\_prompt="你是学习助手，查询偏好时使用 get\_my\_preferences。"
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请读取我的学习偏好。"}]},
        context=UserContext(user\_id="student-001")
    )
    print(result["messages"][-1].content)

这里复用了前面定义的 `get_my_preferences` 工具与 `UserContext`。

需要注意，PostgresStore 和 PostgresSaver 并不是同一个对象：

```
PostgresStore：保存长期数据



PostgresSaver：保存 Agent 检查点
```

如果同时需要长期记忆和持久化对话，应分别配置 Store 与 Checkpointer。

---

## 十五、异步 Store

如果使用 FastAPI 等异步 Web 框架，可以选择支持异步操作的 Store。

例如 PostgreSQL 提供：

```
AsyncPostgresStore
```

常见异步方法：

| 同步方法 | 异步方法 |
| --- | --- |
| `put()` | `aput()` |
| `get()` | `aget()` |
| `search()` | `asearch()` |
| `delete()` | `adelete()` |

### 实例：异步 PostgreSQL Store

## 实例

import os
import asyncio
from dotenv import load\_dotenv
from langgraph.store.postgres import AsyncPostgresStore

load\_dotenv()

async def main():
    async with AsyncPostgresStore.from\_conn\_string(os.environ["POSTGRES\_URI"]) as store:
        await store.setup()

        namespace = ("users", "student-001")
        await store.aput(namespace, "goal", {"text": "掌握 Python Web 开发"})
        item = await store.aget(namespace, "goal")
        print(item.value if item else None)

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(main())

异步方式适合在异步服务中减少数据库 I/O 对事件循环的阻塞。

如果工具内部需要执行异步数据库操作，也应使用相应的异步工具定义和 Store 方法。

---

## 十六、Store 与普通数据库有什么区别？

Store 本质上也是一种数据访问抽象，但它与应用的完整业务数据库并不完全相同。

| 数据场景 | 建议存储 | 原因 |
| --- | --- | --- |
| 用户学习偏好 | Store | 适合 Agent 跨会话读取 |
| 用户长期记忆 | Store | 支持命名空间及可选语义检索 |
| 课程目录缓存 | Store 或业务缓存 | 方便 Agent 查询 |
| 会员余额、支付订单 | 业务数据库 | 需要事务、审计和严格权限 |
| 用户登录与权限 | 认证系统及业务数据库 | 不能交给模型或不受控的记忆数据判断 |
| 对话消息和执行状态 | Checkpointer | 适合线程状态持久化 |

对于学习网站，可以将 Store 视为 Agent 的长期记忆接口，而不是整个网站的唯一数据库。

用户账号、会员订阅、课程购买记录等核心业务数据，应该由可信业务系统负责管理。

---

## 十七、生产环境注意事项

### 1、不要信任模型生成的用户 ID

访问个人 Store 时，应使用由后端认证系统提供的 user\_id，而不是允许模型自行指定任意用户。

### 2、命名空间不等于权限控制

即使数据按用户命名空间存放，也必须在后端验证调用方是否有权访问对应数据。

### 3、不要把 Store 当作自动记忆系统

Store 不会自动记住所有聊天内容。需要明确设计哪些内容值得保存、什么时候保存、如何更新和删除。

### 4、限制记忆写入范围

不要把整段聊天未经筛选就保存到长期记忆。

建议只保存必要、明确且有持续价值的数据，并允许用户查看和修改。

### 5、处理数据过期与删除

长期存储不代表永久保留。应根据业务需求设置数据更新、过期和删除策略。

### 6、避免将 Store 内容当作系统指令

Store 中的数据可能来自用户或外部文档，不应允许它覆盖安全规则、工具权限或系统提示。

### 7、数据库必须持久化与备份

InMemoryStore 重启后数据丢失。需要长期保存时，应选择持久化后端，并考虑备份和恢复机制。

---

## 十八、常见问题

### 1、Store 和 Checkpointer 有什么区别？

Checkpointer 保存线程级 Agent State；Store 保存可以跨线程访问的长期数据。

### 2、Store 是否自动保存用户对话？

不会。需要程序或工具显式调用 put() 等方法写入数据。

### 3、Store 的 namespace 必须包含 user\_id 吗？

不是。公共数据可以使用公共命名空间，用户私人数据则通常应按可信用户 ID 隔离。

### 4、Store 可以跨多个 thread\_id 共享吗？

可以。不同线程只要按授权规则访问相同 namespace 和 key，就能读取相同的 Store 数据。

### 5、Store 的 value 可以是字符串吗？

LangGraph Store 主要以 JSON 文档组织数据，常见写法是保存字典，例如 `{"text": "学习 Python"}`。

### 6、ToolRuntime.store 需要手动注入吗？

不需要。创建 Agent 时传入 store 后，工具可以通过 ToolRuntime 自动访问它。

### 7、为什么 runtime.store 是 None？

可能是创建 Agent 时没有传入 Store，或当前执行环境没有配置对应存储。

### 8、search() 是否自动支持自然语言语义搜索？

不是。语义搜索需要配置 Embedding 和相应索引。

### 9、InMemoryStore 是否支持进程重启后恢复？

不支持。它保存在进程内存中。

### 10、PostgresStore 为什么需要 setup()？

首次使用时需要初始化相关数据库结构。

### 11、Store 可以保存 VIP 会员信息吗？

可以保存业务允许的缓存或展示数据，但不能把未经验证的 Store 内容作为支付、会员权限或账号认证的唯一依据。

### 12、如何删除用户记忆？

可以调用 `store.delete(namespace, key)` 删除指定数据。完整删除用户数据还需要考虑其他键、Checkpointer、业务数据库和备份。

---

## 总结

Store 是 LangChain Agent 实现长期记忆和跨会话数据共享的重要机制。

本章节介绍了：

- `InMemoryStore`：内存 Store。
- `put()`：保存或更新数据。
- `get()`：读取指定数据。
- `search()`：搜索命名空间中的数据。
- `delete()`：删除指定数据。
- `ToolRuntime.store`：在 Agent 工具中访问 Store。
- `runtime.context`：获取可信的用户上下文。
- 跨不同 thread\_id 共享长期偏好。
- `PostgresStore`：持久化长期数据。
- `AsyncPostgresStore`：异步访问数据库。
- Store 与 Checkpointer、普通业务数据库的职责区别。

最常见的写法：

```
store = InMemoryStore()

store.put(("users", "user_001"), "preferences", {"language": "zh-CN"})



item = store.get(("users", "user_001"), "preferences")

print(item.value)
```

在 Agent 中使用：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[my_tool],

    store=store

)
```

在工具中访问：

```
@tool

def my_tool(runtime: ToolRuntime) -> str:

    item = runtime.store.get(("users",), "example")

    return str(item.value) if item else "没有数据"
```

> **核心原则：**Checkpointer 管理单个线程的执行状态，Store 管理跨会话的长期数据。长期记忆需要明确的读写机制，而用户身份、权限和重要业务数据必须由可信后端控制。
