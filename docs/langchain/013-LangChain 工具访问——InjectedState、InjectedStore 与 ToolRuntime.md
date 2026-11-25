# LangChain 工具访问——InjectedState、InjectedStore 与 ToolRuntime

> 来源: https://www.runoob.com/langchain/langchain-state-store.html

---

# LangChain 工具访问——InjectedState、InjectedStore 与 ToolRuntime

前面我们学习了如何使用 `@tool` 定义工具，以及如何将工具注册到 Agent 中。

普通工具主要通过函数参数接收模型提供的数据。例如，查询天气工具需要模型传入城市名称。

但在实际开发中，工具往往还需要访问一些不能由模型直接提供的信息：

- 当前对话的消息历史。
- 用户在对话中已经确认的设置。
- 登录用户的身份信息。
- 跨会话保存的学习进度和个人偏好。
- 当前工具调用的 ID 和执行配置。

LangChain 提供运行时依赖注入机制，让工具可以访问这些信息，而不需要把所有内部数据暴露给语言模型。

早期教程常使用 `InjectedState` 和 `InjectedStore`。在 LangChain 1.x 中，官方更推荐使用 ToolRuntime 统一访问运行时信息。

本章节将介绍三种机制及其关系，并通过完整的 Agent 示例说明如何正确使用。

---

## 一、State、Context 与 Store 的区别

在编写工具之前，需要先理解 Agent 运行时的三种数据。

| 数据类型 | 主要作用 | 典型用途 |
| --- | --- | --- |
| `State` | 管理会话中的可变状态 | 对话历史、当前任务进度、临时选择 |
| `Context` | 传递本次调用的运行上下文 | 用户 ID、请求来源、租户信息 |
| `Store` | 在不同会话或调用之间共享数据 | 用户偏好、学习进度、长期记忆 |

可以通过下面的示意图理解：

```
用户发起请求

      |

      v

  LangChain Agent

      |

      +-- State

      |     当前消息、任务状态

      |

      +-- Context

      |     用户身份、请求配置

      |

      +-- Store

            用户偏好、学习记录

      |

      v

  ToolRuntime

      |

      v

  Python 工具函数
```

其中，State、Context 和 Store 的生命周期不同。

**State：**默认用于当前 Agent 执行过程。如果配置了 Checkpointer，可以按会话线程保存，并在之后的调用中恢复。

**Context：**通常由应用程序在调用 Agent 时传入，提供本次运行需要的上下文信息。

**Store：**可以保存跨会话的数据。实际能保存多久，取决于 Store 的具体实现。

---

## 二、ToolRuntime——官方推荐的工具运行时接口

LangChain 1.x 推荐通过 `ToolRuntime` 访问工具执行上下文。

只需要在工具函数中添加：

```
runtime: ToolRuntime
```

Agent 执行工具时，LangChain 会自动注入运行时对象。

不需要添加 `Annotated`，也不需要让模型传入这个参数。

### ToolRuntime 常用属性

| 属性 | 说明 |
| --- | --- |
| `runtime.state` | 当前 Agent 状态 |
| `runtime.context` | 本次运行的上下文 |
| `runtime.store` | 共享的数据存储对象 |
| `runtime.tool_call_id` | 当前工具调用 ID |
| `runtime.config` | RunnableConfig 运行配置 |
| `runtime.stream_writer` | 用于发送自定义流式更新 |

### 实例：读取当前 Agent 消息数量

## 实例

from langchain.tools import tool, ToolRuntime

@tool
def conversation\_stats(
    runtime: ToolRuntime
) -> str:
    """统计当前 Agent 的消息数量。"""

    messages = runtime.state.get(
        "messages", []
    )

    human\_count = sum(
        message.type == "human"
        for message in messages
    )

    ai\_count = sum(
        message.type == "ai"
        for message in messages
    )

    tool\_count = sum(
        message.type == "tool"
        for message in messages
    )

    return (
        f"当前共 {len(messages)} 条消息；"
        f"用户消息 {human\_count} 条；"
        f"AI 消息 {ai\_count} 条；"
        f"工具结果 {tool\_count} 条"
    )

在这个工具中，`runtime.state` 提供当前 Agent 的状态，其中通常包含 `messages`。

需要注意，`runtime` 不会出现在提供给模型的工具参数列表中。

模型只需要知道工具可以执行对话统计，不需要了解 Agent 内部如何保存消息。

### 在 Agent 中调用

## 实例

from dotenv import load\_dotenv

from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[conversation\_stats],
    system\_prompt=(
        "你是一名对话统计助手。"
        "用户要求统计消息数量时，"
        "使用 conversation\_stats 工具。"
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "请统计当前会话的消息数量。"
        }
    ]
})

print(result["messages"][-1].content)

如果模型调用了 `conversation_stats`，工具就可以访问执行时的消息状态，并返回统计结果。

实际消息数量取决于工具被执行时 Agent 已经产生了多少条消息，因此不能假设总数始终固定。

**注意：**不能简单地直接执行：

```
# 不适合作为运行时注入的测试方式

conversation_stats.invoke({})
```

因为普通的工具调用没有自动提供完整的 Agent 运行时上下文。

如果需要测试这种工具，应通过 `create_agent()` 或正确配置的 LangGraph `ToolNode` 执行。

---

## 三、InjectedState——旧式状态注入方式

`InjectedState` 用于标记需要从 Agent 状态中自动注入的参数。

它仍可用于支持这种注入机制的 LangGraph ToolNode，但在新的 LangChain 教程中，推荐优先使用 `ToolRuntime`。

### InjectedState 的基本语法

```
from typing import Annotated

from langgraph.prebuilt import InjectedState



state: Annotated[dict, InjectedState()]
```

当工具通过支持状态注入的执行节点运行时，LangGraph 会从当前图状态中取得相应数据。

### 实例：读取完整 State

## 实例

from typing import Annotated, Any

from langchain.tools import tool
from langgraph.prebuilt import InjectedState

@tool
def get\_message\_count(
    state: Annotated[
        dict[str, Any],
        InjectedState()
    ]
) -> str:
    """读取当前对话消息数量。"""

    messages = state.get(
        "messages", []
    )

    return (
        f"当前共有 {len(messages)} 条消息"
    )

这里的 `state` 不是由模型填写的普通工具参数。

执行框架会在工具真正运行前，将当前 State 注入对应参数。

### 只注入某个 State 字段

`InjectedState` 也支持只读取状态中的一个字段。

例如：

```
InjectedState("messages")
```

表示只注入 `messages` 字段。

## 实例

from typing import Annotated

from langchain.tools import tool
from langgraph.prebuilt import InjectedState

@tool
def count\_messages(
    messages: Annotated[
        list,
        InjectedState("messages")
    ]
) -> str:
    """统计当前对话消息数量。"""

    return f"共 {len(messages)} 条消息"

相比注入整个 State，这种方式只读取工具需要的字段。

但在新项目中，通常可以直接使用：

```
runtime.state["messages"]
```

避免使用多个不同的注入参数。

---

## 四、读取 State 不等于修改 State

原文中的 `remember_preference()` 只是读取状态并返回文字，并没有真正把用户偏好保存到 Agent 状态中。

如果希望工具修改状态，LangChain 官方推荐返回 `Command`。

`Command` 可以声明需要写入哪些状态字段，并将工具调用结果关联回消息历史。

### 实例：保存当前会话的偏好

## 实例

from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

# 自定义 Agent 状态
class CustomState(AgentState):
    theme: str

@tool
def set\_theme(
    theme: str,
    runtime: ToolRuntime
) -> Command:
    """保存当前会话的界面主题。"""

    return Command(
        update={
            "theme": theme,
            "messages": [
                ToolMessage(
                    content=f"主题已设置为：{theme}",
                    tool\_call\_id=runtime.tool\_call\_id
                )
            ]
        }
    )

@tool
def get\_theme(
    runtime: ToolRuntime
) -> str:
    """查询当前会话的主题设置。"""

    return runtime.state.get(
        "theme", "未设置"
    )

这里定义了一个额外状态字段：

```
theme: str
```

当 `set_theme` 返回 `Command` 时，可以通过 `update` 更新该字段。

其中的 `ToolMessage` 必须与当前工具调用 ID 对应。

如果只有读取操作，就不需要返回 Command。

### 创建支持自定义 State 的 Agent

```
agent = create_agent(

    model=model,

    tools=[set_theme, get_theme],

    state_schema=CustomState,

    system_prompt="你是一名用户设置助手。"

)
```

这样就可以让 Agent 管理自定义会话字段。

**注意：**State 是工作流状态，不建议直接修改 `runtime.state` 内部字典并假设修改一定会持久化。应通过受支持的状态更新方式提交变化。

如果多个工具可能同时更新相同字段，还需要考虑并行写入冲突及 reducer 的配置。

---

## 五、Checkpointer——让 State 跨调用保存

State 不是只能存在于单次 `invoke()` 调用中。

LangGraph 提供 Checkpointer，用于保存会话状态。

例如，同一用户连续发送多个问题时，可以通过相同的 `thread_id` 恢复此前的消息和状态。

### 实例：保存对话历史

## 实例

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver()
)

# 同一段会话使用相同 thread\_id
config = {
    "configurable": {
        "thread\_id": "demo\_session\_001"
    }
}

# 第一轮
agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我正在学习 Python。"
            }
        ]
    },
    config=config
)

# 第二轮
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我刚才说在学习什么？"
            }
        ]
    },
    config=config
)

print(result["messages"][-1].content)

这里使用 `InMemorySaver` 保存会话检查点。

两次调用使用相同的 `thread_id`，因此第二次调用可以访问此前保存的会话历史。

但 `InMemorySaver` 只将数据保存在当前 Python 进程的内存中。

当程序退出或重启后，这些记录通常会丢失。

如果需要持久保存，可以使用数据库支持的 Checkpointer。

---

## 六、InjectedStore——旧式存储注入方式

`InjectedStore` 用于向工具注入 LangGraph 的 Store 对象。

工具可以通过 Store 保存和读取用户偏好、学习进度等长期信息。

其基本语法为：

```
from typing import Annotated

from langgraph.prebuilt import InjectedStore

from langgraph.store.base import BaseStore



store: Annotated[

    BaseStore,

    InjectedStore()

]
```

与 InjectedState 类似，InjectedStore 参数也不会作为普通工具参数暴露给模型。

### 实例：读取 Store 中的数据

## 实例

from typing import Annotated

from langchain.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

@tool
def get\_preference(
    key: str,
    store: Annotated[
        BaseStore,
        InjectedStore()
    ]
) -> str:
    """读取演示用户保存的偏好。"""

    item = store.get(
        ("demo", "preferences"),
        key
    )

    if item is None:
        return "未找到该偏好"

    return str(item.value)

这个工具需要通过配置了 Store 的 Agent 或 ToolNode 执行。

直接调用：

```
get_preference.invoke({"key": "theme"})
```

并不能凭空获得 Store 实例。

因此，这种写法只适合展示旧版注入参数的结构。下面使用官方推荐的 ToolRuntime 实现完整示例。

---

## 七、ToolRuntime.store——访问共享存储

在 LangChain 1.x 中，推荐通过：

```
runtime.store
```

获取当前 Agent 使用的 Store 对象。

Store 通过**命名空间（Namespace）和键（Key）**组织数据。

例如：

```
store.put(

    ("users", "user_001"),

    "profile",

    {"name": "小明"}

)
```

其中：

- `("users", "user_001")` 是命名空间。
- `"profile"` 是键。
- `{"name": "小明"}` 是保存的数据。

读取时使用：

```
item = store.get(

    ("users", "user_001"),

    "profile"

)



print(item.value)
```

### InMemoryStore 的生命周期

`InMemoryStore` 是一个适合教学和测试的内存 Store。

它允许多个 Agent 调用共享同一个 Store 实例中的数据，但不代表数据已经写入磁盘。

| 操作 | InMemoryStore |
| --- | --- |
| 同一个进程内多次调用 | 可以共享数据 |
| 不同会话使用同一个 Store 实例 | 可以共享数据 |
| 程序重新启动 | 内存中的数据通常丢失 |
| 跨多台服务器共享 | 不能自动实现 |

生产环境通常应选择持久化 Store，例如 PostgreSQL 等数据库实现。

---

## 八、完整实例：保存用户学习进度

下面创建一个可以保存和读取学习进度的 Agent。

这个示例使用：

- `ToolRuntime` 获取 Store。
- `runtime.context` 获取当前用户 ID。
- `InMemoryStore` 保存模拟学习记录。
- `create_agent()` 管理工具调用。

需要特别注意，用户 ID 不能直接由模型随意指定，否则可能导致用户之间的数据混淆。

本例由应用程序提供当前用户的身份上下文。

### 完整代码

## 实例

from dataclasses import dataclass

from dotenv import load\_dotenv

from langchain.tools import tool, ToolRuntime
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

from langgraph.store.memory import InMemoryStore

load\_dotenv()

# -------------------------
# 定义运行时用户上下文
# -------------------------

@dataclass
class UserContext:
    user\_id: str

# -------------------------
# 创建 Store
# -------------------------

store = InMemoryStore()

# 初始化模拟用户数据
store.put(
    ("users", "user\_001"),
    "profile",
    {
        "name": "小明",
        "level": "入门",
        "completed\_courses": [
            "HTML 基础教程"
        ]
    }
)

# -------------------------
# 工具一：读取学习档案
# -------------------------

@tool
def get\_user\_profile(
    runtime: ToolRuntime[UserContext]
) -> str:
    """读取当前用户的模拟学习档案。"""

    user\_id = runtime.context.user\_id

    item = runtime.store.get(
        ("users", user\_id),
        "profile"
    )

    if item is None:
        return "没有找到学习档案"

    profile = item.value

    return (
        f"姓名：{profile['name']}；"
        f"水平：{profile['level']}；"
        f"已完成课程："
        + "、".join(
            profile["completed\_courses"]
        )
    )

# -------------------------
# 工具二：保存学习进度
# -------------------------

@tool
def save\_course\_progress(
    course\_name: str,
    runtime: ToolRuntime[UserContext]
) -> str:
    """将课程加入当前用户的模拟已完成列表。"""

    user\_id = runtime.context.user\_id

    namespace = ("users", user\_id)

    item = runtime.store.get(
        namespace,
        "profile"
    )

    if item is None:
        return "用户档案不存在，无法保存"

    profile = dict(item.value)

    completed = list(
        profile.get("completed\_courses", [])
    )

    if course\_name in completed:
        return "该课程已经标记为已完成"

    completed.append(course\_name)

    profile["completed\_courses"] = completed

    runtime.store.put(
        namespace,
        "profile",
        profile
    )

    return (
        f"已保存模拟学习进度：{course\_name}。"
        f"当前完成 {len(completed)} 门课程。"
    )

# -------------------------
# 创建 Agent
# -------------------------

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[
        get\_user\_profile,
        save\_course\_progress
    ],
    store=store,
    context\_schema=UserContext,
    system\_prompt=(
        "你是一名编程学习助手。"
        "用户查询学习进度时使用工具。"
        "用户明确要求标记课程完成时，"
        "才使用保存工具。"
        "所有数据都是教学模拟数据。"
        "没有工具确认成功时，不要声称已保存。"
    )
)

# -------------------------
# 调用 Agent
# -------------------------

context = UserContext(
    user\_id="user\_001"
)

def ask(question: str):
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        context=context
    )

    print("\n用户：", question)
    print(
        "AI：",
        result["messages"][-1].content
    )

if \_\_name\_\_ == "\_\_main\_\_":

    # 查询已完成课程
    ask("查询我的模拟学习档案。")

    # 保存进度
    ask(
        "请把 Python3 基础教程"
        "标记为已完成。"
    )

    # 再次查询
    ask("重新查询我的已完成课程。")

### 运行结果说明

如果模型正确调用相关工具，初始学习档案包括：

```
姓名：小明

水平：入门

已完成课程：HTML 基础教程
```

标记 Python3 基础教程完成后，再次读取 Store，可以得到：

```
姓名：小明

水平：入门

已完成课程：

HTML 基础教程、Python3 基础教程
```

需要注意，以上是预期的业务数据变化，不是固定的 AI 输出文本。

由于 Agent 的工具选择由模型决定，实际是否执行保存操作，应以 Store 中的数据为准。

**安全提示：**示例中的 `UserContext(user_id="user_001")` 是应用程序主动提供的模拟身份。生产环境应从服务器验证过的登录会话或认证信息中获取用户 ID，不能直接信任用户提交的任意 ID。

---

## 九、在 Agent 中结合 Store 查询课程数据

除了用户偏好，Store 还可以保存课程目录等共享信息。

下面使用 Store 保存一组模拟课程资料，再通过工具查询。

### 实例

## 实例

from langchain.tools import tool, ToolRuntime
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

from langgraph.store.memory import InMemoryStore

# 模拟课程目录
store = InMemoryStore()

store.put(
    ("runoob\_demo", "courses"),
    "catalog",
    {
        "Python3 基础教程": {
            "price": "免费",
            "duration": "20小时"
        },
        "Python 数据分析": {
            "price": "模拟会员课程",
            "duration": "30小时"
        },
        "HTML 基础教程": {
            "price": "免费",
            "duration": "15小时"
        }
    }
)

@tool
def query\_course\_info(
    course\_name: str,
    runtime: ToolRuntime
) -> str:
    """查询指定课程的模拟价格和学习时长。"""

    item = runtime.store.get(
        ("runoob\_demo", "courses"),
        "catalog"
    )

    if item is None:
        return "模拟课程目录不存在"

    catalog = item.value

    info = catalog.get(course\_name)

    if info is None:
        return f"未找到课程：{course\_name}"

    return (
        f"课程：{course\_name}；"
        f"价格：{info['price']}；"
        f"时长：{info['duration']}"
    )

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[query\_course\_info],
    store=store,
    system\_prompt=(
        "你是一名课程顾问。"
        "查询课程信息时使用工具。"
        "所有课程资料均为教学模拟数据。"
        "不得将模拟价格当作真实售价。"
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "请查询 Python3 基础教程"
                "和 Python 数据分析的模拟课程信息。"
            )
        }
    ]
})

print(result["messages"][-1].content)

这个例子中，课程目录由 Store 保存，工具通过 `runtime.store` 读取数据。

如果模型分别调用两次 `query_course_info`，就可以获取两门课程的模拟信息。

**注意：**这些价格、学习时长和会员属性都是演示数据，不代表菜鸟教程网站当前实际提供的课程或收费政策。

---

## 十、ToolRuntime.context——获取可信用户上下文

实际应用中，工具经常需要知道当前请求属于哪个用户。

例如，不同用户的学习记录应该保存在不同命名空间中：

```
("users", "user_001")

("users", "user_002")
```

但是，用户 ID 不应该交给模型自由填写。

推荐使用运行时 Context，由应用程序提供已经验证的身份信息。

### 实例

## 实例

from dataclasses import dataclass

from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
    user\_id: str

@tool
def who\_am\_i(
    runtime: ToolRuntime[UserContext]
) -> str:
    """查询当前请求的用户标识。"""

    return (
        f"当前用户："
        f"{runtime.context.user\_id}"
    )

创建 Agent 时指定：

```
agent = create_agent(

    model=model,

    tools=[who_am_i],

    context_schema=UserContext

)
```

调用时提供 Context：

```
result = agent.invoke(

    {

        "messages": [

            {

                "role": "user",

                "content": "查询我的用户标识"

            }

        ]

    },

    context=UserContext(

        user_id="user_001"

    )

)
```

这样，模型不需要获得用户 ID 作为工具参数，工具可以通过可信运行时读取。

**注意：**Context 本身不负责登录认证。应用程序必须先确认用户身份，再创建对应的 Context。

---

## 十一、InjectedToolArg——通用注入参数标记

`InjectedToolArg` 用于标记不应该由语言模型生成的工具参数。

它主要用于声明参数应由程序或执行框架提供，而不是普通 Tool Calling 参数。

例如：

## 实例

from typing import Annotated

from langchain.tools import (
    tool,
    InjectedToolArg
)

@tool
def my\_tool(
    normal\_param: str,
    injected\_param: Annotated[
        str,
        InjectedToolArg
    ]
) -> str:
    """演示普通参数和注入参数的区别。"""

    return (
        f"普通参数：{normal\_param}；"
        f"注入参数：{injected\_param}"
    )

print(
    my\_tool.tool\_call\_schema.model\_json\_schema()
)

这里的 `normal_param` 是提供给模型的普通参数。

`injected_param` 通过 `InjectedToolArg` 标记为注入参数，因此不会作为普通模型工具参数暴露。

但要注意：**InjectedToolArg 只是注入标记，不会自动创建注入值。**

如果没有对应的运行时注入逻辑，工具执行时仍然可能因为缺少参数而失败。

相比自己实现通用参数注入，实际项目通常更推荐使用 `ToolRuntime`。

---

## 十二、State、Checkpointer 与 Store 对比

这三个概念经常被混淆，但用途不同。

| 对比维度 | State | Checkpointer | Store |
| --- | --- | --- | --- |
| 主要作用 | 保存工作流状态 | 保存和恢复 State | 存储跨会话数据 |
| 典型数据 | 消息、临时变量 | 会话检查点 | 用户偏好、学习进度 |
| 数据范围 | 当前工作流或线程 | 与线程等标识关联 | 由命名空间和键组织 |
| 跨调用 | 依赖状态管理 | 支持恢复会话 | 支持共享数据 |
| 持久化 | 依赖执行配置 | 取决于实现 | 取决于实现 |

例如，一个 AI 学习助手可能这样设计：

```
State：

  当前正在讨论 Python 装饰器



Checkpointer：

  保存这段对话的历史和状态



Context：

  当前登录用户是 user_001



Store：

  user_001 已完成 Python 基础课程
```

因此：

- 需要读取当前消息，使用 `runtime.state`。
- 需要知道当前用户，使用 `runtime.context`。
- 需要跨会话保存数据，使用 `runtime.store`。
- 需要会话在后续调用中恢复，配置 `checkpointer`。
- 需要修改自定义 State，返回 `Command`。

---

## 十三、生产环境注意事项

当工具需要访问真实用户数据或长期存储时，还需要考虑安全性和一致性。

### 1、不能信任模型提供的用户 ID

用户身份应来自服务器认证系统，而不是语言模型生成的工具参数。

推荐由应用服务器创建运行时 Context，再通过 `runtime.context` 获取用户身份。

### 2、InMemoryStore 不适合长期保存重要数据

内存 Store 适合学习和测试。

生产环境通常需要使用 PostgreSQL 等持久化实现，保证程序重启后数据仍然可用。

### 3、避免并发写入覆盖

多个工具或多个 Agent 同时修改同一条学习进度时，简单的读取、修改、写回操作可能产生竞态条件。

真实系统应结合数据库事务、版本控制或其他并发控制方案。

### 4、权限检查不能只依赖提示词

即使系统提示词要求只能查询自己的资料，也不能代替后端权限校验。

例如，访问学习记录前，需要确认当前用户有权读取对应命名空间下的数据。

### 5、限制暴露给模型的数据范围

State 和 Store 可能包含用户个人信息、访问令牌或其他敏感数据。

工具应尽量只返回完成当前任务必需的字段，避免把整个用户档案或内部配置直接交给模型。

---

## 十四、常见问题

### 1、InjectedState 和 InjectedStore 还能使用吗？

可以在支持旧式注入机制的 LangGraph ToolNode 场景中使用。

但 LangChain 当前官方更推荐使用 `ToolRuntime`，统一访问状态、上下文和存储。

### 2、为什么直接调用 tool.invoke() 无法自动获得 State？

因为自动注入需要相应的 Agent 或 ToolNode 执行上下文。

直接调用工具不会自动构造完整的 Agent 状态。

### 3、为什么工具读取了 State，却没有保存用户偏好？

因为读取 State 并不等于更新 State。

如果要修改自定义状态字段，应通过受支持的状态更新机制，例如返回 `Command`。

### 4、State 会在 Agent 运行结束后消失吗？

不一定。

如果使用 Checkpointer，并通过相同的线程标识继续调用 Agent，可以恢复此前的状态。

### 5、InMemoryStore 是永久存储吗？

不是。

数据保存在当前 Python 进程内存中，进程结束后通常会丢失。

### 6、Store 和 Checkpointer 可以互相代替吗？

不应该简单互相替代。

Checkpointer 主要用于保存和恢复会话状态，Store 主要用于按照命名空间和键读写共享数据。

### 7、ToolRuntime 可以访问哪些信息？

常见属性包括：

```
runtime.state

runtime.context

runtime.store

runtime.tool_call_id

runtime.config

runtime.stream_writer
```

部分更新版本还提供执行信息等扩展属性，具体以当前 API Reference 为准。

### 8、如何在多个用户之间隔离学习进度？

推荐通过可信 Context 获取用户 ID，再使用用户专属的 Store 命名空间。

例如：

```
namespace = (

    "users",

    runtime.context.user_id

)
```

同时必须在后端验证用户身份，防止访问其他用户的数据。

### 9、InjectedToolArg 会自动创建注入值吗？

不会。

它主要标记工具参数不应暴露给模型。实际注入需要执行框架提供相应逻辑。

### 10、State 和 Store 都需要持久化数据库吗？

不一定。

开发测试可以使用内存实现，但需要在服务重启后保留数据的生产项目，通常应使用数据库支持的 Checkpointer 和 Store。

---

## 总结

LangChain 的工具注入机制，可以让 Agent 工具访问模型参数之外的运行时数据。

本章节介绍了以下内容：

- `InjectedState`：旧式 Agent 状态注入标记。
- `InjectedStore`：旧式 Store 注入标记。
- `ToolRuntime`：官方推荐的统一工具运行时接口。
- `runtime.state`：读取当前 Agent 状态。
- `runtime.context`：读取本次调用的上下文。
- `runtime.store`：读取和保存跨会话数据。
- `Command`：修改 Agent 状态。
- `Checkpointer`：保存与恢复会话状态。
- `InjectedToolArg`：标记不向模型暴露的注入参数。

对于新的 LangChain 项目，推荐优先采用下面的工具定义方式：

```
from langchain.tools import tool, ToolRuntime



@tool

def my_tool(

    keyword: str,

    runtime: ToolRuntime

) -> str:

    """演示访问工具运行时。"""



    messages = runtime.state.get(

        "messages", []

    )



    return (

        f"关键词：{keyword}；"

        f"消息数：{len(messages)}"

    )
```

这种方式比分别声明多个 InjectedState、InjectedStore 参数更统一，也更容易扩展。

**核心原则：**让模型只负责提供完成任务所需的业务参数，而将状态、用户身份、持久化存储和权限控制交给可信的程序运行时。
