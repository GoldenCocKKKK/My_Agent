# LangChain create\_agent() 函数

> 来源: https://www.runoob.com/langchain/langchain-create_agent.html

---

# LangChain create\_agent() 函数

`create_agent()` 是 LangChain 1.x 中创建 AI Agent 的核心函数。

它可以将语言模型、工具、系统提示词和中间件组合起来，创建一个能够自动调用工具并循环执行任务的 Agent。

与普通模型调用不同，Agent 不仅可以回答问题，还能根据任务需要调用 Python 函数、读取外部数据，并根据工具返回的结果决定下一步操作。

在 LangChain 1.x 中，`create_agent()` 基于 LangGraph 构建并编译 Agent 执行图，支持状态管理、流式输出、结构化响应、运行时上下文和会话持久化等功能。

本章节将完整介绍 `create_agent()` 的参数和常用方法，并使用 DeepSeek Flash 模型进行演示。

---

## 一、create\_agent() 的工作原理

使用 `create_agent()` 创建的 Agent，通常按照以下流程运行：

```
用户输入

    |

    v

  Model 节点

    |

    | 模型是否发起工具调用？

    |

    +-- 否 --> 返回最终结果

    |

    +-- 是

          |

          v

       Tools 节点

          |

          | 执行工具

          | 将结果加入消息历史

          |

          v

       Model 节点

          |

          | 继续判断是否调用工具

          v

        ...
```

其中：

- **Model 节点：**负责调用语言模型，生成回复或工具调用请求。
- **Tools 节点：**负责执行模型请求的客户端工具，并返回工具结果。
- **Agent State：**保存当前消息及其他状态字段。
- **Middleware：**可以在执行过程中的特定阶段修改或控制 Agent 行为。

Agent 一般会持续执行，直到模型不再请求工具，或者触发其他停止条件。

因此，`create_agent()` 创建的并不是单纯的聊天模型，而是一个具有执行流程和状态管理能力的应用组件。

---

## 二、安装与环境准备

本教程使用 LangChain 1.x，并以 DeepSeek 为主要模型提供商。

安装依赖：

```
pip install -U langchain langchain-deepseek langgraph python-dotenv pydantic
```

在当前项目目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

Python 程序通过 `load_dotenv()` 加载密钥。

### 创建第一个 Agent

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是一名中文编程助手。"
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "什么是 Python？"
        }
    ]
})

print(result["messages"][-1].content)

这里没有给 Agent 配置任何工具，因此它主要通过模型生成回答。

虽然没有工具，`create_agent()` 依然会返回一个可执行的 LangGraph 对象。

**注意：**DeepSeek 当前推荐的模型名称包括 `deepseek-flash` 和 `deepseek-v4-pro`。新教程不再使用旧的 `deepseek-v4-flash` 作为默认模型名称。

---

## 三、create\_agent() 函数语法

根据 LangChain 官方 API Reference，函数语法如下：

```
from langchain.agents import create_agent



agent = create_agent(

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

    transformers=None,

)
```

参数说明：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `model` | 模型名称字符串或 BaseChatModel 实例 | 必填 |
| `tools` | 工具列表，支持可执行工具、函数和受支持的提供商工具定义 | None |
| `system_prompt` | 系统提示词，可以是字符串或 SystemMessage | None |
| `middleware` | Agent 中间件列表 | () |
| `response_format` | 结构化输出 Schema 或策略 | None |
| `state_schema` | 扩展 AgentState 的自定义状态结构 | None |
| `context_schema` | 运行时 Context 结构 | None |
| `checkpointer` | 保存和恢复会话状态的检查点组件 | None |
| `store` | 用于跨会话共享数据的 Store | None |
| `interrupt_before` | 指定在哪些图节点执行前暂停 | None |
| `interrupt_after` | 指定在哪些图节点执行后暂停 | None |
| `debug` | 是否输出图执行调试信息 | False |
| `name` | 编译后 Agent 图的名称 | None |
| `cache` | LangGraph 图执行缓存配置 | None |
| `transformers` | 为图注册额外的流事件转换器 | None |

对于初学者，最常用的是 `model`、`tools` 和 `system_prompt`。

构建复杂 Agent 时，再逐步使用 `middleware`、`response_format`、`checkpointer` 等高级参数。

---

## 四、model 参数：指定语言模型

`model` 是必填参数，主要支持两种形式：

- 模型名称字符串，例如 `deepseek:deepseek-flash`。
- 已经创建好的 `BaseChatModel` 实例。

### 方式一：直接传入模型名称

## 实例

from langchain.agents import create\_agent

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程助手。"
)

这种写法简单直接，LangChain 会根据模型名称选择对应提供商并初始化模型。

推荐明确使用：

```
provider:model
```

例如：

```
deepseek:deepseek-flash

deepseek:deepseek-v4-pro

openai:gpt-4o-mini
```

实际使用时，需要安装对应提供商集成包并配置 API Key。

### 方式二：传入已初始化的模型

## 实例

from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

# 初始化模型并配置参数
model = init\_chat\_model(
    "deepseek:deepseek-flash",
    max\_tokens=500,
    timeout=30,
    max\_retries=2
)

agent = create\_agent(
    model=model,
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程助手。"
)

这种方式适合需要配置模型超时、重试次数、输出上限或复用模型实例的场景。

**注意：**不推荐在传给 `create_agent()` 之前自行调用 `model.bind_tools()`。

因为 Agent 会根据其工具列表和运行时策略管理工具绑定。预先绑定工具可能与 Agent 的工具注册或动态工具管理机制发生冲突。

---

## 五、tools 参数：注册 Agent 工具

`tools` 用于指定 Agent 可以调用哪些工具。

官方支持的主要类型包括：

- `BaseTool`：LangChain 可执行工具对象。
- 普通 Python Callable：可以被转换为工具。
- 字典形式的提供商原生工具定义：由相应模型服务处理。

需要特别区分：**工具的参数 Schema 不等于工具的实际执行逻辑。**

### 方式一：使用 @tool 装饰器

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent

@tool
def search\_course(keyword: str) -> str:
    """根据关键词搜索模拟编程课程。"""

    courses = {
        "python": "Python 基础、Python 数据分析",
        "java": "Java 基础、Java 面向对象"
    }

    return courses.get(
        keyword.lower(),
        "未找到相关模拟课程"
    )

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course],
    system\_prompt=(
        "你是编程学习助手。"
        "课程数据为教学模拟数据。"
        "需要查询课程时使用工具。"
    )
)

这种方式适合大部分工具定义场景。

模型可以看到工具名称、描述和参数 Schema，并决定是否请求调用工具。

### 方式二：使用普通 Python 函数

LangChain 也可以接受带有适当签名和描述的 Python 函数。

## 实例

from langchain.agents import create\_agent

def calculate\_sum(a: int, b: int) -> int:
    """计算两个整数之和。"""

    return a + b

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[calculate\_sum]
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "请用工具计算 125 + 375。"
        }
    ]
})

print(result["messages"][-1].content)

这里的函数不需要手动添加 `@tool`，Agent 创建流程可以将其转换为可执行工具。

### 方式三：提供商原生工具

`create_agent()` 也支持某些模型提供商的原生工具定义。

这类工具可能由模型服务端执行，而不是由本地 Python 函数执行。

但其字典结构由具体提供商决定。

**注意：**不能简单地将任意 MCP 服务器描述字典直接传入 `tools`，就认为 LangChain 会自动连接 MCP 服务器。

如果需要访问 MCP 工具，应使用兼容的 MCP 集成或适配器，将远程工具转换为 Agent 可用的工具定义，并完成相应的连接、认证和执行配置。

### tools=None：不使用工具

## 实例

from langchain.agents import create\_agent

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=None,
    system\_prompt="你是一名 Python 编程助手。"
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Python 适合初学者吗？"
        }
    ]
})

print(result["messages"][-1].content)

不传工具时，Agent 不会执行普通客户端工具调用循环。

但它依然是一个编译后的 LangGraph 对象，与直接调用 `model.invoke()` 的返回结构并不完全相同。

---

## 六、system\_prompt 参数：系统指令

`system_prompt` 用于定义 Agent 的角色、任务目标和回答规则。

支持字符串和 `SystemMessage` 对象。

### 方式一：字符串

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[],

    system_prompt=(

        "你是菜鸟教程 RUNOOB 的编程助手。"

        "请使用简洁、准确的中文回答。"

    )

)
```

### 方式二：SystemMessage

## 实例

from langchain.messages import SystemMessage
from langchain.agents import create\_agent

system\_message = SystemMessage(
    content=(
        "你是一名 Python 编程教程作者。"
        "尽量使用初学者能够理解的语言。"
    )
)

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt=system\_message
)

系统提示词通常会在模型调用时添加到消息列表前面。

如果需要根据用户、会话或任务动态生成系统提示词，可以使用 Middleware。

**注意：**系统提示词不等于可靠的安全边界。身份认证、用户权限和工具执行限制仍需在应用代码中实现。

---

## 七、middleware 参数：Agent 中间件

`middleware` 是 LangChain 1.x Agent 的重要扩展机制。

它可以在 Agent 执行过程中的特定阶段运行自定义逻辑。

例如：

- 在模型调用前修改消息。
- 根据用户身份选择模型。
- 限制模型调用次数。
- 对工具执行进行拦截。
- 处理部分模型或工具错误。
- 记录模型调用和工具执行信息。

官方提供了多种可直接使用的中间件。

### 实例：限制模型调用次数

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware
)

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            run\_limit=5,
            exit\_behavior="end"
        )
    ],
    system\_prompt="你是一名中文编程助手。"
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "解释什么是 Python 装饰器。"
        }
    ]
})

print(result["messages"][-1].content)

这里使用 `ModelCallLimitMiddleware` 限制一次运行中的模型调用次数。

需要注意，模型调用限制不等同于 Token 预算或 API 费用上限。

中间件还可以通过 `before_model`、`after_model`、`wrap_model_call` 和 `wrap_tool_call` 等机制实现更复杂的行为。

---

## 八、response\_format 参数：结构化输出

如果希望 Agent 返回特定格式的数据，可以使用 `response_format`。

例如，让 Agent 返回教程推荐结果，包含推荐课程和理由。

LangChain 支持 Pydantic 模型，以及 `ProviderStrategy`、`ToolStrategy` 等结构化输出策略。

### 实例：使用 Pydantic 定义结构

## 实例

from pydantic import BaseModel, Field
from langchain.agents import create\_agent

class CourseRecommendation(BaseModel):
    """课程推荐结果。"""

    course: str = Field(
        description="推荐的课程名称"
    )

    reason: str = Field(
        description="推荐理由"
    )

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    response\_format=CourseRecommendation,
    system\_prompt=(
        "你是一名编程学习顾问。"
        "请根据用户的问题返回课程推荐。"
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "我是零基础开发者，"
                "想学习 Python，请给出一个推荐。"
            )
        }
    ]
})

structured = result["structured\_response"]

print("课程：", structured.course)
print("理由：", structured.reason)

如果结构化输出成功，返回状态中会包含：

```
result["structured_response"]
```

该字段通常保存根据指定 Schema 解析后的结果。

例如：

```
{

    "course": "Python 基础教程",

    "reason": "适合初学者了解 Python 的基本语法。"

}
```

这里展示的是结果结构示意，并非模型的固定输出。

**注意：**不同模型对原生结构化输出、Tool Calling 和 JSON Schema 的支持可能不同。

LangChain 可以根据模型能力选择结构化输出策略，但不保证任意模型和任意 Schema 都能成功。

---

## 九、state\_schema 参数：自定义 Agent 状态

Agent 在执行过程中会保存状态。

最常见的状态字段是：

```
messages
```

用于保存用户消息、模型回复和工具执行结果。

AgentState 还可能包含执行控制或结构化响应相关字段，具体以当前 LangChain 版本为准。

如果需要保存额外的应用状态，可以扩展 `AgentState`。

例如，创建一个学习助手，记录当前用户等级和已完成的主题。

### 1、定义自定义 State

## 实例

from langchain.agents import AgentState

class LearningAgentState(AgentState):
    """编程学习 Agent 状态。"""

    user\_level: str
    completed\_topics: list[str]

其中：

- `user_level`：用户当前学习等级。
- `completed_topics`：已完成的学习主题。

但仅仅声明状态字段，并不代表工具执行后这些字段会自动更新。

### 2、使用 Command 修改 State

如果工具需要修改 Agent 状态，可以返回 LangGraph 的 `Command`。

不应只对 `runtime.state` 中的列表执行 `append()`，就假设状态已经正确提交。

## 实例

from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

class LearningAgentState(AgentState):
    user\_level: str
    completed\_topics: list[str]

@tool
def track\_progress(
    topic: str,
    runtime: ToolRuntime
) -> Command:
    """将指定主题加入当前已完成列表。"""

    completed = list(
        runtime.state.get(
            "completed\_topics", []
        )
    )

    if topic not in completed:
        completed.append(topic)

    return Command(
        update={
            "completed\_topics": completed,
            "messages": [
                ToolMessage(
                    content=(
                        f"已处理学习主题：{topic}"
                    ),
                    tool\_call\_id=runtime.tool\_call\_id
                )
            ]
        }
    )

上面使用 `Command(update=...)` 明确声明需要更新的状态字段。

`ToolMessage` 通过当前工具调用 ID 与模型的工具请求关联。

### 3、创建 Agent 并读取更新后的 State

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[track\_progress],
    state\_schema=LearningAgentState,
    system\_prompt=(
        "你是一名学习进度助手。"
        "当用户明确要求记录已完成主题时，"
        "调用 track\_progress 工具。"
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "我已经学完 Python 基础，"
                "请帮我记录这个主题。"
            )
        }
    ],
    "user\_level": "入门",
    "completed\_topics": [
        "HTML 基础"
    ]
})

print("用户等级：", result.get("user\_level"))
print("已完成主题：", result.get("completed\_topics"))

print("最终回答：")
print(result["messages"][-1].content)

如果模型正确调用 `track_progress`，更新后的状态可能是：

```
用户等级： 入门

已完成主题： ['HTML 基础', 'Python 基础']
```

由于工具是否被调用由模型决定，所以不能认为这个结果一定出现。

如果学习进度必须更新，应在应用层明确执行状态变更，或者在工作流中设置强制执行步骤。

**注意：**上述方式适用于相对简单的状态更新。如果存在多个工具并行写入同一字段，需要考虑合适的 reducer 或并发控制，避免发生状态更新冲突。

---

## 十、context\_schema 参数：运行时上下文

`context_schema` 用于定义 Agent 运行时 Context 的结构。

与 State 不同，Context 通常由应用程序在每次调用时提供，用于保存当前运行需要的可信信息。

例如：

- 当前登录用户 ID。
- 用户所在组织。
- 应用当前使用的语言。
- 业务配置。

### 实例：传入当前用户身份

## 实例

from dataclasses import dataclass

from langchain.agents import create\_agent
from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
    user\_id: str

@tool
def get\_current\_user(
    runtime: ToolRuntime[UserContext]
) -> str:
    """返回当前请求的模拟用户标识。"""

    return runtime.context.user\_id

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[get\_current\_user],
    context\_schema=UserContext,
    system\_prompt=(
        "你是用户信息助手。"
        "查询当前用户标识时使用工具。"
    )
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "查询我的用户标识。"
            }
        ]
    },
    context=UserContext(
        user\_id="user\_001"
    )
)

print(result["messages"][-1].content)

这里的 `user_id` 由应用程序提供，而不是模型生成的工具参数。

**注意：**Context 只是传递运行信息的机制，并不会自动完成用户认证。真实项目应先验证用户登录状态，再构建可信 Context。

---

## 十一、checkpointer 参数：会话持久化

如果希望 Agent 能够记住之前的对话，可以配置 `checkpointer`。

Checkpointer 负责保存和恢复 Agent 的执行状态。

LangGraph 提供适合开发和测试的 `InMemorySaver`。

### 实例：保存多轮对话

## 实例

from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    checkpointer=InMemorySaver(),
    system\_prompt="你是一名编程学习助手。"
)

# 使用相同线程 ID
config = {
    "configurable": {
        "thread\_id": "learning\_session\_001"
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
                "content": "我刚才说我在学习什么？"
            }
        ]
    },
    config=config
)

print(result["messages"][-1].content)

由于两次请求使用相同的 `thread_id`，Agent 可以从 Checkpointer 恢复此前的会话消息。

需要注意，`InMemorySaver` 只保存进程内数据。

程序退出后，内存中的会话状态通常会丢失。

对于需要真正持久化的生产项目，可以使用数据库支持的 Checkpointer。

### get\_state()：查看保存的状态

配置 Checkpointer 后，可以使用 `get_state()` 获取指定线程的状态快照。

```
snapshot = agent.get_state(config)



print(snapshot.values)
```

其中，`snapshot.values` 保存当前已记录的状态数据。

如果没有配置 Checkpointer 或没有对应的线程记录，不能假设 `get_state()` 一定返回有效的历史状态。

### update\_state()：更新保存的状态

也可以通过：

```
agent.update_state(

    config,

    {

        "user_level": "进阶"

    }

)
```

修改受状态 Schema 支持的字段。

**注意：**`update_state()` 是图状态管理操作，不是单纯修改普通 Python 字典。使用前应确认图已配置适当的检查点存储，并理解状态合并和 reducer 规则。

---

## 十二、store 参数：跨会话共享数据

`store` 用于向 Agent 提供 LangGraph Store。

与 Checkpointer 不同，Store 主要用于跨线程或跨会话共享数据。

例如：

- 用户的长期偏好。
- 学习进度。
- 课程资料。
- 长期记忆。

开发阶段可以使用 `InMemoryStore`。

### 实例：创建共享 Store

## 实例

from langchain.agents import create\_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 保存模拟课程数据
store.put(
    ("demo", "courses"),
    "python",
    {
        "name": "Python 基础",
        "level": "入门"
    }
)

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    store=store
)

# 直接查看 Store 数据
item = store.get(
    ("demo", "courses"),
    "python"
)

print(item.value)

执行结果：

```
{'name': 'Python 基础', 'level': '入门'}
```

当工具需要访问 Store 时，可以结合上一节介绍的 `ToolRuntime`，使用 `runtime.store` 读取数据。

**注意：**`InMemoryStore` 可以在同一进程内跨调用共享数据，但不是数据库级持久化存储。

---

## 十三、interrupt\_before 与 interrupt\_after

这两个参数用于控制 LangGraph 在指定节点执行前或执行后暂停。

常见节点包括：

```
model

tools
```

例如，可以在工具执行前暂停，让程序检查工具调用请求。

### 实例：工具执行前中断

## 实例

from langchain.agents import create\_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

@tool
def delete\_demo\_file(filename: str) -> str:
    """模拟删除文件，不执行实际文件操作。"""

    return f"模拟删除文件：{filename}"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[delete\_demo\_file],
    checkpointer=InMemorySaver(),
    interrupt\_before=["tools"],
    system\_prompt=(
        "你是文件操作演示助手。"
        "用户要求模拟删除文件时使用工具。"
        "所有操作均为模拟。"
    )
)

config = {
    "configurable": {
        "thread\_id": "demo\_delete\_001"
    }
}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请模拟删除 report.txt"
            }
        ]
    },
    config=config
)

# 查看是否存在待继续执行的节点
snapshot = agent.get\_state(config)

print("待执行节点：", snapshot.next)

如果模型发起了工具调用，图可能在 `tools` 节点执行前暂停。

此时可以检查状态中的工具调用请求。

**注意：**设置中断并不等于自动完成审批。对于真实删除文件、资金操作等敏感行为，还需要在恢复图执行之前完成可靠的人工确认和权限检查。

如果需要继续执行已经暂停的图，应根据 LangGraph 的恢复机制处理，并使用相同的会话配置。

---

## 十四、debug、name、cache 和 transformers

这些参数主要用于调试、集成和扩展执行图。

### debug：开启调试输出

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[],

    debug=True

)
```

启用调试后，LangGraph 可以输出节点执行、状态更新及图流转等信息。

调试输出可能包含用户输入和工具结果，因此生产环境需要注意日志中的敏感数据。

### name：指定 Agent 名称

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[],

    name="runoob_learning_agent"

)
```

名称可以帮助识别执行图，也适用于将 Agent 作为其他 LangGraph 工作流中的子图使用。

### cache：图执行缓存

`cache` 用于配置 LangGraph 的图执行缓存能力。

它不等同于自动缓存所有大语言模型 API 响应。

实际是否发生缓存，还可能取决于节点和缓存策略的配置。

### transformers：流事件转换器

`transformers` 用于注册额外的流事件转换器。

它主要面向需要深度定制流式事件处理流程的开发者。

对于一般的聊天 Agent 或教程示例，通常不需要设置。

---

## 十五、create\_agent() 的返回值

`create_agent()` 返回一个 `CompiledStateGraph` 对象。

这意味着可以使用 LangGraph 提供的同步、异步、流式和状态管理接口。

| 方法 | 作用 |
| --- | --- |
| `invoke()` | 同步执行 Agent，等待任务结果 |
| `ainvoke()` | 异步执行 Agent |
| `stream()` | 同步流式获取执行结果 |
| `astream()` | 异步流式执行 |
| `get_state()` | 读取已保存的图状态快照 |
| `update_state()` | 更新图状态 |
| `get_state_history()` | 获取受支持的状态历史记录 |

### 1、invoke()：同步调用

## 实例

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "什么是 LangChain？"
        }
    ]
})

print(result["messages"][-1].content)

这是最常用的 Agent 调用方式。

### 2、ainvoke()：异步调用

## 实例

import asyncio

from langchain.agents import create\_agent

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[]
)

async def main():
    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "介绍一下 Python。"
            }
        ]
    })

    print(result["messages"][-1].content)

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(main())

适合异步 Web 服务和其他使用 `asyncio` 的应用。

### 3、stream()：流式输出执行过程

使用 `stream_mode="updates"` 可以查看 Agent 的节点更新。

## 实例

from langchain.agents import create\_agent
from langchain.tools import tool

@tool
def calculate(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[calculate]
)

for update in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请使用工具计算 25 + 18。"
            }
        ]
    },
    stream\_mode="updates"
):
    print(update)

如果模型调用了工具，流式更新中可能出现模型节点和工具节点的执行结果。

需要注意，`stream_mode="updates"` 是获取图节点状态更新，而不是逐 Token 输出模型文本。

如果希望实时展示模型生成的 Token，应使用相应的消息流模式。

---

## 十六、完整实例：课程学习 Agent

下面将模型、工具、系统提示词和调用流程整合为一个可运行的实例。

这个 Agent 可以查询模拟课程，并根据用户问题返回学习建议。

文件名：`create_agent_demo.py`

## 完整实例

from dotenv import load\_dotenv

from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# -------------------------
# 模拟课程数据
# -------------------------

COURSES = {
    "python": [
        "Python 基础教程",
        "Python 数据分析"
    ],
    "java": [
        "Java 基础教程",
        "Java 面向对象"
    ],
    "html": [
        "HTML 基础教程"
    ]
}

# -------------------------
# 定义工具
# -------------------------

@tool
def search\_courses(keyword: str) -> str:
    """按编程语言关键词查询模拟课程。"""

    results = COURSES.get(
        keyword.lower()
    )

    if not results:
        return "没有找到对应的模拟课程"

    return "、".join(results)

@tool
def count\_courses(keyword: str) -> str:
    """统计指定语言的模拟课程数量。"""

    results = COURSES.get(
        keyword.lower(),
        []
    )

    return (
        f"{keyword} 共有 "
        f"{len(results)} 门模拟课程"
    )

# -------------------------
# 初始化模型
# -------------------------

model = init\_chat\_model(
    "deepseek:deepseek-flash",
    timeout=30,
    max\_retries=2
)

# -------------------------
# 创建 Agent
# -------------------------

agent = create\_agent(
    model=model,
    tools=[
        search\_courses,
        count\_courses
    ],
    system\_prompt=(
        "你是菜鸟教程 RUNOOB 的学习顾问。"
        "工具中的数据全部是教学模拟数据。"
        "查询课程及数量时优先使用工具。"
        "如果工具没有对应数据，"
        "不要编造课程名称或数量。"
        "请使用简洁、准确的中文回答。"
    ),
    name="runoob\_course\_agent"
)

# -------------------------
# 执行 Agent
# -------------------------

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
        config={
            "recursion\_limit": 15
        }
    )

    print("\n问题：", question)

    for message in result["messages"]:

        # 模型发起工具调用
        if (
            message.type == "ai"
            and message.tool\_calls
        ):
            for call in message.tool\_calls:
                print(
                    "[工具调用]",
                    call["name"],
                    call["args"]
                )

        # 工具结果
        elif message.type == "tool":
            print(
                "[工具结果]",
                message.name,
                message.content
            )

    print(
        "最终回答：",
        result["messages"][-1].content
    )

if \_\_name\_\_ == "\_\_main\_\_":

    ask("有哪些 Python 模拟课程？")

    ask("Java 一共有多少门模拟课程？")

    ask("学习 HTML 适合从哪里开始？")

### 运行程序

```
python create_agent_demo.py
```

如果 Agent 正确调用对应的工具，可以获取模拟课程数据：

```
Python 模拟课程：

Python 基础教程、Python 数据分析



Java 模拟课程数量：

2 门



HTML 模拟课程：

HTML 基础教程
```

实际的工具调用次数、执行顺序和语言模型回答可能不同。

这里展示的课程数据来自 Python 字典，不会访问菜鸟教程的真实课程数据库。

---

## 十七、常见问题

### 1、create\_agent() 和 init\_chat\_model() 有什么区别？

`init_chat_model()` 用于创建聊天模型对象。

`create_agent()` 则在模型基础上增加工具调用、执行循环、状态管理和中间件等能力。

前者主要解决模型初始化，后者主要解决 Agent 执行。

### 2、可以不传 tools 吗？

可以。

当 `tools=None` 或为空列表时，Agent 没有可执行的客户端工具调用循环。

但仍可以使用系统提示词和其他支持的执行图功能。

### 3、create\_agent() 是否自动执行工具？

对于注册的可执行客户端工具，Agent 可以根据模型生成的工具调用请求执行它们。

但是否调用工具仍由模型和 Agent 的配置共同决定。

提供商原生工具可能由远程模型服务执行，不能与本地 Python 工具混为一谈。

### 4、为什么不能直接把 Pydantic 类当作工具使用？

Pydantic 类主要用于定义数据结构和参数 Schema。

它本身不一定提供工具的执行逻辑。

如果希望 Agent 执行 Python 操作，应使用 `@tool`、可调用函数或其他实现了执行接口的工具对象。

### 5、为什么 state\_schema 中的列表没有更新？

因为读取状态或直接修改状态字典，不等于向图提交了状态更新。

如果工具需要更新自定义 State，应使用受支持的状态更新机制，例如返回 `Command`。

### 6、为什么 result["structured\_response"] 不存在？

通常是因为没有配置 `response_format`，或结构化输出过程没有成功完成。

如果需要结构化结果，应正确配置输出 Schema，并确认当前模型支持所需的结构化输出策略。

### 7、如何让 Agent 记住前面的对话？

可以配置 Checkpointer，并在多次调用时使用相同的 `thread_id`。

如果需要程序重启后仍能恢复记录，应使用数据库支持的 Checkpointer。

### 8、store 与 checkpointer 有什么区别？

`checkpointer` 用于保存和恢复会话状态。

`store` 用于跨线程或跨会话共享数据，例如用户偏好和长期学习进度。

二者用途不同，不能简单互相代替。

### 9、interrupt\_before=["tools"] 是否自动实现人工审批？

不会。

它只提供工具节点执行前的暂停机制。

实际审批需要由应用程序读取待执行操作、判断权限并决定是否恢复。

### 10、如何限制 Agent 的执行次数？

可以在运行时通过 `recursion_limit` 限制图执行步数：

```
result = agent.invoke(

    {

        "messages": [

            {

                "role": "user",

                "content": "你好"

            }

        ]

    },

    config={

        "recursion_limit": 15

    }

)
```

需要注意，`recursion_limit` 不是严格的工具调用次数限制，也不是 Token 费用上限。

如果需要限制模型调用次数，可以考虑使用 LangChain 中间件。

### 11、create\_agent() 能连接 MCP 吗？

可以通过兼容的 MCP 集成或适配器，将远程 MCP 工具提供给 Agent。

但不能假设任意 MCP 配置字典都可以直接传入 `tools` 并自动完成远程连接与工具执行。

### 12、什么时候应该使用 LangGraph 而不是 create\_agent()？

如果只是需要模型自动选择工具、处理结果并循环执行，`create_agent()` 通常已经足够。

如果任务需要明确的节点顺序、复杂分支、多个 Agent 协作或严格的执行控制，可以使用 LangGraph 构建自定义工作流。

`create_agent()` 本身就是基于 LangGraph 实现的。

---

## 总结

`create_agent()` 是 LangChain 1.x 创建 Agent 的核心入口。

最简单的用法只需要指定模型和工具：

```
from langchain.agents import create_agent



agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[]

)



result = agent.invoke({

    "messages": [

        {

            "role": "user",

            "content": "你好"

        }

    ]

})
```

随着项目复杂度增加，可以逐步配置：

- `system_prompt`：定义 Agent 的角色和行为。
- `middleware`：扩展模型调用与工具执行过程。
- `response_format`：生成结构化响应。
- `state_schema`：定义额外的工作流状态。
- `context_schema`：传递可信运行时上下文。
- `checkpointer`：保存和恢复会话。
- `store`：管理跨会话共享数据。
- `interrupt_before`：在指定节点前暂停执行。

> 注意：**create\_agent() 负责组织 Agent 的执行流程，模型负责提出工具调用请求，工具负责真正执行操作，而 State 与 Middleware 则帮助管理整个执行过程。**
