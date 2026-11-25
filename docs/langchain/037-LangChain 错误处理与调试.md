# LangChain 错误处理与调试

> 来源: https://www.runoob.com/langchain/langchain-error-handling.html

---

# LangChain 错误处理与调试

开发 LangChain Agent 时，程序不可能始终正常运行。

例如，模型 API 请求超时、工具参数不正确、知识库检索失败、Token 超限，或者 Agent 反复调用同一个工具，都可能导致程序异常。

如果缺少合理的错误处理机制，一个普通的网络故障就可能导致整个任务中断；而错误配置的重试机制，又可能让 Agent 重复发送邮件、创建订单或执行支付。

因此，构建可靠的 Agent 不仅需要让模型能够完成任务，还需要考虑 错误分类、异常捕获、自动重试、执行限制、日志调试和故障恢复。

本章节基于 LangChain 1.x，介绍常见错误排查、ModelRetryMiddleware、ToolRetryMiddleware、ToolErrorMiddleware、Debug 模式、Streaming 调试和生产环境最佳实践。

---

## 一、LangChain 常见错误类型

可以将 Agent 运行错误划分为以下几类：

| 错误类型 | 典型原因 | 处理建议 |
| --- | --- | --- |
| ImportError | 依赖缺失、版本不兼容、导入路径错误 | 检查安装环境和当前 API |
| AuthenticationError | API Key 缺失、错误或失效 | 检查凭据和模型服务权限 |
| RateLimitError | 请求频率或服务额度受限 | 限流、退避重试、检查配额 |
| TimeoutError | 网络或模型响应超时 | 设置超时并选择性重试 |
| Context Length Error | 上下文超过模型限制 | 减少历史消息或总结上下文 |
| ValidationError | 工具参数或结构化输出不符合 Schema | 检查参数与数据模型 |
| Tool Execution Error | 工具内部代码或外部服务异常 | 分类异常并处理失败结果 |
| GraphRecursionError | Agent 执行步骤超过图递归限制 | 检查循环逻辑并限制调用次数 |
| Checkpoint Error | 线程标识、数据库连接或状态恢复异常 | 检查 Checkpointer 和 thread\_id |

不同类型的错误应该采用不同的处理策略。

例如：

- 网络超时：可以在满足条件时重试。
- API Key 错误：应该立即修复凭据。
- 工具参数错误：应该重新校验参数。
- 订单创建失败：必须先确认订单是否已经实际创建，不能直接重复调用。

**核心原则：**不是所有异常都应该重试。首先判断错误是否可能通过再次调用得到解决，以及重复执行是否安全。

---

## 二、环境准备

本教程使用 LangChain 1.x 和 DeepSeek 聊天模型。

### 安装依赖

```
pip install -U langchain langgraph langchain-deepseek python-dotenv pydantic
```

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

在 Python 中加载：

## 实例

import os
from dotenv import load\_dotenv

load\_dotenv()

if not os.getenv("DEEPSEEK\_API\_KEY"):
    raise RuntimeError("缺少 DEEPSEEK\_API\_KEY，请检查 .env 文件")

建议始终在当前项目的虚拟环境中安装依赖，避免系统 Python 和虚拟环境中的包版本不一致。

---

## 三、ImportError：导入失败

LangChain 1.x 将不同功能拆分到多个 Python 包中。

例如，DeepSeek 模型使用：

```
from langchain_deepseek import ChatDeepSeek
```

如果只安装了 `langchain`，而没有安装 `langchain-deepseek`，就可能出现：

```
ModuleNotFoundError: No module named 'langchain_deepseek'
```

解决方式：

```
python -m pip install -U langchain-deepseek
```

### 检查安装版本

## 实例

from importlib.metadata import version

packages = ["langchain", "langgraph", "langchain-deepseek"]

for package in packages:
    try:
        print(package, version(package))
    except Exception as e:
        print(package, "检查失败：", e)

注意，升级 LangChain 版本后，某些旧教程的导入路径可能已经变化。

建议优先检查对应版本的官方 API，而不是继续使用旧版的 AgentExecutor、旧版 Chain 或已经迁移的导入方式。

---

## 四、API Key 与身份认证错误

如果模型服务返回身份认证错误，常见原因包括：

- .env 文件不存在或未成功加载。
- API Key 拼写错误。
- 凭据已经失效或被撤销。
- 当前 Key 无权访问指定模型。
- 配置了错误的服务商 API 地址。

可以先检查环境变量是否存在：

## 实例

import os
from dotenv import load\_dotenv

load\_dotenv()

key = os.getenv("DEEPSEEK\_API\_KEY")
print("API Key 是否存在：", bool(key))

不要直接输出完整 API Key，也不要把 Key 写入程序日志。

**注意：**认证失败通常不是临时网络故障，不应该进行大量自动重试。应先检查凭据、账号和模型访问权限。

---

## 五、模型超时与自动重试

调用大语言模型时，可能遇到网络抖动、请求超时、服务短暂不可用等问题。

LangChain 提供 ModelRetryMiddleware，可以在模型调用失败时进行有限次数的重试。

基本处理流程：

```
调用模型

    ↓

  是否成功？

    ├── 是 → 返回结果

    │

    └── 否

         ↓

     是否属于可重试错误？

         ├── 否 → 立即处理失败

         │

         └── 是

              ↓

          等待退避时间

              ↓

            重新调用

              ↓

          达到重试上限后停止
```

### 实例：ModelRetryMiddleware

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain\_deepseek import ChatDeepSeek

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    timeout=30,
    max\_retries=0
)

agent = create\_agent(
    model=model,
    tools=[],
    middleware=[
        ModelRetryMiddleware(
            max\_retries=2,
            initial\_delay=1.0,
            backoff\_factor=2.0,
            max\_delay=10.0,
            jitter=True,
            on\_failure="error"
        )
    ],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程助手。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "介绍一下 Python。"}]
})

print(result["messages"][-1].content)

这里使用：

```
max_retries=2
```

表示首次调用失败后最多再重试两次。

因此，最多可能执行：

```
首次调用

    ↓ 失败

第一次重试

    ↓ 失败

第二次重试

    ↓

成功或停止
```

总计最多三次尝试。

本示例将模型客户端的 `max_retries` 设为 0，是为了避免 SDK 内部重试和 Middleware 重试叠加，便于教学与排查。

**注意：**生产环境应根据实际模型服务商错误类型调整 retry\_on，避免对认证、权限和永久性参数错误重复请求。

---

## 六、理解指数退避 Backoff

当大量客户端同时遇到服务故障时，如果所有请求都立即重试，可能进一步增加服务压力。

因此，重试中间件支持指数退避。

例如：

```
initial_delay=1.0

backoff_factor=2.0

jitter=False
```

当重试次数允许时，等待时间大致为：

```
第一次重试前：1 秒

第二次重试前：2 秒

第三次重试前：4 秒
```

如果开启：

```
jitter=True
```

等待时间会增加随机扰动，避免多个客户端在完全相同的时间集中重试。

### 常用重试参数

| 参数 | 作用 |
| --- | --- |
| `max_retries` | 首次失败后最多额外重试次数 |
| `initial_delay` | 首次重试前的基础延迟 |
| `backoff_factor` | 指数退避倍数 |
| `max_delay` | 单次等待时间上限 |
| `jitter` | 是否加入随机扰动 |
| `retry_on` | 指定允许重试的异常类型或判断函数 |
| `on_failure` | 全部重试失败后的处理策略 |

其中 `on_failure="error"` 表示重试耗尽后继续抛出异常。

如果使用 `on_failure="continue"`，模型错误可能被转换为 AIMessage 返回，应用需要区分错误回复和正常业务结果。

---

## 七、ToolRetryMiddleware：工具调用重试

Agent 除了调用模型，还会调用数据库、搜索服务和其他外部 API。

例如，一个课程搜索工具可能因为网络问题暂时失败。

如果工具操作可以安全重复执行，可以使用 ToolRetryMiddleware。

### 实例：只重试临时工具故障

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import tool
from langchain\_deepseek import ChatDeepSeek

attempts = 0

@tool
def search\_course(keyword: str) -> str:
    """查询模拟课程目录。"""
    global attempts
    attempts += 1

    # 模拟第一次调用暂时失败
    if attempts == 1:
        raise ConnectionError("模拟：课程服务暂时不可用")

    return f"查询成功：找到 {keyword} 相关模拟课程"

agent = create\_agent(
    model=ChatDeepSeek(model="deepseek-chat", max\_retries=0),
    tools=[search\_course],
    middleware=[
        ToolRetryMiddleware(
            tools=["search\_course"],
            max\_retries=2,
            retry\_on=(ConnectionError, TimeoutError),
            initial\_delay=1.0,
            backoff\_factor=2.0,
            jitter=False,
            on\_failure="continue"
        )
    ],
    system\_prompt="你是课程助手。用户请求查询课程时使用 search\_course。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请使用工具搜索 Python 课程。"}]
})

print(result["messages"][-1].content)

如果模型调用了 `search_course`，第一次工具执行会触发模拟 ConnectionError。

中间件可以进行重试，第二次调用成功后，将结果交给模型。

这里通过 `retry_on` 明确只对指定的临时故障重试。

**注意：**示例中的全局 attempts 计数器只用于单进程教学测试，不适合真实的并发服务。

---

## 八、哪些工具不应该自动重试？

不是所有工具都适合使用 ToolRetryMiddleware。

特别是具有外部副作用的工具，例如：

- 发送邮件。
- 创建订单。
- 执行支付。
- 删除文件。
- 修改数据库记录。

假设订单创建工具发生超时：

```
Agent → 创建订单

             ↓

        服务端已经创建成功

             ↓

        响应在网络中丢失

             ↓

        Agent 认为调用失败

             ↓

             重试

             ↓

        创建第二个订单
```

因此，不能简单地认为"工具抛出异常就表示业务操作没有成功"。

对于真实业务，应优先设计：

- 幂等请求 ID。
- 唯一业务流水号。
- 执行结果查询。
- 重复请求检测。
- 必要的人工审批。

**建议：**ToolRetryMiddleware 优先用于搜索、只读查询等适合重复执行的工具。涉及支付和删除等操作时，必须额外设计幂等保护。

---

## 九、ToolErrorMiddleware：处理工具错误

有些错误没有必要重试，但需要转换成模型可以理解的工具失败信息。

例如，用户输入的课程编号不合法。

这时可以使用 `ToolErrorMiddleware`。

### 实例：处理业务参数错误

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain.tools import tool

@tool
def get\_course(course\_id: int) -> str:
    """根据正整数课程 ID 查询模拟课程。"""
    if course\_id <= 0:
        raise ValueError("课程 ID 必须大于 0")
    return f"模拟课程 {course\_id}：Python 基础"

def handle\_tool\_error(exc, request):
    """将指定业务异常转换成工具失败消息。"""
    if isinstance(exc, ValueError):
        return "课程 ID 不合法，请使用大于 0 的整数。"
    return None

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[get\_course],
    middleware=[ToolErrorMiddleware(on\_error=handle\_tool\_error)]
)

这里仅处理 ValueError。

其他没有明确处理的异常会继续抛出，避免掩盖真正的系统故障。

需要注意，工具内部主动抛出的 ValueError 与模型调用工具时发生的参数绑定、Pydantic 校验错误不一定经过完全相同的处理路径。

**建议：**生产环境应分别处理工具输入校验、业务异常和基础设施故障，不要简单地用一个 except Exception 吞掉所有错误。

---

## 十、组合 Retry 与 Error Middleware

当某个工具既可能发生网络故障，也可能发生业务参数错误时，可以将 Retry 和 Error Middleware 组合使用。

设计思路：

```
工具调用

    ↓

ToolRetryMiddleware

    ↓

临时错误？

    ├── 是 → 有限次数重试

    └── 否 → 不重试

    ↓

ToolErrorMiddleware

    ↓

转换允许公开的业务错误

    ↓

Agent 获得工具结果
```

例如：

## 实例

from langchain.agents.middleware import ToolErrorMiddleware, ToolRetryMiddleware

middleware = [
    ToolErrorMiddleware(on\_error=handle\_tool\_error),
    ToolRetryMiddleware(
        tools=["get\_course"],
        max\_retries=2,
        retry\_on=(ConnectionError, TimeoutError),
        on\_failure="error"
    )
]

这里将 ToolErrorMiddleware 放在外层，ToolRetryMiddleware 放在内层。

当可重试错误最终仍然失败时，异常可以继续传递给外层处理器。

但本例中的 `handle_tool_error` 只处理 ValueError，因此 ConnectionError 等错误仍会向外传播。

如果需要向用户展示"服务暂时不可用"，可以显式增加相应的异常处理逻辑。

---

## 十一、debug=True：开启 Agent 调试模式

LangChain 的 `create_agent()` 支持 `debug=True`。

开启后，底层图执行过程会输出更详细的调试信息，包括节点执行和状态更新等内容。

### 实例：启用调试模式

## 实例

from langchain.agents import create\_agent
from langchain\_deepseek import ChatDeepSeek

agent = create\_agent(
    model=ChatDeepSeek(model="deepseek-chat", temperature=0),
    tools=[],
    debug=True,
    system\_prompt="你是 Python 编程助手。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "你好"}]
})

print("最终回答：", result["messages"][-1].content)

这种模式适合在本地开发时检查 Agent 的执行逻辑。

需要注意，调试输出的具体格式可能随 LangChain 和 LangGraph 版本变化。

因此，不应该编写依赖某一段固定调试文本的程序。

**安全提醒：**debug=True 可能输出消息内容、工具参数和其他状态信息，不建议在包含真实用户隐私或生产凭据的环境中直接开启完整调试。

---

## 十二、stream\_mode="updates"：查看执行步骤

如果希望以结构化方式查看 Agent 的执行步骤，可以使用 Streaming。

对于一般的工具调用调试，推荐先从：

```
stream_mode="updates"
```

开始。

它会在图节点完成执行后返回相应的状态更新。

### 实例：查看模型与工具调用

## 实例

from langchain.agents import create\_agent
from langchain.tools import tool

@tool
def search\_course(keyword: str) -> str:
    """搜索模拟编程课程。"""
    return f"找到 {keyword} 入门课程"

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[search\_course],
    system\_prompt="你是课程助手。课程查询时使用 search\_course。"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": "请搜索 Python 课程。"}]},
    stream\_mode="updates"
):
    for node, update in event.items():
        print("执行节点：", node)

        for msg in update.get("messages", []):
            if msg.type == "ai" and getattr(msg, "tool\_calls", []):
                print("工具调用：", msg.tool\_calls)
            elif msg.type == "tool":
                print("工具结果：", msg.content)
            elif msg.type == "ai" and msg.content:
                print("模型回答：", msg.content)

可能的节点执行流程：

```
执行节点： model

工具调用： search_course



执行节点： tools

工具结果： 找到 Python 入门课程



执行节点： model

模型回答： 为你找到了 Python 入门课程。
```

这样可以检查 Agent 是否实际调用了工具，以及工具是否正确返回结果。

---

## 十三、stream\_mode="debug"：更详细的图调试

LangGraph 还支持：

```
stream_mode="debug"
```

用于输出更细粒度的执行事件。

与 updates 不同，debug 模式主要用于深入观察图执行过程。

### 实例：查看 Debug 事件

## 实例

for event in agent.stream(
    {"messages": [{"role": "user", "content": "你好"}]},
    stream\_mode="debug"
):
    print("事件类型：", event.get("type"))
    print("时间：", event.get("timestamp"))
    print("步骤：", event.get("step"))

    payload = event.get("payload", {})
    if isinstance(payload, dict):
        print("节点：", payload.get("name"))
        print("数据字段：", list(payload.keys()))
    print("-" \* 40)

实际 Debug 事件的字段和嵌套结构可能随 LangGraph 版本变化。

建议先打印完整事件或观察其键结构，再决定如何提取输入、输出等字段。

**注意：**debug 模式的数据量可能较大，适合本地排错，不建议将全部事件直接发送到普通用户的前端页面。

---

## 十四、工具调用无限循环

Agent 在执行时通常会重复以下流程：

```
调用模型

    ↓

模型要求使用工具

    ↓

执行工具

    ↓

将工具结果返回模型

    ↓

再次调用模型
```

如果模型始终认为任务没有完成，就可能反复调用相同工具。

例如：

```
search_course("Python")

    ↓

返回结果

    ↓

search_course("Python")

    ↓

返回结果

    ↓

search_course("Python")

    ↓

...
```

常见原因：

- 工具结果没有说明实际查询状态。
- 模型无法理解工具返回内容。
- 提示词要求反复验证。
- 工具之间相互触发。
- 缺少执行次数限制。

### 使用 ToolCallLimitMiddleware 限制工具调用

## 实例

from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[search\_course],
    middleware=[
        ToolCallLimitMiddleware(
            tool\_name="search\_course",
            run\_limit=3,
            exit\_behavior="end"
        )
    ]
)

这个例子限制单次 Agent 运行中 `search_course` 的调用数量。

当达到限制时，Middleware 会按配置结束相关执行流程，而不是无限循环。

**注意：**不同的 exit\_behavior 会影响触发上限后的处理结果，应根据业务需要选择，不能将被限制的调用视为执行成功。

---

## 十五、限制模型调用次数

除了工具调用次数，还可以限制模型调用次数。

LangChain 提供 `ModelCallLimitMiddleware`。

### 实例：限制单次模型调用

## 实例

from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[search\_course],
    middleware=[
        ModelCallLimitMiddleware(
            run\_limit=5,
            exit\_behavior="end"
        )
    ]
)

这样可以为每次 Agent 运行设置模型调用上限。

但需要注意，限制调用次数并不一定能解决根本原因。

如果工具返回内容存在问题，仍然应该修复工具输出，而不是只依赖上限阻止循环。

---

## 十六、GraphRecursionError：递归限制异常

如果图执行步骤过多，可能遇到：

```
GraphRecursionError
```

可以通过运行配置控制递归限制：

## 实例

from langgraph.errors import GraphRecursionError

try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查询 Python 课程。"}]},
        config={"recursion\_limit": 20}
    )
    print(result["messages"][-1].content)

except GraphRecursionError:
    print("Agent 执行步骤超过限制，请检查循环逻辑。")

其中：

```
recursion_limit=20
```

限制的是底层图执行的步骤数量，不等于最多调用模型 20 次。

因为一次完整工具交互可能包含多个节点执行步骤。

**建议：**不要一遇到 GraphRecursionError 就无限提高 recursion\_limit。应该首先检查工具是否重复调用、路由条件是否正确，以及 Agent 是否能够判断任务完成。

---

## 十七、工具参数校验错误

模型可能生成错误的工具参数。

例如，课程工具要求：

```
course_id: int
```

但模型传入：

```
{"course_id": "abc"}
```

就可能发生参数验证错误。

可以使用 Pydantic 为工具定义明确的输入 Schema。

### 实例：限制工具参数

## 实例

from pydantic import BaseModel, Field
from langchain.tools import tool

class CourseInput(BaseModel):
    course\_id: int = Field(gt=0, description="正整数课程 ID")

@tool(args\_schema=CourseInput)
def get\_course(course\_id: int) -> str:
    """根据课程 ID 查询模拟课程。"""
    return f"课程 {course\_id}：Python 基础"

print(get\_course.invoke({"course\_id": 100}))

这样可以在工具执行前检查参数类型和范围。

例如 `course_id` 必须是大于 0 的整数。

如果参数不符合要求，工具调用可能产生验证失败消息或异常，实际处理方式取决于 ToolNode 和相应错误处理配置。

---

## 十八、结构化输出错误

对于需要固定 JSON 或 Pydantic 格式的任务，模型可能返回不符合 Schema 的内容。

例如，我们要求模型返回：

```
{

    "course_name": "Python 基础",

    "difficulty": "beginner"

}
```

但模型遗漏了必要字段。

可以使用 `ToolStrategy` 定义结构化输出。

### 实例：结构化输出校验

## 实例

from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

class CourseInfo(BaseModel):
    course\_name: str = Field(description="课程名称")
    difficulty: Literal["beginner", "intermediate", "advanced"]

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    response\_format=ToolStrategy(
        CourseInfo,
        handle\_errors=True
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "返回 Python 基础课程的信息。"}]
})

print(result["structured\_response"])

`handle_errors=True` 可以在支持的结构化输出错误场景下，将校验反馈交给模型，让模型尝试修正。

但它不意味着所有模型故障、API 超时或网络异常都能被自动修复。

生产环境还应设置整体请求时间限制，并在校验失败时返回合理的业务错误。

---

## 十九、Token 超限和长对话问题

使用 Checkpointer 保存多轮历史后，对话内容可能持续增长。

例如：

```
第 1 轮：2 条消息

第 10 轮：20 条消息

第 50 轮：100 条消息

第 100 轮：200 条消息
```

如果每次都将全部消息发送给模型，最终可能超过模型的上下文窗口。

处理方式包括：

- 限制传给模型的历史长度。
- 将较早内容总结成摘要。
- 清理已经不需要的工具结果。
- 将长期用户偏好保存在 Store 中。

LangChain 提供 `SummarizationMiddleware`，可以在满足特定条件时对历史消息进行总结。

### 实例：SummarizationMiddleware

## 实例

from langchain.agents.middleware import SummarizationMiddleware

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[search\_course],
    middleware=[
        SummarizationMiddleware(
            model="deepseek:deepseek-chat",
            trigger=("messages", 20),
            keep=("messages", 8)
        )
    ]
)

这里演示基于消息数量触发总结的配置。

需要注意，消息数量与 Token 数量并不等价。正式项目可根据实际模型的 Token 统计能力和上下文预算选择触发策略。

另外，不能简单使用：

```
return {"messages": messages[-6:]}
```

就认为能够从 Agent State 中删除旧消息，因为 messages 通常使用 add\_messages reducer。

如果确实需要修改持久化历史，应使用符合 reducer 语义的消息删除机制，避免破坏 AI 工具调用消息与 ToolMessage 的对应关系。

---

## 二十、对话记忆不生效

如果 Agent 不记得之前聊过什么，通常需要检查 Checkpointer 配置。

首先确认：

- 创建 Agent 时是否传入 checkpointer。
- 不同轮次是否使用相同的 thread\_id。
- Checkpointer 的连接是否仍然有效。
- 程序是否错误地创建了新的存储实例。
- 是否存在主动清理或总结历史消息的 Middleware。

### 实例：检查对话状态

## 实例

from langgraph.checkpoint.memory import InMemorySaver

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread\_id": "debug-memory-001"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "我叫小明。"}]},
    config=config
)

agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)

state = agent.get\_state(config)

print("消息数量：", len(state.values.get("messages", [])))
print("待执行节点：", state.next)

如果两轮调用使用相同 thread\_id，第二轮通常可以访问之前的消息。

**注意：**InMemorySaver 只保存在当前进程内存中，程序重启后会丢失记录。如果需要跨进程持久化，应使用 SQLite 或 PostgreSQL 等 Checkpointer。

---

## 二十一、使用 Python logging 记录业务日志

除 LangChain 自带的 Debug 模式外，还可以使用 Python 标准库 `logging` 记录业务信息。

### 实例：配置日志

## 实例

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger("runoob.agent")

def query\_course\_service(keyword: str) -> str:
    logger.info("开始执行课程查询")
    try:
        result = f"模拟课程查询结果：{keyword}"
        logger.info("课程查询完成")
        return result
    except Exception:
        logger.exception("课程查询失败")
        raise

print(query\_course\_service("Python"))

与直接使用 print() 相比，logging 便于控制日志等级和输出格式。

建议记录：

- 请求 ID。
- 工具名称。
- 调用耗时。
- 错误类别。
- 重试次数。
- 是否成功完成。

不要直接记录完整 API Key、访问令牌或不必要的用户个人信息。

---

## 二十二、LangSmith：可视化调试

对于较复杂的 Agent，仅靠本地 Debug 和 print 日志可能难以完整了解模型与工具的调用关系。

可以启用 LangSmith 追踪。

在 `.env` 中配置：

```
LANGSMITH_TRACING=true

LANGSMITH_API_KEY=your_langsmith_api_key

LANGSMITH_PROJECT=runoob-agent-debug
```

之后正常执行：

```
agent.invoke({

    "messages": [{"role": "user", "content": "搜索 Python 课程。"}]

})
```

在 LangSmith 中可以查看支持追踪的：

- 模型请求与回复。
- 工具调用及结果。
- 执行时间线。
- 错误信息。
- Token 和相关使用数据。

**注意：**LangSmith 可能记录输入输出内容。生产环境应根据需要设置脱敏、隐藏和权限控制。

---

## 二十三、完整实例：带重试与执行限制的 Agent

最后，将本章节的核心功能整合为一个简单的 Agent。

包括：

- 模型请求超时。
- ModelRetryMiddleware。
- ToolRetryMiddleware。
- ToolErrorMiddleware。
- 模型调用次数限制。
- 工具调用次数限制。
- 顶层异常捕获。

文件名：`agent_debug_demo.py`

## 完整实例

import logging
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ToolErrorMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware
)
from langchain.tools import tool
from langchain\_deepseek import ChatDeepSeek
from langgraph.errors import GraphRecursionError

load\_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent\_debug")

# ========== 1. 工具 ==========

@tool
def search\_course(keyword: str) -> str:
    """根据关键词查询模拟课程。"""
    if not keyword.strip():
        raise ValueError("搜索关键词不能为空")
    return f"找到与 {keyword} 相关的模拟课程"

# ========== 2. 业务错误处理 ==========

def handle\_error(exc, request):
    if isinstance(exc, ValueError):
        return "查询参数不合法，请提供非空关键词。"
    return None

# ========== 3. 创建模型 ==========

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    timeout=30,
    max\_retries=0
)

# ========== 4. 创建 Agent ==========

agent = create\_agent(
    model=model,
    tools=[search\_course],
    middleware=[
        ToolErrorMiddleware(on\_error=handle\_error),
        ToolRetryMiddleware(
            tools=["search\_course"],
            retry\_on=(ConnectionError, TimeoutError),
            max\_retries=2,
            on\_failure="error"
        ),
        ModelRetryMiddleware(
            max\_retries=2,
            initial\_delay=1,
            backoff\_factor=2,
            on\_failure="error"
        ),
        ToolCallLimitMiddleware(
            tool\_name="search\_course",
            run\_limit=3,
            exit\_behavior="end"
        ),
        ModelCallLimitMiddleware(
            run\_limit=5,
            exit\_behavior="end"
        )
    ],
    system\_prompt=(
        "你是编程课程助手。"
        "查询课程时使用 search\_course。"
        "只根据工具结果回答，不编造真实课程信息。"
    )
)

# ========== 5. 调用与异常处理 ==========

def run\_agent(question: str):
    logger.info("开始执行 Agent 请求")

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion\_limit": 20}
        )
        print("最终回答：", result["messages"][-1].content)
        logger.info("Agent 执行返回")

    except GraphRecursionError:
        logger.error("Agent 超过图执行步骤限制")
        print("请求未完成：Agent 执行步骤过多。")

    except TimeoutError:
        logger.error("请求超时")
        print("请求未完成：服务暂时响应较慢。")

    except Exception:
        logger.exception("Agent 执行失败")
        print("请求处理失败，请检查服务和日志。")

if \_\_name\_\_ == "\_\_main\_\_":
    run\_agent("请查询 Python 入门课程。")

这套代码为 Agent 提供了基本的故障控制。

但需要注意，Middleware 将错误转换为 AIMessage 或 ToolMessage 时，程序可能正常返回一个结果对象，并不代表业务已经成功执行。

因此，正式应用除了检查是否抛出异常，还需要明确检查业务结果状态。

---

## 二十四、生产环境最佳实践

### 1、按错误类型决定是否重试

网络错误、限流和临时服务故障可以考虑有限重试；认证、权限、参数错误通常应立即处理。

### 2、避免重复副作用

支付、创建订单、发送邮件和删除操作需要幂等保护，不能仅依赖通用重试中间件。

### 3、统一超时与重试预算

模型 SDK、Middleware、HTTP 客户端和应用网关都可能有自己的超时或重试机制，应避免多个层级叠加导致请求长时间无法结束。

### 4、限制 Agent 执行次数

使用 ModelCallLimitMiddleware、ToolCallLimitMiddleware 及 recursion\_limit，防止无限工具循环。

### 5、保留请求追踪信息

建议使用请求 ID 将模型日志、工具日志和外部服务日志关联起来。

### 6、保护敏感数据

Debug 日志和 LangSmith 追踪可能包含用户消息与工具参数。正式环境应对凭据和隐私信息进行脱敏。

### 7、区分技术错误和业务失败

例如"订单不存在"是正常业务结果，不应该自动重试；数据库连接中断则属于技术异常。

### 8、对失败结果提供明确反馈

如果工具没有执行成功，不能让模型声称已经完成。必要时应将状态明确标记为失败或需要人工处理。

### 9、建立回归测试

将真实遇到的错误转为测试案例，例如工具参数错误、API 超时、没有检索结果和重复工具调用。

### 10、不要依赖模型自行修复所有错误

业务校验、数据库事务、授权控制和异常恢复应该由可信代码负责，而不是只依赖 System Prompt。

---

## 二十五、常见问题

### 1、ModelRetryMiddleware 和 ToolRetryMiddleware 有什么区别？

前者处理模型调用失败，后者处理工具执行失败。

### 2、max\_retries=3 表示最多调用三次吗？

不是。它表示首次调用失败后，最多额外重试三次，因此总计最多四次尝试。

### 3、backoff\_factor=2 是否一定等待 2、4、8 秒？

不是。实际延迟还取决于 initial\_delay、max\_delay 和 jitter。

### 4、为什么设置了模型重试，仍然立即报错？

可能是异常类型不符合 retry\_on 条件，或者错误发生在模型调用之外。

### 5、ToolRetryMiddleware 可以用于发送邮件吗？

不建议直接对未做幂等保护的发送邮件操作自动重试，否则可能重复发送。

### 6、ToolErrorMiddleware 会自动重试吗？

不会。它主要用于拦截指定工具错误并转换失败结果。

### 7、debug=True 和 stream\_mode="debug" 有什么区别？

debug=True 让图执行输出更详细的调试日志；stream\_mode="debug" 通过迭代流获取结构化 Debug 事件。

### 8、如何判断模型是否调用了正确工具？

可以检查 AIMessage.tool\_calls、ToolMessage，或者通过 LangSmith 查看调用轨迹。

### 9、GraphRecursionError 应如何处理？

应检查工具重复调用、模型停止条件和图路由，并设置合理的执行限制。

### 10、为什么 Agent 总是重复调用工具？

可能是工具结果不清晰、Prompt 存在冲突、路由逻辑错误，或者模型无法判断任务是否完成。

### 11、Pydantic 能防止所有工具参数错误吗？

不能。它主要验证数据格式和字段约束，无法自动保证参数对应的业务对象真实存在或有权限访问。

### 12、SummarizationMiddleware 能防止所有 Token 超限吗？

不能。仍需要结合模型上下文限制、消息数量、工具输出长度和实际 Token 使用情况进行管理。

### 13、为什么同一个 thread\_id 下仍然没有历史？

需要检查 Checkpointer 是否已配置、存储是否持续存在，以及历史消息是否被清理或覆盖。

### 14、为什么 Agent 没有抛异常，却返回了失败消息？

可能是 Middleware 将异常转换成了 AIMessage 或 ToolMessage，应用应额外检查业务结果。

### 15、是否应该在生产环境开启 debug=True？

通常不建议长期开启完整调试，应优先使用结构化日志、采样追踪和可控的错误监控。

---

## 总结

错误处理与调试是构建可靠 LangChain Agent 的重要环节。

本章节介绍了：

- ImportError、API Key、超时和上下文超限等常见问题。
- `ModelRetryMiddleware`：模型调用重试。
- `ToolRetryMiddleware`：工具执行重试。
- `ToolErrorMiddleware`：工具错误处理。
- 指数退避和重试次数控制。
- `debug=True`：详细执行日志。
- `stream_mode="updates"`：观察节点更新。
- `stream_mode="debug"`：检查图调试事件。
- `ToolCallLimitMiddleware`：限制工具调用次数。
- `ModelCallLimitMiddleware`：限制模型调用次数。
- `GraphRecursionError`：排查过多执行步骤。
- `SummarizationMiddleware`：管理长对话历史。
- Python logging 和 LangSmith 可观测性。

一个实用的错误处理架构可以概括为：

```
用户请求

    ↓

输入验证

    ↓

Agent 执行

    ↓

模型或工具调用

    ↓

错误分类

    |

    ├── 临时故障 → 有限重试

    ├── 参数错误 → 返回校验信息

    ├── 权限错误 → 立即拒绝

    ├── 业务失败 → 返回业务状态

    └── 严重异常 → 终止并记录日志

    ↓

记录执行信息

    ↓

返回结果或失败提示
```

**核心原则：**可靠的 Agent 不只是能够自动重试，而是能够正确判断什么错误值得重试、什么错误必须立即停止，以及如何避免重复执行真实业务操作。

对于正式应用，应将 Middleware、工具参数校验、执行次数限制、结构化日志和 LangSmith 追踪结合起来，建立可诊断、可恢复且安全的错误处理体系。
