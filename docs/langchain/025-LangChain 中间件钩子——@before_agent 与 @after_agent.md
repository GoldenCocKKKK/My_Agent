# LangChain 中间件钩子——@before\_agent 与 @after\_agent

> 来源: https://www.runoob.com/langchain/langchain-before-after-agent.html

---

# LangChain 中间件钩子——@before\_agent 与 @after\_agent

在 LangChain Agent 中，@before\_agent 和 @after\_agent 是两个 Agent 级别的 Middleware 钩子。

它们分别在 Agent 主执行流程开始前和正常完成后运行，适合处理输入预处理、状态初始化、访问检查、结果格式化以及执行统计等任务。

与每次模型调用前后执行的 `@before_model` 和 `@after_model` 不同，这两个钩子作用于整个 Agent 的生命周期。

本章节基于 LangChain 1.x，介绍两个钩子的基本用法、状态更新、执行控制以及实际应用。

---

## 一、before\_agent 与 after\_agent 的区别

一个简单 Agent 的执行流程可以理解为：

```
用户输入

    ↓

before_agent

    ↓

before_model

    ↓

模型调用

    ↓

after_model

    ↓

是否调用工具？

    ├── 是 → 执行工具 → 再次调用模型

    └── 否 → 继续

    ↓

after_agent

    ↓

返回最终结果
```

其中，before\_agent 和 after\_agent 主要控制 Agent 整体执行的开始与结束阶段。

| 对比项 | before\_agent | after\_agent |
| --- | --- | --- |
| 执行时机 | Agent 主流程开始前 | Agent 正常完成后 |
| 正常运行次数 | 通常一次 | 通常一次 |
| 主要用途 | 输入检查、状态初始化 | 结果整理、统计、结束日志 |
| 状态更新 | 支持 | 支持 |
| 流程控制 | 支持配置 can\_jump\_to | 支持配置 can\_jump\_to |

**注意：**如果 Agent 在执行期间发生异常、中断或者通过 Checkpointer 恢复，实际钩子执行情况可能与正常完整调用不同。

---

## 二、两个钩子的基本语法

可以通过装饰器快速定义 Agent 生命周期钩子。

## 实例

from langchain.agents.middleware import before\_agent, after\_agent, AgentState
from langgraph.runtime import Runtime

@before\_agent
def start\_hook(state: AgentState, runtime: Runtime) -> None:
    """Agent 开始前执行。"""
    print("Agent 开始执行")

@after\_agent
def finish\_hook(state: AgentState, runtime: Runtime) -> None:
    """Agent 正常完成后执行。"""
    print("Agent 执行完成")

使用时加入 `middleware`：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[],

    middleware=[start_hook, finish_hook]

)
```

两个钩子接收的主要参数：

- `state`：当前 Agent State，可以读取消息历史和自定义状态。
- `runtime`：当前运行环境，可以读取应用传入的 Context 等信息。

如果不需要修改状态，可以返回 `None`，或者不显式返回。

---

## 三、@before\_agent：输入预处理

`@before_agent` 可以在 Agent 执行前规范化用户输入。

例如，移除问题前后的多余空格，将全角空格转换为普通空格。

需要注意，AgentState 中的 `messages` 使用 `add_messages` Reducer，不能直接返回一条全新的消息就认为原来的消息已经被替换。

正确方法是复用原消息 ID。

### 实例：规范化用户消息

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import before\_agent, AgentState
from langchain.messages import HumanMessage
from langgraph.runtime import Runtime

load\_dotenv()

@before\_agent
def preprocess\_input(state: AgentState, runtime: Runtime) -> dict | None:
    """规范化最近一条用户消息。"""
    users = [msg for msg in state.get("messages", []) if msg.type == "human"]
    if not users:
        return None

    last = users[-1]
    if not isinstance(last.content, str):
        return None

    cleaned = last.content.replace("　", " ").strip()
    if cleaned == last.content:
        return None

    return {"messages": [HumanMessage(id=last.id, content=cleaned)]}

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[preprocess\_input],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程助手。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "　 Python 是什么？  "}]
})
print(result["messages"][-1].content)

在这个示例中，原始用户输入为：

```
　 Python 是什么？
```

经过预处理后，消息内容变成：

```
Python 是什么？
```

模型收到的就是规范化后的问题。

**关键点：**`HumanMessage(id=last.id, ...)` 保留了原消息 ID，因此 add\_messages 可以替换对应的消息，而不是追加重复的用户消息。

这个示例只处理纯文本，不直接修改图片、音频等多模态内容。

---

## 四、@before\_agent：初始化自定义状态

对于需要维护业务状态的 Agent，可以在 before\_agent 中初始化缺失字段。

例如，学习助手需要保存当前学习等级和学习目标。

### 实例：初始化学习状态

## 实例

from typing\_extensions import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import before\_agent
from langgraph.runtime import Runtime

class LearningState(AgentState):
    user\_level: NotRequired[str]
    current\_goal: NotRequired[str]

@before\_agent(state\_schema=LearningState)
def init\_learning\_state(state: LearningState, runtime: Runtime) -> dict | None:
    """为缺失的学习字段设置默认值。"""
    updates = {}
    if "user\_level" not in state:
        updates["user\_level"] = "入门"
    if "current\_goal" not in state:
        updates["current\_goal"] = "学习 Python"
    return updates or None

假设输入状态只有：

```
{

    "messages": [...]

}
```

中间件可以添加：

```
{

    "user_level": "入门",

    "current_goal": "学习 Python"

}
```

这里使用自定义 `LearningState` 声明扩展字段。

**注意：**自定义字段不在 AgentState 默认结构中，需要通过 `state_schema` 等受支持的方式声明。

如果用户已经提供了相关字段，示例不会覆盖原来的值。

---

## 五、@before\_agent：访问控制

在某些应用中，Agent 执行前需要检查用户是否可以使用特定功能。

例如，普通访客可以查询课程，但不能调用管理功能。

可以利用 before\_agent 在流程开始时进行初步检查。

不过，真正的用户身份和权限必须由可信后端确认，不能依赖模型自行判断，也不能只靠用户消息中的关键词。

### 实例：根据运行时角色拦截请求

这里使用 Context 模拟后端已经确认的角色。

## 实例

from dataclasses import dataclass
from langchain.agents import create\_agent
from langchain.agents.middleware import before\_agent, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

@dataclass
class UserContext:
    role: str = "guest"
    allow\_admin: bool = False

@before\_agent(can\_jump\_to=["end"])
def access\_control(state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
    """演示基于可信上下文的访问检查。"""
    ctx = runtime.context
    users = [m for m in state.get("messages", []) if m.type == "human"]
    if not users:
        return None

    text = users[-1].content
    if isinstance(text, str) and text.startswith("/admin"):
        if not ctx.allow\_admin:
            return {
                "jump\_to": "end",
                "messages": [AIMessage(content="当前账户没有管理功能权限。")]
            }
    return None

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    context\_schema=UserContext,
    middleware=[access\_control]
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "/admin 删除课程"}]},
    context=UserContext(role="guest", allow\_admin=False)
)
print(result["messages"][-1].content)

输出：

```
当前账户没有管理功能权限。
```

这里的 `/admin` 只是演示用的命令前缀，不是完整的安全控制。

**重要：**真实管理工具仍然必须在工具执行层或后端 API 中进行权限校验。

即使 before\_agent 拦截了部分请求，也不能据此认为用户无法通过其他提示词触发受限工具。

对于严格的访问控制，更可靠的设计是：

- 先由应用后端完成用户认证。
- 将可信权限信息传给 Agent Runtime。
- 根据权限限制可用工具。
- 在每个有副作用的工具中再次检查权限。

---

## 六、can\_jump\_to：提前结束 Agent

`before_agent` 可以配置 `can_jump_to`，声明允许的流程跳转目标。

例如：

```
@before_agent(can_jump_to=["end"])

def check_request(state, runtime):

    ...
```

如果输入不满足业务规则，可以返回：

```
{

    "jump_to": "end",

    "messages": [AIMessage(content="请求不符合要求。")]

}
```

这样可以在不正常调用模型的情况下提前结束当前执行流程。

支持的跳转目标包括：

| 目标 | 说明 |
| --- | --- |
| `"end"` | 结束主执行流程 |
| `"model"` | 进入模型执行路径 |
| `"tools"` | 进入工具节点，需要有效的工具调用信息 |

通常情况下，before\_agent 最常见的跳转场景是 `"end"`。

**注意：**不应该认为设置 `jump_to="tools"` 就可以直接执行任意工具。工具节点仍然需要合法的工具调用上下文。

---

## 七、@after\_agent：执行完成后统计

`@after_agent` 在 Agent 正常完成执行时运行，可以对消息历史进行统计。

例如：

- 消息总数。
- 用户消息数量。
- 工具调用请求数量。
- 工具结果消息数量。

需要注意，消息数量不等于模型 API 调用次数。

### 实例：统计执行结果

## 实例

from langchain.agents.middleware import after\_agent, AgentState
from langgraph.runtime import Runtime

@after\_agent
def conversation\_stats(state: AgentState, runtime: Runtime) -> None:
    """输出本次 Agent 的消息统计。"""
    messages = state.get("messages", [])
    humans = sum(m.type == "human" for m in messages)
    tool\_requests = sum(len(getattr(m, "tool\_calls", [])) for m in messages)
    tool\_results = sum(m.type == "tool" for m in messages)

    print("消息总数：", len(messages))
    print("用户消息：", humans)
    print("工具请求：", tool\_requests)
    print("工具结果：", tool\_results)

如果 Agent 进行了一次工具调用，可能得到：

```
消息总数： 4

用户消息： 1

工具请求： 1

工具结果： 1
```

实际数据取决于 Agent 的执行路径。

如果 Agent 使用 Checkpointer 保存多轮历史，这里统计的可能是当前 State 中的累计消息，而不是仅本次新增的消息。

如果需要精确统计单次运行的模型 API 调用次数，建议使用专门的 Middleware 计数器或模型调用追踪数据。

---

## 八、通过自定义状态保存统计结果

除了直接打印日志，还可以把统计结果写入 Agent State。

这样调用方就可以从最终结果中读取统计信息。

### 实例：返回自定义统计字段

## 实例

from typing\_extensions import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import after\_agent
from langgraph.runtime import Runtime

class StatsState(AgentState):
    tool\_result\_count: NotRequired[int]
    message\_count: NotRequired[int]

@after\_agent(state\_schema=StatsState)
def save\_stats(state: StatsState, runtime: Runtime) -> dict:
    """将当前消息统计写入 State。"""
    messages = state.get("messages", [])
    return {
        "message\_count": len(messages),
        "tool\_result\_count": sum(m.type == "tool" for m in messages)
    }

Agent 正常执行完成后，可以读取：

```
result["message_count"]

result["tool_result_count"]
```

这种方式适合将统计数据返回应用程序。

但应注意，它们是当前消息状态的统计值，不一定等于本次调用的独立计数。

---

## 九、@after\_agent：格式化最终输出

除了记录统计数据，还可以通过 after\_agent 统一调整最终回答。

例如，自动追加 AI 内容声明。

但是，原文中通过新增 AIMessage 的方式可能导致重复回答。

因为 messages 使用 `add_messages` Reducer，要替换已有消息，应保留原来的消息 ID。

### 实例：为最终回复添加页脚

## 实例

from langchain.agents.middleware import after\_agent, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

FOOTER = "\n\n---\n由菜鸟教程 RUNOOB AI 助手提供，内容仅供参考。"

@after\_agent
def format\_output(state: AgentState, runtime: Runtime) -> dict | None:
    """为最终文本回复添加说明。"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last = messages[-1]
    if last.type != "ai" or getattr(last, "tool\_calls", []):
        return None
    if not isinstance(last.content, str) or not last.content or FOOTER in last.content:
        return None

    updated = last.model\_copy(update={"content": last.content + FOOTER})
    return {"messages": [updated]}

如果模型原始回答为：

```
Python 语法简洁，适合初学者学习。
```

最终返回的内容会变成：

```
Python 语法简洁，适合初学者学习。



---

由菜鸟教程 RUNOOB AI 助手提供，内容仅供参考。
```

这里使用 `model_copy(update=...)` 保留了原始 AIMessage 的 ID 和其他消息字段，并只更新内容。

因此，add\_messages 会替换对应的旧回复，而不是追加一条全新的回复。

**注意：**如果使用 Streaming 实时推送模型文本，那么已发送到浏览器的文本不会因为 after\_agent 修改了最终 State 而自动更新。

如果需要显示格式化后的最终内容，可以在执行结束后发送最终结果事件，或者在前端单独渲染固定页脚。

---

## 十、after\_agent 与自定义流式事件

如果希望将统计信息实时推送给前端，可以在支持的运行时上下文中发送 Custom Streaming 事件。

需要通过 `stream_mode="custom"` 接收，并根据 LangGraph 版本使用相应的事件结构。

### 实例：发送完成统计事件

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import after\_agent, AgentState
from langgraph.runtime import Runtime

@after\_agent
def stream\_stats(state: AgentState, runtime: Runtime) -> None:
    """发送 Agent 完成后的统计事件。"""
    if runtime.stream\_writer:
        messages = state.get("messages", [])
        runtime.stream\_writer({
            "type": "stats",
            "messages": len(messages),
            "tool\_results": sum(m.type == "tool" for m in messages)
        })

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[stream\_stats]
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": "你好"}]},
    stream\_mode="custom",
    version="v2"
):
    if event["type"] == "custom" and event["data"].get("type") == "stats":
        print("统计事件：", event["data"])

正常执行完成后，可能收到：

```
统计事件： {'type': 'stats', 'messages': 2, 'tool_results': 0}
```

这里使用的是 `version="v2"` 流式格式，事件通过以下字段读取：

```
event["type"]

event["ns"]

event["data"]
```

这与旧版直接解包 `(mode, chunk)` 的格式不同。

**注意：**StreamWriter 的可用性依赖执行上下文。如果只是需要在普通 `invoke()` 完成后获取统计信息，可以直接把统计结果写入 State，不必使用 Streaming。

---

## 十一、完整实例：四个钩子协作

下面结合 before\_agent、before\_model、after\_model 和 after\_agent，创建一个简单的 Agent 生命周期演示。

示例中的天气工具返回模拟数据，不是真实天气。

文件名：`agent_lifecycle_demo.py`

安装依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

在 `.env` 中配置：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import before\_agent, after\_agent, before\_model, after\_model, AgentState
from langchain.tools import tool
from langgraph.runtime import Runtime

load\_dotenv()

@before\_agent
def start\_session(state: AgentState, runtime: Runtime) -> None:
    print(">>> Agent 开始")

@before\_model
def before\_call(state: AgentState, runtime: Runtime) -> None:
    print("[模型调用前] 消息数：", len(state.get("messages", [])))

@after\_model
def after\_call(state: AgentState, runtime: Runtime) -> None:
    last = state["messages"][-1]
    names = [c["name"] for c in getattr(last, "tool\_calls", [])]
    print("[模型调用后] 工具请求：", names or "无")

@after\_agent
def finish\_session(state: AgentState, runtime: Runtime) -> None:
    messages = state.get("messages", [])
    print("<<< Agent 完成，消息总数：", len(messages))

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。"""
    weather = {"杭州": "晴，25°C", "北京": "多云，18°C"}
    return weather.get(city, "没有该城市的模拟天气数据")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[get\_weather],
    middleware=[start\_session, before\_call, after\_call, finish\_session],
    system\_prompt="你是中文助手。天气信息必须使用工具查询，并明确说明工具数据是模拟结果。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请用工具查询杭州的模拟天气。"}]
})
print("AI：", result["messages"][-1].content)

### 可能的运行结果

```
>>> Agent 开始

[模型调用前] 消息数： 1

[模型调用后] 工具请求： ['get_weather']

[模型调用前] 消息数： 3

[模型调用后] 工具请求： 无

<<< Agent 完成，消息总数： 4



AI：杭州的模拟天气为晴天，25°C。
```

这个示例展示了四个钩子的职责：

- `before_agent`：整个 Agent 执行开始时运行。
- `before_model`：每次模型调用前运行。
- `after_model`：每次模型调用后运行。
- `after_agent`：Agent 正常完成时运行。

如果模型没有调用工具，那么 before\_model 和 after\_model 可能只执行一次。

如果模型连续调用多个工具，钩子的实际执行次数也可能增加。

---

## 十二、before\_agent 与 before\_model 如何选择？

两个钩子都可以在模型调用之前运行，但它们的作用范围不同。

| 需求 | 推荐钩子 | 原因 |
| --- | --- | --- |
| 检查本次 Agent 的初始输入 | before\_agent | 位于主执行流程开始前 |
| 初始化会话状态 | before\_agent | 避免在每次模型调用前重复初始化 |
| 检查每次模型请求 | before\_model | Agent 可能多次调用模型 |
| 根据当前消息控制下一次模型调用 | before\_model | 适合循环过程中的动态检查 |
| 为本次模型请求临时修改输入 | wrap\_model\_call | 可以修改 ModelRequest |

可以简单理解：

```
before_agent：整个 Agent 运行前的检查



before_model：每次模型调用前的检查
```

---

## 十三、after\_agent 与 after\_model 如何选择？

两个钩子都能在模型产生响应后运行，但它们的执行阶段不同。

| 需求 | 推荐钩子 | 原因 |
| --- | --- | --- |
| 检查每次模型返回 | after\_model | 模型响应后立即执行 |
| 判断模型是否请求工具 | after\_model | 可直接检查 tool\_calls |
| 统计完整 Agent 的消息历史 | after\_agent | 正常完成时获取最终状态 |
| 统一格式化最终回复 | after\_agent | 适合最终结果处理 |
| 执行异常时释放外部资源 | try/finally 等机制 | 不能保证 after\_agent 在异常时运行 |

对于需要审核每一次模型输出的场景，应优先考虑 after\_model。

对于最终统计、添加说明等场景，则更适合 after\_agent。

---

## 十四、六个 Middleware 钩子速查表

| 钩子 | 执行范围 | 主要用途 | 关键机制 |
| --- | --- | --- | --- |
| `before_agent` | Agent 开始阶段 | 输入检查、初始化 | State 更新、路由控制 |
| `before_model` | 每次模型调用前 | 消息管理、流程检查 | State 更新、jump\_to |
| `wrap_model_call` | 模型请求期间 | 重试、降级、请求修改 | handler、ModelRequest |
| `after_model` | 每次模型响应后 | 审核、状态更新 | State Reducer |
| `wrap_tool_call` | 工具执行期间 | 工具重试、缓存、参数修改 | handler、ToolCallRequest |
| `after_agent` | Agent 结束阶段 | 最终统计、结果整理 | State 更新 |

六个钩子可分为两类：

**Node-style：**`before_agent`、`before_model`、`after_model`、`after_agent`。

**Wrap-style：**`wrap_model_call`、`wrap_tool_call`。

Node-style 主要负责状态和流程，Wrap-style 主要负责包裹模型或工具执行。

---

## 十五、生产环境注意事项

### 1、不要只在 before\_agent 中检查工具权限

before\_agent 可以进行初步访问检查，但真实工具还应在执行时通过可信后端验证身份和权限。

### 2、不要重复初始化已有状态

如果使用 Checkpointer 保存多轮对话，before\_agent 应检查字段是否已经存在，避免每次调用覆盖此前保存的数据。

### 3、不要将 after\_agent 当作异常清理钩子

Agent 发生异常或中断时，after\_agent 不一定执行。

对于数据库连接、文件句柄等资源，应使用可靠的上下文管理或 finally 机制。

### 4、不要将消息数量当作 API 调用次数

消息历史可能包含用户消息、AIMessage、ToolMessage，以及多轮会话中的历史内容。

需要准确统计模型请求次数时，应使用模型调用监控或专门的状态计数。

### 5、避免在日志中泄露敏感信息

Middleware 能够访问消息、状态和运行时 Context，日志应避免记录 API Key、密码、访问令牌或其他敏感数据。

### 6、注意流式输出与最终状态的区别

如果前端已经收到流式文本，after\_agent 修改 State 不会自动撤回这些内容。

对于最终格式化结果，可以在流结束后发送完整结果事件，或在前端进行统一展示。

---

## 十六、常见问题

### 1、before\_agent 和 after\_agent 一定各执行一次吗？

正常完整运行时通常各执行一次，但异常、中断、恢复等情况可能影响执行路径。

### 2、before\_agent 可以修改用户输入吗？

可以。可以通过返回状态字典更新消息，并复用原消息 ID 进行替换。

### 3、为什么直接返回新的 HumanMessage 会增加消息数量？

因为 messages 使用 add\_messages Reducer。新消息 ID 不同，通常会被追加。

### 4、如何在 before\_agent 中初始化字段？

可以扩展 AgentState，并在 before\_agent 返回包含默认值的状态更新字典。

### 5、before\_agent 可以提前结束 Agent 吗？

可以。在支持的情况下通过 `can_jump_to=["end"]` 声明结束目标，并返回相应路由指令。

### 6、可以使用 runtime.context 进行权限检查吗？

可以，但 Context 中的身份与权限信息必须由可信应用后端提供。

### 7、after\_agent 可以修改最终回复吗？

可以。对于普通文本 AIMessage，可以复用原消息 ID，提交新的消息内容，让 Reducer 替换旧回复。

### 8、after\_agent 可以发送自定义 Streaming 事件吗？

可以在支持的流式运行上下文中使用 StreamWriter，并通过 custom 模式接收。

### 9、为什么 after\_agent 统计的消息数比本轮多？

如果使用 Checkpointer 保存历史，当前 State 可能包含此前多轮对话的消息。

### 10、after\_agent 可以处理模型调用失败吗？

不能将它视为通用异常捕获机制。模型请求异常应通过对应的包装中间件或上层错误处理完成。

### 11、什么时候应该使用 before\_model 而不是 before\_agent？

当逻辑需要在每次模型调用前运行，而不是仅在整个 Agent 开始阶段执行时，应使用 before\_model。

### 12、什么时候应该使用 after\_model 而不是 after\_agent？

当需要审核每次模型响应、检查工具调用或在执行循环中处理状态时，应使用 after\_model。

---

## 总结

`@before_agent` 和 `@after_agent` 是 LangChain Agent 生命周期的两个重要钩子。

本章节介绍了：

- Agent 级别钩子的执行时机。
- 通过 before\_agent 规范化输入。
- 初始化自定义 AgentState。
- 使用可信 Context 进行初步访问检查。
- 通过 can\_jump\_to 提前结束执行。
- 通过 after\_agent 统计消息和工具执行结果。
- 保存自定义统计字段。
- 保留消息 ID，正确修改最终回答。
- 使用 Custom Streaming 推送完成事件。
- 六个 Middleware 钩子的职责划分。

可以简单记住：

```
before_agent：

    Agent 开始之前

    输入预处理、初始化、访问检查



after_agent：

    Agent 正常完成之后

    结果格式化、统计、结束日志
```

> **核心原则：**Agent 级别的准备工作放在 before\_agent，最终处理放在 after\_agent；每次模型调用都需要执行的逻辑，使用 before\_model 和 after\_model。权限验证与资源管理等关键业务，不能仅依赖 Middleware 生命周期保证。
