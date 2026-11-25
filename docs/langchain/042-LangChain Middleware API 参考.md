# LangChain Middleware API 参考

> 来源: https://www.runoob.com/langchain/langchain-middleware-api.html

---

# LangChain Middleware API 参考

在 LangChain 1.x 中，Middleware（中间件） 是扩展 Agent 行为的重要机制。

通过 Middleware，开发者可以在模型调用之前或之后执行自定义代码，控制工具调用、调整系统提示词、记录日志、处理异常，甚至修改 Agent 的执行流程。

例如，一个智能客服 Agent 可以通过 Middleware 实现：

- 模型调用前检查用户输入。
- 根据用户身份动态调整可用工具。
- 为模型调用设置重试或备用模型。
- 记录模型与工具的调用耗时。
- 执行敏感操作时检查权限。
- 在 Agent 完成后处理最终回答。

本章节基于 LangChain 1.x，整理 `AgentMiddleware` 基类、全部主要 Hook、装饰器参数、ModelRequest、ModelResponse、ExtendedModelResponse，以及 Middleware 的执行顺序和状态更新方式。

---

## 一、Middleware 的工作原理

一个普通 LangChain Agent 的执行过程如下：

```
用户消息

    ↓

调用模型

    ↓

是否需要工具？

    ├── 否 → 返回回答

    └── 是 → 执行工具 → 再次调用模型
```

加入 Middleware 后，流程变成：

```
用户消息

    ↓

before_agent

    ↓

before_model

    ↓

wrap_model_call

    ↓

调用模型

    ↓

after_model

    ↓

是否需要工具？

    ├── 是

    │    ↓

    │  wrap_tool_call

    │    ↓

    │  执行工具

    │    ↓

    │  返回 before_model

    │

    └── 否

         ↓

      after_agent

         ↓

       返回结果
```

这是一幅简化的执行流程图，实际执行还会受到 Middleware 跳转、中断、结构化输出和错误处理等配置的影响。

Middleware 主要有两种类型：

| 类型 | 作用 | 常用场景 |
| --- | --- | --- |
| Node-style Hooks | 在执行节点前后运行逻辑 | 检查、日志、状态更新 |
| Wrap-style Hooks | 包裹模型或工具调用 | 重试、异常处理、修改请求 |

---

## 二、AgentMiddleware 基类

`AgentMiddleware` 是自定义 Middleware 的基类。

开发者可以继承它，并重写需要使用的 Hook。

```
from langchain.agents.middleware import AgentMiddleware



class MyMiddleware(AgentMiddleware):

    def before_model(self, state, runtime):

        return None
```

一个 Middleware 不需要实现全部 Hook。只需要实现实际使用的方法。

### AgentMiddleware 主要属性

| 属性 | 说明 |
| --- | --- |
| `name` | Middleware 的名称 |
| `state_schema` | 声明额外状态字段 |
| `tools` | 随 Middleware 注册的工具 |
| `transformers` | 额外的流事件转换器 |
| `trace_policy` | 控制 Middleware Hook 的追踪数据记录策略 |

其中 transformers 和 trace\_policy 属于较新的扩展能力，使用时需要注意实际安装的 LangChain 版本。

---

## 三、AgentMiddleware 全部主要 Hook

AgentMiddleware 提供同步和异步 Hook。

| Hook | 函数参数 | 执行时机 | 返回值 |
| --- | --- | --- | --- |
| `before_agent` | state, runtime | Agent 开始前 | dict / Command / None |
| `abefore_agent` | state, runtime | 异步开始前 | dict / Command / None |
| `before_model` | state, runtime | 每轮模型调用前 | dict / Command / None |
| `abefore_model` | state, runtime | 异步模型调用前 | dict / Command / None |
| `wrap_model_call` | request, handler | 包裹模型调用 | ModelCallResult |
| `awrap_model_call` | request, handler | 包裹异步模型调用 | ModelCallResult |
| `after_model` | state, runtime | 每轮模型回复后 | dict / Command / None |
| `aafter_model` | state, runtime | 异步模型回复后 | dict / Command / None |
| `wrap_tool_call` | request, handler | 包裹工具调用 | ToolMessage / Command |
| `awrap_tool_call` | request, handler | 包裹异步工具调用 | ToolMessage / Command |
| `after_agent` | state, runtime | Agent 正常结束后 | dict / Command / None |
| `aafter_agent` | state, runtime | 异步 Agent 结束后 | dict / Command / None |

其中：

- `before_agent` 和 `after_agent` 位于 Agent 执行周期的外围。
- `before_model` 和 `after_model` 随模型调用循环执行，可能运行多次。
- `wrap_model_call` 包裹模型请求。
- `wrap_tool_call` 包裹每次工具执行。

**注意：**同步 Hook 与异步 Hook 需要根据执行方式正确实现。一个自定义 Middleware 如果只实现同步逻辑，不应直接假设所有异步操作都能自动适配。

---

## 四、Middleware 装饰器一览

除了继承 AgentMiddleware，也可以直接使用装饰器创建 Middleware。

装饰器更适合逻辑比较简单、只需要单个 Hook 的场景。

| 装饰器 | 主要配置参数 | 作用 |
| --- | --- | --- |
| `@before_agent` | state\_schema, tools, can\_jump\_to, name | Agent 开始前执行 |
| `@after_agent` | state\_schema, tools, can\_jump\_to, name | Agent 结束后执行 |
| `@before_model` | state\_schema, tools, can\_jump\_to, name | 模型调用前执行 |
| `@after_model` | state\_schema, tools, can\_jump\_to, name | 模型调用后执行 |
| `@wrap_model_call` | state\_schema, tools, name | 包裹模型调用 |
| `@wrap_tool_call` | state\_schema, tools, name | 包裹工具调用 |
| `@dynamic_prompt` | 动态提示词相关配置 | 生成当前模型请求的系统提示词 |
| `@hook_config` | can\_jump\_to | 配置类式 Hook 的跳转行为 |

常见导入方式：

```
from langchain.agents.middleware import (

    AgentMiddleware,

    before_agent,

    after_agent,

    before_model,

    after_model,

    wrap_model_call,

    wrap_tool_call,

    dynamic_prompt,

    hook_config

)
```

其中，装饰器参数的支持范围可能随着版本更新而扩展。表中列出了日常开发中最需要掌握的配置。

---

## 五、before\_agent：Agent 开始前执行

`before_agent` 用于在 Agent 运行开始时执行预处理逻辑。

常见用途：

- 初始化自定义状态。
- 记录请求开始信息。
- 检查应用层面的输入条件。
- 准备本次执行所需的数据。

### 实例：记录 Agent 启动

## 实例

from langchain.agents.middleware import before\_agent

@before\_agent
def log\_agent\_start(state, runtime):
    """Agent 开始执行时打印日志。"""
    print("[Middleware] Agent 开始执行")
    print("当前消息数：", len(state.get("messages", [])))
    return None

返回 `None` 表示不修改 Agent State。

如果返回字典，则可以更新对应的状态字段。

**注意：**before\_agent 并不适合替代每轮模型调用前的检查。需要检查每一轮模型请求时，应该使用 before\_model。

---

## 六、before\_model：模型调用前执行

`before_model` 在每次模型调用前运行。

例如，Agent 执行三个模型调用周期，before\_model 就可能执行三次。

### 实例：统计消息数量

## 实例

from langchain.agents.middleware import before\_model

@before\_model
def log\_before\_model(state, runtime):
    """每次模型调用前打印当前消息数量。"""
    count = len(state.get("messages", []))
    print(f"[Middleware] 即将调用模型，消息数量：{count}")
    return None

before\_model 也可以根据条件提前结束执行。

### 实例：提前拦截测试指令

## 实例

from langchain.agents.middleware import before\_model
from langchain.messages import AIMessage

@before\_model(can\_jump\_to=["end"])
def block\_test\_command(state, runtime):
    """遇到测试命令时直接结束执行。"""
    users = [
        msg for msg in state.get("messages", [])
        if msg.type == "human"
    ]

    if users and users[-1].content == "BLOCK\_DEMO":
        return {
            "messages": [
                AIMessage(content="这是一条演示拦截结果。")
            ],
            "jump\_to": "end"
        }

    return None

这里的：

```
can_jump_to=["end"]
```

声明允许 Hook 提前结束执行。

**重要：**如果需要在 Hook 中使用 jump\_to，应明确配置允许的跳转目标。它不是一个可以任意指定节点名称的通用跳转接口。

---

## 七、after\_model：模型回复后执行

`after_model` 在每次模型调用获得响应后执行。

适合：

- 记录模型调用结果。
- 更新模型调用计数。
- 检查模型是否请求了工具。
- 根据模型回复决定是否继续执行。

### 实例：记录工具调用

## 实例

from langchain.agents.middleware import after\_model

@after\_model
def log\_model\_result(state, runtime):
    """检查模型最近一次回复。"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last = messages[-1]

    if last.type == "ai":
        print("[Middleware] 模型已返回")

        for call in getattr(last, "tool\_calls", []):
            print("请求工具：", call["name"])

    return None

需要注意，after\_model 可能多次触发，因此不适合无条件追加一次性客服签名。

如果需要在整个 Agent 正常结束后格式化最终回答，通常可以考虑 after\_agent。

---

## 八、after\_agent：Agent 完成后执行

`after_agent` 通常用于处理 Agent 正常执行完成后的状态。

例如：

- 记录任务完成信息。
- 格式化最终回答。
- 汇总执行结果。
- 更新完成状态。

### 实例：给最终回答添加标记

## 实例

from langchain.agents.middleware import after\_agent

SIGNATURE = "\n\n---\nRUNOOB AI 演示助手"

@after\_agent
def add\_signature(state, runtime):
    """为最终文本回复追加一次标记。"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last = messages[-1]

    if last.type != "ai":
        return None

    if getattr(last, "tool\_calls", []):
        return None

    if not isinstance(last.content, str):
        return None

    if SIGNATURE in last.content:
        return None

    updated = last.model\_copy(
        update={"content": last.content + SIGNATURE}
    )

    return {"messages": [updated]}

这里保留了 AIMessage 的原始 ID，因此在支持 add\_messages Reducer 的状态中，可以更新已有消息，而不是简单追加一条新的回答。

**注意：**after\_agent 不是通用的 finally 处理器。如果执行中发生未处理异常，不应假定它一定执行。需要无论成功或失败都清理资源时，应在应用层使用 try/finally 或上下文管理器。

---

## 九、wrap\_model\_call：包裹模型调用

`wrap_model_call` 是控制模型执行的核心 Hook。

它与 before\_model 的区别在于，wrap\_model\_call 能够直接控制底层模型处理器是否执行。

基本形式：

```
@wrap_model_call

def middleware(request, handler):

    return handler(request)
```

其中：

- `request`：本次模型请求的 ModelRequest。
- `handler`：真正执行下一层模型调用的处理函数。

Middleware 可以：

- 调用 handler 一次，正常执行。
- 在特定错误时调用多次，实现重试。
- 不调用 handler，直接返回自定义结果。
- 通过 request.override() 修改模型或请求内容。

### 实例：记录模型调用耗时

## 实例

import time
from langchain.agents.middleware import (
    wrap\_model\_call,
    ModelRequest,
    ModelResponse
)

@wrap\_model\_call
def measure\_model\_time(request: ModelRequest, handler):
    """统计每次模型调用的耗时。"""
    started = time.perf\_counter()

    try:
        response = handler(request)
        return response
    finally:
        elapsed = time.perf\_counter() - started
        print(f"[Middleware] 模型调用耗时：{elapsed:.3f} 秒")

这里的计时覆盖 handler(request) 的同步执行过程。

如果底层处理器抛出异常，finally 仍会打印耗时。

这类 Middleware 适合记录延迟和性能指标，但不要在生产日志中直接输出包含隐私的完整请求内容。

---

## 十、使用 request.override() 修改模型请求

`ModelRequest.override()` 会创建一个修改后的新请求对象。

原始 request 不会直接被修改。

例如，我们可以动态替换系统消息：

## 实例

from langchain.messages import SystemMessage
from langchain.agents.middleware import wrap\_model\_call

@wrap\_model\_call
def set\_custom\_prompt(request, handler):
    new\_request = request.override(
        system\_message=SystemMessage(
            content="你是 Python 编程老师，请简洁回答。"
        )
    )
    return handler(new\_request)

也可以修改模型：

```
new_request = request.override(

    model=different_model

)
```

或者调整允许使用的工具：

```
new_request = request.override(

    tools=[selected_tool]

)
```

**注意：**tools 只能使用已注册、可以由 Agent 执行的工具。不能简单将任意新工具对象加入 request.tools，就假定底层 ToolNode 自动具备执行能力。

---

## 十一、ModelRequest 关键属性

ModelRequest 封装了当前模型调用所需的信息。

| 属性 | 说明 |
| --- | --- |
| `model` | 当前使用的 BaseChatModel |
| `messages` | 当前模型请求中的消息列表，不包括单独的 system\_message |
| `system_message` | 当前系统消息，可能为空 |
| `tools` | 当前模型可见的工具定义列表 |
| `tool_choice` | 工具选择策略 |
| `response_format` | 结构化输出配置 |
| `model_settings` | 当前模型调用的其他配置 |
| `state` | Agent 当前状态 |
| `runtime` | 运行时上下文及相关能力 |

需要区分：

```
request.messages

request.state["messages"]
```

前者是当前要提供给模型的消息；后者属于 Agent 执行状态。

某些 Middleware 可以调整当前模型请求中的 messages，而不必永久修改持久化历史。

### ModelRequest.override() 支持的常见字段

| 字段 | 用途 |
| --- | --- |
| `model` | 切换模型 |
| `system_message` | 替换系统消息 |
| `messages` | 修改当前模型输入消息 |
| `tools` | 调整模型可见的工具 |
| `tool_choice` | 设置工具选择偏好 |
| `response_format` | 调整结构化输出配置 |
| `model_settings` | 传递模型调用设置 |
| `state` | 替换当前请求所用的状态视图 |

旧接口中的 `system_prompt` 覆盖方式已经不推荐，应该使用 `system_message`。

---

## 十二、dynamic\_prompt：动态系统提示词

对于只需要动态生成 System Prompt 的场景，没必要每次都自己编写 wrap\_model\_call。

LangChain 提供了更简单的：

```
@dynamic_prompt
```

它本质上是基于模型调用包装机制实现的便利装饰器。

### 实例：根据日期生成系统提示词

## 实例

from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def date\_prompt(request: ModelRequest) -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))

    return (
        "你是 Python 学习助手。\n"
        f"当前日期：{now.date().isoformat()}。\n"
        "回答尽量简洁，必要时提供代码。"
    )

dynamic\_prompt 返回的是系统提示词字符串，而不是状态更新字典。

它非常适合：

- 日期和时区提示。
- 根据用户级别调整回答风格。
- 根据业务场景生成角色规则。

相比直接修改 messages，dynamic\_prompt 不需要向聊天历史重复插入 SystemMessage。

---

## 十三、wrap\_tool\_call：包裹工具执行

`wrap_tool_call` 与 wrap\_model\_call 类似，但它拦截的是工具调用。

可以用于：

- 记录工具执行耗时。
- 捕获工具异常。
- 控制工具执行。
- 处理工具返回结果。

基本形式：

```
@wrap_tool_call

def middleware(request, handler):

    return handler(request)
```

### 实例：记录工具执行

## 实例

import time
from langchain.agents.middleware import wrap\_tool\_call

@wrap\_tool\_call
def log\_tool\_execution(request, handler):
    """记录工具名称和执行耗时。"""
    name = request.tool\_call["name"]
    started = time.perf\_counter()

    print("[开始执行工具]", name)

    try:
        return handler(request)
    finally:
        elapsed = time.perf\_counter() - started
        print(f"[工具执行结束] {name}，耗时 {elapsed:.3f} 秒")

工具调用请求可以通过：

```
request.tool_call
```

读取模型提出的工具名称、参数和调用 ID。

如果需要实现工具级异常处理，可以选择性捕获某些异常。

---

## 十四、wrap\_tool\_call：处理工具异常

例如，某个课程查询工具遇到临时错误时，可以将异常转成 ToolMessage。

### 实例：只处理指定异常

## 实例

from langchain.agents.middleware import wrap\_tool\_call
from langchain.messages import ToolMessage

@wrap\_tool\_call
def handle\_course\_error(request, handler):
    try:
        return handler(request)

    except ValueError:
        return ToolMessage(
            content="工具参数不正确，请检查输入。",
            tool\_call\_id=request.tool\_call["id"],
            name=request.tool\_call["name"],
            status="error"
        )

这里不会捕获全部异常，只转换 ValueError。

没有匹配的异常仍然会向上抛出，方便程序识别真正的系统故障。

**注意：**工具调用可能具有真实副作用。对于发送邮件、支付和删除数据等操作，不能简单依靠通用 Middleware 重试，否则可能导致重复执行。

如果只需要标准化处理工具错误，也可以使用内置的 ToolErrorMiddleware。

---

## 十五、ModelResponse 数据结构

`ModelResponse` 表示模型调用处理器返回的完整结果。

其核心结构是：

```
ModelResponse(

    result: list[BaseMessage],

    structured_response: Any | None = None

)
```

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `result` | list[BaseMessage] | 模型调用产生的消息 |
| `structured_response` | ResponseT | None | 已解析的结构化输出 |

通常 result 中会包含一个 AIMessage。

如果模型通过工具机制完成结构化输出，也可能出现额外的 ToolMessage。

### 实例：读取 ModelResponse

## 实例

from langchain.agents.middleware import wrap\_model\_call, ModelResponse

@wrap\_model\_call
def inspect\_model\_response(request, handler):
    response = handler(request)

    if isinstance(response, ModelResponse):
        print("返回消息数：", len(response.result))
        print("是否有结构化结果：", response.structured\_response is not None)

    return response

注意，Middleware 的模型返回类型还可能是 AIMessage 或 ExtendedModelResponse。

因此，编写需要兼容多种 Middleware 的通用代码时，不应假设响应始终只有一个固定类型。

---

## 十六、ModelCallResult 返回类型

LangChain 将模型调用 Hook 支持的返回类型概括为 `ModelCallResult`。

包含：

| 返回类型 | 作用 |
| --- | --- |
| `AIMessage` | 简化的模型回复 |
| `ModelResponse` | 完整模型调用结果 |
| `ExtendedModelResponse` | 模型回复加额外状态更新命令 |

例如，在一个简单的拦截器中，可以直接返回 AIMessage：

## 实例

from langchain.agents.middleware import wrap\_model\_call
from langchain.messages import AIMessage

@wrap\_model\_call
def demo\_short\_circuit(request, handler):
    """拦截固定测试消息。"""
    if (
        request.messages
        and request.messages[-1].type == "human"
        and request.messages[-1].content == "MOCK\_REPLY"
    ):
        return AIMessage(content="这是一条 Middleware 模拟回复。")

    return handler(request)

如果用户输入固定的 MOCK\_REPLY，Middleware 可以直接返回模拟的 AIMessage，而不调用真实模型。

但这种机制不能替代正式的用户授权、工具权限和安全防护。

---

## 十七、ExtendedModelResponse：模型结果与状态更新

`ExtendedModelResponse` 可以同时返回模型结果和额外的 Command 状态更新。

主要结构：

```
ExtendedModelResponse(

    model_response=...,

    command=Command(update={...})

)
```

| 属性 | 说明 |
| --- | --- |
| `model_response` | 底层 ModelResponse |
| `command` | 可选的状态更新 Command |

例如，我们希望在每次模型返回后记录本次模型的 Token 用量。

### 实例：更新模型调用统计

## 实例

from typing import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap\_model\_call,
    ExtendedModelResponse
)
from langgraph.types import Command

class UsageState(AgentState):
    last\_model\_tokens: NotRequired[int]

@wrap\_model\_call(state\_schema=UsageState)
def track\_token\_usage(request, handler):
    response = handler(request)

    tokens = 0
    for msg in response.result:
        usage = getattr(msg, "usage\_metadata", None)
        if usage:
            tokens += usage.get("total\_tokens", 0)

    return ExtendedModelResponse(
        model\_response=response,
        command=Command(
            update={"last\_model\_tokens": tokens}
        )
    )

这样，在模型返回后可以更新：

```
last_model_tokens
```

该字段保存本次模型调用统计，不等于整个会话的累计 Token 数。

**重要：**ExtendedModelResponse 中的 Command 当前主要用于附加状态更新。

不要把它与任意 LangGraph 路由跳转混为一谈。这里不支持通过 command 的 goto、resume、graph 等配置随意控制图执行。

---

## 十八、Node-style 与 Wrap-style 的状态更新区别

两种 Middleware 方式更新状态的机制不同。

| Hook 类型 | 状态更新方式 |
| --- | --- |
| before\_agent | 直接返回 dict 或支持的 Command |
| before\_model | 直接返回 dict 或支持的 Command |
| after\_model | 直接返回 dict 或支持的 Command |
| after\_agent | 直接返回 dict 或支持的 Command |
| wrap\_model\_call | 返回 ExtendedModelResponse 附带 Command |
| wrap\_tool\_call | 可以直接返回 Command |

Node-style 返回的更新会经过 LangGraph 对应字段的 Reducer。

例如，messages 通常使用 add\_messages Reducer。

因此：

```
return {"messages": [AIMessage(content="新消息")]}
```

通常表示追加新消息，而不是替换整个消息列表。

如需修改已有消息，应考虑复用原有消息 ID，或者使用符合 Reducer 语义的专门更新方式。

---

## 十九、Middleware 的执行顺序

当一个 Agent 注册多个 Middleware 时，执行顺序非常重要。

例如：

```
middleware = [

    middleware_a,

    middleware_b,

    middleware_c

]
```

### Before Hook

按列表顺序执行：

```
A.before_model

B.before_model

C.before_model
```

### Wrap Hook

采用嵌套结构：

```
A.wrap_model_call

    ↓

  B.wrap_model_call

      ↓

    C.wrap_model_call

        ↓

      模型执行

        ↑

    C 返回

      ↑

  B 返回

    ↑

A 返回
```

### After Hook

按相反顺序执行：

```
C.after_model

B.after_model

A.after_model
```

对应地，before\_agent 由前向后执行，after\_agent 则由后向前执行。

因此，列表中的第一个 Wrap Middleware 是最外层。

**建议：**需要集中处理其他 Middleware 异常的包装器可以考虑放在外层；但具体顺序仍应根据业务逻辑设计和测试。

---

## 二十、使用 hook\_config 配置跳转

如果通过类继承 AgentMiddleware，可以使用 `@hook_config` 配置允许的跳转目标。

例如：

## 实例

from langchain.agents.middleware import AgentMiddleware, hook\_config
from langchain.messages import AIMessage

class MessageGuardMiddleware(AgentMiddleware):

    @hook\_config(can\_jump\_to=["end"])
    def before\_model(self, state, runtime):
        users = [
            msg for msg in state.get("messages", [])
            if msg.type == "human"
        ]

        if users and users[-1].content == "STOP\_DEMO":
            return {
                "messages": [AIMessage(content="测试请求已拦截。")],
                "jump\_to": "end"
            }

        return None

主要跳转目标：

| 目标 | 作用 |
| --- | --- |
| `end` | 进入 Agent 结束流程 |
| `model` | 跳转到模型执行流程 |
| `tools` | 跳转到工具执行节点 |

注意，跳转到 tools 节点通常需要当前存在有效的工具调用请求。否则可能产生不合法的执行状态。

对于普通业务，最常用的仍然是：

```
can_jump_to=["end"]
```

用于在满足明确条件时提前终止。

---

## 二十一、类式 Middleware：同时实现多个 Hook

如果一个 Middleware 同时需要记录模型调用前后的信息，使用类会更方便。

### 实例：LoggingMiddleware

## 实例

import time
from langchain.agents.middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    """打印模型执行与工具调用的简单日志。"""

    def before\_agent(self, state, runtime):
        print("[Agent] 开始执行")
        return None

    def before\_model(self, state, runtime):
        print("[Model] 即将调用")
        return None

    def wrap\_model\_call(self, request, handler):
        started = time.perf\_counter()
        try:
            return handler(request)
        finally:
            print(
                "[Model] 耗时：",
                round(time.perf\_counter() - started, 3),
                "秒"
            )

    def wrap\_tool\_call(self, request, handler):
        name = request.tool\_call["name"]
        print("[Tool] 开始：", name)
        return handler(request)

    def after\_agent(self, state, runtime):
        print("[Agent] 执行完成")
        return None

注册到 Agent：

```
agent = create_agent(

    model="deepseek:deepseek-chat",

    tools=[my_tool],

    middleware=[LoggingMiddleware()]

)
```

类式 Middleware 特别适合跨项目复用，也可以在 \_\_init\_\_ 中提供自定义参数。

---

## 二十二、自定义 Middleware State

某些 Middleware 需要在多次模型调用之间保存计数器或业务状态。

这时可以扩展 AgentState。

### 实例：记录模型调用次数

## 实例

from typing import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import after\_model

class TrackingState(AgentState):
    model\_call\_count: NotRequired[int]

@after\_model(state\_schema=TrackingState)
def count\_model\_calls(state: TrackingState, runtime):
    """模型每次返回后更新计数。"""
    count = state.get("model\_call\_count", 0)
    return {"model\_call\_count": count + 1}

如果 Agent 执行了三次模型调用，正常情况下可以在最终 State 中看到：

```
model_call_count = 3
```

对于实际生产环境，更推荐使用专门的 ModelCallLimitMiddleware 控制模型调用上限，而不是完全依赖自己编写的计数器。

---

## 二十三、动态限制模型可见工具

有些工具只应对管理员开放。

可以使用 wrap\_model\_call 根据可信 Runtime Context 调整模型可见的工具列表。

例如，普通用户只能查询，管理员可以执行特定管理操作。

### 实例：工具可见性过滤

## 实例

from dataclasses import dataclass
from langchain.agents.middleware import wrap\_model\_call

@dataclass
class UserContext:
    is\_admin: bool = False

@wrap\_model\_call
def filter\_admin\_tools(request, handler):
    is\_admin = request.runtime.context.is\_admin

    if is\_admin:
        return handler(request)

    allowed = [
        tool for tool in request.tools
        if tool.name != "delete\_course"
    ]

    return handler(request.override(tools=allowed))

这里将 delete\_course 从普通用户的模型工具列表中移除。

**重要：**模型不可见某个工具不等于真正完成权限控制。

如果 delete\_course 会修改数据库，还必须在工具执行层根据可信用户身份进行权限验证。

运行时上下文也必须由可信后端提供，不能让浏览器随意提交 is\_admin=True。

---

## 二十四、内置 Middleware 速查

LangChain 还提供了一系列内置 Middleware，可以减少重复开发。

| Middleware | 主要用途 |
| --- | --- |
| `ModelRetryMiddleware` | 模型失败时有限重试 |
| `ToolRetryMiddleware` | 工具执行失败时有限重试 |
| `ToolErrorMiddleware` | 将指定工具异常转换为错误结果 |
| `ModelFallbackMiddleware` | 主要模型失败时尝试备用模型 |
| `ModelCallLimitMiddleware` | 限制模型调用次数 |
| `ToolCallLimitMiddleware` | 限制工具调用次数 |
| `SummarizationMiddleware` | 长对话历史总结 |
| `HumanInTheLoopMiddleware` | 人工审批工具操作 |
| `PIIMiddleware` | 检测和处理特定个人信息 |
| `LLMToolSelectorMiddleware` | 通过模型选择相关工具 |
| `ContextEditingMiddleware` | 管理较长的工具上下文 |
| `TodoListMiddleware` | 任务规划与跟踪 |

例如，普通的模型重试没有必要从零编写 wrap\_model\_call。

可以直接：

```
from langchain.agents.middleware import ModelRetryMiddleware



middleware = [

    ModelRetryMiddleware(max_retries=2)

]
```

相比自定义的 except Exception 无限重试，内置 Middleware 更方便管理重试次数、退避延迟和失败处理。

---

## 二十五、完整实例：Middleware 综合使用

下面创建一个简单的课程查询 Agent，并组合四个 Middleware：

- before\_agent：记录 Agent 开始。
- before\_model：打印当前消息数量。
- wrap\_model\_call：统计模型耗时。
- wrap\_tool\_call：记录工具调用。
- after\_agent：记录完成状态。

文件名：`middleware_api_demo.py`

安装依赖：

```
pip install -U langchain langchain-deepseek python-dotenv
```

创建 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

import time
from dotenv import load\_dotenv

from langchain.agents import create\_agent
from langchain.agents.middleware import (
    before\_agent,
    before\_model,
    wrap\_model\_call,
    wrap\_tool\_call,
    after\_agent
)
from langchain.tools import tool
from langchain\_deepseek import ChatDeepSeek

load\_dotenv()

# ========== 1. 创建工具 ==========

@tool
def search\_course(keyword: str) -> str:
    """查询模拟课程目录。"""
    courses = {
        "Python": "Python 基础、Python 函数",
        "Java": "Java 基础、Java 面向对象"
    }
    return courses.get(keyword, "没有匹配的模拟课程")

# ========== 2. Agent 开始前 ==========

@before\_agent
def log\_start(state, runtime):
    print("[Agent] 开始执行")
    return None

# ========== 3. 每次模型调用前 ==========

@before\_model
def log\_messages(state, runtime):
    print("[Model] 当前消息数：", len(state.get("messages", [])))
    return None

# ========== 4. 模型调用包装 ==========

@wrap\_model\_call
def track\_model\_time(request, handler):
    started = time.perf\_counter()
    try:
        return handler(request)
    finally:
        print(
            "[Model] 耗时：",
            round(time.perf\_counter() - started, 3),
            "秒"
        )

# ========== 5. 工具调用包装 ==========

@wrap\_tool\_call
def track\_tool(request, handler):
    name = request.tool\_call["name"]
    print("[Tool] 正在执行：", name)
    return handler(request)

# ========== 6. Agent 完成后 ==========

@after\_agent
def log\_finish(state, runtime):
    print("[Agent] 执行完成")
    return None

# ========== 7. 创建 Agent ==========

agent = create\_agent(
    model=ChatDeepSeek(
        model="deepseek-chat",
        temperature=0
    ),
    tools=[search\_course],
    middleware=[
        log\_start,
        log\_messages,
        track\_model\_time,
        track\_tool,
        log\_finish
    ],
    system\_prompt=(
        "你是编程课程查询助手。"
        "当用户请求查找课程时，使用 search\_course。"
        "只根据模拟工具数据回答。"
    )
)

# ========== 8. 运行 ==========

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请使用工具查询 Python 课程。"
    }]
})

print("\n最终回答：")
print(result["messages"][-1].content)

可能的日志：

```
[Agent] 开始执行

[Model] 当前消息数： 1

[Model] 耗时： 1.235 秒

[Tool] 正在执行： search_course

[Model] 当前消息数： 3

[Model] 耗时： 0.864 秒

[Agent] 执行完成



最终回答：

模拟课程目录包含 Python 基础和 Python 函数。
```

上面的耗时和回答仅用于展示格式，不是实际测量结果。

如果模型没有选择调用搜索工具，就不会出现对应的 Tool 日志。

---

## 二十六、Middleware 的测试方法

并不是所有 Middleware 测试都需要真实 API Key。

例如，日期格式化、消息条件检查和状态更新可以使用本地代码验证。

但要测试 Middleware 是否在 Agent 执行链中按预期运行，仍需要将其注册到 Agent，并通过真实模型或适合的测试模型触发对应 Hook。

建议分别测试：

| 测试项目 | 检查内容 |
| --- | --- |
| 开始与结束 Hook | 执行顺序是否正确 |
| before\_model | 每次模型调用前是否触发 |
| wrap\_model\_call | 是否正确调用 handler |
| wrap\_tool\_call | 工具名称与返回值是否正确 |
| jump\_to | 是否声明了可用目标 |
| 状态更新 | Reducer 是否按预期合并字段 |
| 异常处理 | 错误是否传播或被转换 |
| 异步执行 | 是否实现对应的异步 Hook |

特别需要验证多工具调用、重复模型调用以及异常中断时的行为，不能仅靠一次"你好"的测试判断 Middleware 已经正确实现。

---

## 二十七、生产环境注意事项

### 1、每个 Middleware 只负责一类功能

尽量将日志、重试、权限、提示词和输出格式化分开，避免一个 Middleware 包含大量互相依赖的逻辑。

### 2、不要无条件修改消息历史

Agent 消息通常使用 add\_messages Reducer。新增消息与替换消息有不同语义，不能直接假定返回较短列表就会删除旧内容。

### 3、不要无限重试模型调用

wrap\_model\_call 可以多次调用 handler，但应该限制次数，并且只重试适合重试的临时故障。

### 4、工具权限必须在执行层落实

仅过滤 request.tools 并不等于真正禁止该操作。数据库写入、邮件发送和订单查询仍需后端授权检查。

### 5、避免重复业务副作用

如果 Middleware 对支付、邮件或订单创建工具执行重试，需要先设计幂等请求与真实操作状态查询。

### 6、正确处理同步和异步逻辑

使用 ainvoke() 或 astream() 时，应确保依赖的 Middleware 和工具支持相应的异步执行路径。

### 7、不要泄露敏感日志

Middleware 可能访问完整对话、用户上下文和工具参数。生产日志与 LangSmith 追踪需要考虑数据脱敏。

### 8、不要滥用 jump\_to

随意跳转到 tools 或 model 可能导致不符合预期的执行状态。只有明确理解 Agent 图结构时才应使用。

### 9、注意多个 Middleware 的顺序

before 先注册先执行，after 后注册先执行，wrap 则层层嵌套。需要针对完整执行链处理异常时，应特别考虑包装层级。

### 10、优先使用内置 Middleware

对于重试、调用次数限制、对话总结和人工审批等通用需求，优先评估 LangChain 内置方案，再决定是否自定义。

---

## 二十八、常见问题

### 1、AgentMiddleware 与普通 Python 装饰器有什么区别？

AgentMiddleware 是集成到 LangChain Agent 执行生命周期的组件，能够访问状态、上下文及模型和工具调用过程。

### 2、before\_agent 每次调用模型都会执行吗？

不会。它处于 Agent 执行周期开始的位置，不是模型调用循环 Hook。

### 3、before\_model 和 wrap\_model\_call 有什么区别？

before\_model 负责执行前状态检查或更新；wrap\_model\_call 可以控制 handler 是否执行以及如何执行。

### 4、after\_model 与 after\_agent 如何选择？

每次模型回复后需要处理时使用 after\_model；正常 Agent 完成后需要处理时使用 after\_agent。

### 5、Middleware 可以完全跳过模型调用吗？

可以。wrap\_model\_call 可以在特定条件下直接返回 AIMessage，而不调用 handler。

### 6、request.override() 会修改原始请求吗？

不会。它返回一个具有覆盖字段的新 ModelRequest。

### 7、ModelRequest.messages 包含 SystemMessage 吗？

系统消息通常由 system\_message 单独表示，不能简单把 request.messages 视为完整最终模型载荷。

### 8、ModelResponse 一定只有一个 AIMessage 吗？

不一定。结构化输出等场景可能包含其他消息。

### 9、ExtendedModelResponse 有什么作用？

它允许在返回模型结果的同时，通过 Command 提供额外状态更新。

### 10、ExtendedModelResponse.command 可以任意跳转节点吗？

不能。当前模型响应扩展机制主要支持附加状态更新，不支持任意 goto 或 resume。

### 11、ModelCallResult 包括哪些类型？

包括 AIMessage、ModelResponse 和 ExtendedModelResponse。

### 12、为什么我返回 {"messages": [...]} 后旧消息还在？

因为 messages 默认使用 add\_messages Reducer，通常会合并或追加，而不是直接替换全部历史。

### 13、Middleware 能动态切换模型吗？

可以。通过 wrap\_model\_call 和 request.override(model=...) 可以在模型调用前选择不同模型。

### 14、Middleware 能动态控制工具吗？

可以调整模型可见的工具集合，但真实工具执行权限仍需要由后端代码控制。

### 15、多个 Middleware 的执行顺序是什么？

before 按注册顺序执行，after 逆序执行，wrap 则以第一个 Middleware 为最外层。

### 16、什么时候使用 AgentMiddleware 类？

当需要多个 Hook、同步和异步实现、初始化参数或跨项目复用时，使用类通常更方便。

### 17、可以记录每次模型的 Token 用量吗？

可以检查模型返回的 usage\_metadata，也可以结合 LangSmith 进行追踪，但实际数据取决于模型服务商是否提供。

### 18、Middleware 能替代业务系统的权限检查吗？

不能。对于用户身份、订单归属、支付与数据库修改，权限必须由可信业务代码执行。

---

## 总结

Middleware 是 LangChain Agent 的重要扩展机制，可以在不修改 Agent 核心执行流程的情况下，为模型和工具增加自定义行为。

本章节主要介绍了：

- `AgentMiddleware`：自定义中间件基类。
- `before_agent`、`after_agent`：执行周期 Hook。
- `before_model`、`after_model`：模型循环 Hook。
- `wrap_model_call`：控制模型调用。
- `wrap_tool_call`：控制工具执行。
- 同步与异步 Hook 的区别。
- `dynamic_prompt`：动态生成系统提示词。
- `ModelRequest`：模型请求信息。
- `request.override()`：创建修改后的请求。
- `ModelResponse`：模型执行结果。
- `ExtendedModelResponse`：模型结果加状态更新。
- `Command`：更新 Agent State。
- `hook_config`：配置允许的图跳转。
- 多个 Middleware 的嵌套与执行顺序。

最简单的 Middleware 写法：

```
from langchain.agents.middleware import before_model



@before_model

def log_request(state, runtime):

    print("即将调用模型")

    return None
```

在 Agent 中注册：

```
agent = create_agent(

    model="deepseek:deepseek-chat",

    tools=[my_tool],

    middleware=[log_request]

)
```

**核心原则：**Node-style Hook 负责执行前后的状态检查与更新，Wrap-style Hook 负责拦截和控制实际调用。只有正确理解两者的区别、执行顺序及 Reducer 语义，才能安全地扩展 Agent 的行为。

## 参考文档

- [LangChain Custom Middleware 官方文档](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [LangChain 内置 Middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [AgentMiddleware API Reference](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware)
- [ModelRequest API Reference](https://reference.langchain.com/python/langchain/agents/middleware/types/ModelRequest)
- [ModelResponse API Reference](https://reference.langchain.com/python/langchain/agents/middleware/types/ModelResponse)
- [ExtendedModelResponse API Reference](https://reference.langchain.com/python/langchain/agents/middleware/types/ExtendedModelResponse)
- [ModelRequest.override() API](https://reference.langchain.com/python/langchain/agents/middleware/types/ModelRequest/override)
- [LangChain Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents)
