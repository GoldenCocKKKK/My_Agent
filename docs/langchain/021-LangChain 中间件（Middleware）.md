# LangChain 中间件（Middleware）

> 来源: https://www.runoob.com/langchain/langchain-middleware-concepts.html

---

# LangChain 中间件（Middleware）

LangChain Middleware（中间件）是扩展 Agent 行为的重要机制。

通过 Middleware，开发者可以在模型调用、工具执行和 Agent 生命周期的特定阶段插入自定义逻辑，实现日志记录、重试、模型降级、内容检查、动态提示词和执行限制等功能。

这些逻辑可以与 Agent 的业务工具分离，不需要每次都修改 Agent 的核心执行代码。

本章节基于 LangChain 1.x，介绍 Middleware 的六个核心钩子、装饰器与类继承两种定义方式、执行顺序、状态更新和常用内置中间件。

---

## 一、什么是 Middleware？

Middleware 可以理解为 Agent 执行流程中的拦截器或钩子（Hook）。

一个普通 Agent 通常按照以下流程运行：

```
用户输入

   ↓

调用模型

   ↓

模型判断是否调用工具

   ↓

执行工具

   ↓

将工具结果返回模型

   ↓

生成最终回答
```

Middleware 可以在这些环节前后运行额外逻辑。

例如：

```
用户输入

   ↓

[before_agent] 初始化和请求检查

   ↓

[before_model] 模型调用前检查

   ↓

[wrap_model_call] 模型请求拦截、重试

   ↓

       Model

   ↓

[after_model] 模型响应检查

   ↓

[wrap_tool_call] 工具执行拦截

   ↓

       Tools

   ↓

重新调用模型（按需循环）

   ↓

[after_agent] 执行结束处理

   ↓

返回结果
```

需要注意，以上只是简化流程。实际执行顺序还受到多个 Middleware 的组合方式、模型工具调用、中断和其他执行控制机制影响。

![LangChain Middleware 中间件生命周期示意图](https://www.runoob.com/wp-content/uploads/2026/05/59b954ef-b70a-4cce-b141-aff8caa6bbfb.png)

---

## 二、Middleware 的六个核心钩子

LangChain Middleware 提供六个主要钩子，分为 Node-style 和 Wrap-style 两类。

| 钩子 | 类型 | 触发时机 | 常见用途 |
| --- | --- | --- | --- |
| `before_agent` | Node | Agent 主流程开始前 | 初始化、输入检查、记录日志 |
| `before_model` | Node | 每次模型调用前 | 消息检查、状态更新、流程控制 |
| `wrap_model_call` | Wrap | 包裹模型调用 | 重试、模型降级、请求改写 |
| `after_model` | Node | 模型响应后 | 结果检查、日志、状态更新 |
| `wrap_tool_call` | Wrap | 包裹工具调用 | 工具错误处理、重试、审计 |
| `after_agent` | Node | Agent 正常执行完成时 | 结果统计、结束日志 |

### Node-style Hooks

节点钩子在指定执行阶段运行，适合日志记录、状态更新和流程检查。

包含：

```
before_agent

before_model

after_model

after_agent
```

通常可以返回：

- `None`：不修改状态。
- `dict`：更新 Agent State。
- `Command`：在支持的执行场景中进行状态或流程控制。

### Wrap-style Hooks

包装钩子负责拦截模型或工具调用。

包含：

```
wrap_model_call

wrap_tool_call
```

这类钩子会收到一个 `handler`，由它执行被包装的操作。

开发者可以选择调用 handler 一次、多次，或者在合适的条件下不调用 handler。

**注意：**Wrap-style Hook 的返回值必须符合相应协议，不能将普通状态字典直接当作模型或工具调用的返回值。

---

## 三、两种 Middleware 定义方式

LangChain 支持使用装饰器或者继承 `AgentMiddleware` 类创建中间件。

### 方式一：装饰器

适合功能简单、逻辑独立的中间件。

## 实例

from langchain.agents.middleware import before\_model, after\_model, AgentState
from langgraph.runtime import Runtime

@before\_model
def log\_before(state: AgentState, runtime: Runtime) -> None:
    """模型调用前打印消息数量。"""
    print("[before\_model] 消息数：", len(state.get("messages", [])))

@after\_model
def log\_after(state: AgentState, runtime: Runtime) -> None:
    """模型调用后打印工具请求。"""
    last = state["messages"][-1]
    tools = [call["name"] for call in getattr(last, "tool\_calls", [])]
    print("[after\_model] 工具请求：", tools or "无")

使用时将中间件加入 `middleware`：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    middleware=[log_before, log_after]

)
```

装饰器定义的中间件可以在多个 Agent 中复用。

### 方式二：继承 AgentMiddleware

如果一个中间件需要多个钩子、内部配置或者状态管理，可以使用类继承。

## 实例

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

class LoggingMiddleware(AgentMiddleware):
    """统一记录 Agent 执行日志。"""

    def before\_agent(self, state: AgentState, runtime: Runtime) -> None:
        print("[Logging] Agent 开始")

    def before\_model(self, state: AgentState, runtime: Runtime) -> None:
        print("[Logging] 准备调用模型")

    def after\_model(self, state: AgentState, runtime: Runtime) -> None:
        print("[Logging] 模型调用完成")

    def after\_agent(self, state: AgentState, runtime: Runtime) -> None:
        print("[Logging] Agent 正常完成")

middleware = LoggingMiddleware()

类中只需要实现实际使用的钩子，不需要把六个方法全部写出来。

默认情况下，中间件名称通常使用类名，也可以根据实际需求配置。

**注意：**对于简单日志，装饰器已经足够。只有逻辑复杂、需要维护配置或组合多个钩子时，才建议使用类继承。

---

## 四、before\_agent：Agent 开始前

`before_agent` 在 Agent 主流程开始前运行，适合进行初始化、输入检查与基础日志记录。

例如，记录用户消息数量：

## 实例

from langchain.agents.middleware import before\_agent, AgentState
from langgraph.runtime import Runtime

@before\_agent
def start\_log(state: AgentState, runtime: Runtime) -> None:
    """记录 Agent 启动信息。"""
    print("Agent 开始执行")
    print("消息数量：", len(state.get("messages", [])))

在正常的完整执行中，`before_agent` 通常在一次 Agent 调用开始时执行。

但如果涉及持久化检查点、执行恢复或中断，不应该简单认为每次恢复都一定重新运行所有生命周期钩子。

---

## 五、before\_model：模型调用前

`before_model` 在模型调用前运行。

与 before\_agent 不同，如果 Agent 需要多次调用模型，before\_model 也可能执行多次。

它适合检查消息、修改状态，或者在满足条件时提前结束当前执行。

### 实例：限制消息数量

## 实例

from langchain.agents.middleware import before\_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

@before\_model(can\_jump\_to=["end"])
def check\_message\_limit(state: AgentState, runtime: Runtime) -> dict | None:
    """消息过多时提前结束。"""
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage(content="当前会话消息较多，请开启新的会话。")],
            "jump\_to": "end"
        }
    return None

这里的 `can_jump_to=["end"]` 声明允许该钩子跳转到结束流程。

如果消息数量达到限制，钩子返回新的 AIMessage，并控制 Agent 结束。

**注意：**这只是演示消息数量限制，不是 Token 数量限制，也不能替代实际上下文窗口管理。

---

## 六、after\_model：模型调用后

`after_model` 在模型获得响应后运行。

可以用于记录模型是否发起工具调用，或者更新执行统计信息。

### 实例：检查工具请求

## 实例

from langchain.agents.middleware import after\_model, AgentState
from langgraph.runtime import Runtime

@after\_model
def inspect\_response(state: AgentState, runtime: Runtime) -> None:
    """检查模型是否请求调用工具。"""
    last = state["messages"][-1]
    calls = getattr(last, "tool\_calls", [])

    if calls:
        print("模型请求工具：", [call["name"] for call in calls])
    else:
        print("模型没有请求工具")

如果模型请求了工具，Agent 之后可能进入工具执行节点。

如果没有工具调用，并且没有其他执行控制要求，Agent 通常会结束当前循环。

另外，after\_model 可以通过返回状态字典更新自定义状态字段，但对应字段需要在 Agent State Schema 中定义。

---

## 七、wrap\_model\_call：拦截模型调用

`wrap_model_call` 是功能较强的 Middleware 钩子。

它可以通过 handler 包裹一次模型调用，并控制请求执行方式。

典型场景包括：

- 模型 API 请求失败后重试。
- 根据条件切换模型。
- 统计模型调用耗时。
- 修改模型请求参数。
- 缓存特定模型请求。

### 实例：记录模型调用耗时

## 实例

import time
from typing import Callable
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

@wrap\_model\_call
def time\_model\_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """统计一次模型调用的耗时。"""
    start = time.perf\_counter()
    try:
        return handler(request)
    finally:
        print(f"[Model] 耗时：{time.perf\_counter() - start:.2f} 秒")

这里的 `handler(request)` 执行实际模型调用。

如果不调用 handler，就不会按通常流程执行被包装的模型请求。

### 实例：失败后重试

下面提供一个简化的重试示例：

## 实例

from typing import Callable
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

@wrap\_model\_call
def retry\_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """演示模型调用失败后的重试。"""
    for attempt in range(3):
        try:
            return handler(request)
        except TimeoutError:
            if attempt == 2:
                raise
            print(f"模型调用超时，准备第 {attempt + 2} 次尝试")
    raise RuntimeError("模型重试失败")

示例只捕获 `TimeoutError`，并非所有提供商超时异常都属于此类型。

真实项目推荐针对当前模型集成实际抛出的异常进行处理，加入指数退避及重试上限。

LangChain 也提供现成的 `ModelRetryMiddleware`，一般不需要从零编写完整的重试机制。

---

## 八、wrap\_tool\_call：拦截工具执行

`wrap_tool_call` 用于包装 Agent 中的工具执行。

可以在调用工具前后记录日志，或者捕获特定工具执行异常。

### 实例：记录工具调用

## 实例

from typing import Callable
from langchain.agents.middleware import wrap\_tool\_call
from langchain.agents.middleware.types import ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

@wrap\_tool\_call
def log\_tool\_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    """记录工具名称与执行结果。"""
    name = request.tool\_call["name"]
    print("[Tool] 开始：", name)
    try:
        return handler(request)
    finally:
        print("[Tool] 结束：", name)

这里使用 `request.tool_call` 获取模型生成的工具调用信息。

工具执行结果通常是 ToolMessage，也可能是用于更新状态的 Command。

**注意：**不是所有异常都会发生在工具函数内部。例如，工具参数绑定或参数校验失败，可能在工具执行之前就由 ToolNode 处理。

对于带有数据库写入、支付或其他副作用的工具，不能盲目重试，否则可能造成重复执行。

---

## 九、after\_agent：Agent 执行结束后

`after_agent` 用于处理 Agent 正常执行完成后的逻辑。

例如统计消息数量或记录结束日志。

## 实例

from langchain.agents.middleware import after\_agent, AgentState
from langgraph.runtime import Runtime

@after\_agent
def finish\_log(state: AgentState, runtime: Runtime) -> None:
    """输出 Agent 完成后的统计信息。"""
    messages = state.get("messages", [])
    tool\_count = sum(msg.type == "tool" for msg in messages)
    print(f"Agent 完成：{len(messages)} 条消息，{tool\_count} 条工具结果")

需要注意，after\_agent 不应该被当作可靠的异常清理机制。

如果模型调用发生异常、任务被取消，或者执行流程被中断，after\_agent 不一定按照正常结束路径运行。

对于数据库连接、文件句柄或其他必须释放的资源，应使用 Python 的上下文管理器、`try/finally` 等机制处理。

---

## 十、完整实例：观察 Middleware 生命周期

下面将四个节点钩子和一个工具结合起来，观察 Agent 的执行过程。

为了避免误导，天气工具返回固定模拟数据，不连接实时天气 API。

文件名：`middleware_demo.py`

安装依赖：

```
pip install -U langchain langchain-deepseek langgraph python-dotenv
```

在 `.env` 配置：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import before\_agent, before\_model, after\_model, after\_agent, AgentState
from langchain.tools import tool
from langgraph.runtime import Runtime

load\_dotenv()

@before\_agent
def start\_log(state: AgentState, runtime: Runtime) -> None:
    print("[before\_agent] Agent 开始")

@before\_model
def pre\_model(state: AgentState, runtime: Runtime) -> None:
    print("[before\_model] 消息数：", len(state.get("messages", [])))

@after\_model
def post\_model(state: AgentState, runtime: Runtime) -> None:
    last = state["messages"][-1]
    calls = [c["name"] for c in getattr(last, "tool\_calls", [])]
    print("[after\_model] 工具请求：", calls or "无")

@after\_agent
def end\_log(state: AgentState, runtime: Runtime) -> None:
    print("[after\_agent] 消息总数：", len(state.get("messages", [])))

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。"""
    data = {"杭州": "晴，25°C", "北京": "多云，18°C"}
    return data.get(city, "没有该城市的模拟天气数据")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[get\_weather],
    middleware=[start\_log, pre\_model, post\_model, end\_log],
    system\_prompt="你是中文助手。查询天气时使用工具，工具返回的是模拟数据，不代表实时天气。"
)

def ask(question: str):
    print("\n用户：", question)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("AI：", result["messages"][-1].content)

if \_\_name\_\_ == "\_\_main\_\_":
    ask("请查询杭州的模拟天气。")
    ask("你好，请介绍一下自己。")

### 可能的运行结果

```
用户：请查询杭州的模拟天气。

[before_agent] Agent 开始

[before_model] 消息数： 1

[after_model] 工具请求： ['get_weather']

[before_model] 消息数： 3

[after_model] 工具请求： 无

[after_agent] 消息总数： 4

AI：杭州的模拟天气为晴天，25°C。



用户：你好，请介绍一下自己。

[before_agent] Agent 开始

[before_model] 消息数： 1

[after_model] 工具请求： 无

[after_agent] 消息总数： 2

AI：你好，我是一名中文助手。
```

从这个示例可以看出：

- `before_agent`：正常执行开始时运行。
- `before_model`：每次模型调用前运行。
- `after_model`：每次模型响应后运行。
- `after_agent`：正常执行完成后运行。

如果模型需要调用工具，before\_model 和 after\_model 可能执行多次。

但实际调用次数由模型决定，不应假设带有工具的请求一定产生四条消息。

---

## 十一、Middleware 的返回值与状态更新

不同类型的 Middleware，返回值的含义并不相同。

| 钩子类型 | 常见返回值 | 作用 |
| --- | --- | --- |
| Node-style | `None` | 不提交状态更新 |
| Node-style | `dict` | 提交状态更新，由 Reducer 合并 |
| Node-style | `Command` | 在受支持的场景中更新状态或控制流程 |
| `wrap_model_call` | `ModelResponse` 或受支持的模型响应类型 | 返回模型执行结果 |
| `wrap_tool_call` | `ToolMessage` 或 `Command` | 返回工具结果或状态更新 |

例如，一个 Node-style Middleware 可以通过返回字典修改自定义状态。

### 实例：统计模型调用次数

## 实例

from typing\_extensions import NotRequired
from langchain.agents import create\_agent
from langchain.agents.middleware import after\_model, AgentState
from langgraph.runtime import Runtime

class TrackingState(AgentState):
    model\_call\_count: NotRequired[int]

@after\_model(state\_schema=TrackingState)
def count\_calls(state: TrackingState, runtime: Runtime) -> dict:
    """统计模型调用次数。"""
    return {"model\_call\_count": state.get("model\_call\_count", 0) + 1}

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[count\_calls]
)

result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print("模型调用次数：", result.get("model\_call\_count"))

如果模型只调用一次，结果通常为：

```
模型调用次数： 1
```

这里的 `model_call_count` 通过 Middleware 的 `state_schema` 声明，再通过返回字典更新。

对于 `messages` 字段，LangGraph 使用 add\_messages Reducer 合并消息。

但并不是所有字段都使用追加逻辑。没有设置特殊 Reducer 的普通字段通常采用覆盖式更新。

---

## 十二、多个 Middleware 的执行顺序

一个 Agent 可以同时使用多个 Middleware：

```
middleware=[middleware_a, middleware_b]
```

不同钩子类型的组合顺序并不完全相同。

一般而言：

- Node-style 的前置钩子按配置顺序执行。
- 后置钩子通常按相反顺序执行，形成对应的生命周期包裹关系。
- Wrap-style 钩子采用嵌套包装结构，第一个 Middleware 通常位于最外层。

例如，两个 Wrap Middleware 的执行过程可以理解为：

```
middleware_a：开始

    middleware_b：开始

        执行模型调用

    middleware_b：结束

middleware_a：结束
```

因此，Middleware 的先后顺序可能影响重试、日志、缓存和错误处理的结果。

**注意：**复杂应用中如果多个 Middleware 修改相同的状态或模型请求，应该明确设计执行顺序，并进行测试。

---

## 十三、内置 Middleware：无需重复造轮子

LangChain 提供了多个常用的内置中间件，可以直接加入 Agent。

| 中间件 | 主要功能 |
| --- | --- |
| `ModelRetryMiddleware` | 模型调用失败后的重试 |
| `ToolRetryMiddleware` | 工具执行失败后的重试 |
| `ModelFallbackMiddleware` | 主模型不可用时切换备用模型 |
| `ModelCallLimitMiddleware` | 限制模型调用次数 |
| `ToolCallLimitMiddleware` | 限制工具调用次数 |
| `SummarizationMiddleware` | 对较长会话进行摘要压缩 |
| `HumanInTheLoopMiddleware` | 在敏感工具操作前进行人工审核 |
| `PIIMiddleware` | 识别和处理个人身份信息 |
| `LLMToolSelectorMiddleware` | 从大量工具中筛选当前相关工具 |

### 实例：限制模型调用次数

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[ModelCallLimitMiddleware(run\_limit=5, exit\_behavior="end")],
    system\_prompt="你是一名中文编程助手。"
)

result = agent.invoke({"messages": [{"role": "user", "content": "什么是 Python 装饰器？"}]})
print(result["messages"][-1].content)

这里限制一次 Agent 运行中的模型调用次数。

需要注意，这不等于限制 Token 消耗，也不是 API 费用的精确上限。

实际应用还可以结合 API 配额、Token 预算和超时控制。

---

## 十四、Middleware 与流式事件

Middleware 还可以与 Streaming 结合，向前端发送自定义执行状态。

例如，在模型调用前向事件流发送提示：

```
正在调用模型……
```

在支持流式事件的运行环境中，可以通过 Runtime 的流式写入器发送事件。

### 实例：自定义状态事件

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import before\_model, AgentState
from langgraph.runtime import Runtime

@before\_model
def notify\_model(state: AgentState, runtime: Runtime) -> None:
    """发送模型调用状态。"""
    if runtime.stream\_writer:
        runtime.stream\_writer({"stage": "model", "message": "正在调用模型"})

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[notify\_model]
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": "介绍 Python。"}]},
    stream\_mode="custom",
    version="v2"
):
    if event["type"] == "custom":
        print(event["data"])

在支持相应 v2 流式协议的 LangGraph 版本中，可能输出：

```
{'stage': 'model', 'message': '正在调用模型'}
```

需要注意，`runtime.stream_writer` 是否可用取决于实际执行上下文。

建议通过 `stream()` 或 `astream()` 消费自定义事件，而不是假设普通的 `invoke()` 会直接显示事件。

---

## 十五、生产环境使用 Middleware 的注意事项

Middleware 虽然方便，但并不是所有业务逻辑都适合放到 Middleware 中。

### 1、不要重复执行有副作用的操作

模型请求重试和工具重试具有不同风险。

对于只读工具，重试通常比较容易控制。

但是，对于支付、删除文件、创建订单等操作，重试可能导致重复执行，需要通过幂等键、事务或其他机制保证安全。

### 2、不要记录敏感信息

Middleware 可以访问 Agent State、工具参数和模型输出。

记录日志时应避免直接保存密码、API Key、访问令牌和敏感个人信息。

### 3、不要把提示词当作安全控制

提示词可以指导模型行为，但不能代替身份验证、工具权限检查和服务器端业务约束。

### 4、避免同步阻塞

对于异步应用，应该使用对应的异步 Middleware 钩子，并避免在异步事件循环中执行长时间阻塞操作。

### 5、控制 Middleware 的数量与复杂度

多个中间件可能同时修改请求、状态和响应，增加排查难度。

建议每个 Middleware 负责一类明确的功能，并为关键逻辑编写测试。

---

## 十六、常见问题

### 1、Middleware 与 Tool 有什么区别？

Tool 负责执行具体业务功能，例如查询课程或读取数据。

Middleware 负责在 Agent 执行流程的特定阶段加入日志、重试、检查等横切逻辑。

### 2、before\_agent 每次都只执行一次吗？

在正常的完整 Agent 调用中通常执行一次。中断、恢复或异常流程可能影响实际生命周期。

### 3、before\_model 会执行多次吗？

会。如果 Agent 需要多轮模型调用，before\_model 会在相应调用前再次运行。

### 4、wrap\_model\_call 与 before\_model 有什么区别？

before\_model 主要用于模型调用前的状态更新和流程控制。

wrap\_model\_call 则通过 handler 包裹实际模型请求，可以控制是否调用、如何重试以及如何处理模型响应。

### 5、wrap\_tool\_call 可以返回 dict 吗？

不能把普通状态字典直接当成工具结果返回。

它应返回符合工具执行协议的 ToolMessage 或 Command。

### 6、如何修改 Agent State？

Node-style Hook 可以返回状态更新字典，LangGraph 会按照 Reducer 合并到 Agent State 中。

### 7、Middleware 可以控制 Agent 提前结束吗？

可以。在支持流程跳转的节点钩子中声明 `can_jump_to`，并返回相应路由指令。

### 8、模型 API 调用失败应该使用什么？

可以使用 wrap\_model\_call 自定义重试，或优先考虑 LangChain 内置的 ModelRetryMiddleware。

### 9、多个 Middleware 的执行顺序重要吗？

重要。尤其是 Wrap-style 中间件采用嵌套执行结构，顺序会影响请求包装、重试和错误处理。

### 10、Middleware 是否支持异步执行？

支持。LangChain 提供对应的异步钩子，如 `abefore_model`、`awrap_model_call` 等，具体用法请参考 API Reference。

### 11、after\_agent 可以保证资源清理吗？

不能完全保证。任务异常、中断或取消时，不应该依赖 after\_agent 一定执行。资源清理应通过独立的可靠机制完成。

### 12、什么时候应该使用 Middleware？

当某种逻辑需要在多个 Agent 或多个模型、工具调用中复用，并且不属于某个工具本身的核心业务时，就可以考虑 Middleware。

---

## 总结

LangChain Middleware 为 Agent 提供了一套灵活的执行扩展机制。

本章节介绍了六个核心钩子：

- `before_agent`：Agent 开始前执行。
- `before_model`：模型调用前执行。
- `wrap_model_call`：包裹模型请求。
- `after_model`：模型响应后执行。
- `wrap_tool_call`：包裹工具执行。
- `after_agent`：Agent 正常完成后执行。

可以简单记住：

```
Node-style：执行检查、更新 State

Wrap-style：拦截调用、重试、处理异常



简单逻辑：使用装饰器

复杂逻辑：继承 AgentMiddleware

常见功能：优先考虑内置 Middleware
```

在实际项目中，建议先从日志记录、执行限制和模型错误处理开始使用 Middleware，再逐步扩展到动态模型选择、人工审核和状态管理。

> **核心原则：**让 Middleware 负责可复用的执行控制和横切逻辑，让 Tool 负责具体业务操作，保持 Agent 结构清晰、易于测试与维护。
