# LangChain 模型调用拦截——@wrap\_model\_call

> 来源: https://www.runoob.com/langchain/langchain-wrap-model-call.html

---

# LangChain 模型调用拦截——@wrap\_model\_call

@wrap\_model\_call 是 LangChain Middleware 中重要的模型调用拦截机制。

与 `@before_model` 和 `@after_model` 不同，`@wrap_model_call` 可以直接包裹模型调用，通过自定义逻辑控制模型请求的执行方式。

例如，可以在调用模型之前检查请求，在执行失败时重试，切换备用模型，读取缓存，或者在特定条件下直接返回预设结果。

本章节基于 LangChain 1.x，介绍 handler 回调、ModelRequest、ModelResponse、请求修改、模型重试、故障转移、缓存和多个中间件组合。

---

## 一、什么是 @wrap\_model\_call？

`@wrap_model_call` 属于 LangChain 的 Wrap-style Middleware。

它通过包装模型执行过程，实现对模型请求和响应的拦截。

基本流程：

```
Agent 准备调用模型

        ↓

  wrap_model_call

        ↓

检查或修改 ModelRequest

        ↓

   handler(request)

        ↓

     模型执行

        ↓

     ModelResponse

        ↓

修改或记录执行结果

        ↓

     返回 Agent
```

其中，`handler` 是关键。

调用 `handler(request)` 会继续执行被包装的模型请求；如果不调用 handler，就可以在符合协议的前提下直接返回结果。

### 基本语法

## 实例

from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

@wrap\_model\_call
def my\_middleware(request: ModelRequest, handler) -> ModelResponse:
    """在模型调用前后执行自定义逻辑。"""
    print("模型即将被调用")
    response = handler(request)
    print("模型调用完成")
    return response

与 Node-style Middleware 不同，这里不能直接返回普通 State 字典代替模型响应。

如果需要修改 Agent State，应使用对应的状态更新机制。

---

## 二、理解 ModelRequest 与 ModelResponse

`wrap_model_call` 中最重要的两个对象分别是：

- `ModelRequest`：当前准备发送给模型的请求。
- `ModelResponse`：模型执行后返回的消息及结构化结果。

### ModelRequest 的常用属性

| 属性 | 说明 |
| --- | --- |
| `request.model` | 当前模型对象 |
| `request.messages` | 本次模型请求的消息列表 |
| `request.system_message` | 当前系统消息对象 |
| `request.tools` | 本次请求可用的工具定义 |
| `request.state` | Agent 当前状态 |
| `request.runtime` | 运行时上下文 |
| `request.response_format` | 当前结构化输出配置 |

其中，`request.messages` 是当前准备提交给模型的消息，不一定等同于完整的 Agent 历史状态。

### ModelResponse 的常用属性

| 属性 | 说明 |
| --- | --- |
| `response.result` | 模型响应的消息列表 |
| `response.structured_response` | 可选的结构化输出结果 |

例如，可以通过：

```
response = handler(request)

for message in response.result:

    print(message.content)
```

读取模型返回的消息。

**注意：**`handler(request)` 通常返回 ModelResponse，而不是直接返回 AIMessage，因此不能在所有场景中使用 `response.content`。

---

## 三、场景一：记录模型调用耗时

最简单的应用是记录模型执行时间。

可以使用 Python 的 `time.perf_counter()` 统计每次调用的耗时。

### 实例

## 实例

import time
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

load\_dotenv()

@wrap\_model\_call
def log\_model\_time(request: ModelRequest, handler) -> ModelResponse:
    """记录模型调用耗时。"""
    start = time.perf\_counter()
    try:
        return handler(request)
    finally:
        print(f"[Model] 耗时：{time.perf\_counter() - start:.2f} 秒")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[log\_model\_time],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程助手。"
)

result = agent.invoke({"messages": [{"role": "user", "content": "什么是 Python？"}]})
print(result["messages"][-1].content)

执行时可能输出：

```
[Model] 耗时：1.82 秒

Python 是一种通用编程语言，语法简洁易读。
```

这里使用 `finally`，即使模型调用抛出异常，也可以记录已经消耗的时间。

不过，如果模型执行失败，程序仍然会抛出原始异常，不会自动生成成功结果。

---

## 四、场景二：模型调用失败后重试

模型 API 可能因为网络中断、请求超时或者服务端临时故障而失败。

对于适合重试的临时错误，可以使用 `wrap_model_call` 调用 handler 多次。

### 实例：限制重试次数

## 实例

import time
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

@wrap\_model\_call
def retry\_on\_timeout(request: ModelRequest, handler) -> ModelResponse:
    """发生标准超时异常时，最多尝试三次。"""
    for attempt in range(3):
        try:
            return handler(request)
        except TimeoutError:
            if attempt == 2:
                raise
            wait = 2 \*\* attempt
            print(f"模型超时，{wait} 秒后重试")
            time.sleep(wait)

这里最多调用 handler 三次，等待时间分别为 1 秒、2 秒。

需要注意，实际模型 SDK 的超时异常不一定是 Python 内置的 TimeoutError。

生产环境应该根据具体模型提供商的异常类型处理，并区分：

- 网络超时、临时服务端错误：通常可以有限重试。
- 限流错误：应结合 Retry-After 或退避策略。
- API Key 无效、请求参数错误：通常不应该重复重试。
- 内容或权限限制：不能简单通过重试绕过。

此外，重复调用模型会增加请求次数和费用。

### 推荐：使用内置 ModelRetryMiddleware

LangChain 已经提供 `ModelRetryMiddleware`，一般不需要自行实现重试和退避机制。

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[ModelRetryMiddleware(max\_retries=2)],
    system\_prompt="你是一名中文编程助手。"
)

这里 `max_retries=2` 表示允许额外重试，具体执行次数还受到异常处理配置影响。

**注意：**模型集成本身也可能配置重试次数。应检查两层重试的叠加效果，避免发生远超预期的 API 请求。

---

## 五、场景三：模型降级与故障转移

如果主模型不可用，可以切换到备用模型。

这种机制称为 Fallback（故障转移）。

例如，主模型因为临时错误失败后，可以尝试使用另一个模型服务。

需要注意，备用模型应该是确实不同的模型或部署，而不是简单地将同一个模型重新初始化。

### 实例：通过 request.override() 切换模型

## 实例

from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

primary\_model = init\_chat\_model("deepseek:deepseek-flash")
fallback\_model = init\_chat\_model("openai:gpt-4o-mini")

@wrap\_model\_call
def fallback\_on\_error(request: ModelRequest, handler) -> ModelResponse:
    """主模型发生异常时尝试备用模型。"""
    try:
        return handler(request)
    except Exception as error:
        print("[Fallback] 主模型失败：", type(error).\_\_name\_\_)
        return handler(request.override(model=fallback\_model))

agent = create\_agent(
    model=primary\_model,
    tools=[],
    middleware=[fallback\_on\_error]
)

这个示例需要同时安装：

```
pip install -U langchain-deepseek langchain-openai
```

并配置：

```
DEEPSEEK_API_KEY=your_deepseek_api_key

OPENAI_API_KEY=your_openai_api_key
```

主模型失败后，代码会重新使用备用模型执行请求。

如果备用模型也失败，异常继续向上抛出。

**注意：**上面的示例为了突出原理，捕获了所有异常。真实项目应限制可触发故障转移的异常类型，避免把鉴权错误或不正确的业务配置掩盖为普通服务故障。

### request.override() 的作用

`request.override()` 返回新的 ModelRequest，不会原地修改旧请求。

例如：

```
new_request = request.override(model=fallback_model)

response = handler(new_request)
```

除了模型，还可以修改系统消息、当前消息列表、工具定义和其他受支持的请求配置。

### 推荐：使用 ModelFallbackMiddleware

对于常见的故障转移需求，可以使用 LangChain 内置中间件。

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create\_agent(
    model="deepseek:deepseek-flash",
    middleware=[ModelFallbackMiddleware("openai:gpt-4o-mini")],
    system\_prompt="你是编程学习助手。"
)

内置中间件可以在主模型调用失败后尝试备用模型。

如果主模型和备用模型来自不同提供商，需要确保二者都支持当前 Agent 所需的工具调用、结构化输出和其他能力。

另外，跨模型故障转移还应考虑数据传输范围、隐私政策和费用差异。

---

## 六、场景四：缓存模型响应

如果多个模型请求完全相同，可以考虑缓存响应，减少重复请求。

但缓存 Agent 模型结果比缓存普通字符串复杂。

因为模型输出可能受到以下信息影响：

- 系统提示词。
- 完整消息历史。
- 工具定义。
- 模型名称与参数。
- 结构化输出配置。
- 运行时上下文。

因此，不能只根据最后一条用户消息创建缓存键。

### 为什么原来的缓存方式有问题？

例如：

```
cache_key = last_content[:200]
```

这种写法可能导致两个不同请求使用相同的缓存键。

同时，用户发送相同的问题，不代表模型应该返回相同的结果。

例如，在不同对话历史下：

```
用户 A：介绍一下 Python。

用户 B：介绍一下 Python。
```

两个用户的系统权限、回答语言或工具数据可能不同。

如果共享错误的缓存结果，可能造成数据泄露或者返回不相关的信息。

### 实例：只缓存简单无工具请求

下面创建一个适合教学的内存缓存示例，仅处理固定模型、没有工具、没有结构化输出的简单请求。

## 实例

import hashlib
import json
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

cache = {}

@wrap\_model\_call
def cache\_simple\_response(request: ModelRequest, handler) -> ModelResponse:
    """仅演示缓存简单、无工具的模型请求。"""
    if request.tools or request.response\_format is not None:
        return handler(request)

    payload = {
        "model": str(request.model),
        "system": str(request.system\_message.content) if request.system\_message else "",
        "messages": [m.model\_dump(mode="json") for m in request.messages]
    }
    key = hashlib.sha256(json.dumps(payload, sort\_keys=True, ensure\_ascii=False).encode()).hexdigest()

    if key in cache:
        print("[缓存命中]")
        return cache[key].model\_copy(deep=True)

    response = handler(request)
    if len(response.result) == 1 and response.result[0].type == "ai" and not response.result[0].tool\_calls:
        cache[key] = response.model\_copy(deep=True)
        print("[缓存写入]")
    return response

这个示例通过完整消息内容、系统提示和模型信息构造缓存键，避免仅使用最后 200 个字符。

但这仍然只是教学实现，不适合直接作为生产级多用户缓存。

真实项目还需要考虑：

- 模型参数、API 版本与结构化输出配置。
- 用户、租户与数据访问隔离。
- 缓存有效期和容量限制。
- 工具数据变化后的缓存失效。
- 并发请求去重。
- 响应中可能包含的敏感信息。

**注意：**上述示例使用进程内字典，程序重启后缓存会消失，也不适合多进程共享。

如果只是希望缓存普通模型调用结果，也可以考虑 LangChain 的模型缓存机制，而不必专门使用 Middleware 实现。

---

## 七、场景五：修改模型请求

`ModelRequest.override()` 不仅可以切换模型，还可以修改当前请求的消息、系统提示词以及部分模型参数。

常见需求包括：

- 根据用户等级调整提示词。
- 为模型增加当前任务说明。
- 在特殊情况下调整模型设置。
- 只修改本次模型请求，而不直接修改 AgentState。

### 实例：动态注入时间信息

## 实例

from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.messages import SystemMessage
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

@wrap\_model\_call
def inject\_time\_context(request: ModelRequest, handler) -> ModelResponse:
    """为模型注入北京时间。"""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    base = str(request.system\_message.content) if request.system\_message else "你是编程助手。"
    extra = f"当前北京时间：{now:%Y-%m-%d %H:%M}。"

    new\_request = request.override(system\_message=SystemMessage(content=base + "\n" + extra))
    return handler(new\_request)

这里使用 `ZoneInfo("Asia/Shanghai")` 明确指定北京时间。

但如果需要根据用户所在地展示时间，应该从可信的用户设置中读取时区，而不是将服务器时间直接当作用户时间。

**注意：**如果只需要动态调整 System Prompt，可以优先使用上一章节介绍的 `@dynamic_prompt`，不一定需要直接编写 wrap\_model\_call。

---

## 八、短路返回：不调用模型

`wrap_model_call` 可以选择不执行 handler，直接返回一个符合协议的模型响应。

例如，当用户提出某个固定的演示请求时，直接返回预设文本。

### 实例：返回固定回复

## 实例

from langchain.messages import AIMessage
from langchain.agents.middleware import wrap\_model\_call, ModelRequest

@wrap\_model\_call
def fixed\_reply(request: ModelRequest, handler):
    """对演示关键词直接返回固定答案。"""
    messages = request.messages
    if messages and messages[-1].type == "human":
        if messages[-1].content == "PING\_DEMO":
            return AIMessage(content="PONG\_DEMO")
    return handler(request)

由于返回了 AIMessage，Agent 可以将其作为模型响应处理。

这种方式适合简单的固定答案、测试或特定请求短路。

但如果当前请求启用了结构化输出或存在特殊工具调用要求，不能直接将任意文本 AIMessage 当作完全兼容的结果。

另外，生产环境不应该通过这种方式伪装真实 API 调用或隐瞒服务故障。

---

## 九、多个 wrap\_model\_call 的组合顺序

一个 Agent 可以注册多个模型包装中间件。

例如：

```
middleware=[outer_middleware, inner_middleware]
```

LangChain 会按照嵌套包装的方式组合它们。

第一个 Middleware 位于最外层。

### 实例：观察执行顺序

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse

@wrap\_model\_call
def outer\_middleware(request: ModelRequest, handler) -> ModelResponse:
    print("[外层] 开始")
    response = handler(request)
    print("[外层] 结束")
    return response

@wrap\_model\_call
def inner\_middleware(request: ModelRequest, handler) -> ModelResponse:
    print("  [内层] 开始")
    response = handler(request)
    print("  [内层] 结束")
    return response

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[outer\_middleware, inner\_middleware]
)

result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print(result["messages"][-1].content)

正常执行时，日志顺序为：

```
[外层] 开始

  [内层] 开始

  [内层] 结束

[外层] 结束
```

执行过程类似：

```
outer_middleware

    ↓

inner_middleware

    ↓

模型调用

    ↓

inner_middleware 返回

    ↓

outer_middleware 返回
```

如果某个中间件发生异常且没有处理，对应的后续日志可能不会执行。

因此，需要始终记录结束事件时，可以使用 `try/finally`。

---

## 十、缓存与重试的执行顺序

多个中间件的顺序会影响实际行为。

例如：

```
middleware=[cache_middleware, retry_middleware]
```

其执行过程可以理解为：

```
缓存检查

    ↓

未命中

    ↓

重试逻辑

    ↓

模型调用

    ↓

返回结果

    ↓

写入缓存
```

如果缓存命中，外层中间件可以直接返回结果，不再执行内部重试和模型调用。

相反，如果把重试放到外层，重试可能重新执行整个内部缓存处理逻辑。

因此，在设计 Middleware 顺序时，应该明确每个中间件的职责。

一般来说：

| 功能 | 适合的位置 | 说明 |
| --- | --- | --- |
| 请求校验 | 靠外层 | 尽早阻断不合法请求 |
| 缓存查询 | 重试逻辑外层 | 缓存命中时可以直接返回 |
| 模型重试 | 靠近模型调用 | 处理实际请求失败 |
| 日志与监控 | 按统计目标选择 | 外层统计总耗时，内层统计实际调用 |

这只是常见组织方式，具体还需要考虑缓存是否覆盖备用模型、请求参数是否改变，以及不同提供商的行为差异。

---

## 十一、同步与异步 wrap\_model\_call

前面的示例都使用同步 `handler(request)`。

如果应用使用异步 Agent，可以通过 `@awrap_model_call` 实现异步模型拦截。

### 实例：异步记录模型调用时间

## 实例

import time
from langchain.agents.middleware import awrap\_model\_call, ModelRequest, ModelResponse

@awrap\_model\_call
async def async\_timer(request: ModelRequest, handler) -> ModelResponse:
    """记录异步模型调用耗时。"""
    start = time.perf\_counter()
    try:
        return await handler(request)
    finally:
        print(f"[Async Model] 耗时：{time.perf\_counter() - start:.2f} 秒")

异步模型调用需要使用 `await handler(request)`。

如果在异步中间件中实现重试等待，应使用：

```
await asyncio.sleep(seconds)
```

而不是 `time.sleep()`，避免阻塞事件循环。

**注意：**中间件的同步与异步钩子必须与 Agent 的执行方式匹配。使用 `ainvoke()` 或 `astream()` 时，应确保相关 Middleware 支持异步路径。

---

## 十二、完整实例：模型调用监控与重试

下面将模型调用耗时统计和重试机制组合起来，构建一个简单的编程助手。

文件名：`wrap_model_demo.py`

安装依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

配置 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

import time
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import wrap\_model\_call, ModelRequest, ModelResponse, ModelRetryMiddleware

load\_dotenv()

@wrap\_model\_call
def monitor\_model(request: ModelRequest, handler) -> ModelResponse:
    """统计模型执行耗时与结果。"""
    start = time.perf\_counter()
    try:
        response = handler(request)
        print("[Model] 返回消息数：", len(response.result))
        return response
    finally:
        print(f"[Model] 调用耗时：{time.perf\_counter() - start:.2f} 秒")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[monitor\_model, ModelRetryMiddleware(max\_retries=2)],
    system\_prompt="你是菜鸟教程 RUNOOB 的中文编程助手，回答准确、简洁。"
)

def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("用户：", question)
    print("AI：", result["messages"][-1].content)

if \_\_name\_\_ == "\_\_main\_\_":
    ask("什么是 Python 装饰器？")
    ask("Python 和 Java 有什么主要区别？")

执行时可能出现：

```
[Model] 返回消息数： 1

[Model] 调用耗时： 1.75 秒

用户： 什么是 Python 装饰器？

AI： 装饰器是一种用于扩展函数行为的机制……
```

如果模型请求发生可重试的错误，内层的 ModelRetryMiddleware 可以根据配置进行处理。

这里使用：

```
middleware=[monitor_model, ModelRetryMiddleware(max_retries=2)]
```

意味着外层监控统计的是包括内层重试在内的整段执行耗时，而不是每一次单独的底层 API 请求耗时。

如果需要精确统计单次 API 尝试，应调整监控中间件的放置位置。

---

## 十三、生产环境注意事项

### 1、不要无条件重试所有错误

API Key 无效、请求参数错误和权限错误，通常不应该反复重试。

建议结合异常类型、服务端状态码以及重试上限处理。

### 2、不要使用不安全的缓存键

只根据用户最后一条消息缓存结果，可能造成跨用户数据混淆。

生产环境应结合完整请求语义、用户隔离、模型参数和缓存有效期设计。

### 3、降级模型需要兼容当前 Agent

如果 Agent 依赖 Tool Calling 或结构化输出，备用模型必须支持对应能力。

否则即使模型 API 可以调用，也可能无法完成任务。

### 4、不要把模型错误伪装成成功

模型服务失败时，如果业务必须得到可靠结果，应明确向应用返回错误状态或进行受控降级。

直接返回"处理成功"等固定文本，可能让上层程序错误判断任务状态。

### 5、谨慎处理敏感信息

Middleware 可以读取消息历史和请求上下文，不应该随意记录用户密码、API Key、访问令牌或其他敏感数据。

### 6、监控实际请求费用

重试和模型降级可能导致额外 API 调用，应监控模型调用次数、Token 消耗、执行耗时和错误率。

---

## 十四、常见问题

### 1、@wrap\_model\_call 和 @before\_model 有什么区别？

`@before_model` 是模型调用前的节点钩子，主要负责状态检查、更新和路由控制。

`@wrap_model_call` 则通过 handler 包裹模型请求，适合重试、降级、短路返回和修改请求。

### 2、handler(request) 返回 AIMessage 吗？

通常返回 ModelResponse，其 `result` 保存模型响应消息。

但 Middleware 允许简化返回 AIMessage 等受支持的模型调用结果类型。

### 3、可以不调用 handler 吗？

可以。这样可以直接返回预设 AIMessage 或其他符合协议的结果。

### 4、handler 可以执行多次吗？

可以。重试和降级都可能多次调用 handler，但会增加实际 API 请求数量。

### 5、request.override() 会修改原始请求吗？

不会。它返回新的 ModelRequest，原始请求对象保持不变。

### 6、如何读取 ModelResponse 的文本？

通常可以遍历 `response.result`，读取其中 AIMessage 的文本内容。

### 7、为什么不能只使用最后一条消息作为缓存键？

因为模型输出还受到系统提示、历史消息、工具配置、模型参数和用户上下文影响。

### 8、备用模型与主模型可以相同吗？

技术上可以重复请求同一模型，但这通常属于重试，不是真正的跨模型故障转移。

### 9、ModelRetryMiddleware 与自定义重试有什么区别？

ModelRetryMiddleware 提供现成的重试控制。自定义重试适合具有特殊异常识别、额外监控或业务要求的场景。

### 10、多个 wrap\_model\_call 的执行顺序是什么？

第一个位于最外层，之后依次向内包装。执行返回时按相反方向退出。

### 11、能在 wrap\_model\_call 中修改 AgentState 吗？

可以使用 LangChain 支持的 `ExtendedModelResponse` 和相关 Command 更新机制，但不能简单将普通状态字典作为 ModelResponse 返回。

### 12、异步 Agent 应使用什么钩子？

可以使用 `@awrap_model_call`，并通过 `await handler(request)` 执行异步模型调用。

---

## 总结

`@wrap_model_call` 是 LangChain Agent 进行模型调用拦截的重要扩展机制。

它可以用于：

- 记录模型调用耗时。
- 失败后重试。
- 主模型故障时切换备用模型。
- 缓存符合条件的模型响应。
- 修改当前模型请求。
- 在特定条件下跳过模型调用。
- 组合多个独立的模型执行中间件。

基本写法：

```
@wrap_model_call

def my_middleware(request: ModelRequest, handler):

    # 模型调用前

    response = handler(request)

    # 模型调用后

    return response
```

需要修改模型时：

```
new_request = request.override(model=fallback_model)

response = handler(new_request)
```

需要直接返回固定回复时：

```
return AIMessage(content="预设回复")
```

> **核心原则：**通过 handler 控制模型执行，通过 request.override() 修改当前请求，通过符合协议的返回值将结果交给 Agent。对于常见的模型重试和故障转移，优先考虑 LangChain 提供的内置 Middleware。
