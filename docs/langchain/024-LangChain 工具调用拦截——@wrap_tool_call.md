# LangChain 工具调用拦截——@wrap\_tool\_call

> 来源: https://www.runoob.com/langchain/langchain-wrap-tool-call.html

---

# LangChain 工具调用拦截——@wrap\_tool\_call

在 LangChain Agent 中，@wrap\_tool\_call 用于拦截和控制工具的执行过程。

与 `@wrap_model_call` 类似，它也使用 handler 回调函数，但处理的对象不是大语言模型，而是 Agent 调用的工具。

通过这个中间件，可以实现工具重试、参数规范化、结果缓存、日志监控、错误处理以及自定义工具响应等功能。

本章节基于 LangChain 1.x，介绍 `ToolCallRequest`、`request.override()`、`ToolMessage`、`Command` 及工具执行相关的内置 Middleware。

---

## 一、什么是 @wrap\_tool\_call？

Agent 调用工具时，通常按照以下流程执行：

```
用户发送问题

      ↓

模型生成工具调用请求

      ↓

wrap_tool_call：执行前检查

      ↓

handler(request)

      ↓

执行 Python 工具

      ↓

ToolMessage 或 Command

      ↓

wrap_tool_call：结果处理

      ↓

返回 Agent，继续执行
```

其中，`handler(request)` 负责执行当前工具调用。

你可以在 handler 前后加入自定义逻辑，也可以在特定条件下不调用 handler，直接返回符合协议的结果。

### 基本结构

## 实例

from langchain.agents.middleware import wrap\_tool\_call
from langchain.agents.middleware.types import ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

@wrap\_tool\_call
def my\_tool\_wrapper(request: ToolCallRequest, handler) -> ToolMessage | Command:
    """拦截一次工具调用。"""
    print("准备执行：", request.tool\_call["name"])
    result = handler(request)
    print("工具执行完成")
    return result

这里的返回值通常有两种：

- `ToolMessage`：工具执行结果。
- `Command`：工具执行产生的状态更新或受支持的图执行控制。

**注意：**不能直接用普通字符串或任意字典替代这两种返回类型。

---

## 二、ToolCallRequest 包含哪些信息？

`ToolCallRequest` 描述当前准备执行的工具调用。

| 属性 | 说明 |
| --- | --- |
| `request.tool_call` | 工具调用信息，包括名称、参数和调用 ID |
| `request.tool` | 当前工具对象，某些情况下可能为 None |
| `request.state` | 当前 Agent State |
| `request.runtime` | 当前 ToolRuntime，可访问运行时上下文 |
| `request.override()` | 生成修改后的工具调用请求 |

其中最常用的是：

```
request.tool_call["name"]

request.tool_call["args"]

request.tool_call["id"]
```

例如，模型准备执行：

```
search_course(keyword="python")
```

对应的工具调用信息可能为：

```
{

    "name": "search_course",

    "args": {"keyword": "python"},

    "id": "call_123",

    "type": "tool_call"

}
```

`id` 用于关联模型发起的工具请求与后续工具响应。

### 实例：查看工具调用信息

## 实例

from langchain.agents.middleware import wrap\_tool\_call

@wrap\_tool\_call
def inspect\_tool(request, handler):
    """打印当前工具名称和参数名。"""
    call = request.tool\_call
    print("工具：", call["name"])
    print("参数名：", list(call.get("args", {})))
    print("调用 ID：", call["id"])
    return handler(request)

这里默认只打印参数名称，而不打印完整参数值，避免将密码、Token 等敏感数据写入日志。

---

## 三、场景一：工具调用失败后重试

工具可能依赖外部 HTTP API、数据库或其他服务。

如果外部服务出现临时故障，可以考虑通过工具中间件进行有限重试。

但并不是所有错误都适合重试。

例如，查询课程失败可以尝试重试，而创建订单、扣款等操作，如果没有幂等性保护，重复执行可能带来严重问题。

### 实例：只重试指定异常

## 实例

import time
from langchain.agents.middleware import wrap\_tool\_call

class TemporaryServiceError(Exception):
    """模拟外部服务临时故障。"""

@wrap\_tool\_call
def retry\_tool(request, handler):
    """临时故障时最多执行三次。"""
    for attempt in range(3):
        try:
            result = handler(request)
            if isinstance(result, ToolMessage) and result.status == "error":
                return result
            return result
        except TemporaryServiceError:
            if attempt == 2:
                raise
            delay = 2 \*\* attempt
            print(f"工具临时故障，{delay} 秒后重试")
            time.sleep(delay)

这里最多执行三次工具调用：

```
第一次调用

    ↓ 发生 TemporaryServiceError

等待 1 秒

    ↓

第二次调用

    ↓ 发生 TemporaryServiceError

等待 2 秒

    ↓

第三次调用
```

只有明确抛出 `TemporaryServiceError` 才会触发这个自定义重试逻辑。

**注意：**如果工具返回的是 `ToolMessage(status="error")`，这里会直接返回错误消息，不会自动将其转换为异常。

工具抛出的 Python 异常和返回的错误 ToolMessage，属于不同的错误处理方式。

---

## 四、完整实例：模拟工具失败并重试

下面创建一个模拟课程查询服务，第一次、第二次调用失败，第三次返回结果。

为了方便理解，这里使用 Python 内存计数器模拟故障，不连接真实网站。

## 实例

import time
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import wrap\_tool\_call
from langchain.tools import tool
from langchain.messages import ToolMessage

load\_dotenv()

class TemporaryServiceError(Exception):
    pass

attempts = {}

@tool
def fetch\_course\_data(course\_id: str) -> str:
    """查询指定课程 ID 的模拟资料。"""
    attempts[course\_id] = attempts.get(course\_id, 0) + 1
    if attempts[course\_id] < 3:
        raise TemporaryServiceError("模拟课程服务暂时不可用")
    return f"{course\_id}：Python3 基础教程，免费"

@wrap\_tool\_call
def retry\_course\_tool(request, handler):
    """仅重试指定课程查询工具。"""
    if request.tool\_call["name"] != "fetch\_course\_data":
        return handler(request)

    for attempt in range(3):
        try:
            return handler(request)
        except TemporaryServiceError:
            if attempt == 2:
                raise
            print(f"第 {attempt + 1} 次失败，准备重试")
            time.sleep(0.2)

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[fetch\_course\_data],
    middleware=[retry\_course\_tool],
    system\_prompt="你是课程助手。用户要求查询课程时调用工具，数据仅供演示。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请使用工具查询课程 python-001。"}]
})
print(result["messages"][-1].content)

如果模型按预期调用工具，可能出现：

```
第 1 次失败，准备重试

第 2 次失败，准备重试



Python3 基础教程是一门免费的模拟课程。
```

这里的第三次工具调用返回了成功结果。

不过，模型是否调用工具、最终回复内容以及是否产生其他步骤，都由实际执行情况决定。

**注意：**这个计数器仅用于单进程教学示例，不适合并发生产环境。

---

## 五、推荐：ToolRetryMiddleware

对于常见的工具重试，LangChain 已经提供了 ToolRetryMiddleware。

它支持：

- 指定最大重试次数。
- 限定需要重试的工具。
- 指定哪些异常需要重试。
- 指数退避和随机抖动。
- 控制重试全部失败后的行为。

### 实例：使用内置工具重试

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[fetch\_course\_data],
    middleware=[ToolRetryMiddleware(
        tools=["fetch\_course\_data"],
        max\_retries=2,
        retry\_on=(TemporaryServiceError,),
        on\_failure="error",
        initial\_delay=1.0,
        backoff\_factor=2.0
    )]
)

这里的 `max_retries=2` 表示最多额外重试两次，加上第一次执行，总共最多三次。

常用参数如下：

| 参数 | 说明 |
| --- | --- |
| `max_retries` | 第一次执行失败后，允许的额外重试次数 |
| `tools` | 指定需要应用重试的工具 |
| `retry_on` | 允许重试的异常类型或判断函数 |
| `on_failure` | 重试失败后继续返回错误消息或抛出异常 |
| `initial_delay` | 初始等待时间 |
| `backoff_factor` | 指数退避系数 |
| `max_delay` | 最大等待时间 |
| `jitter` | 是否添加随机等待抖动 |

`on_failure` 常用值：

- `"continue"`：返回错误 ToolMessage，让 Agent 继续处理。
- `"error"`：重新抛出异常，停止当前执行路径。

一般建议优先使用内置中间件，而不是重复实现复杂重试逻辑。

---

## 六、场景二：修改工具参数

有时候模型生成的工具参数需要在执行前进行规范化。

例如，用户输入：

```
杭州

 杭州

杭州　
```

这些内容可能包含普通空格、全角空格或其他格式差异。

可以通过 `request.override(tool_call=...)` 在工具执行前修改参数。

### 实例：规范化城市名称

## 实例

from langchain.agents.middleware import wrap\_tool\_call

@wrap\_tool\_call
def normalize\_city(request, handler):
    """规范化 get\_weather 工具的城市参数。"""
    call = request.tool\_call
    if call["name"] != "get\_weather":
        return handler(request)

    args = dict(call["args"])
    city = args.get("city")

    if isinstance(city, str):
        args["city"] = city.replace("　", " ").strip()
        call = {\*\*call, "args": args}
        request = request.override(tool\_call=call)

    return handler(request)

如果模型传入：

```
{"city": "  杭州　"}
```

工具实际收到的参数会被规范化为：

```
{"city": "杭州"}
```

这里的 `request.override()` 返回新的 ToolCallRequest，不会直接修改原始请求对象。

**注意：**参数规范化不能代替参数校验。对于数据库 ID、文件路径、权限标识等敏感参数，还需要由可信业务逻辑进行验证。

---

## 七、场景三：缓存工具调用结果

某些工具的执行成本比较高，例如读取稳定的技术文档、查询静态配置或访问更新频率较低的公共资料。

如果工具名称和参数相同，可以考虑缓存结果。

但并不是所有工具都适合缓存。

| 工具类型 | 是否适合缓存 | 说明 |
| --- | --- | --- |
| 静态教程目录 | 适合 | 内容变化较少 |
| 公共配置查询 | 视情况 | 需要设置缓存有效期 |
| 实时天气、股票价格 | 谨慎 | 数据具有时效性 |
| 用户私人信息 | 谨慎 | 必须考虑用户隔离 |
| 创建订单、发送邮件 | 不宜简单缓存 | 属于有副作用的操作 |

### 实例：缓存模拟课程查询

下面仅缓存 `search_course` 工具的成功文本结果。

## 实例

import json
from langchain.agents.middleware import wrap\_tool\_call
from langchain.messages import ToolMessage
from langgraph.types import Command

tool\_cache = {}

@wrap\_tool\_call
def cache\_course(request, handler):
    """缓存只读课程搜索结果。"""
    call = request.tool\_call
    if call["name"] != "search\_course":
        return handler(request)

    key = json.dumps(call["args"], sort\_keys=True, ensure\_ascii=False)
    if key in tool\_cache:
        print("[缓存命中] search\_course")
        return ToolMessage(content=tool\_cache[key], tool\_call\_id=call["id"], name=call["name"])

    result = handler(request)
    if isinstance(result, ToolMessage) and result.status != "error" and isinstance(result.content, str):
        tool\_cache[key] = result.content
        print("[缓存写入] search\_course")
    return result

这个示例有几个关键设计：

- 只缓存指定的只读工具。
- 使用规范化 JSON 参数生成缓存键。
- 只缓存成功的文本 ToolMessage。
- 不缓存 Command 状态更新。
- 缓存命中时使用当前 `tool_call_id` 创建新的工具回复。

为什么不能直接复用旧的 ToolMessage？

因为旧消息关联的是此前的 Tool Call ID，新的工具请求必须有对应的新 ToolMessage。

**注意：**这个示例使用 Python 字典作为缓存，没有过期时间、容量限制或并发保护，仅适合教学。

生产环境还应考虑用户隔离、权限、缓存失效和工具版本变化。

---

## 八、场景四：工具调用日志与监控

`@wrap_tool_call` 也非常适合记录工具执行时间、执行状态和调用次数。

例如，可以记录：

- 调用了哪个工具。
- 是否执行成功。
- 执行耗时。
- 是否发生异常。

但默认不建议记录完整工具参数，因为参数可能包含用户隐私或认证信息。

### 实例：记录工具耗时

## 实例

import time
from langchain.agents.middleware import wrap\_tool\_call
from langchain.messages import ToolMessage

@wrap\_tool\_call
def monitor\_tool(request, handler):
    """记录工具名称、耗时和状态。"""
    name = request.tool\_call["name"]
    start = time.perf\_counter()

    try:
        result = handler(request)
        status = "失败" if isinstance(result, ToolMessage) and result.status == "error" else "完成"
        print(f"[Tool] {name}：{status}，{time.perf\_counter() - start:.2f}s")
        return result
    except Exception:
        print(f"[Tool] {name}：异常，{time.perf\_counter() - start:.2f}s")
        raise

如果模型调用了课程查询工具，可能输出：

```
[Tool] search_course：完成，0.02s
```

如果工具返回 `ToolMessage(status="error")`，日志会记录为失败。

如果工具抛出异常，则进入 `except` 分支，并继续向外抛出异常。

这里使用 `time.perf_counter()`，更适合测量代码耗时。

---

## 九、场景五：工具错误结果处理

工具执行失败时，有两种常见情况：

- 工具抛出 Python 异常。
- 工具正常返回一个错误结果，例如"未找到课程"。

两者需要区别处理。

### 情况一：将工具异常转换为错误消息

可以在工具中间件中捕获特定异常，并返回 `ToolMessage(status="error")`。

## 实例

from langchain.agents.middleware import wrap\_tool\_call
from langchain.messages import ToolMessage

@wrap\_tool\_call
def handle\_course\_error(request, handler):
    """将课程服务临时故障转换为错误消息。"""
    try:
        return handler(request)
    except TemporaryServiceError:
        call = request.tool\_call
        return ToolMessage(
            content="课程服务暂时不可用，请稍后重试。",
            tool\_call\_id=call["id"],
            name=call["name"],
            status="error"
        )

这样，Agent 可以获得明确的工具失败消息，而不是将异常直接传到最外层。

**注意：**只应该捕获明确希望处理的异常类型。不能无条件吞掉所有异常，否则可能掩盖程序 Bug 和重要故障。

### 情况二：工具返回"未找到"

如果搜索工具返回：

```
未找到相关课程
```

这不一定代表工具执行失败，而可能是一次正常的空查询结果。

可以在工具执行后统一整理消息内容：

## 实例

from langchain.agents.middleware import wrap\_tool\_call
from langchain.messages import ToolMessage

@wrap\_tool\_call
def format\_empty\_result(request, handler):
    """规范化课程搜索的空结果。"""
    result = handler(request)
    if request.tool\_call["name"] != "search\_course":
        return result

    if isinstance(result, ToolMessage) and result.content == "未找到相关课程":
        return ToolMessage(
            content="课程目录中没有匹配结果，可以请用户更换关键词。",
            tool\_call\_id=request.tool\_call["id"],
            name=request.tool\_call["name"]
        )
    return result

这里返回新的 ToolMessage，随后 Agent 可以继续根据工具结果生成回答。

与原文不同，我们没有返回：

```
Command(update={

    "messages": [AIMessage(content="没有找到结果")]

})
```

因为这样并不能保证 Agent 直接结束，而且还可能破坏原本应该提供的工具调用响应关系。

---

## 十、Command：通过工具更新 AgentState

对于需要修改 Agent State 的工具，可以返回 `Command`。

例如，学习助手可以将某个主题标记为已完成。

但需要注意，`Command` 并不是随意跳过 Agent 循环的通用快捷方式。

### 实例：保存学习进度

## 实例

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def complete\_topic(topic: str, runtime: ToolRuntime) -> Command:
    """记录已完成的学习主题。"""
    topics = list(runtime.state.get("completed\_topics", []))
    if topic not in topics:
        topics.append(topic)

    return Command(update={
        "completed\_topics": topics,
        "messages": [ToolMessage(
            content=f"已记录完成：{topic}",
            tool\_call\_id=runtime.tool\_call\_id
        )]
    })

这个工具通过 Command 提交两个更新：

- `completed_topics`：新的学习主题列表。
- `messages`：与当前工具调用对应的 ToolMessage。

如果工具已经返回 Command，`@wrap_tool_call` 应保留其状态更新语义，不要随意把它转成普通文本。

对于工具执行后的继续、跳转和结束行为，应使用 LangGraph 支持的控制流程，而不能简单认为添加一条 AIMessage 就能立即结束 Agent。

---

## 十一、内置 ToolErrorMiddleware

如果只需要将特定工具异常转换为错误消息，不一定需要自己编写 `@wrap_tool_call`。

LangChain 提供 `ToolErrorMiddleware`，用于集中处理工具执行异常。

### 实例：自定义异常提示

## 实例

from langchain.agents.middleware import ToolErrorMiddleware

def on\_tool\_error(error: Exception, request) -> str | None:
    """只处理模拟课程服务异常。"""
    if isinstance(error, TemporaryServiceError):
        return "课程服务暂时不可用，请稍后重试。"
    return None

middleware = ToolErrorMiddleware(
    on\_error=on\_tool\_error,
    tools=["fetch\_course\_data"]
)

这里返回字符串时，中间件会将异常转换为对应的错误 ToolMessage。

返回 `None` 时，不处理该异常，让异常继续传播。

需要注意，ToolErrorMiddleware 的职责是**错误转换，而不是重试**。

如果既需要重试，又需要在最终失败时返回友好提示，可以组合 ToolRetryMiddleware 和 ToolErrorMiddleware。

---

## 十二、多个工具中间件的组合

一个 Agent 可以配置多个 `@wrap_tool_call` 中间件。

它们采用嵌套包装方式，第一个中间件通常位于最外层。

例如：

```
middleware=[

    monitor_tool,

    cache_course,

    retry_course_tool

]
```

执行过程大致为：

```
monitor_tool

    ↓

cache_course

    ↓

缓存未命中

    ↓

retry_course_tool

    ↓

执行工具

    ↓

返回结果

    ↓

写入缓存

    ↓

记录耗时

    ↓

返回 Agent
```

如果缓存命中，内层重试和实际工具执行可能被直接跳过。

因此，中间件顺序会影响缓存、重试和监控的行为。

### 常见组合建议

| 组合 | 建议顺序 | 原因 |
| --- | --- | --- |
| 监控 + 缓存 | 监控在外 | 统计缓存命中和未命中的整体耗时 |
| 缓存 + 重试 | 缓存在外 | 命中缓存时避免执行重试逻辑 |
| 异常转换 + 重试 | 异常转换在外 | 先允许内部完成重试，最终再转换异常 |
| 参数规范化 + 缓存 | 规范化在外 | 提高缓存键一致性 |

具体顺序仍应根据业务要求测试。

---

## 十三、异步工具调用拦截

如果 Agent 使用 `ainvoke()` 或 `astream()`，可以通过 `@awrap_tool_call` 编写异步工具中间件。

### 实例：异步工具监控

## 实例

import time
from langchain.agents.middleware import awrap\_tool\_call

@awrap\_tool\_call
async def async\_tool\_monitor(request, handler):
    """统计异步工具调用耗时。"""
    start = time.perf\_counter()
    try:
        return await handler(request)
    finally:
        name = request.tool\_call["name"]
        print(f"[Async Tool] {name}：{time.perf\_counter() - start:.2f}s")

异步中间件需要使用：

```
await handler(request)
```

如果需要重试等待，应使用：

```
await asyncio.sleep(seconds)
```

避免在异步事件循环中直接使用 `time.sleep()`。

---

## 十四、完整实例：课程工具的参数处理与监控

下面把参数规范化、工具重试和执行监控组合起来，创建一个简单的课程查询 Agent。

文件名：`wrap_tool_demo.py`

安装依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

在 `.env` 中配置：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

import time
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import wrap\_tool\_call, ToolRetryMiddleware
from langchain.tools import tool
from langchain.messages import ToolMessage

load\_dotenv()

COURSES = {
    "python": "Python3 基础教程",
    "java": "Java 基础教程",
    "html": "HTML 基础教程"
}

@tool
def search\_course(keyword: str) -> str:
    """根据编程语言查询模拟课程。"""
    return COURSES.get(keyword, "未找到相关课程")

@wrap\_tool\_call
def normalize\_keyword(request, handler):
    """将搜索关键词转换为统一格式。"""
    call = request.tool\_call
    if call["name"] == "search\_course":
        args = dict(call["args"])
        if isinstance(args.get("keyword"), str):
            args["keyword"] = args["keyword"].strip().lower()
            request = request.override(tool\_call={\*\*call, "args": args})
    return handler(request)

@wrap\_tool\_call
def monitor\_tool(request, handler):
    """监控工具执行耗时。"""
    start = time.perf\_counter()
    try:
        result = handler(request)
        status = "失败" if isinstance(result, ToolMessage) and result.status == "error" else "成功"
        print(f"[Tool] {request.tool\_call['name']}：{status}")
        return result
    finally:
        print(f"[Tool] 耗时：{time.perf\_counter() - start:.2f}s")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course],
    middleware=[
        normalize\_keyword,
        monitor\_tool,
        ToolRetryMiddleware(tools=["search\_course"], max\_retries=2)
    ],
    system\_prompt="你是课程助手。查询课程时调用 search\_course，课程目录仅为教学模拟数据。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请使用工具查询 PYTHON 课程。"}]
})
print("最终回答：", result["messages"][-1].content)

### 可能的运行结果

```
[Tool] search_course：成功

[Tool] 耗时：0.01s

最终回答：可以学习 Python3 基础教程。
```

这个示例展示了三个中间件的协作：

1. `normalize_keyword`：统一工具输入参数。
2. `monitor_tool`：记录工具执行结果和耗时。
3. `ToolRetryMiddleware`：在符合条件的工具异常发生时重试。

这里的课程目录是静态模拟数据，实际不会出现网络故障，所以正常情况下不会触发重试。

是否调用工具取决于模型行为，运行结果仅供参考。

---

## 十五、@wrap\_tool\_call 与 @wrap\_model\_call 的区别

| 对比项 | @wrap\_model\_call | @wrap\_tool\_call |
| --- | --- | --- |
| 拦截对象 | 模型请求 | 工具执行 |
| 请求对象 | ModelRequest | ToolCallRequest |
| 主要参数 | model、messages、tools、system\_message | tool\_call、tool、state、runtime |
| 常见返回值 | ModelResponse、AIMessage 等 | ToolMessage、Command |
| 典型用途 | 模型重试、降级、动态请求修改 | 工具重试、参数规范化、结果缓存 |
| 内置重试 | ModelRetryMiddleware | ToolRetryMiddleware |

可以简单记住：

```
@wrap_model_call：管理模型怎么调用



@wrap_tool_call：管理工具怎么执行
```

两者都支持通过 handler 包装执行过程，但请求对象和返回协议并不相同。

---

## 十六、生产环境注意事项

### 1、不要盲目重试有副作用的工具

发送邮件、扣款、创建订单和删除文件等工具，可能存在重复执行风险。

如果必须重试，应在可信后端使用幂等键、事务或其他去重机制。

### 2、缓存必须考虑用户隔离

用户私人数据、订单信息和权限相关查询不能直接共享一个全局缓存。

缓存键需要包含合适的用户或租户范围，并设置缓存有效期。

### 3、工具参数需要后端校验

模型生成的工具参数不一定可靠。

Middleware 可以进行参数规范化，但真正的权限检查和业务验证必须由可信程序完成。

### 4、日志不要输出敏感参数

工具参数可能包含数据库查询条件、访问令牌和用户个人信息。

建议默认记录工具名称、调用状态和耗时，而不是完整参数。

### 5、不要吞掉所有异常

工具执行异常可能来自临时网络故障，也可能来自代码 Bug。

建议只捕获明确需要处理的异常类型，避免影响问题排查。

### 6、Command 需要保持工具消息协议

如果工具通过 Command 更新 State，应保留对应工具调用 ID 的 ToolMessage，避免产生不完整的工具调用历史。

---

## 十七、常见问题

### 1、@wrap\_tool\_call 与普通 @tool 有什么区别？

`@tool` 定义工具功能；`@wrap_tool_call` 拦截工具的执行过程。

### 2、handler(request) 返回什么？

通常返回 ToolMessage，使用状态更新的工具也可能返回 Command。

### 3、可以不调用 handler 吗？

可以。例如缓存命中时，可以直接构造与当前调用 ID 对应的 ToolMessage。

### 4、request.override() 会修改原始请求吗？

不会。它返回新的 ToolCallRequest，可以修改工具调用参数或受支持的状态字段。

### 5、ToolRetryMiddleware 的 max\_retries 是总执行次数吗？

不是。它表示第一次执行之后允许的额外重试次数。

### 6、为什么工具返回错误消息却没有触发异常重试？

因为 ToolMessage 的错误状态和 Python 异常不同。应根据具体工具行为设计错误处理逻辑。

### 7、可以缓存 Command 吗？

不建议简单缓存并复用 Command。它可能包含状态修改、消息关联和执行控制信息。

### 8、为什么缓存命中要创建新的 ToolMessage？

因为每次工具调用都有自己的 tool\_call\_id，需要返回与当前工具调用对应的消息。

### 9、怎样处理工具的空查询结果？

可以将其视为正常业务结果，并返回清晰的 ToolMessage，让 Agent 根据结果继续回答。

### 10、ToolErrorMiddleware 和 ToolRetryMiddleware 有什么区别？

前者负责将指定异常转换为错误 ToolMessage；后者负责对指定异常进行重试。

### 11、多个工具中间件的顺序重要吗？

重要。第一个中间件通常位于最外层，缓存、重试和错误处理的先后顺序会影响实际行为。

### 12、异步工具调用使用什么？

可以使用 `@awrap_tool_call`，并通过 `await handler(request)` 执行异步工具。

---

## 总结

`@wrap_tool_call` 是 LangChain Agent 进行工具执行拦截的重要机制。

通过它可以实现：

- 工具执行失败后的重试。
- 工具参数规范化和请求修改。
- 只读工具结果缓存。
- 工具调用日志和耗时统计。
- 工具异常处理和错误消息转换。
- 工具执行结果后处理。
- 多个工具 Middleware 的组合。

最基本的写法：

```
@wrap_tool_call

def my_wrapper(request, handler):

    # 执行前处理

    result = handler(request)

    # 执行后处理

    return result
```

需要修改参数时：

```
new_request = request.override(tool_call=new_tool_call)

result = handler(new_request)
```

需要返回缓存结果时：

```
return ToolMessage(

    content=cached_content,

    tool_call_id=request.tool_call["id"]

)
```

> **核心原则：**让工具负责业务功能，让 Middleware 负责执行控制。工具重试、缓存和错误处理必须保持 ToolMessage 与当前工具调用的正确关联，不能为了简化代码而破坏 Agent 的执行协议。
