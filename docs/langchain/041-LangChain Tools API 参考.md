# LangChain Tools API 参考

> 来源: https://www.runoob.com/langchain/langchain-tools-api.html

---

# LangChain Tools API 参考

在 LangChain 中，大语言模型主要负责理解问题和生成回答，而 Tools（工具） 则让模型具备执行具体操作的能力。

例如，Agent 可以通过工具查询数据库、搜索知识库、计算数学表达式，或者调用外部 API。

LangChain 1.x 提供了统一的 Tools API，最常用的是 `@tool` 装饰器和 `BaseTool` 工具抽象。

本章节系统整理 @tool 的参数、BaseTool 属性与方法、Pydantic 参数校验、ToolRuntime 依赖注入、工具返回值、异常处理以及 Agent 工具调用流程。

---

## 一、什么是 LangChain Tool？

Tool 是提供给模型调用的函数或操作接口。

一个 Tool 通常包含：

- **name：**工具名称，帮助模型识别功能。
- **description：**工具描述，告诉模型何时使用。
- **args\_schema：**工具参数结构。
- **执行逻辑：**真正执行查询、计算或其他操作的 Python 代码。

例如：

```
用户：帮我计算 25 + 17

         ↓

      Agent

         ↓

   选择 add_numbers

         ↓

      执行工具

         ↓

      返回 42

         ↓

   模型生成回答
```

工具调用有两个重要步骤：

1. 模型生成 ToolCall，提出调用哪个工具以及传入什么参数。
2. Agent 或应用程序执行工具，并通过 ToolMessage 返回结果。

**注意：**单纯使用 `model.bind_tools()` 并不会自动执行 Python 函数。如果希望自动执行工具循环，通常使用 `create_agent()`。

---

## 二、@tool 装饰器

`@tool` 是创建 LangChain Tool 最简单的方法。

它可以将普通 Python 函数转换为 BaseTool 对象，并根据函数签名推断工具输入参数。

### 基本语法

```
from langchain.tools import tool



@tool

def tool_name(param: str) -> str:

    """描述工具的用途。"""

    return f"结果：{param}"
```

函数的文档字符串非常重要，因为模型会根据工具名称、描述和参数 Schema 判断什么时候应该调用工具。

### 实例：创建第一个工具

## 实例

from langchain.tools import tool

@tool
def greet(name: str) -> str:
    """根据用户姓名生成一句问候语。"""
    return f"你好，{name}！"

print(greet.name)
print(greet.description)
print(greet.invoke({"name": "小明"}))

运行结果：

```
greet

根据用户姓名生成一句问候语。

你好，小明！
```

这里的 greet 已经不再只是普通 Python 函数，而是一个具有名称、描述、参数 Schema 和调用接口的 LangChain Tool。

---

## 三、@tool 完整参数

当前 LangChain 的 @tool 装饰器支持以下主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `name_or_callable` | None | 工具名称或需要转换的函数 |
| `runnable` | None | 将 Runnable 转换为工具 |
| `description` | None | 自定义工具描述，默认使用 docstring |
| `return_direct` | False | 工具返回后是否直接结束 Agent 工具循环 |
| `args_schema` | None | Pydantic 或 JSON Schema 参数定义 |
| `infer_schema` | True | 是否根据函数签名推断参数 Schema |
| `response_format` | "content" | 工具返回值的组织形式 |
| `parse_docstring` | False | 是否解析 Google 风格的参数文档 |
| `error_on_invalid_docstring` | True | 文档字符串解析失败时是否抛出异常 |
| `extras` | None | 附加提供商专用工具字段 |

常用形式包括：

```
@tool



@tool("custom_name")



@tool(description="自定义工具描述")



@tool(args_schema=MyInput)



@tool(return_direct=True)
```

**注意：**自定义工具名称通常使用 `@tool("custom_name")`，而不是假定当前 @tool 装饰器具有独立的 name 关键字参数。

---

## 四、自定义工具名称

默认情况下，工具名称就是 Python 函数名称。

如果希望修改名称，可以给 @tool 传入字符串。

### 实例

## 实例

from langchain.tools import tool

@tool("search\_python\_docs")
def search\_docs(keyword: str) -> str:
    """搜索模拟 Python 文档。"""
    return f"找到关于 {keyword} 的参考资料"

print(search\_docs.name)
print(search\_docs.invoke({"keyword": "装饰器"}))

输出：

```
search_python_docs

找到关于 装饰器 的参考资料
```

建议使用清晰的 snake\_case 名称，例如：

```
search_knowledge

get_weather

query_order

calculate_sum

send_email
```

避免使用包含空格、特殊符号或含义不明确的工具名称。

---

## 五、自定义工具描述

工具描述会影响模型的工具选择。

例如，如果只是写：

```
"""查询数据"""
```

模型可能不知道应该在什么情况下使用它。

更好的描述应该说明用途、适用范围和返回内容。

### 实例：自定义 description

## 实例

@tool(
    "search\_course",
    description="在模拟编程课程目录中查找课程。当用户询问 Python、Java 等课程时使用。"
)
def find\_course(keyword: str) -> str:
    """查找课程。"""
    return f"模拟查询结果：{keyword} 入门课程"

print(find\_course.description)

`description` 显式指定时，优先使用这个描述，而不是函数文档字符串。

**建议：**工具描述应包含"工具能做什么"和"什么时候调用"，但不要把工具描述写成冗长的系统提示词。

---

## 六、infer\_schema：自动生成参数 Schema

默认情况下：

```
infer_schema=True
```

LangChain 会根据 Python 函数的类型注解和签名生成参数 Schema。

### 实例

## 实例

@tool
def calculate\_sum(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b

print(calculate\_sum.args)
print(calculate\_sum.invoke({"a": 10, "b": 25}))

可能得到：

```
{

    "a": {"title": "A", "type": "integer"},

    "b": {"title": "B", "type": "integer"}

}



35
```

参数结构由函数定义自动生成。

建议给工具函数添加明确的类型注解，否则模型可能难以准确理解参数要求。

---

## 七、args\_schema：Pydantic 参数校验

当工具参数较复杂时，可以使用 Pydantic 定义输入结构。

例如，一个课程查询工具需要：

- 课程关键词不能为空。
- 查询数量为 1～20。
- 课程类别只能选择 Python、Java 或 JavaScript。

可以使用 `BaseModel`、`Field` 和 `Literal` 实现。

### 实例：参数校验

## 实例

from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool

class SearchInput(BaseModel):
    keyword: str = Field(
        min\_length=1,
        description="课程搜索关键词"
    )
    category: Literal["Python", "Java", "JavaScript"] = Field(
        description="课程类别"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最多返回多少条结果"
    )

@tool(args\_schema=SearchInput)
def search\_course(
    keyword: str,
    category: str,
    limit: int = 5
) -> str:
    """根据类别和关键词搜索模拟编程课程。"""
    return f"搜索 {category} / {keyword}，最多 {limit} 条"

result = search\_course.invoke({
    "keyword": "入门",
    "category": "Python",
    "limit": 10
})

print(result)

运行结果：

```
搜索 Python / 入门，最多 10 条
```

如果传入：

```
{

    "keyword": "",

    "category": "PHP",

    "limit": 100

}
```

则可能触发 Pydantic 参数验证错误，因为这些值不符合 Schema。

**注意：**Pydantic 可以校验参数类型和取值范围，但不能代替用户权限检查或实际业务规则校验。

---

## 八、parse\_docstring：解析参数说明

工具函数可以使用 Google 风格的 docstring 描述参数。

开启：

```
parse_docstring=True
```

后，LangChain 会尝试将 Args 中的说明提取到工具 Schema。

### 实例

## 实例

@tool(parse\_docstring=True)
def search\_article(keyword: str, limit: int = 5) -> str:
    """搜索模拟教程文章。

    Args:
        keyword: 需要查询的关键词。
        limit: 最多返回的文章数量。
    """
    return f"查询 {keyword}，最多 {limit} 篇"

print(search\_article.args)

需要注意，docstring 的格式必须符合解析要求。

如果格式不正确，并且：

```
error_on_invalid_docstring=True
```

则可能在创建工具时抛出 ValueError。

对于输入较复杂的工具，推荐优先使用 Pydantic args\_schema。

---

## 九、return\_direct：直接返回工具结果

正常情况下，Agent 调用工具后，会把工具结果返回给模型，再由模型生成最终回复。

```
模型请求工具

    ↓

执行工具

    ↓

工具返回结果

    ↓

再次调用模型

    ↓

最终回答
```

如果设置：

```
return_direct=True
```

工具执行后，Agent 可以直接结束当前工具循环，不再进行下一次普通模型回复。

### 实例：直接返回

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent

@tool(return\_direct=True)
def get\_order\_status(order\_id: str) -> str:
    """查询模拟订单状态。"""
    return f"模拟订单 {order\_id}：已完成"

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[get\_order\_status],
    system\_prompt="用户查询模拟订单时使用 get\_order\_status。"
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请查询模拟订单 ORD-001。"
    }]
})

print(result["messages"][-1].content)

如果模型调用了这个工具，Agent 可以在工具返回后结束执行。

**注意：**`return_direct=True` 并不代表 `agent.invoke()` 的返回值直接变成字符串。

create\_agent() 通常仍然返回 Agent State，工具结果可以从其中的 messages 读取。

此外，对于并行工具调用和复杂 Middleware，最终执行行为还需要结合实际 Agent 配置判断。

---

## 十、response\_format：工具返回值格式

@tool 的 `response_format` 支持两种常见形式：

| 值 | 说明 |
| --- | --- |
| `"content"` | 工具返回结果作为 ToolMessage.content |
| `"content_and_artifact"` | 工具返回 (content, artifact) 二元组 |

默认值是：

```
response_format="content"
```

适用于普通字符串、字典等工具结果。

### 实例：content\_and\_artifact

有时，我们希望模型只看到必要的摘要，而程序可以保留更完整的原始数据。

## 实例

from langchain.tools import tool

@tool(response\_format="content\_and\_artifact")
def search\_docs(keyword: str):
    """搜索模拟文档，返回摘要和原始资料。"""
    documents = [
        {
            "id": "doc-001",
            "title": "Python 函数",
            "content": "Python 函数用于封装重复逻辑。"
        }
    ]

    content = f"找到 {len(documents)} 条关于 {keyword} 的资料。"
    return content, documents

tool\_call = {
    "type": "tool\_call",
    "name": "search\_docs",
    "args": {"keyword": "Python"},
    "id": "call\_docs\_001"
}

result = search\_docs.invoke(tool\_call)

print("给模型的内容：", result.content)
print("原始资料：", result.artifact)

这里使用带 `tool_call_id` 的工具调用输入，因此返回 ToolMessage，可以读取：

```
result.content

result.artifact
```

其中：

- `content`：主要提供给模型阅读。
- `artifact`：可供应用程序保留和处理的附加原始数据。

**注意：**artifact 不应被当成秘密保险箱。它仍可能保存在运行状态或追踪数据中，敏感内容需要独立的访问控制。

---

## 十一、BaseTool 主要属性

通过 @tool 创建的工具通常是 BaseTool 的具体实现。

BaseTool 提供统一的工具描述、参数校验、调用和错误处理接口。

| 属性 | 说明 |
| --- | --- |
| `name` | 工具名称 |
| `description` | 工具功能描述 |
| `args_schema` | 工具输入 Schema |
| `args` | 模型可使用的参数结构 |
| `tool_call_schema` | 提供给模型的工具调用 Schema |
| `return_direct` | 是否直接结束 Agent 工具循环 |
| `response_format` | 工具结果格式 |
| `handle_tool_error` | ToolException 处理配置 |
| `handle_validation_error` | 参数校验异常处理配置 |
| `tags` | 追踪标签 |
| `metadata` | 工具附加元数据 |

### 实例：查看工具属性

## 实例

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a \* b

print("名称：", multiply.name)
print("描述：", multiply.description)
print("参数：", multiply.args)
print("直接返回：", multiply.return\_direct)
print("返回格式：", multiply.response\_format)

---

## 十二、BaseTool 常用方法

| 方法 | 说明 |
| --- | --- |
| `invoke(input, config=None)` | 同步执行工具 |
| `ainvoke(input, config=None)` | 异步执行工具 |
| `run(tool_input)` | 底层同步执行接口 |
| `arun(tool_input)` | 底层异步执行接口 |
| `get_input_schema()` | 获取输入 Schema |
| `model_copy()` | 复制工具配置对象 |

### 实例：invoke()

## 实例

result = multiply.invoke({"a": 6, "b": 7})
print(result)

运行结果：

```
42
```

### 实例：ainvoke()

## 实例

import asyncio

async def main():
    result = await multiply.ainvoke({"a": 8, "b": 9})
    print(result)

asyncio.run(main())

输出：

```
72
```

对于简单工具，直接使用 invoke() 或 ainvoke() 最方便。

---

## 十三、ToolRuntime：统一运行时依赖注入

在 LangChain 1.x 中，推荐使用 ToolRuntime 获取工具执行时的上下文。

此前的旧版示例可能使用：

```
InjectedState

InjectedStore

InjectedToolCallId
```

这些接口属于较早的依赖注入模式。

新项目推荐统一使用：

```
from langchain.tools import ToolRuntime
```

### ToolRuntime 主要属性

| 属性 | 用途 |
| --- | --- |
| `runtime.state` | 读取当前 Agent 状态 |
| `runtime.context` | 读取可信运行时上下文 |
| `runtime.store` | 访问跨会话长期存储 |
| `runtime.tool_call_id` | 获取当前工具调用 ID |
| `runtime.config` | 读取 RunnableConfig |
| `runtime.stream_writer` | 发送自定义流式进度 |
| `runtime.execution_info` | 读取执行标识与重试信息，需满足版本要求 |
| `runtime.server_info` | LangGraph Server 环境下的服务元数据 |

ToolRuntime 参数会在 Agent 执行工具时注入，不会作为普通工具参数暴露给模型。

---

## 十四、ToolRuntime.state：读取对话状态

如果工具需要查看当前会话中的消息数量，可以通过 `runtime.state` 获取。

### 实例：读取当前消息数量

## 实例

from langchain.tools import tool, ToolRuntime

@tool
def count\_messages(runtime: ToolRuntime) -> str:
    """统计当前对话状态中的消息数量。"""
    messages = runtime.state.get("messages", [])
    return f"当前共有 {len(messages)} 条消息。"

这里模型看到的工具参数 Schema 不包含 runtime。

如果在 Agent 中调用该工具，LangGraph 会自动提供对应状态。

**注意：**这种工具通常需要在 Agent 或 ToolNode 的执行上下文中调用。不要像普通工具那样假定 `count_messages.invoke({})` 一定能够独立获得正确的运行时状态。

---

## 十五、ToolRuntime.context：注入可信用户信息

对于用户个人资料、订单查询等业务，需要知道当前是谁在调用工具。

这类身份信息不应该由模型生成，而应该由后端认证系统提供。

可以通过 `context_schema` 和 `ToolRuntime` 结合实现。

### 实例：获取当前用户资料

## 实例

from dataclasses import dataclass
from langchain.agents import create\_agent
from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
    user\_id: str

profiles = {
    "user\_001": {"name": "小明", "level": "Python 初级"}
}

@tool
def get\_my\_profile(runtime: ToolRuntime[UserContext]) -> str:
    """查询当前已认证用户的模拟学习资料。"""
    user\_id = runtime.context.user\_id
    profile = profiles.get(user\_id)

    if not profile:
        return "没有找到当前用户的资料。"

    return f"姓名：{profile['name']}，学习阶段：{profile['level']}"

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[get\_my\_profile],
    context\_schema=UserContext
)

result = agent.invoke(
    {"messages": [{
        "role": "user",
        "content": "请用工具查看我的学习资料。"
    }]},
    context=UserContext(user\_id="user\_001")
)

print(result["messages"][-1].content)

这里的 user\_id 来自 Agent 调用方，而不是模型工具参数。

**安全提醒：**实际网站必须根据登录用户生成 context，不能允许浏览器传入任意 user\_id 查询他人信息。

---

## 十六、ToolRuntime.store：跨会话存储

ToolRuntime 也可以访问通过 Agent 配置的 Store。

Store 适合保存跨会话共享的偏好和长期数据。

### 实例：读取用户偏好

## 实例

from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime

store = InMemoryStore()

store.put(
    ("users", "user\_001"),
    "preference",
    {"language": "Python"}
)

@tool
def get\_preference(runtime: ToolRuntime[UserContext]) -> str:
    """读取当前用户的学习偏好。"""
    if runtime.store is None:
        return "长期存储未配置。"

    user\_id = runtime.context.user\_id
    item = runtime.store.get(("users", user\_id), "preference")

    return str(item.value) if item else "没有保存偏好。"

这里需要将 Store 传给 create\_agent()，并通过可信 context 获取当前用户 ID。

完整创建方式：

```
agent = create_agent(

    model="deepseek:deepseek-chat",

    tools=[get_preference],

    context_schema=UserContext,

    store=store

)
```

**注意：**InMemoryStore 只适合教学和本地测试。需要跨进程持久化时，应使用 PostgreSQL 等持久化 Store 实现。

---

## 十七、ToolRuntime.tool\_call\_id：获取调用 ID

每次模型调用工具时，通常会为 ToolCall 分配一个 ID。

这个 ID 用于关联：

```
AIMessage.tool_calls

          ↓

    ToolMessage

          ↓

   相同 tool_call_id
```

工具可以通过以下属性获取当前调用 ID：

```
runtime.tool_call_id
```

当工具需要返回 Command 更新 Agent 状态时，该 ID 尤其重要。

---

## 十八、工具更新 Agent State：Command

普通工具返回字符串，只会把结果传给模型。

如果希望修改 Agent State 中的其他字段，可以通过 `Command(update=...)` 返回状态更新。

同时应该返回与当前工具调用 ID 匹配的 ToolMessage，避免消息历史出现未完成的工具调用。

### 实例：设置用户偏好

## 实例

from typing import NotRequired
from langchain.agents import AgentState, create\_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

class CustomState(AgentState):
    preferred\_language: NotRequired[str]

@tool
def set\_language(
    language: str,
    runtime: ToolRuntime[None, CustomState]
) -> Command:
    """将当前会话的首选编程语言保存到 Agent 状态。"""
    return Command(
        update={
            "preferred\_language": language,
            "messages": [
                ToolMessage(
                    content=f"首选编程语言已设置为 {language}",
                    tool\_call\_id=runtime.tool\_call\_id,
                    name="set\_language"
                )
            ]
        }
    )

agent = create\_agent(
    model="deepseek:deepseek-chat",
    tools=[set\_language],
    state\_schema=CustomState
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请调用工具，把我的首选编程语言设置为 Python。"
    }]
})

print("当前状态：", result.get("preferred\_language"))

如果模型调用了 set\_language，该工具会更新状态字段：

```
preferred_language = "Python"
```

**注意：**普通工具返回值不会自动写入任意自定义 State 字段。需要明确返回 Command 或通过其他图节点更新状态。

---

## 十九、ToolRuntime.stream\_writer：工具执行进度

对于执行时间较长的工具，可以在执行过程中发送自定义进度消息。

这适合文件解析、批量处理和知识库索引等任务。

### 实例：发送自定义进度

## 实例

from langchain.tools import tool, ToolRuntime

@tool
def process\_documents(runtime: ToolRuntime) -> str:
    """模拟处理多个知识库文档。"""
    runtime.stream\_writer({"progress": 0, "message": "开始处理"})
    runtime.stream\_writer({"progress": 50, "message": "正在建立索引"})
    runtime.stream\_writer({"progress": 100, "message": "处理完成"})
    return "已模拟处理文档"

Agent 可以使用：

```
stream_mode="custom"
```

接收自定义事件。

**注意：**stream\_writer 应在 LangGraph 执行上下文中使用。这些进度数据仅为演示，不代表真实任务已经完成。

---

## 二十、ToolException：工具业务异常

工具执行时可能出现两类常见错误：

- 业务异常：例如课程 ID 不合法、查询条件无效。
- 系统异常：例如数据库连接断开、网络超时。

LangChain 提供 `ToolException`，可以表示工具执行错误。

但 ToolException 并不意味着异常一定会自动转换成可读结果。

实际行为取决于 `handle_tool_error` 或 Agent Middleware 的配置。

### 实例：ToolException

## 实例

from langchain.tools import tool, ToolException

@tool
def get\_course(course\_id: int) -> str:
    """查询指定 ID 的模拟课程。"""
    if course\_id <= 0:
        raise ToolException("课程 ID 必须大于 0")

    return f"模拟课程 {course\_id}：Python 入门"

get\_course.handle\_tool\_error = True

print(get\_course.invoke({"course\_id": -1}))

工具在配置了 handle\_tool\_error=True 后，可以将 ToolException 的错误信息作为工具结果返回。

例如：

```
课程 ID 必须大于 0
```

如果没有启用相应处理，异常可能继续向上抛出。

---

## 二十一、handle\_tool\_error 与 handle\_validation\_error

BaseTool 提供两个容易混淆的配置：

| 参数 | 用途 |
| --- | --- |
| `handle_tool_error` | 处理工具执行时抛出的 ToolException |
| `handle_validation_error` | 处理 Pydantic 参数校验异常 |

例如：

## 实例

from pydantic import BaseModel, Field
from langchain.tools import tool, ToolException

class QueryInput(BaseModel):
    course\_id: int = Field(gt=0)

@tool(args\_schema=QueryInput)
def query\_course(course\_id: int) -> str:
    """根据 ID 查询模拟课程。"""
    if course\_id == 999:
        raise ToolException("课程不存在")

    return f"课程编号：{course\_id}"

query\_course.handle\_tool\_error = True
query\_course.handle\_validation\_error = "课程编号必须是正整数"

print(query\_course.invoke({"course\_id": 1}))
print(query\_course.invoke({"course\_id": -1}))
print(query\_course.invoke({"course\_id": 999}))

示意输出：

```
课程编号：1

课程编号必须是正整数

课程不存在
```

两个配置可以分别处理参数验证错误和工具主动抛出的 ToolException。

**注意：**它们不能处理所有 Python 异常。对于数据库异常、网络错误等，还可以使用 ToolRetryMiddleware 或 ToolErrorMiddleware。

---

## 二十二、旧版 InjectedState / InjectedStore 如何迁移？

早期 LangGraph 示例可能使用依赖注入标记：

| 旧接口 | 推荐替代方式 |
| --- | --- |
| `InjectedState` | `ToolRuntime.state` |
| `InjectedStore` | `ToolRuntime.store` |
| `InjectedToolCallId` | `ToolRuntime.tool_call_id` |
| `InjectedToolArg` | 优先考虑 ToolRuntime 或具体运行时接口 |

### 旧写法

```
from typing import Annotated

from langgraph.prebuilt import InjectedState



@tool

def count_messages(

    state: Annotated[dict, InjectedState]

) -> str:

    return str(len(state["messages"]))
```

### 新写法

```
from langchain.tools import tool, ToolRuntime



@tool

def count_messages(runtime: ToolRuntime) -> str:

    """统计当前对话消息数量。"""

    return str(len(runtime.state["messages"]))
```

新写法更加统一，也方便获取 context、store 和 tool\_call\_id。

对于正在维护的旧版 LangGraph 项目，不一定需要立即迁移，但新建的 LangChain 1.x 项目建议使用 ToolRuntime。

---

## 二十三、工具参数中的保留名称

LangChain 的工具执行系统会自动注入某些运行时参数。

因此，普通业务参数不应该使用这些名称：

| 名称 | 问题 |
| --- | --- |
| `config` | 与 RunnableConfig 自动注入冲突 |
| `runtime` | 保留给 ToolRuntime |
| `run_manager` | 与内部回调管理机制冲突 |
| `callbacks` | 与内部回调参数冲突 |

例如，不推荐：

```
@tool

def search_docs(config: str) -> str:

    """搜索文档。"""

    return config
```

推荐：

```
@tool

def search_docs(query: str) -> str:

    """搜索文档。"""

    return query
```

如果确实需要获取 RunnableConfig，应通过 ToolRuntime 提供的对应属性访问。

---

## 二十四、Tool 与 Agent 的调用关系

直接调用工具和在 Agent 中调用工具，行为不完全相同。

| 调用方式 | 执行行为 |
| --- | --- |
| `tool.invoke()` | 应用直接执行工具 |
| `model.bind_tools()` | 模型可以产生 ToolCall，但不自动执行工具 |
| `agent.invoke()` | Agent 可以自动管理工具执行循环 |

典型 Agent 工具调用流程：

```
用户消息

    ↓

AIMessage(tool_calls)

    ↓

匹配工具名称

    ↓

校验 args_schema

    ↓

注入 ToolRuntime

    ↓

执行 Python 工具

    ↓

生成 ToolMessage

    ↓

返回模型

    ↓

最终回答
```

工具返回的内容默认属于模型参考数据，不应该被赋予系统指令权限。

---

## 二十五、完整实例：创建多工具 Agent

下面通过一个可以直接运行的示例，演示：

- @tool 创建工具。
- Pydantic 参数校验。
- return\_direct 配置。
- BaseTool 属性读取。
- ToolException 异常处理。
- ToolRuntime.context。
- create\_agent() 组合多个工具。

文件名：`tools_api_demo.py`

安装依赖：

```
pip install -U langchain langchain-deepseek pydantic python-dotenv
```

在 `.env` 中配置：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dataclasses import dataclass
from typing import Literal
from dotenv import load\_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create\_agent
from langchain.tools import tool, ToolRuntime, ToolException
from langchain\_deepseek import ChatDeepSeek

load\_dotenv()

# ========== 1. 普通工具 ==========

@tool
def add\_numbers(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b

# ========== 2. 参数校验工具 ==========

class CourseInput(BaseModel):
    language: Literal["Python", "Java"] = Field(description="编程语言")
    level: Literal["beginner", "advanced"] = Field(
        default="beginner",
        description="学习难度"
    )

@tool(args\_schema=CourseInput)
def search\_course(
    language: str,
    level: str = "beginner"
) -> str:
    """搜索模拟编程课程。"""
    return f"找到 {language} / {level} 模拟课程"

# ========== 3. 直接返回工具 ==========

@tool(return\_direct=True)
def get\_demo\_status() -> str:
    """返回模拟系统状态。"""
    return "模拟系统运行正常"

# ========== 4. ToolException ==========

@tool
def get\_course\_by\_id(course\_id: int) -> str:
    """根据课程编号查询模拟课程。"""
    if course\_id <= 0:
        raise ToolException("课程编号必须大于 0")
    return f"模拟课程 {course\_id}：Python 入门"

get\_course\_by\_id.handle\_tool\_error = True

# ========== 5. ToolRuntime Context ==========

@dataclass
class UserContext:
    user\_id: str

@tool
def get\_current\_user(runtime: ToolRuntime[UserContext]) -> str:
    """查询当前用户的模拟标识。"""
    return f"当前用户：{runtime.context.user\_id}"

# ========== 6. 直接测试工具 ==========

print("工具名称：", add\_numbers.name)
print("工具描述：", add\_numbers.description)
print("参数 Schema：", add\_numbers.args)

print("计算结果：", add\_numbers.invoke({"a": 12, "b": 30}))

print("课程查询：", search\_course.invoke({
    "language": "Python",
    "level": "beginner"
}))

print("错误处理：", get\_course\_by\_id.invoke({"course\_id": -1}))

# ========== 7. 创建 Agent ==========

agent = create\_agent(
    model=ChatDeepSeek(model="deepseek-chat", temperature=0),
    tools=[
        add\_numbers,
        search\_course,
        get\_demo\_status,
        get\_course\_by\_id,
        get\_current\_user
    ],
    context\_schema=UserContext,
    system\_prompt="""你是编程学习助手。

根据问题选择适当的工具。
计算问题使用 add\_numbers。
课程查询使用 search\_course。
查询当前用户时使用 get\_current\_user。
不要编造真实课程和系统状态。"""
)

# ========== 8. 测试 Agent ==========

questions = [
    "请使用工具计算 12 + 30。",
    "请使用工具搜索 Python 入门课程。",
    "请通过工具查看当前用户标识。"
]

for question in questions:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        context=UserContext(user\_id="student-001")
    )

    print("\n问题：", question)
    print("回答：", result["messages"][-1].content)

    tool\_msgs = [
        msg for msg in result["messages"]
        if msg.type == "tool"
    ]
    print("工具执行次数：", len(tool\_msgs))

在这个示例中，前面几个工具可以直接使用 invoke() 测试。

而 get\_current\_user 需要由 Agent 注入 ToolRuntime，不应该在没有运行时上下文的情况下直接调用。

同时，Agent 是否调用某个工具，仍取决于模型实际产生的工具调用请求。

---

## 二十六、常见问题

### 1、@tool 和普通 Python 函数有什么区别？

@tool 会将函数转换成具有名称、描述、参数 Schema 和统一调用接口的 LangChain Tool。

### 2、@tool 必须写 docstring 吗？

通常应提供描述。可以使用函数 docstring，也可以显式设置 description。

### 3、如何修改工具名称？

可以使用 `@tool("custom_name")`。

### 4、args\_schema 不填写会怎样？

默认根据函数签名和类型注解自动推断工具参数。

### 5、Pydantic 能阻止所有错误参数吗？

不能。它主要负责数据格式和约束校验，业务权限和实际数据有效性仍需单独检查。

### 6、return\_direct=True 会直接返回字符串吗？

在 Agent 中它可以跳过后续普通模型回复，但 create\_agent.invoke() 通常仍返回 Agent State。

### 7、ToolCall 和 ToolMessage 有什么区别？

ToolCall 是模型提出的工具调用请求；ToolMessage 是工具执行后的结果消息。

### 8、ToolRuntime 是模型自己生成的吗？

不是。ToolRuntime 由 LangChain 执行环境自动注入，不暴露为模型的普通工具参数。

### 9、InjectedState 和 InjectedStore 还能使用吗？

旧项目中仍可能使用，但新教程推荐 ToolRuntime 提供的统一接口。

### 10、ToolRuntime.state 和 context 有什么区别？

state 保存执行中的状态；context 提供当前调用的可信外部信息。

### 11、ToolRuntime.store 会自动保存聊天历史吗？

不会。Store 主要用于长期数据；聊天状态通常由 Checkpointer 管理。

### 12、工具如何修改 AgentState？

可以返回 Command(update=...)，并包含与当前工具调用匹配的 ToolMessage。

### 13、ToolException 会自动重试吗？

不会。需要根据 handle\_tool\_error 或 Middleware 设置决定处理方式。

### 14、handle\_tool\_error 和 handle\_validation\_error 有什么区别？

前者处理工具执行时的 ToolException，后者处理输入参数校验错误。

### 15、response\_format="content\_and\_artifact" 有什么作用？

它允许工具同时返回给模型的内容，以及供应用处理的额外原始数据。

### 16、工具能发送执行进度吗？

可以。通过 ToolRuntime.stream\_writer 发送自定义流事件。

### 17、工具调用需要注意哪些安全问题？

需要校验参数、验证身份与权限、限制资源消耗，并为邮件、支付、删除等副作用操作加入确认和幂等保护。

### 18、工具一定会被模型调用吗？

不一定。工具选择由模型决定，需要强制执行的业务流程应该使用确定性工作流。

---

## 总结

LangChain Tools API 是 Agent 与外部系统交互的基础。

本章节主要介绍了：

- `@tool`：将 Python 函数转换为工具。
- `args_schema`：通过 Pydantic 或 JSON Schema 定义参数。
- `infer_schema`：自动推断工具参数。
- `parse_docstring`：解析参数文档。
- `return_direct`：直接结束工具执行循环。
- `response_format`：配置工具返回结构。
- `BaseTool`：工具统一抽象接口。
- `invoke()` 和 `ainvoke()`：同步与异步执行。
- `ToolRuntime`：运行时依赖注入。
- `runtime.state`：读取当前状态。
- `runtime.context`：读取可信用户上下文。
- `runtime.store`：访问跨会话长期存储。
- `runtime.tool_call_id`：关联工具请求与结果。
- `Command`：修改 Agent State。
- `ToolException`：处理工具执行异常。
- `handle_tool_error` 和 `handle_validation_error`：分类处理错误。

最常见的工具写法：

```
from langchain.tools import tool



@tool

def search_course(keyword: str) -> str:

    """搜索模拟编程课程。"""

    return f"找到与 {keyword} 相关的课程"
```

在 Agent 中注册：

```
agent = create_agent(

    model="deepseek:deepseek-chat",

    tools=[search_course]

)
```

**核心原则：**让模型负责选择工具，让 Tool 负责执行可验证的操作，让 ToolRuntime 提供可信上下文。涉及真实订单、支付、发送邮件或文件删除时，必须由应用程序负责权限校验、幂等控制与必要的人工审批。

## 参考文档

- [LangChain Tools 官方文档](https://docs.langchain.com/oss/python/langchain/tools)
- [@tool API Reference](https://reference.langchain.com/python/langchain-core/tools/convert/tool)
- [BaseTool API Reference](https://reference.langchain.com/python/langchain-core/tools/base/BaseTool)
- [StructuredTool API Reference](https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool)
- [LangChain Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain 内置 Middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
