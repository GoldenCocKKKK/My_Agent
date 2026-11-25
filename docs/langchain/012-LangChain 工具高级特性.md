# LangChain 工具高级特性

> 来源: https://www.runoob.com/langchain/langchain-tool-advanced.html

---

# LangChain 工具高级特性

上一节我们学习了如何通过 `@tool` 装饰器定义工具，并将工具注册到 LangChain Agent 中。

实际开发时，除了定义工具的名称、参数和执行逻辑，还需要解决一些问题：

- 工具执行完成后，能不能直接返回结果，不再调用模型？
- 如何获取当前工具调用的唯一 ID？
- 工具运行失败时，如何避免整个 Agent 中断？
- 如何让模型根据工具错误信息调整下一步操作？

本章节介绍四个重要的工具高级特性：

- `return_direct`：控制工具执行后是否直接结束 Agent 循环。
- `InjectedToolCallId`：在工具执行时注入当前调用 ID。
- `ToolException`：用于表示工具执行过程中的可预期错误。
- `handle_tool_error`：控制工具如何处理 ToolException。

本教程采用 LangChain 1.x 的 `create_agent()` API，示例使用 DeepSeek Flash 模型。

---

## 一、安装与环境准备

首先安装依赖：

```
pip install -U langchain langchain-deepseek python-dotenv
```

在项目目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

在 Python 中读取环境变量并初始化模型：

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

本章的工具示例均使用模拟数据，不会实际访问菜鸟教程数据库、真实订单系统或天气服务。

需要注意，Tool Calling 的具体行为由模型能力与 LangChain 执行框架共同决定。模型是否选择工具、一次选择多少个工具，并不是完全固定的。

---

## 二、return\_direct：直接返回工具结果

默认情况下，Agent 执行工具之后，会将工具结果发送回模型，让模型根据结果继续回答。

执行流程通常如下：

```
用户提问

   ↓

模型请求调用工具

   ↓

工具执行

   ↓

结果返回模型

   ↓

模型生成最终回答
```

对于一些查询类工具，工具结果本身已经是完整的回答，不一定需要模型再次加工。

这时可以设置：

```
@tool(return_direct=True)
```

在支持该功能的 Agent 执行框架中，工具执行完成后可以直接结束当前 Agent 循环。

### 1、普通工具与 return\_direct 工具

## 实例

from langchain.tools import tool

# 普通工具
@tool
def search\_normal(keyword: str) -> str:
    """搜索指定关键词的模拟编程课程。"""

    courses = {
        "python": "Python 基础、Python 数据分析",
        "java": "Java 基础、Java 面向对象"
    }

    return courses.get(
        keyword.lower(),
        "没有找到匹配的模拟课程"
    )

# 直接返回工具
@tool(return\_direct=True)
def search\_direct(keyword: str) -> str:
    """搜索模拟课程并直接返回搜索结果。"""

    courses = {
        "python": "Python 基础、Python 数据分析",
        "java": "Java 基础、Java 面向对象"
    }

    return courses.get(
        keyword.lower(),
        "没有找到匹配的模拟课程"
    )

print("普通工具：", search\_normal.return\_direct)
print("直接返回工具：", search\_direct.return\_direct)

执行结果：

```
普通工具： False

直接返回工具： True
```

可以通过工具对象的 `return_direct` 属性检查是否启用直接返回。

### 2、在 Agent 中比较两种模式

下面分别创建两个 Agent，一个使用普通工具，另一个使用直接返回工具。

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 普通工具 Agent
agent\_normal = create\_agent(
    model=model,
    tools=[search\_normal],
    system\_prompt=(
        "你是一个课程查询助手。"
        "查询课程时使用提供的工具。"
        "所有课程数据均为模拟数据。"
    )
)

# 直接返回工具 Agent
agent\_direct = create\_agent(
    model=model,
    tools=[search\_direct],
    system\_prompt=(
        "你是一个课程查询助手。"
        "查询课程时使用提供的工具。"
        "所有课程数据均为模拟数据。"
    )
)

question = {
    "messages": [
        {
            "role": "user",
            "content": "请使用工具搜索 python 课程"
        }
    ]
}

def show\_result(agent):
    result = agent.invoke(question)

    for message in result["messages"]:
        print(
            f"[{message.type}]",
            message.content
        )

    print("最后消息类型：", result["messages"][-1].type)
    print("最后消息内容：", result["messages"][-1].content)

print("=== 普通工具 ===")
show\_result(agent\_normal)

print("\n=== return\_direct 工具 ===")
show\_result(agent\_direct)

如果模型选择调用对应工具，普通模式通常会让模型再次处理工具结果；直接返回模式则可能在工具执行后结束当前 Agent 循环。

两者的区别如下：

| 模式 | 工具执行后的处理 | 适用场景 |
| --- | --- | --- |
| `return_direct=False` | 通常将工具结果交给模型继续处理 | 需要分析、总结或继续调用工具 |
| `return_direct=True` | 满足直接返回条件时结束 Agent 循环 | 工具结果已经可以直接展示 |

### 3、最终消息不一定是 AIMessage

使用普通工具时，Agent 的最后一条消息通常是模型生成的 `AIMessage`。

但使用 `return_direct=True` 时，Agent 可能直接在工具执行后结束，因此最后一条消息可能是 `ToolMessage`。

例如：

```
result = agent_direct.invoke(question)



last_message = result["messages"][-1]



print(last_message.type)

print(last_message.content)
```

如果工具执行成功并触发直接返回，可能输出：

```
tool

Python 基础、Python 数据分析
```

因此，不能假设所有 Agent 的最终输出都必须是 AIMessage。

### 4、多个工具与 return\_direct

对于当前 LangChain 1.x 的 `create_agent()`，如果模型在同一轮生成多个客户端工具调用，执行框架会结合这一批工具的 `return_direct` 属性判断是否结束循环。

当前实现要求这一批已执行的客户端工具都设置了 `return_direct=True`，才会通过该条件直接结束。

例如：

| 本轮工具调用 | 是否满足 return\_direct 结束条件 |
| --- | --- |
| 只有一个直接返回工具 | 是 |
| 两个工具均为 return\_direct=True | 是 |
| 一个普通工具、一个直接返回工具 | 否 |
| 只有普通工具 | 否 |

**注意：**这是当前 `create_agent()` 的执行行为，不应简单推广到旧版 `AgentExecutor` 或自定义 Agent 框架。

此外，即使工具被设置为直接返回，也不代表工具执行一定成功。生产环境仍然需要检查工具执行状态。

如果业务要求某个步骤一定执行、一定终止或一定完成后续汇总，应该通过程序逻辑明确编排。

---

## 三、InjectedToolCallId：注入工具调用 ID

当模型请求调用工具时，每一次工具调用通常会有一个唯一标识：

```
tool_call_id
```

例如：

```
{

    "name": "log_user_action",

    "args": {

        "action": "查询 Python 课程"

    },

    "id": "call_log_001",

    "type": "tool_call"

}
```

有时候，工具内部需要获取当前这一次调用的 ID，以便关联日志、工具结果或 Agent 状态更新。

LangChain 提供 `InjectedToolCallId`，可以通过参数注解，让运行时注入对应的工具调用 ID。

### 1、基本用法

使用 `Annotated` 标注工具参数：

```
from typing import Annotated

from langchain.tools import InjectedToolCallId



tool_call_id: Annotated[str, InjectedToolCallId]
```

这个参数与普通参数不同，不需要由模型生成。

### 实例：记录工具调用 ID

## 实例

from typing import Annotated

from langchain.tools import (
    tool,
    InjectedToolCallId
)

@tool
def log\_user\_action(
    action: str,
    tool\_call\_id: Annotated[str, InjectedToolCallId]
) -> str:
    """记录一条模拟操作日志。"""

    return (
        f"操作：{action}；"
        f"工具调用 ID：{tool\_call\_id}"
    )

# 手动构造一次 ToolCall
tool\_call = {
    "name": "log\_user\_action",
    "args": {
        "action": "查询 Python 课程"
    },
    "id": "call\_log\_001",
    "type": "tool\_call"
}

# 将完整 ToolCall 交给工具
result = log\_user\_action.invoke(tool\_call)

print("返回类型：", type(result).\_\_name\_\_)
print("返回内容：", result.content)
print("关联 ID：", result.tool\_call\_id)

执行结果类似：

```
返回类型： ToolMessage

返回内容： 操作：查询 Python 课程；工具调用 ID：call_log_001

关联 ID： call_log_001
```

这里传入的是完整的 ToolCall 字典，其中包含 `id`。

LangChain 根据它注入 `tool_call_id`，因此工具函数可以获取当前调用的标识。

### 2、InjectedToolCallId 不属于用户参数

普通工具参数由模型提供，例如：

```
action="查询 Python 课程"
```

但 `tool_call_id` 属于运行时上下文。

使用 `InjectedToolCallId` 标记后，该字段不会作为普通参数暴露给模型。

可以通过以下代码查看工具向模型公开的参数：

```
print(log_user_action.tool_call_schema.model_json_schema())
```

公开参数通常只包含 `action`，而不会要求模型填写 `tool_call_id`。

**注意：**工具调用 ID 只是当前 Tool Call 的标识，不代表用户身份、登录状态或权限信息。

如果业务需要识别真实用户，应从经过认证的应用上下文获取身份，不能使用工具调用 ID 代替用户 ID。

### 3、适用场景

`InjectedToolCallId` 常用于：

- 关联模型请求与工具执行日志。
- 跟踪某次工具调用的执行结果。
- 构造与当前 Tool Call 对应的 ToolMessage。
- 结合 LangGraph Command 更新 Agent 状态。

如果工具需要更多运行时信息，例如状态、上下文或存储对象，也可以进一步学习 LangChain 的 `ToolRuntime`。

---

## 四、ToolException：工具执行异常

在实际开发中，工具并不总能成功执行。

例如：

- 查询的课程不存在。
- 用户提供了无效参数。
- 外部 API 请求失败。
- 数据库没有找到相关记录。

对于工具执行过程中预期可能出现的错误，可以使用 `ToolException` 表示。

`ToolException` 是 LangChain 定义的工具异常类型，用于配合工具错误处理机制。

### 实例：模拟用户信息查询

## 实例

from langchain.tools import (
    tool,
    ToolException
)

@tool
def get\_user\_info(user\_id: int) -> str:
    """查询指定用户的模拟资料。"""

    if user\_id <= 0:
        raise ToolException(
            "用户 ID 必须是正整数"
        )

    users = {
        1: "张三，普通用户",
        2: "李四，普通用户"
    }

    if user\_id not in users:
        raise ToolException(
            f"未找到 ID 为 {user\_id} 的用户"
        )

    return users[user\_id]

# 正常调用
print(get\_user\_info.invoke({
    "user\_id": 1
}))

# 无效参数
try:
    get\_user\_info.invoke({
        "user\_id": -1
    })

except ToolException as error:
    print("工具异常：", error)

# 用户不存在
try:
    get\_user\_info.invoke({
        "user\_id": 999
    })

except ToolException as error:
    print("工具异常：", error)

执行结果：

```
张三，普通用户

工具异常： 用户 ID 必须是正整数

工具异常： 未找到 ID 为 999 的用户
```

这里的 `ToolException` 没有被工具内部处理，因此向调用方抛出。

我们通过 `try/except` 捕获错误。

### ToolException 与普通异常

| 类型 | 主要用途 |
| --- | --- |
| `ToolException` | 表示希望通过工具错误处理机制管理的执行错误 |
| `ValueError` | 表示参数值不符合要求等一般 Python 错误 |
| `TimeoutError` | 表示超时问题 |
| `ValidationError` | 表示 Pydantic 输入验证失败 |

**注意：**抛出 `ToolException` 不代表 Agent 一定会自动恢复错误。

是否将异常转换为工具结果，取决于工具自身的 `handle_tool_error` 设置，以及 Agent 的工具执行与中间件处理机制。

---

## 五、handle\_tool\_error：处理工具异常

`handle_tool_error` 是 LangChain `BaseTool` 的属性，用于控制工具遇到 `ToolException` 时如何处理。

默认值为：

```
handle_tool_error = False
```

也就是说，默认情况下，ToolException 会继续向外抛出。

如果希望工具将错误转换为可读取的结果，可以设置：

```
my_tool.handle_tool_error = True
```

### 1、正确设置 handle\_tool\_error

需要特别注意，在当前 LangChain 官方 API 中，`@tool` 装饰器并没有 `handle_tool_error` 这个直接参数。

因此，不要写成：

```
# 错误写法：当前 @tool 不接受这个参数

@tool(handle_tool_error=True)
```

推荐先创建工具对象，再设置相应属性。

### 实例：转换为工具错误信息

## 实例

from langchain.tools import (
    tool,
    ToolException
)

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。"""

    weather\_data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C",
        "上海": "小雨，22°C"
    }

    if city not in weather\_data:
        raise ToolException(
            f"未收录城市：{city}。"
            "可查询杭州、北京和上海。"
        )

    return (
        f"{city}的模拟天气："
        f"{weather\_data[city]}"
    )

# 设置工具错误处理策略
get\_weather.handle\_tool\_error = True

# 输入未收录的城市
result = get\_weather.invoke({
    "city": "北境"
})

print(result)

执行结果：

```
未收录城市：北境。可查询杭州、北京和上海。
```

由于设置了 `handle_tool_error=True`，工具会将 `ToolException` 的错误文本作为结果返回，而不是继续向调用方抛出该异常。

这里调用工具时传入的是普通参数字典，因此结果以普通内容形式返回。

如果通过完整 ToolCall 执行，包含工具调用 ID，LangChain 可以生成带有错误状态的 `ToolMessage`。

### 2、使用 ToolCall 查看错误状态

## 实例

# 复用已经定义的 get\_weather 工具

tool\_call = {
    "name": "get\_weather",
    "args": {
        "city": "北境"
    },
    "id": "call\_weather\_001",
    "type": "tool\_call"
}

result = get\_weather.invoke(tool\_call)

print("消息类型：", type(result).\_\_name\_\_)
print("执行状态：", result.status)
print("工具结果：", result.content)
print("工具调用 ID：", result.tool\_call\_id)

执行结果：

```
消息类型： ToolMessage

执行状态： error

工具结果： 未收录城市：北境。可查询杭州、北京和上海。

工具调用 ID： call_weather_001
```

`status="error"` 表示这次工具调用发生了错误。

它与正常的工具执行结果不同，即使错误信息已经转换为文本，也不应该将其当作业务操作成功的证明。

---

## 六、handle\_tool\_error 的四种配置方式

工具对象的 `handle_tool_error` 支持多种配置。

| 配置 | 行为 |
| --- | --- |
| `False` | 不处理 ToolException，继续向外抛出 |
| `True` | 将 ToolException 的错误信息作为工具结果 |
| `str` | 返回指定的固定错误提示 |
| `Callable` | 调用自定义错误处理函数生成结果 |

### 1、False：默认抛出异常

```
get_weather.handle_tool_error = False
```

工具出现 ToolException 时，需要由调用方或执行框架处理。

### 2、True：返回原始错误信息

```
get_weather.handle_tool_error = True
```

这种方式适合需要把业务错误交给模型理解的场景。

但如果原始异常包含内部信息或敏感数据，不建议直接返回。

### 3、str：返回固定错误提示

```
get_weather.handle_tool_error = (

    "天气查询失败，请检查城市名称。"

)
```

此时即使内部 ToolException 的错误内容不同，对外也会返回指定的统一提示。

这种方式适合隐藏内部错误细节。

### 4、Callable：自定义错误处理

## 实例

from langchain.tools import ToolException

def handle\_weather\_error(
    error: ToolException
) -> str:
    """将工具异常转换为安全的提示。"""

    return (
        "天气查询未成功。"
        "请检查城市名称，"
        "或改为查询杭州、北京、上海。"
    )

get\_weather.handle\_tool\_error = (
    handle\_weather\_error
)

result = get\_weather.invoke({
    "city": "北境"
})

print(result)

执行结果：

```
天气查询未成功。请检查城市名称，或改为查询杭州、北京、上海。
```

通过自定义函数，可以统一处理错误内容，避免向模型暴露内部系统信息。

### handle\_tool\_error 与 handle\_validation\_error

还需要区分两种错误处理属性：

| 属性 | 处理对象 |
| --- | --- |
| `handle_tool_error` | 工具执行时抛出的 ToolException |
| `handle_validation_error` | 工具输入 Schema 验证失败产生的相关验证错误 |

例如，如果 Pydantic 要求 `user_id` 为整数，而模型传入了不合法的数据，可能在执行函数之前就触发验证错误。

此时不能假设设置 `handle_tool_error=True` 就可以自动捕获所有参数验证错误。

---

## 七、工具错误处理与 Agent 的关系

当工具注册到 `create_agent()` 后，Agent 会根据模型产生的工具调用请求执行工具。

如果工具将 ToolException 转换为工具结果，Agent 就可以把这条结果加入消息历史，再交给模型处理。

例如：

```
用户请求查询城市

    ↓

模型调用 get_weather

    ↓

工具发现城市不存在

    ↓

抛出 ToolException

    ↓

handle_tool_error 转换错误

    ↓

ToolMessage(status="error")

    ↓

模型读取错误信息

    ↓

向用户解释查询失败
```

### 实例：让 Agent 处理天气查询错误

## 实例

from dotenv import load\_dotenv
from langchain.tools import (
    tool,
    ToolException
)
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。"""

    data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C",
        "上海": "小雨，22°C"
    }

    if city not in data:
        raise ToolException(
            f"未收录城市：{city}。"
            "目前只支持杭州、北京和上海。"
        )

    return (
        f"{city}的模拟天气为{data[city]}"
    )

# 在注册到 Agent 之前设置
get\_weather.handle\_tool\_error = True

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[get\_weather],
    system\_prompt=(
        "你是一名天气查询助手。"
        "工具提供的是模拟数据，不是真实天气。"
        "查询天气时使用 get\_weather。"
        "如果工具返回错误，应解释原因，"
        "不要编造没有的数据。"
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "请查询北境的模拟天气。"
        }
    ]
})

# 查看工具结果
for message in result["messages"]:

    if message.type == "tool":
        print("工具名称：", message.name)
        print("执行状态：", message.status)
        print("工具结果：", message.content)

print("\nAgent 最终回答：")
print(result["messages"][-1].content)

如果模型调用了 `get_weather`，工具会返回未收录城市的错误信息。

在工具错误被正确处理的情况下，Agent 可以继续调用模型，生成类似回答：

```
目前没有收录“北境”的模拟天气数据。

支持查询的城市有杭州、北京和上海。
```

**注意：**模型是否调用工具，以及如何组织最后的回复，并不是完全确定的。

另外，如果需要统一控制 Agent 所有工具的错误处理，还可以使用 LangChain 的工具执行中间件，而不是为每个工具重复编写异常处理逻辑。

### 不能使用 with\_config() 修改工具错误策略

例如，下面的写法不是设置工具错误处理的正确方式：

```
# 不推荐：不能据此修改 BaseTool 的错误处理属性

my_tool.with_config(

    handle_tool_errors=True

)
```

`with_config()` 是 Runnable 的通用运行时配置机制，不会自动把 `handle_tool_errors` 转换成工具本身的 `handle_tool_error` 属性。

这里还要区分两个名称：

- `handle_tool_error`：单个 BaseTool 的属性，单数。
- `handle_tool_errors`：LangGraph ToolNode 的错误处理配置名称，复数。

如果使用 `create_agent()`，推荐优先使用工具对象属性或 Agent 中间件来处理错误，而不是依赖不受支持的参数传递方式。

---

## 八、完整示例：带异常处理的课程登记 Agent

下面实现一个课程登记 Agent，演示如何将 `ToolException` 和 `handle_tool_error` 结合使用。

这里通过 Python 字典模拟用户和课程信息，所有登记结果只在程序内存中保存，不会创建真实订单或修改实际网站数据。

完整执行流程如下：

```
用户提出登记请求

    ↓

Agent 选择 register_course

    ↓

检查用户是否存在

    ↓

检查课程是否存在

    ↓

验证是否重复登记

    ↓

成功则写入模拟记录

    ↓

失败则返回工具错误

    ↓

Agent 整理最终回答
```

### 完整代码

文件名：`tool_advanced_demo.py`

## 完整实例

# 文件：tool\_advanced\_demo.py

from dotenv import load\_dotenv

from langchain.tools import (
    tool,
    ToolException
)
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 模拟用户数据
USERS = {
    "张三",
    "李四",
    "王五"
}

# 模拟课程数据
COURSES = {
    "Python3 基础教程",
    "HTML 基础教程",
    "Java 面向对象"
}

# 内存中的模拟登记记录
REGISTRATIONS = set()

# -------------------------
# 定义工具
# -------------------------

@tool
def register\_course(
    user\_name: str,
    course\_name: str
) -> str:
    """为指定用户登记一门模拟课程。

    Args:
        user\_name: 用户姓名。
        course\_name: 课程名称。
    """

    # 检查用户
    if user\_name not in USERS:
        raise ToolException(
            f"用户不存在：{user\_name}"
        )

    # 检查课程
    if course\_name not in COURSES:
        raise ToolException(
            f"课程不存在：{course\_name}"
        )

    record = (
        user\_name,
        course\_name
    )

    # 防止重复登记
    if record in REGISTRATIONS:
        raise ToolException(
            "该用户已经登记过这门课程"
        )

    # 写入内存模拟记录
    REGISTRATIONS.add(record)

    return (
        f"模拟登记成功："
        f"{user\_name} - {course\_name}"
    )

# 配置单个工具的异常处理
register\_course.handle\_tool\_error = (
    "课程登记未完成。"
    "请检查用户、课程名称或重复登记情况。"
)

# -------------------------
# 创建 Agent
# -------------------------

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[register\_course],
    system\_prompt=(
        "你是一名课程登记助手。"
        "这里是教学模拟系统，不是真实订单系统。"
        "用户明确要求登记时，"
        "才可以使用 register\_course。"
        "不要声称已经产生真实订单或支付。"
        "如果工具返回错误，"
        "应说明登记没有成功。"
    )
)

# -------------------------
# 运行与调试
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
            "recursion\_limit": 12
        }
    )

    print("\n问题：", question)

    for message in result["messages"]:

        # 查看模型发起的工具调用
        if (
            message.type == "ai"
            and message.tool\_calls
        ):
            for call in message.tool\_calls:
                print(
                    "[工具请求]",
                    call["name"],
                    call["args"]
                )

        # 查看工具执行结果
        elif message.type == "tool":
            print(
                "[工具结果]",
                message.name,
                message.status,
                message.content
            )

    print(
        "最终回答：",
        result["messages"][-1].content
    )

if \_\_name\_\_ == "\_\_main\_\_":

    # 正常登记
    ask(
        "请在模拟系统中为张三登记"
        "Python3 基础教程。"
    )

    # 用户不存在
    ask(
        "请在模拟系统中为赵六登记"
        "Python3 基础教程。"
    )

    # 重复登记
    ask(
        "请再次为张三登记"
        "Python3 基础教程。"
    )

    # 查看模拟记录
    print("\n模拟登记记录：")
    print(REGISTRATIONS)

### 运行结果说明

如果模型正确调用工具，第一次登记会成功：

```
模拟登记成功：张三 - Python3 基础教程
```

第二次请求中，用户 `赵六` 不在模拟用户列表中，工具会抛出 ToolException。

由于工具设置了 `handle_tool_error`，异常会转换为工具错误响应：

```
课程登记未完成。请检查用户、课程名称或重复登记情况。
```

第三次请求中，由于张三已经登记过同一课程，工具同样会返回错误。

最后的模拟登记集合正常情况下只包含一条记录：

```
{('张三', 'Python3 基础教程')}
```

不过，模型实际的工具选择和调用次数并非严格固定。如果模型未调用工具，或者生成了不同的参数，执行结果也会相应变化。

### 为什么这里使用固定错误提示？

上面的工具采用：

```
register_course.handle_tool_error = (

    "课程登记未完成。"

    "请检查用户、课程名称或重复登记情况。"

)
```

这样可以避免将内部错误细节直接交给模型或用户。

如果需要让 Agent 知道具体失败原因，可以改为：

```
register_course.handle_tool_error = True
```

此时工具会返回 ToolException 的错误内容。

实际业务系统需要根据安全要求选择合适的错误提示方式。

**重要：**登记、支付、删除文件等具有副作用的工具，不能只依赖模型判断来保证正确执行。生产环境还需要认证、授权、参数校验、幂等处理及必要的人工确认。

---

## 九、return\_direct 与错误处理的注意事项

工具可以同时配置直接返回和错误处理。

例如：

```
@tool(return_direct=True)

def lookup_course(course_name: str) -> str:

    """查询指定模拟课程。"""

    if course_name != "Python3 基础教程":

        raise ToolException("课程不存在")

    return "课程：Python3 基础教程"



lookup_course.handle_tool_error = True
```

但必须理解两个选项控制的是不同的行为。

| 配置 | 控制的问题 |
| --- | --- |
| `return_direct` | 工具执行后是否通过直接返回条件结束 Agent 循环 |
| `handle_tool_error` | ToolException 是否转换为工具结果 |

需要特别注意：**工具产生了错误响应，不代表 Agent 一定会继续尝试修复。**

某些 LangChain 版本或执行路径中，`return_direct=True` 的工具即使返回了错误 ToolMessage，也可能满足直接结束的条件。

因此，如果业务希望模型看到工具失败后重新选择参数或工具，建议不要给该工具设置 `return_direct=True`，或者应显式控制 Agent 的错误恢复流程。

---

## 十、常见问题

### 1、return\_direct=True 是否保证工具一定被执行？

不保证。

它只控制工具被执行之后的返回行为，不会强制模型选择该工具。

如果业务必须执行某个工具，应通过程序调用或明确的工作流控制。

### 2、为什么 return\_direct=True 后，最后一条消息是 ToolMessage？

因为 Agent 可能在工具执行后直接结束，不再调用模型生成新的 AIMessage。

因此，读取结果时可以检查消息类型：

```
result = agent.invoke(inputs)



last = result["messages"][-1]



print(last.type)

print(last.content)
```

### 3、InjectedToolCallId 是用户 ID 吗？

不是。

它是当前一次工具调用的标识，主要用于关联请求和工具执行结果。

真实用户身份应从可信认证上下文中获取。

### 4、为什么使用 @tool(handle\_tool\_error=True) 会报错？

因为当前官方 `@tool` 装饰器不接受 `handle_tool_error` 作为直接参数。

应先定义工具：

```
@tool

def my_tool(value: str) -> str:

    """测试工具。"""

    return value
```

然后设置：

```
my_tool.handle_tool_error = True
```

### 5、ToolException 会自动被 Agent 捕获吗？

不一定。

工具自身的错误策略、执行节点以及 Agent 中间件都可能影响最终行为。

建议明确配置工具的错误处理，而不是依赖默认异常兜底。

### 6、handle\_tool\_error 能处理所有 Python 异常吗？

不能。

它主要处理 `ToolException`。

普通程序异常、输入参数校验错误及模型 API 错误，需要根据实际类型和执行层分别处理。

### 7、ToolMessage 的 status="error" 表示什么？

表示这次工具调用产生了错误结果。

即使工具错误被转换为文本返回，也不代表业务操作成功。

实际程序可以利用 `status` 区分成功与失败。

### 8、为什么不能用 with\_config() 配置 handle\_tool\_errors？

因为 `with_config()` 用于设置 Runnable 的运行时配置，并不会自动修改工具对象或 ToolNode 的错误处理属性。

单个工具的 ToolException 处理应通过 `handle_tool_error` 属性设置。

### 9、ToolException 与参数校验错误有什么区别？

`ToolException` 通常是工具函数执行时主动抛出的业务错误。

参数校验错误可能发生在工具函数真正执行之前，例如模型提供了不符合 Pydantic Schema 的参数。

这两类错误不一定由同一个错误处理设置负责。

### 10、是否建议将所有异常直接交给模型？

不建议。

数据库连接信息、文件系统路径、内部服务地址和异常堆栈可能包含敏感信息。

生产环境应将内部错误写入可信日志，对模型只返回经过筛选的必要信息。

---

## 总结

LangChain 的工具高级特性，主要用于控制 Agent 的执行流程与异常处理。

| 功能 | 核心作用 |
| --- | --- |
| `return_direct` | 控制工具结果是否可以直接结束 Agent 循环 |
| `InjectedToolCallId` | 注入当前工具调用 ID |
| `ToolException` | 表示工具执行过程中的错误 |
| `handle_tool_error` | 决定如何处理工具抛出的 ToolException |

实际开发时，需要重点区分三个概念：

- **模型的决策：**是否调用工具，以及传递什么参数。
- **工具的执行：**真正执行 Python 代码，并返回结果或错误。
- **Agent 的调度：**决定是否将工具结果交回模型、继续循环或结束任务。

对于简单查询工具，可以考虑使用 `return_direct=True`，减少不必要的模型调用。

对于需要错误恢复的工具，可以使用 `ToolException` 配合 `handle_tool_error`，让错误以结构化工具结果的形式返回。

而对于涉及真实用户、数据库、订单或支付的工具，还需要在应用程序中实现权限控制、数据验证和操作确认，不能只依靠 Agent 的自然语言判断。
