# 第一个 LangChain Agent

> 来源: https://www.runoob.com/langchain/langchain-first-agent.html

---

# 第一个 LangChain Agent

我们要完成一个任务，有时需要查询外部数据、进行计算，或者连续执行多个操作，单纯依靠模型生成文本就不够了。

这时可以使用 **Agent（智能体）**。

Agent 可以根据用户的问题决定是否调用工具，在获得工具执行结果后再次调用模型，并重复这个过程，直到生成最终回答或达到停止条件。

本章节将使用 LangChain 的 `create_agent()` 创建第一个 Agent，并通过天气查询和数学计算两个工具，理解 Agent 的完整执行过程。

---

## 一、什么是 Agent？

Agent 是一种能够结合语言模型与外部工具完成任务的程序。

Agent 不只是生成回答，还可以根据任务需要选择工具，并利用工具返回的结果继续处理问题。

### 普通模型与 Agent 的区别

| 对比项目 | 普通模型调用 | Agent |
| --- | --- | --- |
| 基本流程 | 输入问题 → 生成回答 | 输入问题 → 判断是否调用工具 → 执行工具 → 继续处理 |
| 工具使用 | 需要额外编排 | 可以根据模型的工具调用请求自动执行 |
| 外部数据 | 无法仅凭模型参数保证获取实时数据 | 可以通过工具访问数据库、API 等外部数据源 |
| 执行次数 | 通常完成一次模型调用 | 可能进行多轮模型调用及工具执行 |
| 适用场景 | 聊天、改写、摘要、翻译 | 数据查询、任务自动化、多步骤操作 |

### Agent 的工作循环

Agent 的工作过程通常可以概括为：

```
用户输入

   ↓

调用语言模型

   ↓

模型是否请求调用工具？

   ├── 否 → 生成最终回答

   │

   └── 是

        ↓

      执行工具

        ↓

      获得工具结果

        ↓

      再次调用模型

        ↓

      继续判断是否需要工具
```

这种循环常被概括为**推理、行动、观察**（Reasoning、Action、Observation）。

例如，用户提问：

```
杭州今天的天气怎么样？
```

Agent 可能先调用天气查询工具，将返回的天气数据提供给模型，再由模型组织成自然语言回答。

**注意：**Agent 能否获取实时天气，取决于是否接入了真实的天气 API。本章节首先使用模拟数据，目的是学习 Agent 的运行机制，而不是查询真实天气。

---

## 二、安装与环境准备

本教程使用 LangChain 1.x 风格的 `create_agent()` API。

首先安装所需依赖：

```
pip install -U langchain langchain-deepseek python-dotenv
```

其中：

- `langchain`：提供 Agent、工具以及统一模型调用接口。
- `langchain-deepseek`：提供 DeepSeek 模型集成。
- `python-dotenv`：读取项目中的环境变量配置。

### 配置 DeepSeek API Key

在项目目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

将 `your_deepseek_api_key` 替换为实际的 API Key。

![](https://www.runoob.com/wp-content/uploads/2026/05/runoob_1778729604970.png)

然后在 Python 中加载：

```
from dotenv import load_dotenv



load_dotenv()
```

本教程使用 DeepSeek 官方推荐的 `deepseek-flash` 模型。

对应的 LangChain 初始化方式为：

```
from langchain.chat_models import init_chat_model



model = init_chat_model(

    "deepseek:deepseek-flash"

)
```

使用前请确认模型服务可访问，并且所用模型支持工具调用。

DeepSeek 官方同时提供 `deepseek-v4-pro`，如果需要，可以将模型名称修改为：

```
model = init_chat_model(

    "deepseek:deepseek-v4-pro"

)
```

---

## 三、创建第一个 Agent

创建 Agent 主要分为三个步骤：

1. 定义 Agent 可以调用的工具。
2. 使用 `create_agent()` 创建 Agent。
3. 调用 `agent.invoke()` 执行任务。

### 步骤 1：定义工具函数

LangChain 提供 `@tool` 装饰器，可以将普通 Python 函数转换为 Agent 可以使用的工具。

这里创建两个工具：

- `get_weather()`：查询指定城市的模拟天气。
- `calculate()`：计算简单的数学表达式。

#### 工具一：查询天气

## 实例

from langchain.tools import tool

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气数据。

    Args:
        city: 城市名称，例如杭州、北京、上海。
    """
    weather\_data = {
        "杭州": "晴，25°C，湿度60%",
        "北京": "多云，18°C，湿度45%",
        "上海": "小雨，22°C，湿度80%",
    }

    return weather\_data.get(
        city,
        f"没有找到{city}的模拟天气数据"
    )

这里使用的是预先定义的字典数据。

如果用户查询杭州，工具返回：

```
晴，25°C，湿度60%
```

如果查询没有收录的城市，就返回未找到数据的提示。

**注意：**这些数值是教学样例，并不代表真实天气。实际项目可以将字典替换为天气 API 请求。

#### @tool 装饰器的作用

`@tool` 会将 Python 函数转换为 LangChain 工具对象，并根据函数签名和描述信息建立工具定义。

例如：

```
@tool

def get_weather(city: str) -> str:

    """查询指定城市的模拟天气数据。"""

    ...
```

其中：

- `get_weather`：工具名称。
- `city: str`：工具接收的参数及类型。
- 文档字符串：描述工具的用途。
- 返回值：工具执行后提供给 Agent 的结果。

模型会利用这些信息判断何时调用工具，以及应该传入什么参数。

因此，工具描述应尽量明确，并说明参数的含义。

#### 工具二：安全计算器

第二个工具用于执行简单的数学计算。

原先使用 `eval()` 直接计算模型生成的表达式并不安全。即使移除了内置函数，也不建议将不可信表达式直接交给 `eval()`。

这里改用 Python 的 `ast` 模块，只允许数字、括号、加减乘除和一元正负号。

## 实例

import ast
import operator

from langchain.tools import tool

# 允许使用的数学运算符
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def safe\_calculate(expression: str):
    """仅计算简单的数学表达式。"""

    # 限制表达式长度，防止过大输入
    if len(expression) > 100:
        raise ValueError("表达式过长")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        # 数字常量
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                if abs(node.value) > 1\_000\_000\_000:
                    raise ValueError("数字过大")
                return node.value

        # 二元运算
        if isinstance(node, ast.BinOp):
            op\_type = type(node.op)

            if op\_type not in OPERATORS:
                raise ValueError("不支持的运算符")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if op\_type is ast.Div and right == 0:
                raise ValueError("除数不能为零")

            result = OPERATORS[op\_type](left, right)

            if abs(result) > 1\_000\_000\_000:
                raise ValueError("计算结果过大")

            return result

        # 一元正负号
        if isinstance(node, ast.UnaryOp):
            op\_type = type(node.op)

            if op\_type not in (ast.UAdd, ast.USub):
                raise ValueError("不支持的运算符")

            return OPERATORS[op\_type](
                evaluate(node.operand)
            )

        raise ValueError("不支持的表达式")

    return evaluate(tree.body)

@tool
def calculate(expression: str) -> str:
    """计算简单数学表达式，支持加减乘除和括号。

    Args:
        expression: 例如 "3 \* 7 + 2" 或 "(25 - 18)"。
    """
    try:
        result = safe\_calculate(expression)

        return f"计算结果：{expression} = {result}"

    except (
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError
    ) as error:
        return f"计算失败：{error}"

例如，计算：

```
3 * 7 + 2
```

工具返回：

```
计算结果：3 * 7 + 2 = 23
```

这个计算器只支持有限的数学语法，不允许调用函数、访问对象属性或执行任意 Python 代码。

它适合教学演示。正式应用中仍应根据业务要求限制输入长度、计算复杂度和执行资源。

---

### 步骤 2：创建 Agent

定义工具之后，通过 `create_agent()` 将语言模型和工具组合起来。

## 实例

from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

# 初始化 DeepSeek 模型
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 创建 Agent
agent = create\_agent(
    model=model,
    tools=[get\_weather, calculate],
    system\_prompt=(
        "你是一名中文 AI 助手。"
        "回答天气问题时，必须先调用 get\_weather。"
        "get\_weather 返回的是模拟数据，不是实时天气。"
        "涉及需要精确计算的算术题时，优先使用 calculate。"
        "如果工具没有相关数据，不要编造结果。"
    ),
)

这里的 `create_agent()` 会创建一个由 LangGraph 支持的 Agent 执行图。

Agent 会根据模型返回的工具调用请求执行对应工具，并将工具结果加入消息历史。

#### create\_agent() 常用参数

| 参数 | 说明 | 是否必填 |
| --- | --- | --- |
| `model` | 聊天模型实例或支持的模型标识字符串 | 是 |
| `tools` | Agent 可使用的工具列表 | 否 |
| `system_prompt` | 系统提示词，定义 Agent 的角色与行为 | 否 |
| `middleware` | 用于扩展 Agent 的运行行为，例如日志、重试、模型选择等 | 否 |
| `response_format` | 用于配置结构化输出 | 否 |
| `checkpointer` | 配置状态持久化与会话检查点 | 否 |
| `debug` | 开启调试信息 | 否 |

对于初学者，通常只需要关注 `model`、`tools` 和 `system_prompt` 三个参数。

**注意：**工具列表为空时，Agent 不会执行工具调用循环，但仍可以使用语言模型完成回复。

---

### 步骤 3：运行 Agent

Agent 创建完成后，可以通过 `invoke()` 执行任务。

输入格式为：

```
{

    "messages": [

        {

            "role": "user",

            "content": "杭州天气怎么样？"

        }

    ]

}
```

其中 `messages` 表示发送给 Agent 的消息列表。

## 实例

# 执行 Agent
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "杭州的模拟天气怎么样？"
            }
        ]
    }
)

# 输出最终回答
print(result["messages"][-1].content)

示例回答可能为：

```
杭州的模拟天气为晴天，气温25°C，湿度60%。
```

这里的回答基于 `get_weather()` 中的模拟数据。

由于 Agent 的工具选择及模型输出具有一定动态性，实际回答可能有所不同。

如果希望查看完整执行过程，可以打印消息历史。

### 查看完整消息历史

## 实例

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "杭州的模拟天气怎么样？"
        }
    ]
})

for message in result["messages"]:

    print("消息类型：", message.type)

    # 查看模型请求的工具调用
    if message.type == "ai" and message.tool\_calls:
        for call in message.tool\_calls:
            print("工具名称：", call["name"])
            print("工具参数：", call["args"])

    # 查看工具返回结果
    if message.type == "tool":
        print("工具名称：", message.name)
        print("工具结果：", message.content)

    # 查看模型回答
    if message.type == "ai" and not message.tool\_calls:
        print("AI 回复：", message.content)

    print("-" \* 40)

一次典型执行过程可能类似：

```
消息类型： human

----------------------------------------

消息类型： ai

工具名称： get_weather

工具参数： {'city': '杭州'}

----------------------------------------

消息类型： tool

工具名称： get_weather

工具结果： 晴，25°C，湿度60%

----------------------------------------

消息类型： ai

AI 回复： 杭州的模拟天气是晴天，气温25°C，湿度60%。

----------------------------------------
```

这里可以看到四条消息：

1. 用户消息 `HumanMessage`。
2. 模型发起工具调用的 `AIMessage`。
3. 工具执行结果 `ToolMessage`。
4. 模型生成最终回答的 `AIMessage`。

**注意：**请求调用工具的 AI 消息，其 `content` 可能为空，但 `tool_calls` 中仍然包含工具名称、参数等信息。

---

## 四、Agent 执行流程解析

下面结合天气查询示例，解释 Agent 的内部执行流程。

![LangChain Agent 模型调用与工具执行流程](https://www.runoob.com/wp-content/uploads/2026/05/6fde20ee-22ee-4ddf-9824-0d4c667f2e99.png)

### 第一步：接收用户问题

用户输入：

```
杭州的模拟天气怎么样？
```

Agent 将用户消息发送给语言模型，同时提供当前可使用的工具定义。

### 第二步：模型判断是否调用工具

模型根据问题、系统提示词和工具描述，可能决定使用 `get_weather`。

此时模型返回的消息可能包含：

```
{

    "name": "get_weather",

    "args": {

        "city": "杭州"

    }

}
```

这称为工具调用请求（Tool Call）。

这里要注意：**模型负责提出工具调用请求，但不等于模型自己直接执行 Python 函数。**

### 第三步：Agent 执行工具

LangChain 根据工具调用请求找到对应的工具，并传入参数。

例如：

```
get_weather(city="杭州")
```

工具执行后返回：

```
晴，25°C，湿度60%
```

该结果作为 `ToolMessage` 加入消息历史。

### 第四步：再次调用模型

模型读取工具执行结果，决定是否还需要调用其他工具。

如果不需要继续执行工具，就生成最终回答。

如果还需要其他数据，则可以再次发起工具调用。

因此，Agent 的核心机制可以概括为：

```
Model → Tool Call → Tool Result → Model
```

这个循环通常持续到模型不再请求工具，或者达到执行限制、错误等停止条件。

---

## 五、让 Agent 调用多个工具

Agent 不仅可以调用一个工具，也可以根据任务需要组合多个工具。

例如：

```
杭州和北京的模拟气温相差多少度？
```

对于这个问题，Agent 可能需要：

1. 查询杭州的模拟天气。
2. 查询北京的模拟天气。
3. 计算两地气温差。
4. 生成最终回答。

## 实例

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "请分别查询杭州和北京的模拟天气，"
                "再使用 calculate 工具计算两地气温差。"
            )
        }
    ]
})

for message in result["messages"]:

    if message.type == "ai" and message.tool\_calls:
        for call in message.tool\_calls:
            print(
                f"[调用工具] {call['name']} "
                f"参数：{call['args']}"
            )

    elif message.type == "tool":
        print(
            f"[工具结果] {message.name}: "
            f"{message.content}"
        )

print("\n最终回答：")
print(result["messages"][-1].content)

一次可能的执行过程如下：

```
[调用工具] get_weather 参数：{'city': '杭州'}

[调用工具] get_weather 参数：{'city': '北京'}

[工具结果] get_weather: 晴，25°C，湿度60%

[工具结果] get_weather: 多云，18°C，湿度45%

[调用工具] calculate 参数：{'expression': '25 - 18'}

[工具结果] calculate: 计算结果：25 - 18 = 7



最终回答：

根据模拟数据，杭州为25°C，北京为18°C，

两地气温相差7°C。
```

这里的重点是：用户不需要编写一套固定的工具调用顺序，模型可以根据当前任务和已返回的结果继续选择工具。

### 工具一定按顺序执行吗？

不一定。

如果模型一次返回多个工具调用请求，LangChain 的工具执行节点可以处理这些调用，其中独立调用可能并发执行。

但也可能出现以下情况：

- 模型先查询杭州，再查询北京。
- 模型一次请求查询两个城市。
- 模型获得天气数据后，自己完成简单减法，没有调用计算器。
- 模型根据用户请求调用计算器，再生成最终回答。

因此，**不能假设 Agent 一定会调用三个工具，也不能保证两个天气工具调用一定并行。**

如果某个步骤必须执行，或者必须按照特定顺序完成，应该通过明确的程序流程、工具编排或 LangGraph 工作流控制，而不是只依赖提示词。

---

## 六、工具调用与消息类型

LangChain Agent 执行过程中，主要涉及以下消息类型：

| 消息类型 | 作用 |
| --- | --- |
| `HumanMessage` | 表示用户输入 |
| `SystemMessage` | 向模型提供系统级指令 |
| `AIMessage` | 表示模型输出，可能包含工具调用请求 |
| `ToolMessage` | 表示工具执行后返回的结果 |

### AIMessage.tool\_calls

模型需要调用工具时，通常会在 `AIMessage` 中提供 `tool_calls`。

例如：

```
[

    {

        "name": "get_weather",

        "args": {"city": "杭州"},

        "id": "call_example_001",

        "type": "tool_call"

    }

]
```

其中：

- `name`：工具名称。
- `args`：传给工具的参数。
- `id`：工具调用标识，用于关联请求和工具结果。
- `type`：消息内容的类型。

实际的 `id` 由模型提供商或接口生成，不应在业务代码中固定写死。

### ToolMessage

当工具执行完成后，LangChain 会使用 `ToolMessage` 保存工具结果。

一个工具结果通常包含：

```
name: get_weather

content: 晴，25°C，湿度60%

tool_call_id: call_example_001
```

`tool_call_id` 用于关联此前的工具调用请求。

这使模型能够知道每条工具结果对应哪一次调用。

---

## 七、完整可运行示例

下面将前面的工具定义、模型初始化和 Agent 运行代码整合为一个 Python 文件。

**文件名：**`04_first_agent.py`

运行前，请确保已安装：

```
pip install -U langchain langchain-deepseek python-dotenv
```

并在项目目录的 `.env` 文件中配置 `DEEPSEEK_API_KEY`。

## 完整实例

# 文件：04\_first\_agent.py

import ast
import operator

from dotenv import load\_dotenv

from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

# 读取环境变量
load\_dotenv()

# -------------------------
# 工具一：模拟天气查询
# -------------------------

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气数据。

    Args:
        city: 城市名称，例如杭州、北京、上海。
    """
    weather\_data = {
        "杭州": "晴，25°C，湿度60%",
        "北京": "多云，18°C，湿度45%",
        "上海": "小雨，22°C，湿度80%",
    }

    return weather\_data.get(
        city,
        f"没有找到{city}的模拟天气数据"
    )

# -------------------------
# 工具二：安全数学计算
# -------------------------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def safe\_calculate(expression: str):
    """只计算有限的简单算术表达式。"""

    if len(expression) > 100:
        raise ValueError("表达式过长")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                if abs(node.value) > 1\_000\_000\_000:
                    raise ValueError("数字过大")
                return node.value

        if isinstance(node, ast.BinOp):
            op\_type = type(node.op)

            if op\_type not in OPERATORS:
                raise ValueError("不支持的运算符")

            a = evaluate(node.left)
            b = evaluate(node.right)

            if op\_type is ast.Div and b == 0:
                raise ValueError("除数不能为零")

            result = OPERATORS[op\_type](a, b)

            if abs(result) > 1\_000\_000\_000:
                raise ValueError("计算结果过大")

            return result

        if isinstance(node, ast.UnaryOp):
            op\_type = type(node.op)

            if op\_type not in (ast.UAdd, ast.USub):
                raise ValueError("不支持的运算符")

            return OPERATORS[op\_type](
                evaluate(node.operand)
            )

        raise ValueError("不支持的表达式")

    return evaluate(tree.body)

@tool
def calculate(expression: str) -> str:
    """计算简单算术表达式，支持加减乘除和括号。

    Args:
        expression: 例如 "3 \* 7 + 2"。
    """
    try:
        result = safe\_calculate(expression)

        return f"计算结果：{expression} = {result}"

    except (
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError
    ) as error:
        return f"计算失败：{error}"

# -------------------------
# 创建 Agent
# -------------------------

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[get\_weather, calculate],
    system\_prompt=(
        "你是一名中文 AI 助手。"
        "天气工具中的数据全部为模拟数据，"
        "不能当作真实天气。"
        "回答天气问题时先使用 get\_weather。"
        "遇到需要精确计算的算术任务时，"
        "优先使用 calculate 工具。"
        "如果工具返回错误或没有数据，"
        "应明确说明，不要编造。"
    )
)

# -------------------------
# 执行 Agent
# -------------------------

def ask(question: str):
    print("\n" + "=" \* 50)
    print("问题：", question)

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

    # 打印工具调用过程
    for message in result["messages"]:

        if message.type == "ai" and message.tool\_calls:
            for call in message.tool\_calls:
                print(
                    f"[工具调用] {call['name']} "
                    f"{call['args']}"
                )

        elif message.type == "tool":
            print(
                f"[工具结果] {message.name}: "
                f"{message.content}"
            )

    print("最终回答：")
    print(result["messages"][-1].content)

    return result

# -------------------------
# 测试
# -------------------------

if \_\_name\_\_ == "\_\_main\_\_":

    # 测试天气查询
    ask("杭州的模拟天气怎么样？")

    # 测试多个工具
    ask(
        "查询杭州和北京的模拟天气，"
        "然后使用 calculate 计算两地气温差。"
    )

    # 测试数学计算
    ask(
        "请使用 calculate 工具计算 "
        "(3 + 2) \* 7 的结果。"
    )

### 运行代码

在终端执行：

```
python 04_first_agent.py
```

程序会依次向 Agent 发送三个问题，并打印：

- 用户问题。
- 模型发起的工具调用及参数。
- 工具执行结果。
- Agent 最终回答。

由于模型决定工具调用的方式可能不同，实际输出的顺序和文本不一定与示例完全一致。

不过，如果 Agent 正确调用了工具，应当能够在输出中看到类似：

```
[工具调用] get_weather {'city': '杭州'}

[工具结果] get_weather: 晴，25°C，湿度60%
```

以及：

```
[工具调用] calculate {'expression': '(3 + 2) * 7'}

[工具结果] calculate: 计算结果：(3 + 2) * 7 = 35
```

如果没有出现预期的工具调用，应检查模型是否支持 Tool Calling、工具描述是否清晰，以及相关系统提示词是否合适。

---

## 八、同步与异步调用

LangChain Agent 同时支持同步和异步调用。

### 同步调用：invoke()

同步方式会等待 Agent 执行完成后返回结果。

```
result = agent.invoke({

    "messages": [

        {

            "role": "user",

            "content": "查询杭州的模拟天气"

        }

    ]

})
```

### 异步调用：ainvoke()

在异步程序中，可以使用 `ainvoke()`。

## 实例

import asyncio

async def main():

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "杭州的模拟天气怎么样？"
            }
        ]
    })

    print(result["messages"][-1].content)

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(main())

这段代码需要在已经定义 `agent` 的程序中运行。

异步调用适合 Web 服务、聊天系统和其他使用 `asyncio` 的应用。

需要注意，如果工具内部包含耗时的同步 I/O，仍然可能影响并发性能。实际项目中应根据需要使用异步工具实现。

---

## 九、Agent 流式输出

除了等待 Agent 完成任务，LangChain 还支持流式观察执行过程。

可以使用 `agent.stream()` 配合 `stream_mode="updates"`，查看 Agent 各执行节点的状态更新。

## 实例

inputs = {
    "messages": [
        {
            "role": "user",
            "content": "查询杭州的模拟天气。"
        }
    ]
}

for update in agent.stream(
    inputs,
    stream\_mode="updates"
):
    for node\_name, state in update.items():

        print("执行节点：", node\_name)

        messages = state.get("messages", [])

        for message in messages:
            if message.type == "ai" and message.tool\_calls:
                print("工具调用：", message.tool\_calls)

            elif message.type == "tool":
                print("工具结果：", message.content)

            elif message.type == "ai":
                print("模型回答：", message.content)

使用 `stream_mode="updates"` 时，可以观察模型节点和工具节点执行后的更新。

这适合调试 Agent 的运行流程。

如果需要逐 Token 展示模型生成的内容，可以进一步使用 LangChain 支持的消息流模式，但不同模型对流式工具调用和输出内容的支持可能存在差异。

---

## 十、限制 Agent 执行次数

由于 Agent 可能在多个工具之间循环调用，因此需要考虑执行次数和成本。

如果模型反复请求工具，可能产生：

- 不必要的 API 调用费用。
- 过长的响应时间。
- 重复的外部工具请求。
- 超出预期的任务执行。

可以在执行时设置 `recursion_limit`，限制 LangGraph 的执行步数。

```
result = agent.invoke(

    {

        "messages": [

            {

                "role": "user",

                "content": "查询杭州的模拟天气。"

            }

        ]

    },

    config={

        "recursion_limit": 15

    }

)
```

**注意：**`recursion_limit` 限制的是图执行步骤，并不等同于精确限制工具调用次数，也不是 Token 或费用上限。

实际应用还应设置 API 超时、调用预算、并发限制，以及必要的工具权限。

---

## 十一、常见问题

### 1、为什么 Agent 没有调用工具？

工具调用由模型根据问题和工具定义决定，并不是每个问题都会自动触发工具。

常见原因包括：

- 问题本身不需要外部工具。
- 工具描述不清楚。
- 模型缺乏可靠的工具调用能力。
- 模型认为可以直接回答。

可以通过改进工具描述和系统提示词，提高工具调用的准确性。

但如果某个工具执行属于强制业务步骤，应由程序逻辑明确控制，而不是仅依赖提示词。

### 2、为什么 AIMessage 的 content 为空？

当模型请求调用工具时，可能不会生成普通文本，而是直接返回 `tool_calls`。

例如：

```
message.content == ""

message.tool_calls != []
```

因此，需要同时查看 `content` 和 `tool_calls`，不能只根据 `content` 判断模型是否有输出。

### 3、为什么工具调用顺序与示例不同？

Agent 的工具选择和执行路径具有动态性。

同一个问题可能产生一次工具调用，也可能产生多次工具调用。

如果多个工具调用互不依赖，模型可能在同一轮返回多个工具调用请求。

因此，工具执行日志的顺序不应被当作固定接口规范。

### 4、为什么工具返回结果后，Agent 还会再次调用模型？

因为工具只负责执行具体操作，例如查询数据或计算表达式。

模型还需要读取工具结果，并判断任务是否完成。

如果任务尚未完成，模型可以继续请求调用工具；如果已经完成，则生成最终回答。

### 5、为什么 Agent 可能一直调用工具？

当工具返回的信息无法满足任务，或者模型持续选择错误的工具时，可能出现反复调用。

应通过清晰的工具描述、错误反馈和执行限制减少这种情况。

对于实际应用，还可以结合 Agent 中间件控制工具调用和模型行为。

### 6、如何接入真实天气 API？

可以将 `get_weather()` 中的模拟字典查询替换为真实的 HTTP 请求。

实际接入时需要处理：

- 城市名称与位置编码的转换。
- API Key 和请求认证。
- 请求超时和错误响应。
- 数据时间、温度单位与时区。
- API 调用配额及异常重试。

工具返回的应是明确标注查询时间的天气数据，避免模型将历史或缓存信息误认为实时结果。

### 7、create\_agent() 与旧版 initialize\_agent() 有什么区别？

LangChain 1.x 推荐使用 `create_agent()` 创建 Agent。

它基于 LangGraph 的执行机制，支持工具循环、状态管理以及中间件等扩展能力。

一些旧教程中的 `initialize_agent()`、`AgentExecutor` 或旧版 ReAct Agent API，属于此前版本的用法，不应与本教程混用。

### 8、如何记住多轮对话？

本教程中每次调用 `ask()` 时，都会传入新的消息列表，因此示例默认将每次请求作为独立任务处理。

如果需要在多个请求之间保存会话状态，可以使用 LangGraph 的 `checkpointer` 等机制，并为不同会话指定相应的线程标识。

会话持久化属于后续进阶内容，本章节先关注 Agent 的基本工作流程。

---

## 总结

本章节使用 LangChain 创建了一个能够调用天气工具和计算工具的简单 Agent。

整个过程主要分为三步：

1. 使用 `@tool` 装饰器定义工具。
2. 使用 `create_agent()` 将模型和工具组合起来。
3. 使用 `agent.invoke()` 执行 Agent 并读取结果。

理解 Agent 时，需要记住几个关键概念：

- **模型负责决策：**根据消息与工具定义选择是否调用工具。
- **工具负责执行：**真正访问数据、执行计算或完成其他操作。
- **Agent 负责循环：**将工具结果重新提供给模型，直到满足停止条件。
- **工具调用不是固定流程：**模型可能调用一次、多次，也可能不调用工具。
- **工具输出不一定可信：**生产环境需要做好权限控制、数据验证和错误处理。

对于初学者，建议先掌握 `@tool`、`create_agent()` 和 `invoke()`，再进一步学习多轮对话记忆、流式输出、Agent 中间件以及 LangGraph 工作流。
