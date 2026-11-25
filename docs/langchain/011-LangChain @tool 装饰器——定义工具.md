# LangChain @tool 装饰器——定义工具

> 来源: https://www.runoob.com/langchain/langchain-tool-basics.html

---

# LangChain @tool 装饰器——定义工具

工具（Tool）是 LangChain Agent 与外部系统交互的重要桥梁。

通过工具，Agent 可以查询数据库、调用 API、执行计算、读取文件，或者完成其他由程序提供的操作。

LangChain 提供 @tool 装饰器，可以将 Python 函数快速转换为 Agent 能够识别和调用的工具。

本章节将介绍 `@tool` 的基本用法、参数定义、Pydantic 校验、工具注册，以及多个工具协作的方法。

---

## 一、@tool 基本语法

`@tool` 是 LangChain 提供的工具定义装饰器。

只需要在普通 Python 函数前添加 `@tool`，LangChain 就可以根据函数名称、参数类型和描述信息创建工具。

基本语法：

```
from langchain.tools import tool



@tool

def tool_name(parameter: str) -> str:

    """工具的功能描述。"""

    return "工具执行结果"
```

使用 `@tool` 后，函数会被包装为 LangChain 的工具对象。

它不再只是原来的普通 Python 函数，而是具有名称、描述、参数 Schema 和执行接口的对象。

### 实例：第一个工具

## 实例

from langchain.tools import tool

@tool
def hello\_tool(name: str) -> str:
    """向指定的人打招呼。

    Args:
        name: 要打招呼的人的名字。
    """
    return f"你好，{name}！欢迎来到菜鸟教程 RUNOOB。"

# 调用工具
result = hello\_tool.invoke({
    "name": "小明"
})

print(result)

# 查看工具信息
print("工具名称：", hello\_tool.name)
print("工具描述：", hello\_tool.description)
print("工具参数：", hello\_tool.args)

执行结果中的主要内容：

```
你好，小明！欢迎来到菜鸟教程 RUNOOB。



工具名称： hello_tool
```

还可以通过 `description` 和 `args` 查看工具的描述与参数结构。

**注意：**普通工具输入通常使用 `invoke()`：

```
hello_tool.invoke({"name": "小明"})
```

不要把经过装饰的工具对象直接当作原始 Python 函数调用。

### @tool 转换后有什么变化？

LangChain 工具通常具有以下属性和方法：

| 属性或方法 | 说明 |
| --- | --- |
| `name` | 工具名称，默认使用函数名称 |
| `description` | 工具描述，通常来自函数文档字符串 |
| `args` | 工具参数的 Schema 信息 |
| `args_schema` | 参数验证 Schema |
| `invoke()` | 同步执行工具 |
| `ainvoke()` | 异步调用工具接口 |

其中，工具名称、描述和参数结构会帮助模型理解工具的用途。

执行接口则负责真正运行对应的 Python 代码。

---

## 二、工具描述与文档字符串

工具的文档字符串（docstring）非常重要。

模型需要通过工具描述判断：

- 工具能够完成什么任务。
- 什么情况下适合调用它。
- 工具需要哪些参数。
- 参数分别代表什么。

例如：

```
@tool

def get_weather(city: str) -> str:

    """查询指定城市的天气信息。"""

    ...
```

这个描述虽然简短，但已经明确了工具的基本用途。

如果工具比较复杂，还可以使用 Google 风格文档字符串描述参数。

### 使用 parse\_docstring

默认情况下，`@tool` 不会自动将 Google 风格文档字符串中的 `Args` 内容解析为单独的参数描述。

如果需要解析，可以设置：

```
@tool(parse_docstring=True)
```

### 实例

## 实例

from langchain.tools import tool

@tool(parse\_docstring=True)
def get\_course\_info(
    language: str,
    level: str = "入门"
) -> str:
    """查询指定编程语言的模拟课程信息。

    Args:
        language: 编程语言名称，例如 Python、Java。
        level: 学习难度，默认为入门。
    """
    return f"查询 {language} 的{level}课程"

print("工具名称：")
print(get\_course\_info.name)

print("\n工具描述：")
print(get\_course\_info.description)

print("\n参数 Schema：")
print(get\_course\_info.args)

这里通过 `parse_docstring=True`，让 LangChain 尝试从文档字符串中提取参数描述。

**注意：**开启该选项后，文档字符串应符合支持的 Google 风格格式。如果格式不正确，可能抛出 `ValueError`。

对于简单工具，使用清晰的函数描述和类型注解通常已经足够；对于参数较多的工具，推荐结合 `parse_docstring` 或 Pydantic 的 `Field` 提供描述信息。

---

## 三、使用不同的参数类型

`@tool` 支持根据 Python 类型注解生成参数 Schema。

常见类型包括：

- `str`：字符串。
- `int`：整数。
- `float`：浮点数。
- `bool`：布尔值。
- `Literal`：限定允许的参数值。
- `list`：列表。

### 实例：模拟课程搜索

下面定义一个课程搜索工具，支持关键词、难度、最大结果数和是否只显示免费课程。

为避免与真实网站数据混淆，以下课程列表均为教学模拟数据。

## 实例

from typing import Literal
from langchain.tools import tool

@tool(parse\_docstring=True)
def search\_courses(
    keyword: str,
    level: Literal["入门", "进阶", "高级"],
    max\_results: int = 5,
    free\_only: bool = True
) -> str:
    """搜索模拟课程列表。

    Args:
        keyword: 搜索关键词，例如 Python、HTML。
        level: 课程难度级别。
        max\_results: 最多返回多少条，范围为1到20。
        free\_only: 是否只返回免费课程。
    """
    courses = [
        {
            "title": "Python 基础",
            "keyword": "python",
            "level": "入门",
            "free": True
        },
        {
            "title": "Python 入门练习",
            "keyword": "python",
            "level": "入门",
            "free": True
        },
        {
            "title": "Python 数据分析",
            "keyword": "python",
            "level": "进阶",
            "free": True
        },
        {
            "title": "Python 高级实战",
            "keyword": "python",
            "level": "高级",
            "free": False
        },
        {
            "title": "HTML 基础",
            "keyword": "html",
            "level": "入门",
            "free": True
        }
    ]

    if not 1 <= max\_results <= 20:
        return "max\_results 必须在1到20之间"

    results = []

    for course in courses:
        if keyword.lower() not in course["keyword"]:
            continue

        if course["level"] != level:
            continue

        if free\_only and not course["free"]:
            continue

        results.append(course["title"])

    results = results[:max\_results]

    if not results:
        return "未找到符合条件的模拟课程"

    return "搜索结果：" + "、".join(results)

# 查询入门课程
print(search\_courses.invoke({
    "keyword": "python",
    "level": "入门"
}))

# 查询高级课程，包括付费项目
print(search\_courses.invoke({
    "keyword": "python",
    "level": "高级",
    "free\_only": False
}))

执行结果：

```
搜索结果：Python 基础、Python 入门练习

搜索结果：Python 高级实战
```

其中：

- `keyword` 和 `level` 没有默认值，属于必填参数。
- `max_results` 默认值为 5。
- `free_only` 默认值为 True。
- `Literal` 用于限制课程难度的允许值。

上面的搜索只在 Python 列表中执行，不会连接真实网站或数据库。

---

## 四、注册工具到 Agent

工具定义完成后，可以通过 `create_agent()` 的 `tools` 参数将其注册到 Agent。

例如，一个学习顾问 Agent 可以拥有两个工具：

- `search_courses`：搜索课程。
- `get_course_detail`：查询课程详情。

### 实例：创建课程推荐 Agent

## 实例

from dotenv import load\_dotenv

from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 模拟课程数据库
COURSES = {
    "Python3 基础教程": {
        "language": "python",
        "chapters": 30,
        "hours": 20
    },
    "Python 数据分析": {
        "language": "python",
        "chapters": 18,
        "hours": 15
    },
    "HTML 基础教程": {
        "language": "html",
        "chapters": 25,
        "hours": 12
    }
}

@tool
def search\_courses(keyword: str) -> str:
    """根据编程语言关键词搜索模拟课程。"""

    results = [
        name
        for name, info in COURSES.items()
        if keyword.lower() in info["language"]
    ]

    if not results:
        return "没有找到相关的模拟课程"

    return "、".join(results)

@tool
def get\_course\_detail(course\_name: str) -> str:
    """获取指定模拟课程的章节数和预计学习时长。"""

    info = COURSES.get(course\_name)

    if info is None:
        return f"没有找到课程：{course\_name}"

    return (
        f"课程：{course\_name}；"
        f"章节数：{info['chapters']}；"
        f"预计学习时长：{info['hours']}小时"
    )

# 初始化模型
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 创建 Agent
agent = create\_agent(
    model=model,
    tools=[
        search\_courses,
        get\_course\_detail
    ],
    system\_prompt=(
        "你是编程学习顾问。"
        "工具中的课程数据是教学模拟数据。"
        "查询课程时优先使用工具。"
        "如果未找到课程，应明确说明，"
        "不要编造课程信息。"
    )
)

# 调用 Agent
def ask(question: str):

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    print("问题：", question)
    print("回答：", result["messages"][-1].content)
    print("-" \* 40)

ask("有哪些 Python 模拟课程？")

ask("Python3 基础教程预计学多久？")

运行时，Agent 可以根据问题选择合适的工具。

例如，查询课程列表时，模型可能调用：

```
search_courses(keyword="python")
```

查询课程详情时，可能调用：

```
get_course_detail(

    course_name="Python3 基础教程"

)
```

工具返回结果后，Agent 会再次调用模型生成回答。

**注意：**工具选择并不是固定的。模型可能调用一个工具、多个工具，也可能直接回答。

如果某个工具必须执行，应该通过程序流程或工作流明确控制，而不是只依靠提示词。

---

## 五、多个工具协作

Agent 可以组合不同工具完成任务。

例如，用户希望知道某只股票的模拟价格，并将金额转换成人民币。

这个过程可能涉及：

1. 调用股票价格查询工具。
2. 获取股票价格。
3. 调用货币换算工具。
4. 生成最终回答。

下面仍然使用模拟数据，演示多个工具如何协作。

### 实例：模拟股票价格与货币换算

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

@tool
def get\_stock\_price(symbol: str) -> str:
    """查询指定股票的模拟美元价格。

    Args:
        symbol: 股票代码，例如 AAPL、GOOGL。
    """

    prices = {
        "AAPL": 185.50,
        "GOOGL": 142.30,
        "TSLA": 245.80
    }

    price = prices.get(symbol.upper())

    if price is None:
        return "没有找到该股票的模拟价格"

    return (
        f"{symbol.upper()} 的模拟价格为 "
        f"{price:.2f} USD"
    )

@tool
def convert\_currency(
    amount: float,
    from\_currency: str,
    to\_currency: str
) -> str:
    """根据固定的模拟汇率换算货币金额。"""

    # 教学用模拟汇率
    rates = {
        ("USD", "CNY"): 7.25
    }

    source = from\_currency.upper()
    target = to\_currency.upper()

    if source == target:
        return f"{amount:.2f} {target}"

    rate = rates.get((source, target))

    if rate is None:
        return "不支持指定的模拟货币换算"

    result = amount \* rate

    return (
        f"{amount:.2f} {source} "
        f"= {result:.2f} {target}"
        f"（模拟汇率 {rate}）"
    )

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

agent = create\_agent(
    model=model,
    tools=[
        get\_stock\_price,
        convert\_currency
    ],
    system\_prompt=(
        "你是一名金融数据演示助手。"
        "股票价格和汇率全部为模拟数据，"
        "不能当作实时行情。"
        "如果需要计算换算结果，请调用工具。"
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "查询 AAPL 的模拟价格，"
                "并根据模拟汇率换算成人民币。"
            )
        }
    ]
})

# 查看工具执行结果
for message in result["messages"]:

    if message.type == "tool":
        print(
            f"[工具 {message.name}] "
            f"{message.content}"
        )

print("\n最终回答：")
print(result["messages"][-1].content)

如果 Agent 依次调用两个工具，可能得到：

```
[工具 get_stock_price] AAPL 的模拟价格为 185.50 USD

[工具 convert_currency] 185.50 USD = 1344.88 CNY（模拟汇率 7.25）
```

这里的价格与汇率都是固定的教学数据，不代表真实市场行情。

**注意：**模型可能先获取价格，再根据返回值调用换算工具，也可能采用其他路径。不要将示例中的工具调用次数和顺序视为固定结果。

---

## 六、工具的可选参数与默认值

Python 函数可以为参数设置默认值，LangChain 工具同样支持。

没有默认值的参数通常是必填参数；有默认值的参数通常可以省略。

### 实例

## 实例

from langchain.tools import tool

@tool
def recommend\_tutorial(
    language: str,
    level: str = "入门",
    count: int = 3
) -> str:
    """从模拟教程列表中推荐学习内容。

    Args:
        language: 编程语言。
        level: 学习级别。
        count: 最多推荐的数量。
    """

    tutorials = {
        "python": [
            "Python 基础",
            "Python 面向对象",
            "Python 数据分析",
            "Python 自动化"
        ],
        "java": [
            "Java 基础",
            "Java 面向对象",
            "Java 集合框架",
            "Java 多线程"
        ]
    }

    if not 1 <= count <= 10:
        return "count 必须在1到10之间"

    available = tutorials.get(
        language.lower(),
        []
    )

    if not available:
        return "没有相关教程"

    selected = available[:count]

    return (
        f"{language}（{level}）推荐："
        + "、".join(selected)
    )

# 只提供必填参数
print(recommend\_tutorial.invoke({
    "language": "Python"
}))

# 覆盖可选参数
print(recommend\_tutorial.invoke({
    "language": "Java",
    "level": "进阶",
    "count": 2
}))

可能输出：

```
Python（入门）推荐：Python 基础、Python 面向对象、Python 数据分析

Java（进阶）推荐：Java 基础、Java 面向对象
```

这里的 `level` 只用于标记请求的学习级别，并没有真正筛选课程难度。

在实际项目中，如果需要精确按照课程难度推荐，还应在数据中定义对应的难度字段并执行筛选。

默认参数能减少工具调用时需要传入的字段，但不应该为用户身份、权限、支付金额等关键参数随意设置默认值。

---

## 七、args\_schema：使用 Pydantic 校验参数

对于参数较多、需要严格验证输入的工具，可以通过 `args_schema` 指定 Pydantic 模型。

例如，可以限制：

- 字符串最短和最长长度。
- 整数允许的范围。
- 参数必须属于某几个指定值。
- 字段是否必须提供。

### 实例：自定义搜索参数

## 实例

from pydantic import BaseModel, Field
from typing import Literal

from langchain.tools import tool

class CourseSearchInput(BaseModel):
    """课程搜索参数。"""

    keyword: str = Field(
        min\_length=1,
        max\_length=50,
        description="搜索关键词"
    )

    category: Literal[
        "all",
        "frontend",
        "backend",
        "data"
    ] = Field(
        default="all",
        description="课程类别"
    )

    page: int = Field(
        default=1,
        ge=1,
        le=100,
        description="页码，从1开始"
    )

@tool(args\_schema=CourseSearchInput)
def search\_course(
    keyword: str,
    category: str = "all",
    page: int = 1
) -> str:
    """搜索模拟课程数据。"""

    return (
        f"搜索关键词：{keyword}；"
        f"分类：{category}；"
        f"页码：{page}"
    )

# 正常调用
print(search\_course.invoke({
    "keyword": "Python",
    "category": "backend",
    "page": 1
}))

# 参数不符合 Schema
try:
    search\_course.invoke({
        "keyword": "Python",
        "category": "invalid"
    })

except Exception as error:
    print(
        "参数校验失败：",
        type(error).\_\_name\_\_
    )

正常调用返回：

```
搜索关键词：Python；分类：backend；页码：1
```

第二次调用中，`category="invalid"` 不在允许的值列表中，会触发参数校验错误。

这样可以在工具函数执行前，拦截部分不符合要求的输入。

**注意：**即使使用 Pydantic 校验，也不能代替业务权限检查、SQL 参数化查询、文件路径限制等安全措施。

### 查看 JSON Schema

可以通过工具的参数 Schema 了解它接收哪些参数。

```
print(search_course.args)
```

如果使用 Pydantic 模型，也可以直接获取其 JSON Schema：

```
print(

    CourseSearchInput.model_json_schema()

)
```

这对于工具调试、前端参数表单生成和 API 文档编写很有帮助。

---

## 八、@tool 的其他常用参数

除了 `args_schema`，LangChain 的 `@tool` 还支持多个可选参数。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `description` | None | 显式指定工具描述，优先级高于函数文档字符串 |
| `args_schema` | None | 指定工具输入参数的 Schema |
| `infer_schema` | True | 是否根据函数签名推断参数 Schema |
| `parse_docstring` | False | 是否解析 Google 风格的参数描述 |
| `return_direct` | False | 配置工具结果是否可直接作为 Agent 的最终输出，具体取决于执行框架 |
| `response_format` | content | 指定工具返回普通内容，或内容与 artifact 的组合 |

这些选项的行为应结合当前 LangChain 版本和所使用的 Agent 执行框架理解。

### 指定自定义名称和描述

可以直接通过装饰器指定工具名称与说明。

## 实例

from langchain.tools import tool

@tool(
    "search\_python\_docs",
    description="在本地模拟文档中查询 Python 相关知识"
)
def search\_docs(keyword: str) -> str:
    """根据关键词查找文档。"""

    return f"模拟搜索：{keyword}"

print(search\_docs.name)
print(search\_docs.description)

这里的实际 Python 函数名是 `search_docs`，但暴露给模型的工具名称是 `search_python_docs`。

### return\_direct 参数

对于部分场景，工具执行结果不需要再经过模型组织，可以直接作为结果返回。

可以设置：

```
@tool(return_direct=True)

def get_help(topic: str) -> str:

    """获取帮助内容。"""

    return f"帮助主题：{topic}"
```

`return_direct=True` 主要用于告诉支持该机制的 Agent 执行框架：工具调用完成后可以直接返回结果。

**注意：**如果只是手动调用 `tool.invoke()`，该参数不会改变 Python 函数本身的返回值。不同 Agent 框架对它的处理也可能存在差异。

---

## 九、工具返回内容与 artifact

有时工具需要返回两类信息：

- 供模型阅读的简短文本。
- 供程序保存或后续处理的完整数据。

可以使用 `response_format="content_and_artifact"`。

### 实例

## 实例

from langchain.tools import tool

@tool(response\_format="content\_and\_artifact")
def search\_documents(keyword: str):
    """查询模拟文档并返回摘要与完整记录。"""

    records = [
        {
            "id": 1,
            "title": "Python 基础教程"
        },
        {
            "id": 2,
            "title": "Python 数据分析"
        }
    ]

    content = (
        f"找到与 {keyword} 相关的 "
        f"{len(records)} 条模拟记录"
    )

    artifact = {
        "keyword": keyword,
        "records": records
    }

    return content, artifact

# 直接执行，查看工具返回值
result = search\_documents.invoke({
    "keyword": "Python"
})

print(result)

当工具作为 Agent 的 Tool Call 执行，并具有对应工具调用 ID 时，LangChain 可以将内容和 artifact 分别放入 `ToolMessage.content` 和 `ToolMessage.artifact`。

其中，`content` 供模型读取；`artifact` 可以保存完整的结构化数据。

**注意：**工具直接调用与通过带 ID 的 Tool Call 执行时，返回值的封装形式可能不同。不要假设直接调用 `invoke()` 一定返回带有 artifact 属性的 ToolMessage。

---

## 十、同步工具与异步工具

LangChain 工具也支持异步调用。

例如，当工具需要访问网络 API 时，可以使用异步函数减少等待期间的线程占用。

### 实例：异步工具

## 实例

import asyncio

from langchain.tools import tool

@tool
async def async\_search(keyword: str) -> str:
    """执行异步模拟搜索。"""

    # 模拟异步等待
    await asyncio.sleep(0.1)

    return f"异步搜索结果：{keyword}"

async def main():

    result = await async\_search.ainvoke({
        "keyword": "Python"
    })

    print(result)

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(main())

执行结果：

```
异步搜索结果：Python
```

对于网络请求、数据库查询等 I/O 密集型任务，异步工具通常更适合与异步 Agent 或 Web 服务配合。

需要注意，使用 `async def` 并不会让阻塞代码自动变为非阻塞。实际网络请求仍需要使用对应的异步客户端。

---

## 十一、工具调用中的错误处理

工具执行时可能发生异常，例如：

- 参数校验失败。
- 外部 API 请求超时。
- 数据库无法连接。
- 用户提供了不存在的资源。
- 外部服务返回错误。

可以根据需要将可恢复的业务错误转换为工具返回结果，方便 Agent 继续处理。

### 实例

## 实例

from langchain.tools import tool

@tool
def get\_course\_detail(course\_name: str) -> str:
    """获取指定模拟课程的详细信息。"""

    courses = {
        "Python 基础": "适合初学者",
        "Java 基础": "介绍 Java 基础语法"
    }

    if course\_name not in courses:
        return (
            f"未找到课程：{course\_name}。"
            "请检查名称或重新搜索。"
        )

    return courses[course\_name]

print(get\_course\_detail.invoke({
    "course\_name": "不存在的课程"
}))

执行结果：

```
未找到课程：不存在的课程。请检查名称或重新搜索。
```

这里通过明确的返回信息告诉 Agent：课程不存在。

对于真正的程序异常，还可以使用 LangChain 的工具错误处理机制，例如 `ToolException` 及相应的错误处理配置。

生产环境不要直接向模型或用户暴露数据库密码、服务器内部路径、完整异常堆栈等敏感信息。

---

## 十二、工具定义方式对比

LangChain 可以通过多种方式向模型声明工具，但需要区分**可执行工具**和**仅用于描述接口的工具 Schema**。

| 定义方式 | 特点 | 适用场景 |
| --- | --- | --- |
| `@tool` | 根据 Python 函数生成可执行工具 | 常规 Agent 工具 |
| `@tool(args_schema=...)` | 增加精细参数描述和校验 | 复杂参数、业务工具 |
| `StructuredTool` | 显式创建支持多个参数的工具对象 | 需要更灵活控制同步或异步执行逻辑 |
| Pydantic 模型类 | 主要用于定义工具参数结构 | 与模型的 `bind_tools()` 配合 |
| JSON Schema / 字典 | 直接描述工具名称及参数协议 | 已有工具 Schema、兼容接口 |

**注意：**将 Pydantic 类或工具 Schema 传给 `model.bind_tools()`，并不代表它包含了真正的执行逻辑。

如果希望 Agent 自动执行 Python 工具，通常推荐使用 `@tool` 创建工具对象，再传给 `create_agent()`。

---

## 十三、完整实例：模拟课程查询助手

下面将工具定义、参数校验、模型初始化和 Agent 调用组合成一个完整示例。

**文件名：**`langchain_tools_demo.py`

首先安装依赖：

```
pip install -U langchain langchain-deepseek pydantic python-dotenv
```

项目目录下的 `.env` 文件需要包含：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from typing import Literal

from dotenv import load\_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 模拟课程数据
COURSES = [
    {
        "name": "Python 基础",
        "language": "python",
        "level": "入门"
    },
    {
        "name": "Python 数据分析",
        "language": "python",
        "level": "进阶"
    },
    {
        "name": "Java 基础",
        "language": "java",
        "level": "入门"
    }
]

class SearchInput(BaseModel):
    """课程搜索参数。"""

    language: str = Field(
        min\_length=1,
        description="编程语言，例如 python、java"
    )

    level: Literal["入门", "进阶"] = Field(
        default="入门",
        description="学习难度"
    )

@tool(args\_schema=SearchInput)
def search\_courses(
    language: str,
    level: str = "入门"
) -> str:
    """根据编程语言和难度搜索模拟课程。"""

    results = [
        course["name"]
        for course in COURSES
        if course["language"] == language.lower()
        and course["level"] == level
    ]

    if not results:
        return "未找到匹配的模拟课程"

    return "、".join(results)

@tool
def count\_courses(language: str) -> str:
    """统计指定编程语言的模拟课程数量。"""

    total = sum(
        1
        for course in COURSES
        if course["language"] == language.lower()
    )

    return f"{language} 共有 {total} 门模拟课程"

# 初始化模型
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 创建 Agent
agent = create\_agent(
    model=model,
    tools=[
        search\_courses,
        count\_courses
    ],
    system\_prompt=(
        "你是一名编程学习顾问。"
        "当前工具全部使用模拟课程数据。"
        "查询课程时优先使用工具。"
        "没有找到数据时明确说明。"
        "不要编造课程名称或统计结果。"
    )
)

def ask(question: str):

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    print("\n问题：", question)

    # 输出工具调用过程
    for message in result["messages"]:

        if message.type == "ai" and message.tool\_calls:
            for call in message.tool\_calls:
                print(
                    "[工具请求]",
                    call["name"],
                    call["args"]
                )

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

    ask("有哪些 Python 入门模拟课程？")

    ask("Python 一共有多少门模拟课程？")

运行：

```
python langchain_tools_demo.py
```

程序会显示模型发起的工具调用、工具执行结果和 Agent 的最终回答。

由于工具调用路径由模型决定，实际执行日志可能略有不同。

但在正确使用模拟工具时，可以查询到：

```
Python 入门模拟课程：Python 基础



Python 模拟课程总数：2
```

这些结果来自程序中的 `COURSES` 列表，不是实时网站数据。

---

## 十四、常见问题

### 1、@tool 装饰器会立即执行函数吗？

不会。

`@tool` 负责创建工具对象，只有调用 `invoke()`、`ainvoke()`，或者由 Agent 执行相应工具调用时，才会运行底层逻辑。

### 2、为什么工具没有被 Agent 调用？

可能因为模型认为不需要该工具，也可能是工具描述不够清晰。

还应确认模型支持 Tool Calling，工具已经正确传入 `create_agent()`。

如果某个业务步骤必须执行，应在程序中明确编排，不应只依赖模型自主选择。

### 3、@tool 为什么提示缺少文档字符串？

默认情况下，如果没有显式提供描述，工具需要从函数文档字符串中获取描述信息。

例如：

```
@tool

def hello(name: str) -> str:

    """向指定用户打招呼。"""

    return f"你好，{name}"
```

也可以使用装饰器的 `description` 参数显式指定说明。

### 4、如何修改工具名称？

可以使用：

```
@tool("custom_tool_name")

def my_function(value: str) -> str:

    """工具描述。"""

    return value
```

这样工具名称将使用 `custom_tool_name`。

### 5、args\_schema 是必须的吗？

不是。

对于参数简单的函数，LangChain 通常可以根据类型注解自动生成 Schema。

如果需要字段取值范围、字符串长度等精细校验，可以使用 Pydantic 模型。

### 6、能否直接把 Pydantic 类传给 Agent？

不应该将仅描述参数的 Pydantic 类当作包含执行逻辑的工具。

在模型工具绑定场景中，Pydantic 类可以描述 Tool Calling Schema；但在 Agent 自动执行场景中，通常需要可执行的工具对象。

### 7、工具参数校验后就安全了吗？

不是。

Pydantic 可以检查数据类型和部分取值约束，但不能代替权限控制、网络访问限制、数据库参数化查询和敏感操作确认。

对于具有文件删除、支付或数据库写入能力的工具，还需要在应用层增加安全控制。

### 8、如何查看工具的参数定义？

可以使用：

```
print(my_tool.args)
```

还可以查看：

```
print(my_tool.name)

print(my_tool.description)
```

对于使用 Pydantic 定义的参数，可以通过 `model_json_schema()` 查看完整的 JSON Schema。

### 9、多个工具能并行调用吗？

如果模型在同一轮生成多个工具调用请求，Agent 的工具执行节点可能并发处理独立工具。

但实际能否并发取决于模型、工具实现和执行框架。

如果工具之间存在依赖关系，应该明确控制执行顺序。

### 10、工具返回值必须是字符串吗？

不一定。

工具可以返回其他受支持的数据类型，LangChain 会根据执行方式进行转换或封装。

在 Agent 场景中，建议让提供给模型的结果简洁、清晰，并尽量避免返回不必要的大量数据。

---

## 总结

`@tool` 是 LangChain 中定义工具的常用方式。

它可以将 Python 函数转换为具有名称、描述、参数 Schema 和执行接口的工具对象。

本章节重点介绍了：

- 使用 `@tool` 创建基本工具。
- 通过类型注解定义参数。
- 使用 `parse_docstring=True` 解析参数说明。
- 使用 `args_schema` 结合 Pydantic 校验输入。
- 将工具注册到 `create_agent()`。
- 让多个工具根据任务需要协作。
- 通过 `response_format` 返回摘要和附加数据。
- 使用异步工具和错误处理机制。

对于大多数项目，可以从下面的基本写法开始：

```
from langchain.tools import tool



@tool

def search(keyword: str) -> str:

    """根据关键词执行搜索。"""

    return f"搜索结果：{keyword}"
```

随后使用：

```
agent = create_agent(

    model=model,

    tools=[search]

)
```

将工具交给 Agent。

需要记住：**模型负责决定何时请求工具，LangChain 负责调度，工具函数负责真正执行任务。**
