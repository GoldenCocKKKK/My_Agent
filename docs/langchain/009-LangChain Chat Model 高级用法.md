# LangChain Chat Model 高级用法

> 来源: https://www.runoob.com/langchain/langchain-model-advanced.html

---

# LangChain Chat Model 高级用法

前面我们已经学习了如何通过 `init_chat_model()` 初始化模型，以及如何使用 `invoke()` 调用模型。

本章节介绍 LangChain Chat Model 的两个重要功能：

- **bind\_tools()：**将工具定义绑定到模型，使模型能够根据任务生成工具调用请求。
- **with\_structured\_output()：**按照指定的数据结构（Schema）解析模型输出，方便程序直接使用。

这两个功能经常用于 Agent、数据提取、自动化任务和 AI 应用开发。

需要注意，`bind_tools()` 与 `with_structured_output()` 的用途不同，但底层实现可能都涉及模型的工具调用能力。

---

## 一、安装与环境准备

本教程使用 LangChain 1.x，并以 DeepSeek 官方推荐的 `deepseek-flash` 模型为主要示例。

首先安装依赖：

```
pip install -U langchain langchain-deepseek pydantic python-dotenv
```

在项目目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

然后初始化模型：

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

response = model.invoke("你好，LangChain！")

print(response.content)

以下示例默认已经完成上述初始化，除非示例中重新创建模型。

需要注意，工具调用和结构化输出功能是否可用，以及具体支持的参数，取决于所使用的模型、API 和 LangChain 集成版本。

---

## 二、bind\_tools()：为模型绑定工具

`bind_tools()` 用于向聊天模型提供工具定义，让模型知道当前有哪些工具可以使用，以及工具需要哪些参数。

例如，一个天气查询工具可以接收城市名称：

```
get_weather(city="杭州")
```

当用户询问天气时，模型可以生成一个工具调用请求，指定工具名称和参数。

**注意：**`bind_tools()` 不负责真正执行 Python 函数。

它只是向模型提供工具定义，并允许模型在响应中返回 `tool_calls`。

### bind\_tools() 基本语法

```
model_with_tools = model.bind_tools(tools)
```

其中，`tools` 是工具定义列表。

LangChain 支持多种工具定义方式，包括：

- 使用 `@tool` 装饰器定义的 LangChain 工具。
- 普通 Python 函数（由集成转换其定义）。
- Pydantic 模型类。
- 符合要求的 OpenAI Tool Calling 格式字典。

### 实例 1：使用字典定义工具

下面通过 OpenAI Function Calling 风格的字典描述一个天气查询工具。

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 描述一个天气查询工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get\_weather",
            "description": "查询指定城市的模拟天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如杭州、北京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 将工具定义绑定到模型
model\_with\_tools = model.bind\_tools(tools)

# 请求模型使用工具
response = model\_with\_tools.invoke(
    "请使用 get\_weather 查询杭州的模拟天气。"
)

# 查看工具调用
if response.tool\_calls:

    for tool\_call in response.tool\_calls:
        print("工具名称：", tool\_call["name"])
        print("工具参数：", tool\_call["args"])
        print("调用 ID：", tool\_call["id"])

else:
    print("模型直接回答：", response.content)

如果模型选择调用工具，可能输出类似：

```
工具名称： get_weather

工具参数： {'city': '杭州'}

调用 ID： call_xxxxx
```

其中，调用 ID 由模型服务生成，不同请求的值通常不同。

还需要注意，即使已经绑定工具，模型也不保证每次都会调用工具。是否调用由模型和相关配置共同决定。

### tool\_calls 返回结构

LangChain 会将模型返回的工具调用请求标准化到 `AIMessage.tool_calls` 中。

典型结构如下：

```
[

    {

        "name": "get_weather",

        "args": {

            "city": "杭州"

        },

        "id": "call_xxxxx",

        "type": "tool_call"

    }

]
```

| 字段 | 说明 |
| --- | --- |
| `name` | 工具名称 |
| `args` | 模型生成的工具参数字典 |
| `id` | 工具调用标识，用于关联工具返回结果 |
| `type` | 调用内容类型 |

如果模型返回了格式不正确的工具调用，还可能出现 `invalid_tool_calls`。因此，在实际开发中不能假设每一次工具调用都有效。

---

## 三、使用 @tool 定义工具

如果工具需要真正执行 Python 代码，通常推荐使用 `@tool` 装饰器定义工具。

例如：

## 实例

from langchain.tools import tool

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。

    Args:
        city: 城市名称，如杭州、北京。
    """
    weather\_data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C",
        "上海": "小雨，22°C",
    }

    return weather\_data.get(
        city,
        f"没有找到{city}的模拟天气"
    )

# 绑定工具
model\_with\_tools = model.bind\_tools([
    get\_weather
])

response = model\_with\_tools.invoke(
    "查询杭州的模拟天气。"
)

print(response.tool\_calls)

使用 `@tool` 时，LangChain 会利用函数名称、参数类型与文档字符串构建工具定义。

在这个示例中：

- 工具名称为 `get_weather`。
- 工具需要一个字符串参数 `city`。
- 文档字符串向模型解释工具用途。

模型可以根据这些信息生成工具调用请求。

但是，此时天气查询函数仍然没有被自动执行。

---

## 四、手动执行工具：完整 Tool Calling 流程

仅使用 `bind_tools()`，模型会告诉程序需要调用哪个工具，但不会自动执行。

如果不使用 Agent，就需要自己编写工具调度代码。

基本流程如下：

```
用户提问

   ↓

模型生成工具调用请求

   ↓

程序检查 tool_calls

   ↓

调用对应 Python 工具

   ↓

将结果封装为 ToolMessage

   ↓

再次调用模型

   ↓

生成最终回答
```

### 完整实例

下面展示如何手动完成一次工具调用闭环。

## 实例

from dotenv import load\_dotenv

from langchain.chat\_models import init\_chat\_model
from langchain.tools import tool
from langchain.messages import (
    HumanMessage,
    ToolMessage
)

load\_dotenv()

# 1. 定义工具
@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。

    Args:
        city: 城市名称。
    """
    data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C"
    }

    return data.get(city, "没有该城市的模拟数据")

# 2. 初始化模型并绑定工具
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

model\_with\_tools = model.bind\_tools([
    get\_weather
])

# 3. 创建消息列表
messages = [
    HumanMessage(
        content="请调用工具查询杭州的模拟天气。"
    )
]

# 4. 第一次调用模型
response = model\_with\_tools.invoke(messages)

# 保留模型消息，包括工具调用请求
messages.append(response)

# 5. 手动执行模型请求的工具
tool\_map = {
    "get\_weather": get\_weather
}

for tool\_call in response.tool\_calls:

    tool\_name = tool\_call["name"]

    if tool\_name not in tool\_map:
        tool\_result = f"未知工具：{tool\_name}"
    else:
        try:
            # 执行对应工具
            tool\_result = tool\_map[tool\_name].invoke(
                tool\_call["args"]
            )
        except Exception as error:
            tool\_result = (
                f"工具执行失败：{type(error).\_\_name\_\_}"
            )

    # 将结果与调用 ID 关联
    messages.append(
        ToolMessage(
            content=str(tool\_result),
            tool\_call\_id=tool\_call["id"]
        )
    )

# 6. 将工具结果传回模型
if response.tool\_calls:

    final\_response = model\_with\_tools.invoke(
        messages
    )

    print("最终回答：")
    print(final\_response.content)

else:
    print("模型直接回答：")
    print(response.content)

如果第一次模型调用产生了 `get_weather` 工具调用，程序会执行对应的 Python 函数。

工具返回模拟天气后，通过 `ToolMessage` 将结果传回模型。

模型就可以利用这个结果组织最终回答。

例如：

```
杭州的模拟天气为晴天，气温25°C。
```

**注意：**这个例子演示一次工具执行与一次后续模型调用。如果第二次模型响应仍然包含工具调用请求，还需要继续循环处理。

如果需要完整的多轮工具调用、状态管理和执行控制，建议使用上一节介绍的 `create_agent()`，而不是自己实现复杂的调度逻辑。

另外，工具参数由模型生成，执行数据库修改、文件操作或网络请求等敏感工具前，还应进行参数校验和权限检查。

---

## 五、使用 Pydantic 模型描述工具参数

对于参数较多的工具，可以使用 Pydantic 定义参数结构。

Pydantic 可以提供类型声明、字段描述和数据校验，减少手写 JSON Schema 的工作量。

### 实例

## 实例

from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model

class GetWeather(BaseModel):
    """查询指定城市的模拟天气。"""

    city: str = Field(
        description="城市名称，例如杭州、北京"
    )

    unit: str = Field(
        default="celsius",
        description="温度单位：celsius 或 fahrenheit"
    )

class Calculate(BaseModel):
    """计算简单的数学表达式。"""

    expression: str = Field(
        description="数学表达式，例如 123 \* 456"
    )

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

model\_with\_tools = model.bind\_tools([
    GetWeather,
    Calculate
])

response = model\_with\_tools.invoke(
    "查询北京的模拟天气，并计算123乘以456。"
)

for tool\_call in response.tool\_calls:
    print("工具：", tool\_call["name"])
    print("参数：", tool\_call["args"])

如果模型选择调用两个工具，可能返回：

```
工具： GetWeather

参数： {'city': '北京', 'unit': 'celsius'}



工具： Calculate

参数： {'expression': '123 * 456'}
```

这里有一个重要区别：**工具名称默认来自 Pydantic 类名**。

因此：

- `GetWeather` 对应工具名称 `GetWeather`。
- `Calculate` 对应工具名称 `Calculate`。

如果希望工具名称为 `get_weather` 或 `calculate`，可以改用对应名称的 `@tool` 函数。

**注意：**这里的 Pydantic 类用于描述工具输入结构，不包含天气查询或数学计算的具体执行逻辑。模型返回工具调用请求后，仍需要应用程序提供对应的工具实现。

Pydantic 类的字段描述可以帮助模型生成符合要求的参数，但模型生成的参数并不一定始终有效。实际执行前仍需要进行校验。

---

## 六、with\_structured\_output()：结构化输出

`with_structured_output()` 用于将聊天模型的输出转换为指定的数据结构。

例如，普通模型可能返回以下文字：

```
张三今年28岁，是一名全栈工程师，

掌握 Python、React 和 Docker。
```

如果程序需要提取姓名、年龄、职业和技能，就可以通过结构化输出得到：

```
{

    "name": "张三",

    "age": 28,

    "occupation": "全栈工程师",

    "skills": ["Python", "React", "Docker"]

}
```

这样就不必自己从自然语言中解析字段。

### 基本语法

```
structured_model = model.with_structured_output(

    schema

)



result = structured_model.invoke(text)
```

其中，`schema` 用于定义期望的输出结构。

常用的结构定义方式包括：

- Pydantic 模型。
- JSON Schema 字典。
- TypedDict 类型。

**注意：**`with_structured_output()` 并不保证底层一定使用独立的原生结构化输出 API。部分模型集成会使用 Tool Calling 或 JSON Mode 实现。

### 实例：通过 Pydantic 提取人物信息

## 实例

from dotenv import load\_dotenv
from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 定义结构化输出格式
class PersonInfo(BaseModel):
    """从文本中提取人物信息。"""

    name: str = Field(
        description="人物姓名"
    )

    age: int = Field(
        description="年龄"
    )

    occupation: str = Field(
        description="职业"
    )

    skills: list[str] = Field(
        description="技能列表"
    )

# 初始化模型
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 使用 Pydantic Schema
structured\_model = model.with\_structured\_output(
    PersonInfo
)

text = (
    "张三今年28岁，是一名全栈工程师，"
    "精通 Python、React 和 Docker。"
)

result = structured\_model.invoke(
    "请根据文本提取人物信息，不要编造：\n" + text
)

print("姓名：", result.name)
print("年龄：", result.age)
print("职业：", result.occupation)
print("技能：", result.skills)

print("返回类型：", type(result))

如果模型正确提取信息，输出可能为：

```
姓名： 张三

年龄： 28

职业： 全栈工程师

技能： ['Python', 'React', 'Docker']



返回类型： &lt;class '__main__.PersonInfo'&gt;
```

使用 Pydantic Schema 且解析成功时，返回值通常是对应的 Pydantic 模型实例。

因此，可以通过：

```
result.name

result.age

result.skills
```

直接读取字段。

还可以转换为字典：

```
data = result.model_dump()



print(data)
```

Pydantic 会检查返回字段是否满足相应类型和验证规则。

但需要注意，**类型校验不等于事实校验**。即使返回结构合法，也需要根据业务需求检查模型提取的信息是否准确。

---

## 七、bind\_tools() 与 with\_structured\_output() 的区别

这两个方法都可以利用模型的结构化响应能力，但用途不同。

| 对比项目 | bind\_tools() | with\_structured\_output() |
| --- | --- | --- |
| 主要用途 | 让模型选择并请求调用工具 | 让模型输出符合指定结构的数据 |
| 典型返回值 | AIMessage | Pydantic 实例或 dict |
| 工具执行 | 需要程序或 Agent 执行 | 主要关注数据生成与解析，不负责执行外部业务工具 |
| 典型场景 | Agent、搜索、天气查询、数据库操作 | 信息提取、分类、JSON 数据生成 |
| 底层机制 | 依赖模型的工具调用支持 | 可能使用 Tool Calling、JSON Mode 或原生结构化输出 |

可以这样理解：

- **bind\_tools()：**让模型告诉程序下一步需要执行什么操作。
- **with\_structured\_output()：**让模型按照程序需要的数据格式返回内容。

例如：

```
用户：查询杭州天气



bind_tools()

    ↓

模型请求调用 get_weather

    ↓

程序执行工具

    ↓

获得天气数据
```

而结构化输出更像：

```
输入：张三今年28岁，职业是程序员



with_structured_output()

    ↓

{

    "name": "张三",

    "age": 28,

    "occupation": "程序员"

}
```

这两种能力可以在更复杂的 Agent 工作流中组合使用，但不能简单理解为完全相同的操作。

---

## 八、嵌套结构化输出

`with_structured_output()` 不仅可以提取简单字段，还可以描述嵌套对象和列表。

例如，可以从一段菜谱文字中提取：

- 菜名。
- 制作难度。
- 食材名称与用量。
- 烹饪步骤和预计时间。

### 实例：提取菜谱信息

## 实例

from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model

class Ingredient(BaseModel):
    """食材信息。"""

    name: str = Field(
        description="食材名称"
    )

    amount: str = Field(
        description="食材用量"
    )

class CookingStep(BaseModel):
    """烹饪步骤。"""

    step\_number: int = Field(
        description="步骤编号"
    )

    description: str = Field(
        description="步骤说明"
    )

    duration\_minutes: int = Field(
        description="步骤所需分钟数"
    )

class Recipe(BaseModel):
    """菜谱信息。"""

    dish\_name: str = Field(
        description="菜名"
    )

    difficulty: str = Field(
        description="难度"
    )

    ingredients: list[Ingredient] = Field(
        description="食材列表"
    )

    steps: list[CookingStep] = Field(
        description="烹饪步骤"
    )

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

structured\_model = model.with\_structured\_output(
    Recipe
)

recipe\_text = """
今天介绍番茄炒蛋。

食材：
番茄2个、鸡蛋3个、葱花少许、
盐适量、糖少许。

制作步骤：
1. 番茄切块，鸡蛋打散，约5分钟。
2. 将鸡蛋炒熟盛出，约3分钟。
3. 炒番茄至出汁，加盐和糖，约5分钟。
4. 加入鸡蛋翻炒，撒上葱花，约2分钟。
"""

result = structured\_model.invoke(
    "请将下面的菜谱整理为结构化数据。\n"
    "难度可根据描述进行合理判断，"
    "其他字段不要编造。\n\n"
    + recipe\_text
)

print("菜名：", result.dish\_name)
print("难度：", result.difficulty)

print("\n食材：")
for ingredient in result.ingredients:
    print(
        ingredient.name,
        ingredient.amount
    )

print("\n步骤：")
for step in result.steps:
    print(
        step.step\_number,
        step.description,
        step.duration\_minutes
    )

如果解析成功，可以得到类似结果：

```
菜名： 番茄炒蛋

难度： 简单



食材：

番茄 2个

鸡蛋 3个

葱花 少许

盐 适量

糖 少许



步骤：

1 番茄切块，鸡蛋打散 5

2 将鸡蛋炒熟盛出 3

3 炒番茄至出汁，加盐和糖 5

4 加入鸡蛋翻炒，撒上葱花 2
```

这里的 `Recipe` 包含两个嵌套对象列表：

```
ingredients: list[Ingredient]



steps: list[CookingStep]
```

模型可以根据 Schema 构建层级化的数据结构。

不过，嵌套结构越复杂，越需要关注模型的结构化输出能力、字段完整性和解析失败情况。

---

## 九、JSON Schema 结构化输出

如果不希望使用 Pydantic 类，也可以直接通过 JSON Schema 字典描述输出结构。

这对于需要与其他语言、前端应用或外部接口共享数据格式的项目非常有用。

### 实例：情感分析

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 定义 JSON Schema
json\_schema = {
    "title": "SentimentAnalysis",
    "description": "文本情感分析结果",
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": [
                "positive",
                "negative",
                "neutral"
            ],
            "description": "情感倾向"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "模型给出的置信度估计"
        },
        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "关键情感词"
        }
    },
    "required": [
        "sentiment",
        "confidence",
        "keywords"
    ]
}

structured\_model = model.with\_structured\_output(
    json\_schema
)

text = (
    "菜鸟教程 RUNOOB 真的太棒了，"
    "强烈推荐给编程新手！"
)

result = structured\_model.invoke(
    "请分析下面文本的情感，并返回结构化数据：\n"
    + text
)

print("情感：", result["sentiment"])
print("置信度：", result["confidence"])
print("关键词：", result["keywords"])

如果模型按照要求输出，可以得到类似：

```
情感： positive

置信度： 0.95

关键词： ['太棒了', '强烈推荐']
```

这里的置信度只是模型生成的估计值，不代表经过统计校准的真实概率。

与 Pydantic 模式不同，传入 JSON Schema 字典时，LangChain 通常返回普通的 `dict`。

例如：

```
result["sentiment"]

result["confidence"]
```

**注意：**使用 JSON Schema 并不自动保证模型输出完全符合所有规则，也不意味着 LangChain 一定会执行完整的 JSON Schema 验证。

如果业务需要严格验证数据类型、范围和必填字段，建议使用 Pydantic 或单独执行 Schema 校验。

---

## 十、include\_raw：同时保留原始响应

默认情况下，`with_structured_output()` 会直接返回解析后的结果。

如果还需要查看原始模型响应、Token 使用情况或解析错误，可以设置：

```
include_raw=True
```

### 实例

## 实例

from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model

class PersonInfo(BaseModel):
    """人物信息。"""

    name: str = Field(
        description="姓名"
    )

    age: int = Field(
        description="年龄"
    )

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

structured\_model = model.with\_structured\_output(
    PersonInfo,
    include\_raw=True
)

result = structured\_model.invoke(
    "提取人物信息：张三今年28岁。"
)

print("解析结果：", result["parsed"])

print("解析错误：", result["parsing\_error"])

print("原始响应：", result["raw"])

设置 `include_raw=True` 后，返回结果通常包含以下三个字段：

| 字段 | 含义 |
| --- | --- |
| `raw` | 模型的原始消息响应 |
| `parsed` | 解析完成的结构化结果 |
| `parsing_error` | 解析阶段发生的错误，没有错误时通常为 None |

如果解析成功，`parsed` 会包含对应的 Pydantic 实例。

如果解析失败，`parsing_error` 可以帮助定位问题。

**注意：**`include_raw=True` 主要帮助观察解析过程，不代表会捕获所有网络错误、认证错误或模型服务异常。

---

## 十一、结构化输出的实现方式

LangChain 的 `with_structured_output()` 是一个统一接口，实际使用的底层机制由模型集成决定。

常见方式包括：

| 方式 | 说明 |
| --- | --- |
| `function_calling` | 通过工具或函数调用机制产生符合指定结构的参数 |
| `json_mode` | 使用模型服务的 JSON 输出模式 |
| `json_schema` | 使用支持原生 JSON Schema 约束的模型接口 |

并不是每个提供商都支持这三种方式。

另外，不同 LangChain 集成对 `method` 参数的默认值和实现方式也可能不同。

### DeepSeek 的 JSON 输出能力

DeepSeek 官方提供 JSON Output 功能，可在对应接口中设置：

```
{

    "response_format": {

        "type": "json_object"

    }

}
```

根据 DeepSeek 官方说明，使用 JSON Output 时，还应该在提示词中明确要求输出 JSON，并提供期望的结构示例。

例如：

```
请将下列人物信息转换为 JSON。



JSON 字段：

{

    "name": "姓名",

    "age": 0

}



输入：张三今年28岁。
```

需要注意：**JSON Mode 主要保证输出采用合法 JSON 格式，并不等于所有字段都严格符合指定的 Schema。**

如果需要字段类型校验，仍然建议在程序中使用 Pydantic 等机制验证。

此外，DeepSeek 官方还提示 JSON Mode 在部分情况下可能出现空输出或内容截断，因此需要合理设置输出上限并做好异常处理。

---

## 十二、结构化输出失败时如何处理？

实际调用中，模型可能返回缺少字段、类型错误或不符合格式的数据。

例如，程序要求：

```
{

    "name": "张三",

    "age": 28

}
```

但模型实际返回：

```
{

    "name": "张三",

    "age": "未知"

}
```

此时 `age` 不满足预期的整数类型。

如果使用 Pydantic 模型，可能在解析验证阶段发生错误。

### 实例：捕获结构化输出错误

## 实例

from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model

class UserInfo(BaseModel):
    """用户信息。"""

    name: str = Field(
        description="用户姓名"
    )

    age: int = Field(
        ge=0,
        le=150,
        description="用户年龄"
    )

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

structured\_model = model.with\_structured\_output(
    UserInfo,
    include\_raw=True
)

try:
    result = structured\_model.invoke(
        "提取用户信息：张三今年28岁。"
    )

    if result["parsing\_error"]:
        print(
            "解析失败：",
            result["parsing\_error"]
        )
    else:
        user = result["parsed"]

        print("姓名：", user.name)
        print("年龄：", user.age)

except Exception as error:
    print(
        "模型请求失败：",
        type(error).\_\_name\_\_,
        str(error)
    )

这个示例包含两层检查：

- 通过 `parsing_error` 检查解析过程。
- 通过异常捕获处理请求执行过程中发生的其他错误。

对于生产环境，建议进一步增加日志、有限次数的重试和结果业务校验。

**注意：**不要因为 JSON 能够被解析，就直接信任其中所有数据。模型仍可能生成事实不准确的内容。

---

## 十三、ConfigurableModel 中使用 bind\_tools()

除了固定模型，LangChain 通过 `init_chat_model()` 创建的可配置模型也支持工具绑定等声明式操作。

这意味着可以先配置模型和工具，再在运行时选择使用哪个模型。

### 实例：动态切换 DeepSeek Flash 和 Pro

## 实例

from langchain.chat\_models import init\_chat\_model
from langchain.tools import tool

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。

    Args:
        city: 城市名称。
    """
    data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C"
    }

    return data.get(
        city,
        "没有该城市的模拟数据"
    )

# 创建可配置模型
configurable\_model = init\_chat\_model(
    "deepseek:deepseek-flash",

    configurable\_fields=[
        "model"
    ]
)

# 绑定工具
model\_with\_tools = configurable\_model.bind\_tools([
    get\_weather
])

# 使用默认 Flash
response = model\_with\_tools.invoke(
    "查询杭州的模拟天气。"
)

print("Flash 工具调用：")
print(response.tool\_calls)

# 切换到 Pro
response = model\_with\_tools.invoke(
    "查询北京的模拟天气。",
    config={
        "configurable": {
            "model": "deepseek:deepseek-v4-pro"
        }
    }
)

print("Pro 工具调用：")
print(response.tool\_calls)

上述代码中，`configurable_model` 可以在运行时选择 Flash 或 Pro。

绑定工具后，仍然可以通过 `config` 指定当前模型。

但需要注意，目标模型必须支持相应的工具调用功能。

同时，`bind_tools()` 仍然只让模型生成工具调用请求，不会自动执行工具。

---

## 十四、ConfigurableModel 中使用结构化输出

可配置模型同样可以在支持的集成上使用 `with_structured_output()`。

需要注意，多个模型提供商的结构化输出能力可能不同。因此，动态切换模型时，必须确保目标模型支持所使用的输出 Schema 和解析方式。

### 实例

## 实例

from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model

class SummaryInfo(BaseModel):
    """文本摘要信息。"""

    title: str = Field(
        description="标题"
    )

    summary: str = Field(
        description="简短摘要"
    )

configurable\_model = init\_chat\_model(
    "deepseek:deepseek-flash",
    configurable\_fields=["model"]
)

structured\_model = (
    configurable\_model.with\_structured\_output(
        SummaryInfo
    )
)

# 使用默认 Flash
result = structured\_model.invoke(
    "请提取标题和摘要："
    "Python 是一门简单易学的编程语言，"
    "广泛用于 Web 开发和数据分析。"
)

print(result.model\_dump())

# 运行时切换为 Pro
result = structured\_model.invoke(
    "请提取标题和摘要："
    "LangChain 可以帮助开发者构建 AI 应用。",
    config={
        "configurable": {
            "model": "deepseek:deepseek-v4-pro"
        }
    }
)

print(result.model\_dump())

这类声明式操作能够与可配置模型组合，让模型选择和输出结构保持相对独立。

不过，切换提供商时不能假设全部参数与 Schema 都能原样使用。对于不同模型，应分别测试工具调用和结构化输出兼容性。

---

## 十五、完整实例：提取结构化文章信息

下面使用 `with_structured_output()` 创建一个简单的文章信息提取程序。

假设用户输入一段介绍 Python 的文章，程序需要提取标题、摘要和关键词。

文件名：`chat_model_advanced.py`

## 完整实例

# 文件：chat\_model\_advanced.py

from dotenv import load\_dotenv
from pydantic import BaseModel, Field

from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 定义文章输出结构
class ArticleInfo(BaseModel):
    """文章信息。"""

    title: str = Field(
        description="文章标题"
    )

    summary: str = Field(
        description="100字以内的文章摘要"
    )

    keywords: list[str] = Field(
        description="文章关键词"
    )

# 初始化模型
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 配置结构化输出
structured\_model = model.with\_structured\_output(
    ArticleInfo,
    include\_raw=True
)

# 待分析文章
article = """
Python 是一种通用的高级编程语言。

Python 的语法简洁，适合初学者学习。

它广泛应用于 Web 开发、数据分析、
人工智能和自动化脚本等领域。

Python 拥有丰富的第三方库和活跃的社区。
"""

# 调用模型
try:
    response = structured\_model.invoke(
        "请从下面的文章中提取标题、摘要和关键词。"
        "不要编造原文中不存在的事实。\n\n"
        + article
    )

    # 检查解析是否成功
    if response["parsing\_error"]:
        print(
            "解析失败：",
            response["parsing\_error"]
        )

    else:
        result = response["parsed"]

        print("标题：", result.title)
        print("摘要：", result.summary)
        print("关键词：", result.keywords)

        # 转换为 JSON
        print("\nJSON：")
        print(
            result.model\_dump\_json(indent=2)
        )

except Exception as error:
    print(
        "模型调用失败：",
        type(error).\_\_name\_\_,
        str(error)
    )

如果模型成功解析，程序可以生成类似的 JSON 数据：

```
{

  "title": "Python 编程语言简介",

  "summary": "Python 语法简洁，应用广泛，拥有丰富的第三方库和活跃的社区。",

  "keywords": [

    "Python",

    "Web开发",

    "数据分析",

    "人工智能",

    "自动化"

  ]

}
```

这个例子展示了结构化输出在实际应用中的用途。

例如，可以将提取结果用于：

- 自动生成文章摘要。
- 提取文章关键词。
- 构建内容分类系统。
- 将非结构化文本转换为数据库记录。
- 为搜索和推荐系统生成结构化元数据。

---

## 十六、常见问题

### 1、为什么 bind\_tools() 没有执行 Python 函数？

因为 `bind_tools()` 只负责向模型声明工具，并不自动执行。

模型通常通过 `AIMessage.tool_calls` 返回工具调用请求。

要真正执行工具，可以自己编写调度代码，也可以使用 LangChain 的 `create_agent()`。

### 2、为什么 tool\_calls 为空？

可能有以下原因：

- 模型认为不需要工具。
- 工具描述不清晰。
- 所使用的模型不支持工具调用。
- 模型选择直接生成文本。

如果需要提高工具调用的准确率，可以完善工具描述和提示词，并检查提供商是否支持相关的 Tool Choice 参数。

需要注意，强制工具调用参数并不是所有提供商都支持。

### 3、Pydantic 工具类为什么没有自动执行？

Pydantic 类主要用于定义工具参数结构，不包含实际执行逻辑。

如果使用 `model.bind_tools([GetWeather])`，模型可以生成名为 `GetWeather` 的工具调用，但程序仍需要提供真正的查询逻辑。

### 4、为什么 with\_structured\_output() 会报错？

可能是模型不支持当前的结构化输出方式，也可能是模型返回的数据无法通过解析或验证。

建议：

- 确认模型是否支持结构化输出。
- 检查 Schema 定义是否过于复杂。
- 尝试减少字段或嵌套层级。
- 设置 `include_raw=True` 查看原始响应。
- 检查 `parsing_error` 和请求异常。

### 5、JSON Schema 和 Pydantic 哪个更好？

如果主要在 Python 中使用，Pydantic 通常更方便，可以通过属性访问字段，并提供类型验证。

如果需要在 Python、JavaScript、Java 等不同语言之间共享数据结构，JSON Schema 更容易与多种语言和系统集成。

具体选择取决于项目需求。

### 6、结构化输出是否保证内容正确？

不能。

结构化输出主要控制返回数据的形式，并不能保证模型提取或生成的事实全部正确。

对于合同、金融、医疗或其他高要求场景，应进行必要的数据校验和人工审核。

### 7、为什么结构化输出返回 dict，而不是 Pydantic 对象？

返回值类型与传入的 Schema 有关。

一般情况下：

- 使用 Pydantic 模型类：解析成功时返回 Pydantic 实例。
- 使用 JSON Schema 或 TypedDict：通常返回字典。
- 使用 `include_raw=True`：返回包含 `raw`、`parsed` 和 `parsing_error` 的字典。

### 8、是否可以同时使用工具和结构化输出？

可以在支持相关能力的模型和 Agent 框架中组合使用。

例如，Agent 可以先查询数据库，再把结果整理为结构化数据。

但不同模型集成支持的组合方式可能不同，不应假设可以任意链式调用 `bind_tools()` 和 `with_structured_output()` 而始终正常工作。

对于较复杂的应用，可以使用 `create_agent()` 提供工具，并通过 Agent 的结构化响应配置处理最终输出。

---

## 总结

本章节介绍了 LangChain Chat Model 的两个高级功能。

**bind\_tools()** 主要用于工具调用：

```
model_with_tools = model.bind_tools([

    get_weather

])



response = model_with_tools.invoke(

    "查询杭州天气"

)



print(response.tool_calls)
```

**with\_structured\_output()** 主要用于结构化数据生成：

```
structured_model = model.with_structured_output(

    PersonInfo

)



result = structured_model.invoke(

    "张三今年28岁"

)



print(result.name)
```

它们的主要区别是：

- `bind_tools()` 让模型能够请求执行外部工具，但不负责真正执行。
- `with_structured_output()` 将模型输出解析为指定数据结构。
- Pydantic 适合需要 Python 类型验证的应用。
- JSON Schema 适合需要跨语言共享数据格式的场景。
- `include_raw=True` 可以帮助调试结构化输出结果和解析错误。
- 可配置模型可以与声明式工具和结构化输出操作组合，但需要检查目标模型的兼容性。

掌握这两个方法后，就具备了进一步学习 LangChain Agent、工具编排、数据提取和 AI 工作流的基础。
