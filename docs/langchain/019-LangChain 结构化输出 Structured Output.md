# LangChain 结构化输出 Structured Output

> 来源: https://www.runoob.com/langchain/langchain-structured-output.html

---

# LangChain 结构化输出 Structured Output

大多数时候，我们需要的并不是一段自由文本，而是程序能够直接处理的数据，例如 JSON 对象、用户信息、商品数据或课程推荐结果。

LangChain 提供 Structured Output（结构化输出） 功能，可以让模型按照指定的数据结构返回结果，减少手动解析文本的工作。

在 LangChain 1.x 中，可以通过 `create_agent(response_format=...)` 配置 Agent 的结构化输出，也可以直接使用模型的 `with_structured_output()` 方法。

本章节基于 LangChain 1.x 和 Pydantic 2，介绍 Pydantic、TypedDict、ProviderStrategy、ToolStrategy、嵌套结构以及错误处理。

---

## 一、为什么需要结构化输出？

假设用户提供了一段描述：

```
张三今年 28 岁，是一名 Python 工程师。
```

普通模型可能返回：

```
张三今年 28 岁，职业是 Python 工程师。
```

如果需要将这些信息写入数据库，还需要从自然语言中提取姓名、年龄和职业。

而结构化输出可以直接返回：

```
{

    "name": "张三",

    "age": 28,

    "job": "Python 工程师"

}
```

| 方式 | 返回内容 | 后续处理 |
| --- | --- | --- |
| 普通文本 | 自然语言回答 | 需要解析和提取字段 |
| 结构化输出 | Pydantic 对象或字典 | 可以按字段访问和校验 |

结构化输出的优势在于：

- 明确字段名称和数据类型。
- 减少对正则表达式和文本解析的依赖。
- 支持字段范围、枚举及其他约束。
- 便于与数据库、API 和前端组件集成。

**注意：**结构化输出可以检查数据格式，但不能保证数据内容一定真实。例如，模型可以生成符合 Schema 的错误日期或不存在的课程名称。

---

## 二、安装与环境准备

使用 Python 3.10 或更新版本，安装 LangChain 1.x、DeepSeek 集成和相关依赖：

```
python -m pip install -U "langchain&gt;=1.2,&lt;2" langchain-deepseek python-dotenv "pydantic&gt;=2,&lt;3" typing_extensions
```

项目目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

在调用模型前加载环境变量：

```
from dotenv import load_dotenv

load_dotenv()
```

本教程使用 `ChatDeepSeek(model="deepseek-flash")`。其中 `deepseek-flash` 是 DeepSeek API 模型名；使用 `init_chat_model()` 时，对应的完整写法是 `deepseek:deepseek-flash`。

为了便于演示信息提取，示例统一关闭思考模式，并显式使用工具调用方式。DeepSeek 当前的思考模式也支持工具调用，但多轮工具调用需要正确保留 `reasoning_content`，不能简单认为"思考模型不支持结构化输出"。

各节完整代码示例均加载 `.env` 并初始化模型；标明"复用"的片段需要先保留对应的 Schema 定义。第十二节提供可单独保存运行的完整程序。

---

## 三、最简单的用法：Pydantic 模型

使用 Pydantic 定义需要的输出结构，然后通过 `response_format` 传给 `create_agent()`。

例如，从一段用户描述中提取课程名称、难度、学习时长和是否免费。

### 实例：提取课程信息

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

class CourseInfo(BaseModel):
    """从文本中提取课程信息。"""
    course\_name: str = Field(description="课程名称")
    difficulty: Literal["入门", "进阶", "高级"]
    estimated\_hours: int = Field(ge=1, description="预计学习小时数")
    is\_free: bool = Field(description="是否免费")

agent = create\_agent(
    model=model,
    tools=[],
    response\_format=ToolStrategy(CourseInfo),
    system\_prompt="只提取用户明确提供的课程信息，不要编造。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我在学 Python3 基础教程，属于入门课程，预计学习20小时，而且是免费的。"}]
})

course = result["structured\_response"]
print("课程名：", course.course\_name)
print("难度：", course.difficulty)
print("时长：", course.estimated\_hours, "小时")
print("免费：", course.is\_free)
print("对象类型：", type(course).\_\_name\_\_)

如果模型成功提取，输出类似：

```
课程名： Python3 基础教程

难度： 入门

时长： 20 小时

免费： True

对象类型： CourseInfo
```

其中：

- `BaseModel`：定义数据结构。
- `Field`：配置字段说明及约束。
- `Literal`：限制允许的字段值。
- `response_format`：指定 Agent 输出结构；这里用 `ToolStrategy` 明确通过工具调用生成结构化结果。
- `structured_response`：获取最终结构化结果。

使用 Pydantic Schema 且解析成功后，`structured_response` 是 Pydantic 模型实例，可以通过属性直接访问。Pydantic 默认允许部分类型转换，例如将数字字符串转换为整数；需要严格拒绝此类转换时，可以使用 `Field(strict=True)`。

### 转换为 JSON

如果需要发送给前端，可以使用：

```
print(course.model_dump())

print(course.model_dump_json(indent=2))
```

`model_dump()` 返回 Python 字典，`model_dump_json()` 返回 JSON 字符串。

**注意：**如果原文没有提供某个字段，不应该让模型随意猜测。可以使用可选字段或要求补充信息。

---

## 四、ProviderStrategy 与 ToolStrategy

LangChain 1.x 提供两种主要的 Agent 结构化输出策略：

| 策略 | 实现方式 | 适用场景 |
| --- | --- | --- |
| `ProviderStrategy` | 使用模型服务商原生结构化输出能力 | 提供商支持原生 Schema 约束 |
| `ToolStrategy` | 通过 Tool Calling 生成结构化数据 | 支持工具调用的模型 |
| 直接传 Schema | 由 LangChain 根据模型能力选择策略 | 希望自动选择的常规场景 |

### 1、自动选择策略

下面复用上一节的 `model` 和 `CourseInfo`：

```
agent = create_agent(

    model=model,

    response_format=CourseInfo

)
```

LangChain 会根据当前模型的能力信息，选择支持的结构化输出方式。

直接传入 Schema 类型时，LangChain 在识别到原生结构化输出能力后使用 `ProviderStrategy`，否则使用 `ToolStrategy`。从 LangChain 1.1 起，该判断会读取模型的 profile 能力信息。

自动选择依赖模型能力信息，并不保证所有模型都能完成结构化输出。

### 2、显式使用 ToolStrategy

对于使用 Tool Calling 完成结构化输出的模型，可以显式配置。下面继续复用第三节的 `CourseInfo`：

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

agent = create\_agent(
    model=model,
    tools=[],
    response\_format=ToolStrategy(CourseInfo),
    system\_prompt="从用户描述中提取课程信息。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Python3 基础教程是免费入门课程，预计学习20小时。"}]
})

print(result["structured\_response"].model\_dump())

ToolStrategy 会通过模型的工具调用机制传递结构化数据，再由 LangChain 进行解析和校验。

这里的结构化输出工具主要用于传递数据 Schema，不等于执行了一个真实的课程查询 Python 函数。

### 3、显式使用 ProviderStrategy

只有模型、API 端点和 LangChain 集成都支持原生结构化输出时，才应显式使用 `ProviderStrategy`。下面使用支持该能力的 OpenAI 模型演示，不复用前面的 DeepSeek 模型：

```
python -m pip install -U langchain-openai
```

在 `.env` 中另行配置 `OPENAI_API_KEY`，然后运行：

```
from dotenv import load_dotenv

from pydantic import BaseModel

from langchain_openai import ChatOpenAI

from langchain.agents import create_agent

from langchain.agents.structured_output import ProviderStrategy



load_dotenv()



class ContactInfo(BaseModel):

    """联系人信息。"""

    name: str

    email: str



native_model = ChatOpenAI(model="gpt-5.5")

agent = create_agent(

    model=native_model,

    response_format=ProviderStrategy(ContactInfo, strict=True),

)



result = agent.invoke({

    "messages": [{"role": "user", "content": "张三的邮箱是 zhangsan@example.com。"}]

})

print(result["structured_response"].model_dump())
```

`ProviderStrategy(strict=True)` 需要 LangChain 1.2 或更新版本，而且只适用于支持该参数的提供商。它控制服务端的 Schema 约束，与 Pydantic 的 `Field(strict=True)` 类型校验不是同一设置。

JSON 模式主要约束返回合法 JSON，不等于按指定 JSON Schema 生成。本教程的 `ChatDeepSeek.with_structured_output(..., method="function_calling")` 走工具调用路径；不能仅凭参数名称推断请求使用了服务商原生 Schema 接口。

### 4、JSON Schema 字典

如果 Schema 来自配置文件，也可以使用 JSON Schema 字典。传给 Agent 时应显式包在策略中，不能直接依赖 `response_format=schema_dict` 自动选择：

```
person_schema = {

    "title": "PersonInfo",

    "description": "从文本中提取姓名与年龄。",

    "type": "object",

    "properties": {

        "name": {"type": "string"},

        "age": {"type": "integer", "minimum": 0}

    },

    "required": ["name", "age"],

    "additionalProperties": False

}



agent = create_agent(

    model=model,

    response_format=ToolStrategy(person_schema),

)
```

这里复用前面的 DeepSeek `model` 和 `ToolStrategy` 导入。返回结果为字典；原始 JSON Schema 字典在 ToolStrategy 路径中不会自动执行完整的本地 Schema 校验，也不能依靠 `handle_errors` 修复所有字段违规。需要本地验证和自动纠错时，优先使用 Pydantic。

---

## 五、与工具共存的 Structured Output

结构化输出和普通工具可以组合使用。

例如，课程推荐 Agent 可以先搜索课程，再将最终结果整理成指定的结构。

需要注意，模型必须支持当前结构化输出策略及普通工具调用的组合。

### 实例：搜索课程并生成推荐

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

COURSES = {
    "python": "Python3 基础教程，入门，模拟学习时长20小时",
    "java": "Java 基础教程，入门，模拟学习时长25小时"
}

@tool
def search\_course(keyword: str) -> str:
    """根据编程语言查询模拟课程。"""
    return COURSES.get(keyword.lower(), "未找到相关模拟课程")

class CourseRecommendation(BaseModel):
    """课程推荐结果。"""
    course\_name: str = Field(description="推荐课程")
    reason: str = Field(description="推荐理由")
    difficulty: str = Field(description="课程难度")

agent = create\_agent(
    model=model,
    tools=[search\_course],
    response\_format=ToolStrategy(CourseRecommendation),
    system\_prompt="你是编程学习顾问。查询课程时使用工具，仅推荐工具实际返回的课程。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我零基础想学 Python，请查询课程并推荐。"}]
})

rec = result["structured\_response"]
print("课程：", rec.course\_name)
print("理由：", rec.reason)
print("难度：", rec.difficulty)

for msg in result["messages"]:
    if msg.type == "tool" and msg.name == "search\_course":
        print("课程查询结果：", msg.content)

如果模型正确调用查询工具，可能得到：

```
课程： Python3 基础教程

理由： 适合没有编程基础的学习者

难度： 入门
```

代码按工具名筛选 `search_course` 的消息，单独显示课程查询结果。

`result["messages"]` 中还可能包含结构化输出策略生成的 ToolMessage，因此不能把每条工具消息都当作业务查询。

**注意：**模型不一定严格按照预期调用普通工具。如果必须查询真实数据库并保证数据来源，应在业务工作流中强制执行查询。

---

## 六、复杂嵌套结构

Pydantic 支持列表、嵌套模型、枚举和字段限制，适合用于学习计划、订单信息等复杂数据结构。

### 实例：生成 Python 学习计划

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from typing import Literal
from pydantic import BaseModel, Field, model\_validator
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

class Topic(BaseModel):
    """单个学习主题。"""
    name: str
    order: int = Field(ge=1)
    minutes: int = Field(gt=0)

class LearningPlan(BaseModel):
    """学习计划。"""
    goal: str
    level: Literal["入门", "进阶", "高级"]
    total\_hours: float = Field(gt=0, le=10)
    topics: list[Topic] = Field(min\_length=1)

    @model\_validator(mode="after")
    def check\_duration(self):
        total = sum(topic.minutes for topic in self.topics)
        if abs(total / 60 - self.total\_hours) > 0.01:
            raise ValueError("total\_hours 必须等于全部主题分钟数之和除以60")
        return self

agent = create\_agent(
    model=model,
    response\_format=ToolStrategy(LearningPlan),
    system\_prompt="制定合理的学习计划。所有主题分钟数之和必须等于 total\_hours × 60。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "制定一个总时长不超过10小时的 Python 入门学习计划。"}]
})

plan = result["structured\_response"]
print("目标：", plan.goal)
print("难度：", plan.level)
print("总时长：", plan.total\_hours, "小时")

for topic in plan.topics:
    print(f"{topic.order}. {topic.name}（{topic.minutes}分钟）")

成功执行后，可能得到：

```
目标：掌握 Python 基础语法

难度：入门

总时长：6.0 小时



1. Python 环境与变量（60分钟）

2. 数据类型（60分钟）

3. 条件判断（60分钟）

4. 循环（60分钟）

5. 函数（60分钟）

6. 综合练习（60分钟）
```

这里不仅验证 `total_hours` 不超过 10 小时，还通过 `model_validator` 校验总时长与各主题分钟数是否一致。自定义验证器在 Python 本地执行，其逻辑不会自动变成服务商能够执行的 JSON Schema 规则；本例允许总时长存在不超过 0.01 小时的误差。

如果模型返回不符合约束的数据，Pydantic 会产生验证错误。使用 ToolStrategy 时，LangChain 可以根据错误处理策略要求模型重新生成。

---

## 七、使用 TypedDict 返回字典

除了 Pydantic，LangChain 也支持使用 `TypedDict` 定义结构化输出。

与 Pydantic 模型不同，TypedDict 方案通常返回普通 Python 字典。

### 实例：提取人员信息

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from typing\_extensions import TypedDict
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

class PersonInfo(TypedDict):
    """人员信息。"""
    name: str
    age: int
    job: str

agent = create\_agent(
    model=model,
    response\_format=ToolStrategy(PersonInfo),
    system\_prompt="提取用户明确提供的人员信息，不要猜测。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "张三今年28岁，是一名Python工程师。"}]
})

person = result["structured\_response"]
print(person["name"])
print(person["age"])
print(person["job"])
print(type(person).\_\_name\_\_)

可能输出：

```
张三

28

Python工程师

dict
```

两种方式的区别：

| 方式 | 返回类型 | 主要特点 |
| --- | --- | --- |
| Pydantic | BaseModel 实例 | 支持丰富的字段约束与自定义验证 |
| TypedDict | dict | 声明简单，适合普通结构化数据 |

`TypedDict` 本身主要提供类型声明；当前 Agent 结构化输出路径会使用 Pydantic TypeAdapter 进行解析和类型验证。但直接调用模型的 `with_structured_output(TypedDict)` 时，不应假定具有同样的本地验证行为。涉及自定义业务规则时，优先选择 Pydantic。

---

## 八、可选字段与缺失数据

实际文本中经常存在信息不完整的情况。

例如：

```
我正在学习 Python 基础教程，属于入门级别。
```

用户没有提供预计学习时长，也没有说明是否收费。

此时不应该让模型自行编造这两个字段。

可以将字段定义为可选：

## 实例

from typing import Literal
from pydantic import BaseModel, Field

class OptionalCourseInfo(BaseModel):
    """允许缺失字段的课程信息。"""
    course\_name: str | None = Field(default=None, min\_length=1)
    difficulty: Literal["入门", "进阶", "高级"] | None = None
    estimated\_hours: int | None = Field(default=None, ge=1)
    is\_free: bool | None = None

将本节的 `OptionalCourseInfo` 传给 `ToolStrategy` 即可使用这个结构。`str | None` 表示允许空值，`= None` 才表示字段可省略；在 Pydantic 2 中，只写 `name: str | None` 仍是必填字段。

Python 中用 `None` 表示未知值，序列化为 JSON 后对应 `null`。

例如：

```
{

    "course_name": "Python 基础教程",

    "difficulty": "入门",

    "estimated_hours": null,

    "is_free": null

}
```

**注意：**声明可选字段只是允许缺失值，并不能彻底防止模型猜测。因此仍需在 System Prompt 中说明只提取明确提供的信息。

---

## 九、直接使用 with\_structured\_output()

如果任务只是从文本中提取信息，不需要 Agent 调用工具，可以直接使用聊天模型的 `with_structured_output()`。

这通常比创建 Agent 更简单。下面显式指定 `method="function_calling"`，通过 DeepSeek 的工具调用机制返回结构化数据。

### 实例：情感分析

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from typing import Literal
from pydantic import BaseModel, Field

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

class SentimentResult(BaseModel):
    """情感分析结果。"""
    sentiment: Literal["积极", "消极", "中性"]
    score: float = Field(ge=0, le=1, description="情感判断强度")
    keywords: list[str]

structured\_model = model.with\_structured\_output(SentimentResult, method="function\_calling")

texts = [
    "菜鸟教程真的很好用，强烈推荐！",
    "这个教程内容比较少，有些失望。",
    "我今天学习了 Python 变量。"
]

for text in texts:
    result = structured\_model.invoke(text)
    print(text)
    print(result.model\_dump())

可能得到：

```
菜鸟教程真的很好用，强烈推荐！

{'sentiment': '积极', 'score': 0.95, 'keywords': ['好用', '推荐']}

这个教程内容比较少，有些失望。

{'sentiment': '消极', 'score': 0.7, 'keywords': ['内容少', '失望']}

我今天学习了 Python 变量。

{'sentiment': '中性', 'score': 0.2, 'keywords': ['Python', '变量']}
```

其中的情感分类和评分由模型判断，不具有绝对客观性。

尤其要注意，`score` 并不自动代表经过统计校准的概率或置信度。

如果需要用于真实业务决策，应结合测试数据评估模型表现。

### 与 Agent 的区别

| 方式 | 返回值 | 适用场景 |
| --- | --- | --- |
| `create_agent(response_format=...)` | Agent 状态，包含 structured\_response | 需要工具调用、多步骤处理 |
| `model.with_structured_output(...)` | 解析后的结构化数据 | 文本提取、分类、简单解析 |

`with_structured_output()` 的内部实现由模型集成和 `method` 参数决定。直接调用不会因为字段校验失败就自动启动 Agent 式的对话纠错；解析异常需要调用方处理。

### JSON 模式与工具调用的区别

| 方式 | 主要作用 | 本地校验 |
| --- | --- | --- |
| `method="function_calling"` | 将 Schema 转换为工具参数结构 | 传入 Pydantic 时验证字段 |
| `method="json_mode"` | 要求返回合法 JSON，需在提示词中说明 JSON 格式和字段 | 传入 Pydantic 时验证解析结果，不等于服务端强制遵守 Schema |
| `ProviderStrategy` | 由支持原生结构化输出的服务商约束结构 | 仍需执行无法用 Schema 表达的业务校验 |

---

## 十、include\_raw：查看原始模型响应

使用 `with_structured_output()` 时，有时需要同时查看模型原始消息及解析结果。

可以设置：

```
include_raw=True
```

### 实例

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

class Person(BaseModel):
    name: str
    age: int = Field(ge=0)

structured\_model = model.with\_structured\_output(
    Person, method="function\_calling", include\_raw=True
)

result = structured\_model.invoke("张三今年28岁。")

print("解析结果：", result["parsed"])
print("原始消息：", result["raw"])
print("解析错误：", result["parsing\_error"])

该调用返回包含以下三个键的字典：

| 字段 | 说明 |
| --- | --- |
| `raw` | 模型生成的原始 AIMessage |
| `parsed` | 解析成功的结构化对象；解析失败时为 None。未获得预期工具调用时也可能为 None |
| `parsing_error` | 捕获的解析异常对象；没有解析异常时为 None |

`include_raw=True` 适合调试模型输出格式。使用结果前既要检查 `parsing_error`，也要检查 `parsed` 是否为 None，不能仅凭没有异常就认为已经取得结构化对象。

**注意：**它并不意味着所有错误都会转化为 `parsing_error`。例如，网络连接失败、API 鉴权错误和请求超时仍可能直接抛出异常。

---

## 十一、结构化输出的错误处理

模型生成结构化数据时，可能出现以下问题：

- 缺少必填字段。
- 字段类型不正确。
- 枚举值不在允许范围内。
- 数值超出限制。
- 一次返回多个结构化结果，即使内容相同，也可能触发 `MultipleStructuredOutputsError`。

对于 ToolStrategy，LangChain 提供 `handle_errors` 参数，可以控制 Schema 验证失败后的处理方式。

| 配置 | 行为 |
| --- | --- |
| `True` | 自动处理受支持的结构化输出错误，并尝试让模型修正 |
| `False` | 不进行该策略的自动错误恢复，异常继续抛出 |
| 字符串 | 使用指定的错误提示反馈给模型 |
| 异常类型或异常类型元组 | 只处理指定的结构化输出异常，其余继续抛出 |
| Callable | 通过自定义函数生成错误提示 |

### 实例：自定义错误提示

## 实例

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

load\_dotenv()

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

class ProductRating(BaseModel):
    """商品评分。"""
    rating: int = Field(ge=1, le=5)
    comment: str

agent = create\_agent(
    model=model,
    response\_format=ToolStrategy(
        ProductRating,
        handle\_errors="评分必须为1到5的整数，并且需要提供评论。"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "这个商品不错，我给它4分。"}]
})

print(result["structured\_response"])

如果模型返回非法评分，LangChain 可以将错误提示交给模型，让其尝试重新生成。

`handle_errors` 默认是 `True`，处理的是 Agent 结构化输出的解析或验证错误，不负责处理所有网络或鉴权异常，也不保证最终成功。

需要限制反复纠错时，可以在调用 Agent 时设置图执行步数上限：

```
result = agent.invoke(

    {"messages": [{"role": "user", "content": "这个商品不错，我给它4分。"}]},

    config={"recursion_limit": 12},

)
```

`recursion_limit` 限制图执行步数，包含模型和工具等节点，不等于"最多重试12次"。模型初始化时的 `max_retries` 则主要控制请求层重试，二者用途不同。

**注意：**这里的 `ToolStrategy(handle_errors=...)` 与普通工具的 `BaseTool.handle_tool_error` 是不同机制，不应混淆。

---

## 十二、完整实例：从用户描述中提取课程信息

下面整合 Pydantic、可空字段、字段约束、ToolStrategy、执行步数限制和 JSON 输出。所有业务字段都允许未知值，避免"缺失信息返回 null"与必填课程名发生冲突。

文件名：`structured_output_demo.py`

## 完整实例

from typing import Literal
from dotenv import load\_dotenv
from pydantic import BaseModel, ConfigDict, Field
from langchain\_deepseek import ChatDeepSeek
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy, StructuredOutputError
from langgraph.errors import GraphRecursionError
from openai import APIError

load\_dotenv()

class CourseInfo(BaseModel):
    """从用户文本中提取课程资料；未明确说明的字段返回 null。"""
    model\_config = ConfigDict(extra="forbid")

    course\_name: str | None = Field(default=None, min\_length=1, description="课程名称")
    difficulty: Literal["入门", "进阶", "高级"] | None = None
    estimated\_hours: int | None = Field(default=None, ge=1, le=1000)
    is\_free: bool | None = Field(default=None, description="是否免费")

model = ChatDeepSeek(
    model="deepseek-flash",
    extra\_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max\_retries=2,
)

agent = create\_agent(
    model=model,
    response\_format=ToolStrategy(CourseInfo, handle\_errors=True),
    system\_prompt=(
        "从用户描述中提取课程信息，只使用用户明确提供的数据。"
        "缺失字段返回 null；不要猜测课程价格、会员权限或学习时长。"
        "不要因为课程名含有'基础'就推断难度。"
        "如果文本没有明确课程名称，course\_name 也返回 null。"
    ),
)

texts = [
    "Python3 基础教程是入门课程，预计20小时，免费。",
    "Java 进阶课程需要大约35小时。",
    "我最近正在学习 HTML 基础教程。",
    "我想学编程，还没选好课程。",
]

for text in texts:
    print("\n原文：", text)
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"recursion\_limit": 12},
        )
        course = result.get("structured\_response")
        if not isinstance(course, CourseInfo):
            print("提取失败：未返回有效的课程结构化对象")
            continue
        print(course.model\_dump\_json(indent=2))
    except GraphRecursionError:
        print("提取失败：达到执行步数上限，请检查输入或简化 Schema")
    except StructuredOutputError as error:
        print("结构化输出失败：", type(error).\_\_name\_\_)
    except APIError as error:
        print("模型请求失败：", type(error).\_\_name\_\_)

配置好 `.env` 后，在终端执行：

```
python structured_output_demo.py
```

第一段文本如果提取成功，结构化结果应接近：

```
{

  "course_name": "Python3 基础教程",

  "difficulty": "入门",

  "estimated_hours": 20,

  "is_free": true

}
```

第三段文本没有说明难度、时长和收费信息，因此期望返回：

```
{

  "course_name": "HTML 基础教程",

  "difficulty": null,

  "estimated_hours": null,

  "is_free": null

}
```

第四段文本没有明确任何课程资料，期望所有业务字段均为 `null`。`extra="forbid"` 用于拒绝 Schema 之外的额外字段，并不能阻止模型为合法字段编造内容，重要数据仍需核验来源。

---

## 十三、常见问题

### 1、structured\_response 是字符串还是对象？

本章的 Pydantic 示例返回对应模型实例；TypedDict 和 JSON Schema 示例返回字典，不是 JSON 字符串。需要字符串时，再调用序列化方法。

### 2、为什么 structured\_response 不存在？

先确认配置了 `response_format`，并且读取的是 `agent.invoke()` 返回的最终状态，而不是流式执行过程中的单个事件；再检查流程是否正常结束。`with_structured_output()` 的结果也不放在这个键中。

### 3、Pydantic 和 TypedDict 应该选哪个？

如果需要复杂字段校验、嵌套模型或自定义验证器，推荐 Pydantic；如果只需要简单字典结构，可以考虑 TypedDict。

### 4、ProviderStrategy 和 ToolStrategy 有什么区别？

ProviderStrategy 使用模型服务商原生结构化输出机制；ToolStrategy 通过模型工具调用完成结构化数据生成。

### 5、所有模型都支持结构化输出吗？

不是。不同模型支持的功能存在差异，尤其是原生 JSON Schema、Tool Calling 和复杂嵌套结构。

### 6、response\_format 可以和 tools 同时使用吗？

可以，但需要模型和所使用的结构化输出方式支持相应组合。

### 7、如何防止模型编造缺失字段？

可以将字段定义为可选，在系统提示词中要求缺失信息返回 null，并在应用层进行验证。

### 8、结构化输出能保证数据正确吗？

不能。Schema 主要约束数据格式及部分数值关系，不负责验证所有业务事实。

### 9、为什么模型一直重试结构化输出？

可能因为模型反复生成不符合 Schema 的内容。可简化约束、改进描述，用 `handle_errors=False` 暴露错误，或通过 `recursion_limit` 限制执行步数。约束互相矛盾时，应先修正 Schema。

### 10、什么时候不需要 create\_agent()？

如果任务只是文本分类、信息提取或简单结构化转换，可以直接使用模型的 `with_structured_output()`。

---

## 总结

LangChain 结构化输出可以让模型按照指定 Schema 返回数据，方便程序直接使用。

本章节介绍了：

- `response_format`：配置 Agent 结构化输出。
- `Pydantic`：定义数据结构和字段校验。
- `TypedDict`：定义字典格式的输出。
- `ProviderStrategy`：使用模型原生结构化输出。
- `ToolStrategy`：通过工具调用实现结构化输出。
- `structured_response`：获取 Agent 结构化结果。
- `with_structured_output()`：直接让聊天模型输出结构化数据。
- `include_raw=True`：检查原始响应与解析异常。
- `handle_errors`：处理 ToolStrategy 的结构化输出错误。

复用已经初始化的 `model`、`CourseInfo` 和 `ToolStrategy`，核心写法为：

```
agent = create_agent(

    model=model,

    response_format=ToolStrategy(CourseInfo)

)



result = agent.invoke({

    "messages": [{"role": "user", "content": "Python 入门课，20小时，免费。"}]

})



data = result["structured_response"]
```

如果不需要 Agent 工具调用，也可以直接使用：

```
structured_model = model.with_structured_output(

    CourseInfo, method="function_calling"

)

data = structured_model.invoke("Python 入门课，20小时，免费。")
```

> **核心原则：**使用 Schema 约束数据格式，使用 Pydantic 检查字段和业务规则，并通过可信数据源验证重要事实。结构化输出解决的是数据组织问题，而不是模型内容准确性的全部问题。

---

## 参考文档

- [LangChain：Agent 结构化输出](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangChain：模型结构化输出](https://docs.langchain.com/oss/python/langchain/models#structured-output)
- [ChatDeepSeek.with\_structured\_output API](https://reference.langchain.com/python/langchain-deepseek/chat_models/ChatDeepSeek/with_structured_output)
- [DeepSeek：模型名称与能力](https://api-docs.deepseek.com/quick_start/pricing/)
- [DeepSeek：思考模式与工具调用](https://api-docs.deepseek.com/guides/thinking_mode/)
