# LangChain 输出策略——ToolStrategy、ProviderStrategy 与 AutoStrategy

> 来源: https://www.runoob.com/langchain/langchain-structured-output-strategies.html

---

# LangChain 输出策略——ToolStrategy、ProviderStrategy 与 AutoStrategy

上一节我们学习了 LangChain 的 [结构化输出（Structured Output）](https://www.runoob.com/langchain/langchain-structured-output.html)。

结构化输出可以让 Agent 按照 Pydantic、TypedDict 或 JSON Schema 定义的数据结构返回结果，方便程序直接处理。

在 LangChain 1.x 中，结构化输出主要有三种策略：ToolStrategy、ProviderStrategy 和 AutoStrategy。

本章节介绍它们的工作原理、参数配置、兼容性、错误处理和选择方法。

---

## 一、三种结构化输出策略

三种策略的主要区别在于：结构化数据通过哪种机制生成，以及由谁负责保证输出符合 Schema。

| 策略 | 工作原理 | 使用条件 | 主要特点 |
| --- | --- | --- | --- |
| `ToolStrategy` | 使用模型 Tool Calling 生成结构化数据 | 模型支持工具调用 | 支持结构化输出错误反馈与重试 |
| `ProviderStrategy` | 使用模型服务商原生结构化输出机制 | 模型和服务商支持原生结构化输出 | 可利用提供商的 Schema 约束能力 |
| `AutoStrategy` | 根据模型能力信息选择策略 | 模型能力信息可正确识别 | 减少手动选择策略的工作 |

基本写法：

```
# 使用 ToolStrategy

response_format=ToolStrategy(CourseInfo)



# 使用 ProviderStrategy

response_format=ProviderStrategy(CourseInfo)



# 自动选择

response_format=CourseInfo
```

通常推荐先直接传入 Schema。如果需要明确控制结构化输出机制，再手动指定 ToolStrategy 或 ProviderStrategy。

**注意：**三种策略并不是固定的性能排行榜。实际请求耗时、Token 消耗和稳定性取决于模型、Schema、API 及是否发生重试。

---

## 二、安装与环境准备

安装 LangChain 及相关依赖：

```
pip install -U langchain langchain-deepseek python-dotenv pydantic
```

在项目目录创建 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

加载环境变量：

```
from dotenv import load_dotenv

load_dotenv()
```

本教程中 ToolStrategy 和自动策略主要使用 DeepSeek Flash 演示。

需要注意，DeepSeek 模型是否支持原生 Schema 约束，应以实际服务能力为准，不能直接假设它支持 ProviderStrategy。

---

## 三、ToolStrategy：通过工具调用输出数据

`ToolStrategy` 使用模型的 Tool Calling 能力实现结构化输出。

LangChain 会将 Schema 转换成用于结构化响应的工具定义，让模型生成符合工具参数要求的数据。

这里的工具主要负责表达输出结构，不代表真正执行了一个外部 API 或 Python 业务函数。

执行过程可以理解为：

```
用户输入

    ↓

模型接收结构化输出 Schema

    ↓

模型生成 Tool Call

    ↓

LangChain 解析并验证参数

    ↓

structured_response
```

### 实例：提取天气描述

下面从用户提供的天气描述中提取城市、温度、天气状况和湿度。

所有信息来自输入文本，不会调用真实天气服务。

## 实例

from dotenv import load\_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

load\_dotenv()

class WeatherReport(BaseModel):
    """从文本中提取天气信息。"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="摄氏温度")
    condition: str = Field(description="天气状况")
    humidity: int = Field(ge=0, le=100, description="湿度百分比")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    response\_format=ToolStrategy(WeatherReport),
    system\_prompt="只提取用户描述中明确提供的天气数据，不要查询或编造实时天气。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "杭州晴天，温度25度，湿度60%。"}]
})

report = result["structured\_response"]
print(report.model\_dump())

成功执行时，结果可能为：

```
{

    'city': '杭州',

    'temperature': 25.0,

    'condition': '晴天',

    'humidity': 60

}
```

由于传入的是 Pydantic 模型，解析成功后可以通过属性访问字段：

```
print(report.city)

print(report.temperature)

print(report.humidity)
```

其中，`humidity` 使用 `ge=0` 和 `le=100` 约束湿度范围。

---

## 四、查看 ToolStrategy 的消息过程

使用 ToolStrategy 时，模型通常会生成一条结构化输出 Tool Call。

可以通过消息历史检查实际执行情况。

## 实例

for msg in result["messages"]:
    if msg.type == "ai" and msg.tool\_calls:
        print("结构化调用：", [(c["name"], c["args"]) for c in msg.tool\_calls])
    elif msg.type == "tool":
        print("工具消息：", msg.content)
    else:
        print("消息类型：", msg.type)

print("消息总数：", len(result["messages"]))

如果模型第一次就生成了正确的结构化数据，可能看到：

```
消息类型： human

结构化调用： [('WeatherReport', {...})]

工具消息： Returning structured response: ...

消息总数： 3
```

**注意：**不能将消息数量固定写成 4 条。

对于一次成功的 ToolStrategy 结构化响应，可能只包含用户消息、带 Tool Call 的 AIMessage，以及对应的 ToolMessage。

如果模型生成了无效数据，LangChain 可能反馈错误并再次请求模型，消息数量就会增加。

同样，ToolStrategy 也不代表每次一定额外调用一次模型 API。工具调用请求本身可以由一次模型响应生成。

---

## 五、ToolStrategy 的 handle\_errors 参数

`ToolStrategy` 支持处理结构化输出错误，例如字段缺失、数值越界、Schema 验证失败，以及模型错误地产生多个结构化输出。

handle\_errors 默认值为 True。

也就是说，默认情况下，LangChain 会尝试将受支持的结构化输出错误反馈给模型，让它重新生成符合要求的数据。

| 配置 | 作用 |
| --- | --- |
| `True` | 使用默认错误提示，允许模型尝试修正 |
| `False` | 不进行该策略的自动错误恢复，异常向外抛出 |
| 字符串 | 使用指定的固定错误提示 |
| 异常类型 | 仅处理指定异常类型 |
| 异常类型元组 | 处理指定的多个异常类型 |
| Callable | 调用自定义函数生成错误提示 |

### 实例：使用默认错误处理

## 实例

from pydantic import BaseModel, Field
from langchain.agents.structured\_output import ToolStrategy

class ProductRating(BaseModel):
    """商品评分。"""
    rating: int = Field(ge=1, le=5)
    comment: str

strategy = ToolStrategy(
    schema=ProductRating,
    handle\_errors=True
)

如果模型输出 `rating=10`，校验会失败，LangChain 可以向模型反馈错误，让它尝试修改。

但模型不一定每次都能修正成功。

### 实例：自定义错误提示

## 实例

strategy = ToolStrategy(
    schema=ProductRating,
    handle\_errors="评分必须在1到5之间，并且必须提供评论内容。"
)

这里的字符串会作为固定错误提示。

**注意：**不要写成：

```
handle_errors="格式有误：{error}"
```

然后假设 LangChain 会自动把 `{error}` 替换成具体异常。

如果需要根据错误内容生成提示，应传入 Callable。

### 实例：自定义错误处理函数

## 实例

from langchain.agents.structured\_output import (
    ToolStrategy, StructuredOutputValidationError, MultipleStructuredOutputsError
)

def handle\_format\_error(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "字段格式或取值不符合要求，请修正后重新输出。"
    if isinstance(error, MultipleStructuredOutputsError):
        return "只能返回一种结构化结果，请选择最符合任务的一种。"
    return "结构化输出失败，请检查格式。"

strategy = ToolStrategy(ProductRating, handle\_errors=handle\_format\_error)

实际使用时，可以把 `strategy` 传给 `create_agent(response_format=strategy)`。

需要注意，结构化输出重试不等于普通工具执行失败后的重试，也不等于网络请求失败后的 API 自动重试。

---

## 六、tool\_message\_content：自定义工具消息

ToolStrategy 还有一个有用的参数：

```
tool_message_content
```

它可以设置结构化输出成功后，添加到消息历史中的 ToolMessage 内容。

### 实例

## 实例

agent = create\_agent(
    model="deepseek:deepseek-flash",
    response\_format=ToolStrategy(
        schema=WeatherReport,
        tool\_message\_content="天气描述已成功提取。"
    )
)

这样结构化输出成功后，对应的工具消息可以显示为：

```
天气描述已成功提取。
```

该参数只改变工具消息的文本，不会替代或修改最终的 `structured_response` 数据。

---

## 七、ProviderStrategy：使用原生结构化输出

`ProviderStrategy` 使用模型服务商提供的原生结构化输出接口。

例如，部分 OpenAI、Anthropic、Google 和 xAI 模型支持通过服务商能力约束输出结构。

这种方式通常能够利用提供商自己的 Schema 约束机制。

但不能简单认为某个模型系列的所有版本都支持，也不能保证 ProviderStrategy 在所有情况下都比 ToolStrategy 更快。

### 实例：使用支持原生结构化输出的模型

下面采用 OpenAI 兼容示例。运行前需要安装 `langchain-openai` 并配置相应 API Key。

## 实例

from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ProviderStrategy

class CourseInfo(BaseModel):
    """课程信息。"""
    name: str
    level: Literal["入门", "进阶", "高级"]
    price: str = Field(description="用户明确描述的价格或收费方式")

agent = create\_agent(
    model="openai:gpt-4o",
    tools=[],
    response\_format=ProviderStrategy(CourseInfo),
    system\_prompt="只提取用户明确提供的课程资料。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Python3 基础教程是入门级的免费课程。"}]
})

course = result["structured\_response"]
print(course.model\_dump())

成功执行时，结果可能为：

```
{

    'name': 'Python3 基础教程',

    'level': '入门',

    'price': '免费'

}
```

这里使用的是明确支持相应原生结构化输出功能时的调用方式。

**注意：**如果你使用其他提供商或模型，应先核对该模型的原生结构化输出支持情况。

### strict 参数

ProviderStrategy 支持可选的 `strict` 参数：

```
response_format=ProviderStrategy(

    CourseInfo,

    strict=True

)
```

但 `strict=True` 需要对应模型提供商支持严格 Schema 约束，并不是所有模型都支持。

如果服务端不接受相应设置，可能直接返回 API 错误。

---

## 八、ProviderStrategy 是否一定比 ToolStrategy 快？

不一定。

ProviderStrategy 不需要将结构化输出表示为 Tool Calling 请求，但这不代表它一定能减少网络请求或获得更低延迟。

ToolStrategy 在一次成功调用中，也可以通过一次模型响应生成完整的结构化数据。

实际性能取决于：

- 模型生成速度与响应延迟。
- Schema 的复杂度。
- 是否需要多轮工具调用。
- 结构化输出失败后的重试次数。
- 模型提供商和网络环境。

因此，应通过真实请求测量，而不是仅根据消息数量判断。

### 性能比较应该看哪些指标？

| 指标 | 说明 |
| --- | --- |
| 请求耗时 | 一次 Agent 调用从开始到结束的时间 |
| 模型调用次数 | 实际请求模型 API 的次数 |
| Token 消耗 | 输入和输出 Token 数量 |
| Schema 成功率 | 输出通过数据校验的比例 |
| 重试次数 | 结构化输出错误后重新请求的次数 |

如果业务非常关注成本与时延，建议使用相同的模型、Schema 和测试数据，对不同策略做多次对比。

---

## 九、AutoStrategy：自动选择结构化输出策略

LangChain 提供 `AutoStrategy`，用于根据模型能力自动选择结构化输出方式。

但在常规开发中，不需要显式创建 AutoStrategy。

只需要将 Pydantic 模型直接传给 `response_format` 即可。

### 实例：课程评价分析

## 实例

from pydantic import BaseModel, Field
from langchain.agents import create\_agent

class Analysis(BaseModel):
    """课程评价分析。"""
    summary: str = Field(description="简短总结")
    score: int = Field(ge=1, le=10, description="评分1到10")
    pros: list[str] = Field(description="优点")
    cons: list[str] = Field(description="不足")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    response\_format=Analysis,
    system\_prompt="只根据用户提供的评价进行分析，不要添加未经提供的事实。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "这个Python教程内容系统，例子很多，但高级专题相对较少。"}]
})

analysis = result["structured\_response"]
print("总结：", analysis.summary)
print("评分：", analysis.score)
print("优点：", analysis.pros)
print("不足：", analysis.cons)

模型可能返回：

```
总结：内容系统、示例较多，但高级专题相对有限

评分：8

优点：['内容系统', '示例丰富']

不足：['高级专题相对较少']
```

需要注意，评分由模型根据用户描述主观判断，并不是经过客观测试得到的质量分数。

### 自动策略如何选择？

通常遵循以下思路：

```
response_format=Analysis

          |

          v

读取模型能力信息

          |

          +-- 支持原生结构化输出

          |       |

          |       v

          |  ProviderStrategy

          |

          +-- 不支持或未识别

                  |

                  v

             ToolStrategy
```

LangChain 可以根据模型的 Profile 信息判断是否支持原生结构化输出。

但 Profile 信息可能缺失或与实际 API 能力不完全一致，因此自动选择不代表一定成功，也不保证性能最优。

---

## 十、通过 model.profile 检查模型能力

在 LangChain 1.x 中，模型对象可以提供 Profile 信息，用于描述当前模型支持的功能。

例如：

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model("deepseek:deepseek-flash")
profile = model.profile or {}

print("原生结构化输出：", profile.get("structured\_output"))
print("工具调用：", profile.get("tool\_calling"))

如果 Profile 返回：

```
structured_output: True
```

通常表示 LangChain 的模型元数据认为该模型具备原生结构化输出能力。

如果返回 `False` 或 `None`，不能直接断言所有结构化输出方法都不可用，因为 ToolStrategy 可能仍然可用。

**注意：**Profile 是能力参考信息，并不代替真实请求测试。

---

## 十一、ToolStrategy 与 ProviderStrategy 的选择

可以根据实际模型能力和业务需求选择结构化输出策略。

| 场景 | 建议方案 | 原因 |
| --- | --- | --- |
| 一般结构化输出 | 直接传入 Pydantic Schema | 让 LangChain 自动选择 |
| 明确需要工具调用模式 | ToolStrategy | 显式使用 Tool Calling |
| 需要处理 Schema 错误并重试 | ToolStrategy | 提供 handle\_errors |
| 模型明确支持原生结构化输出 | ProviderStrategy | 使用服务商原生 Schema 机制 |
| 强调复杂数据验证 | Pydantic Schema | 支持字段范围和自定义验证器 |
| 追求低延迟与成本 | 实际基准测试 | 不同模型和任务的表现不同 |

通常可以先从自动策略开始；如果发现输出校验频繁失败，再考虑显式使用 ToolStrategy 进行错误恢复。

如果模型提供商明确支持严格的原生结构化输出，可以进一步测试 ProviderStrategy。

---

## 十二、JSON Schema 与策略的注意事项

除了 Pydantic，LangChain 还支持使用 JSON Schema 字典定义输出结构。

但 JSON Schema 字典与 Pydantic 类的处理方式有所不同。

使用 JSON Schema 字典时，应明确包装在策略对象中：

## 实例

from langchain.agents.structured\_output import ToolStrategy

course\_schema = {
    "title": "CourseInfo",
    "description": "课程信息",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "hours": {"type": "integer", "minimum": 1}
    },
    "required": ["name", "hours"]
}

strategy = ToolStrategy(course\_schema)

**注意：**当 ToolStrategy 使用原始 JSON Schema 字典时，LangChain 不会像处理 Pydantic 模型那样执行完整的客户端 Schema 校验。

因此，不能仅凭 `minimum` 等字段就认定程序已经验证了实际结果。

如果需要严格的 Python 端字段校验、交叉字段验证和错误重试，建议优先选择 Pydantic。

---

## 十三、完整实例：动态选择输出策略

下面实现一个可以根据配置选择自动策略或 ToolStrategy 的课程评价分析工具。

为了保证示例模型兼容性，这里不强制将 DeepSeek 配置为 ProviderStrategy。

文件名：`output_strategy_demo.py`

## 完整实例

from typing import Literal
from dotenv import load\_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

class ReviewResult(BaseModel):
    """教程评价结果。"""
    sentiment: Literal["积极", "消极", "中性"]
    score: int = Field(ge=1, le=10)
    advantages: list[str]
    disadvantages: list[str]

model = init\_chat\_model("deepseek:deepseek-flash")
profile = model.profile or {}
print("模型原生结构化输出支持：", profile.get("structured\_output"))

def build\_agent(strategy: str = "auto"):
    response\_format = ReviewResult if strategy == "auto" else ToolStrategy(
        ReviewResult, handle\_errors=True
    )
    return create\_agent(
        model=model,
        response\_format=response\_format,
        system\_prompt="只根据用户提供的教程评价提取信息。评分是主观判断，不是客观测评。"
    )

agent = build\_agent("tool")  # 改为 "auto" 可以测试自动策略

try:
    result = agent.invoke({
        "messages": [{"role": "user", "content": "这个Python教程内容清晰、实例丰富，但部分高级主题覆盖较少。"}]
    }, config={"recursion\_limit": 15})

    review = result["structured\_response"]
    print("情感：", review.sentiment)
    print("评分：", review.score)
    print("优点：", review.advantages)
    print("缺点：", review.disadvantages)
    print("模型消息数：", sum(m.type == "ai" for m in result["messages"]))
except Exception as error:
    print("结构化输出失败：", type(error).\_\_name\_\_)

这里可以通过修改：

```
build_agent("auto")

build_agent("tool")
```

比较自动策略与显式 ToolStrategy。

但这只是一个功能演示。如果需要严谨比较性能，应使用相同测试集、记录实际 API 耗时、Token 消耗以及结构化输出成功率。

---

## 十四、常见问题

### 1、AutoStrategy 需要手动创建吗？

一般不需要。直接将 Pydantic、TypedDict 或其他受支持的 Schema 类型传入 `response_format` 即可。

### 2、ToolStrategy 是否所有模型都支持？

不是。它依赖模型及对应集成支持 Tool Calling。

### 3、ToolStrategy 一定多调用一次模型吗？

不一定。一次模型响应就可能包含结构化输出的 Tool Call。

如果需要错误重试，模型调用次数才可能增加。

### 4、handle\_errors 的默认值是什么？

当前官方 ToolStrategy 默认是 `True`。

### 5、ProviderStrategy 一定比 ToolStrategy 快吗？

不一定。需要结合模型、请求和实际的网络环境进行测试。

### 6、ProviderStrategy 不支持时一定自动降级吗？

不能对所有模型和集成作这样的保证。

如果希望 LangChain 按模型能力自动选择，应直接传入 Schema 类型；明确使用 ProviderStrategy 前，应先核对模型支持情况。

### 7、为什么 ProviderStrategy 调用失败？

可能是当前模型不支持原生结构化输出，或者提供商不支持当前 Schema、strict 参数或相关 API 配置。

### 8、如何查看模型是否支持原生结构化输出？

可以查看 `model.profile` 中的 `structured_output` 信息，并结合实际 API 测试。

### 9、ToolStrategy 可以自定义工具消息吗？

可以，通过 `tool_message_content` 配置结构化输出成功后的工具消息文本。

### 10、JSON Schema 是否等于 Pydantic 校验？

不是。对于 ToolStrategy 的原始 JSON Schema 字典，不能假设 LangChain 会执行与 Pydantic 相同的客户端验证。

### 11、结构化输出可以与普通工具同时使用吗？

可以，但模型必须支持同时使用普通工具及所选择的结构化输出机制。

### 12、为什么 structured\_response 不存在？

可能是没有配置 `response_format`，或者模型未成功完成结构化输出流程。

---

## 总结

LangChain 1.x 主要通过三种策略管理 Agent 的结构化输出。

- `ToolStrategy`：利用 Tool Calling 生成结构化数据，支持输出校验错误反馈与重试。
- `ProviderStrategy`：使用模型服务商原生结构化输出能力。
- `AutoStrategy`：根据模型能力选择合适的策略，通常只需直接传入 Schema。

最常用的写法是：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    response_format=CourseInfo

)



result = agent.invoke({

    "messages": [{"role": "user", "content": "提取课程信息"}]

})



data = result["structured_response"]
```

如果需要明确使用工具调用模式：

```
response_format=ToolStrategy(

    CourseInfo,

    handle_errors=True

)
```

如果当前模型明确支持原生结构化输出：

```
response_format=ProviderStrategy(CourseInfo)
```

**核心原则：**普通场景优先使用自动策略；需要明确控制错误处理时使用 ToolStrategy；确认模型具备原生结构化输出能力后，再考虑 ProviderStrategy。

不要只根据消息数量判断性能，也不要将符合 Schema 的输出直接等同于真实、可靠的业务数据。
