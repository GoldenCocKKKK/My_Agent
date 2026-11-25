# LangChain Chat Model API 参考

> 来源: https://www.runoob.com/langchain/langchain-chat-model-api.html

---

# LangChain Chat Model API 参考

LangChain 提供统一的 Chat Model 接口，让开发者可以使用相似的代码调用 OpenAI、Anthropic、DeepSeek、Google Gemini、Ollama 等不同模型。

其中，init\_chat\_model() 用于初始化聊天模型，BaseChatModel 则定义了模型调用、流式输出、批量执行、工具绑定和结构化输出等常用接口。

本章节基于 LangChain 1.x，系统整理模型初始化参数、BaseChatModel 常用方法、模型提供商、动态切换、AIMessage 数据结构及常见问题。

---

## 一、init\_chat\_model() 是什么？

`init_chat_model()` 是 LangChain 提供的统一聊天模型初始化函数。

例如，使用 DeepSeek：

```
from langchain.chat_models import init_chat_model



model = init_chat_model("deepseek:deepseek-chat")

response = model.invoke("你好")

print(response.content)
```

如果改用其他服务商，通常只需要修改模型标识，并安装对应集成包。

模型名称建议使用：

```
provider:model_name
```

例如：

```
deepseek:deepseek-chat

openai:gpt-4o

anthropic:claude-sonnet-4-5-20250929

google_genai:gemini-2.5-flash

ollama:llama3.2
```

以上为模型标识格式示例，具体模型是否仍可调用，应以对应服务商的模型列表和账号权限为准。

如果不提供 provider，LangChain 会尝试从模型名称推断服务商，但这种推断并非适用于所有模型。

**建议：**在正式项目中明确指定 provider，避免模型名称相似时被错误识别。

---

## 二、init\_chat\_model() 函数签名

当前 API 的核心函数签名如下：

```
init_chat_model(

    model: str | None = None,

    *,

    model_provider: str | None = None,

    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,

    config_prefix: str | None = None,

    **kwargs

)
```

它主要包含四个显式参数，以及用于传递具体模型配置的 `**kwargs`。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model` | str | None | None | 模型名称，推荐 provider:model\_name 格式 |
| `model_provider` | str | None | None | 单独指定模型服务商 |
| `configurable_fields` | "any" / list / tuple / None | None | 指定哪些模型参数可在运行时修改 |
| `config_prefix` | str | None | None | 为动态配置键添加前缀 |
| `**kwargs` | Any | 因模型而异 | 传递给具体模型构造函数的其他参数 |

函数通常返回一个 BaseChatModel 实例；如果启用动态配置，则返回支持运行时选择模型参数的包装对象。

---

## 三、常用模型配置参数

除了核心参数，init\_chat\_model() 还可以通过 `**kwargs` 传递模型配置。

需要注意：这些并非所有服务商都统一支持的固定参数。

| 参数 | 常见类型 | 说明 |
| --- | --- | --- |
| `temperature` | float | 控制采样随机性，合法范围因模型而异 |
| `max_tokens` | int | None | 限制模型生成的最大输出 Token 数 |
| `timeout` | float | None | 模型请求超时设置，通常以秒计 |
| `max_retries` | int | 模型客户端的失败重试配置 |
| `base_url` | str | None | 自定义模型 API 服务地址 |
| `api_key` | 因集成而异 | 服务商 API 凭据 |
| `rate_limiter` | BaseRateLimiter | 限制模型请求速率 |
| `top_p` | float | None | 核采样参数，是否支持取决于模型 |
| `stop` | list[str] | None | 停止序列，通常作为调用时参数提供 |

例如：

## 实例：配置模型

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    temperature=0.3,
    max\_tokens=1024,
    timeout=30,
    max\_retries=2
)

response = model.invoke("用100字介绍 Python。")
print(response.content)

这个示例设置了输出长度、超时和重试等参数。

**注意：**`temperature=0` 不保证每次生成结果都完全相同。部分推理模型还可能限制或忽略 temperature 等采样参数。

---

## 四、固定模型与动态模型

init\_chat\_model() 支持两种常见模式：

- **固定模型：**创建时确定模型，后续正常调用。
- **动态模型：**运行时通过 config 选择模型或修改允许的参数。

### 方式一：固定模型

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    temperature=0
)

response = model.invoke("Python 是什么？")
print(response.content)

没有配置 configurable\_fields 时，显式指定的模型默认是固定模型。

### 方式二：动态调整参数

如果希望不同请求使用不同的 temperature，可以明确配置：

## 实例

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    temperature=0,
    configurable\_fields=("temperature",)
)

response = model.invoke(
    "写一段 Python 学习建议。",
    config={
        "configurable": {
            "temperature": 0.7
        }
    }
)

print(response.content)

这里允许每次调用通过 config 修改 temperature，其他模型配置仍使用默认值。

### 方式三：动态切换模型

可以允许模型名称在运行时变化：

## 实例

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    configurable\_fields=("model",)
)

response = model.invoke(
    "你好",
    config={
        "configurable": {
            "model": "deepseek:deepseek-chat"
        }
    }
)

print(response.content)

如果要在不同提供商之间切换，需要安装对应集成包，并配置所需的 API Key。

例如切换到 OpenAI 时，还需要 `langchain-openai` 和有效的 OpenAI API Key。

---

## 五、model=None 的特殊行为

如果创建模型时没有指定 model：

```
model = init_chat_model()
```

LangChain 会创建一个可以在运行时指定模型的配置对象。

这时默认允许配置：

```
model

model_provider
```

### 实例：运行时确定模型

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model()

response = model.invoke(
    "你好，请介绍自己。",
    config={
        "configurable": {
            "model": "deepseek:deepseek-chat"
        }
    }
)

print(response.content)

**注意：**如果创建时没有默认模型，调用时又没有在 config 中指定模型，通常会出现配置错误。

---

## 六、config\_prefix：动态配置前缀

当同一个应用包含多个可配置模型时，可以通过 `config_prefix` 区分它们。

例如：

```
chat_model

summary_model
```

这适用于一个模型负责普通对话，另一个模型负责摘要的场景。

### 实例：使用配置前缀

## 实例

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    temperature=0,
    configurable\_fields=("temperature",),
    config\_prefix="chat"
)

response = model.invoke(
    "介绍 Python。",
    config={
        "configurable": {
            "chat\_temperature": 0.5
        }
    }
)

print(response.content)

这里的动态配置键变成：

```
chat_temperature
```

如果没有 config\_prefix，则直接使用：

```
temperature
```

**安全提醒：**不要在接受用户输入的服务中随意设置 `configurable_fields="any"`。因为它可能允许在运行时修改 api\_key、base\_url 等敏感配置，带来请求重定向或凭据泄露风险。

---

## 七、BaseChatModel 是什么？

`BaseChatModel` 是 LangChain 聊天模型的核心抽象类。

不同服务商的模型虽然 API 地址、参数和返回格式存在差异，但通过 BaseChatModel 提供了统一的调用方式。

例如：

```
model.invoke()

model.ainvoke()

model.stream()

model.astream()

model.batch()

model.bind_tools()

model.with_structured_output()
```

因此，可以在不同模型之间复用大量应用代码。

---

## 八、BaseChatModel 常用方法速查

| 方法 | 作用 | 典型返回值 |
| --- | --- | --- |
| `invoke(input, config=None, **kwargs)` | 同步调用模型 | AIMessage |
| `ainvoke(input, config=None, **kwargs)` | 异步调用 | AIMessage |
| `stream(input, config=None, **kwargs)` | 同步流式调用 | Iterator[AIMessageChunk] |
| `astream(input, config=None, **kwargs)` | 异步流式调用 | AsyncIterator[AIMessageChunk] |
| `batch(inputs, config=None, **kwargs)` | 批量处理多个输入 | list[AIMessage] |
| `abatch(inputs, config=None, **kwargs)` | 异步批量处理 | list[AIMessage] |
| `batch_as_completed()` | 按完成顺序获取批量结果 | 带索引的结果迭代器 |
| `bind_tools(tools, **kwargs)` | 绑定工具 Schema | 支持工具调用的模型 Runnable |
| `with_structured_output(schema, **kwargs)` | 结构化输出 | Runnable |
| `bind(**kwargs)` | 预先绑定模型调用参数 | Runnable |
| `with_retry(**kwargs)` | 为调用包装重试逻辑 | Runnable |
| `with_fallbacks(...)` | 失败时尝试备用 Runnable | Runnable |

这里的返回值是常见类型。使用 bind\_tools、with\_structured\_output 等包装方法后，具体返回结构会受到模型和配置影响。

---

## 九、invoke()：同步调用模型

`invoke()` 是最常见的模型调用方法。

它支持普通字符串，也支持消息列表。

### 实例：字符串输入

## 实例

model = init\_chat\_model("deepseek:deepseek-chat")

response = model.invoke("什么是 Python？")

print(response.content)
print(type(response).\_\_name\_\_)

典型返回对象为 `AIMessage`。

### 实例：消息列表

## 实例

from langchain.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是 Python 编程老师，回答要简洁。"),
    HumanMessage(content="解释 Python 列表。")
]

response = model.invoke(messages)
print(response.content)

与单个字符串相比，消息列表可以明确区分系统指令和用户输入。

---

## 十、ainvoke()：异步调用

如果应用基于 asyncio、FastAPI 等异步框架，可以使用 `ainvoke()`。

### 实例

## 实例

import asyncio
from langchain.chat\_models import init\_chat\_model

async def main():
    model = init\_chat\_model("deepseek:deepseek-chat")
    response = await model.ainvoke("介绍一下 Python。")
    print(response.content)

asyncio.run(main())

`ainvoke()` 在异步应用中通常更容易与其他网络 I/O 任务协同。

不过，是否使用原生异步 HTTP 请求，以及具体性能表现，取决于模型集成实现。

---

## 十一、stream()：流式输出

模型生成较长内容时，可以使用 `stream()` 逐步获取结果。

返回的通常是 `AIMessageChunk`。

### 实例：逐步打印回答

## 实例

model = init\_chat\_model("deepseek:deepseek-chat")

for chunk in model.stream("介绍 Python 的主要特点。"):
    if isinstance(chunk.content, str):
        print(chunk.content, end="", flush=True)

print()

注意，流式输出不一定每次返回一个汉字，也不保证每次只返回一个 Token。

不同模型可能将多个字符组合成一个 Chunk。

如果提供商不支持真正的流式接口，部分模型实现可能退化为一次性返回完整内容。

---

## 十二、astream()：异步流式输出

## 实例

import asyncio
from langchain.chat\_models import init\_chat\_model

async def main():
    model = init\_chat\_model("deepseek:deepseek-chat")

    async for chunk in model.astream("介绍 Python 函数。"):
        if isinstance(chunk.content, str):
            print(chunk.content, end="", flush=True)

    print()

asyncio.run(main())

异步流式输出适合 Web 服务、聊天界面和支持异步处理的程序。

对于复杂的多模型、工具执行事件，还可以进一步研究 LangChain 的 Stream Events 接口。

---

## 十三、batch()：批量调用模型

如果需要处理多个相互独立的问题，可以使用 `batch()`。

### 实例：批量处理问题

## 实例

model = init\_chat\_model("deepseek:deepseek-chat")

questions = [
    "Python 是什么？",
    "Java 是什么？",
    "Go 语言有什么特点？"
]

results = model.batch(
    questions,
    config={"max\_concurrency": 2}
)

for question, response in zip(questions, results):
    print("问题：", question)
    print("回答：", response.content)

`batch()` 会按照输入顺序返回结果。

`max_concurrency` 可以限制并发数量。

需要注意，批量调用不代表一定使用服务商专门的批处理 API。默认实现可能通过并行调用 invoke() 完成。

---

## 十四、AIMessage：模型返回对象

直接调用 Chat Model 后，通常得到 `AIMessage`，而不只是一个普通字符串。

它可以包含正文、工具调用、Token 用量和模型响应信息。

### 常用字段

| 字段 | 说明 |
| --- | --- |
| `content` | 模型消息内容，可能是字符串或内容块 |
| `text` | 提取文本内容的方法接口 |
| `tool_calls` | 解析后的工具调用列表 |
| `invalid_tool_calls` | 未能正确解析的工具调用 |
| `usage_metadata` | 输入、输出和总 Token 使用量 |
| `response_metadata` | 模型或提供商响应元数据 |
| `id` | 消息标识，可能为空 |

### 实例：查看响应详情

## 实例

response = model.invoke("你好")

print("内容：", response.content)
print("类型：", response.type)
print("消息 ID：", response.id)
print("工具调用：", response.tool\_calls)
print("Token 用量：", response.usage\_metadata)
print("响应元数据：", response.response\_metadata)

例如，usage\_metadata 可能包含：

```
{

    "input_tokens": 15,

    "output_tokens": 28,

    "total_tokens": 43

}
```

以上数字仅作为格式示例。

实际 Token 用量取决于服务商是否返回相关统计，某些字段可能为空。

---

## 十五、bind\_tools()：绑定工具

`bind_tools()` 可以将工具定义提供给聊天模型，使模型能够返回结构化的工具调用请求。

例如：

## 实例：工具绑定

from langchain.chat\_models import init\_chat\_model
from langchain.tools import tool

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。"""
    return f"{city}：模拟晴天，25°C"

model = init\_chat\_model("deepseek:deepseek-chat")
model\_with\_tools = model.bind\_tools([get\_weather])

response = model\_with\_tools.invoke("请调用工具查询杭州天气。")

print("模型回答：", response.content)
print("工具调用：", response.tool\_calls)

如果模型选择调用工具，tool\_calls 可能类似：

```
[

    {

        "name": "get_weather",

        "args": {"city": "杭州"},

        "id": "call_xxx",

        "type": "tool_call"

    }

]
```

**重要：**`bind_tools()` 不会自动执行 Python 工具函数。

它主要负责将工具定义传递给模型，并让模型生成工具调用请求。

实际工具执行、ToolMessage 返回以及后续模型调用，需要由应用代码或 Agent 完成。

如果希望自动完成整个工具调用循环，更推荐使用 `create_agent()`。

---

## 十六、with\_structured\_output()：结构化输出

普通模型返回的是自然语言文本。

如果希望直接得到符合指定结构的数据，可以使用 `with_structured_output()`。

常见 Schema 包括：

- Pydantic BaseModel。
- TypedDict。
- JSON Schema。

### 实例：使用 Pydantic

## 实例

from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model

class CourseInfo(BaseModel):
    name: str = Field(description="课程名称")
    level: str = Field(description="课程难度")
    language: str = Field(description="编程语言")

model = init\_chat\_model("deepseek:deepseek-chat")
structured\_model = model.with\_structured\_output(CourseInfo)

result = structured\_model.invoke(
    "提取课程信息：Python3 基础教程，入门难度，Python 语言。"
)

print(result)
print("课程名称：", result.name)

当使用 Pydantic Schema，且模型成功生成并校验结果时，通常返回对应的 Pydantic 对象。

如果使用 JSON Schema 或 TypedDict，返回结果可能是字典。

**注意：**不同模型对结构化输出的支持方式不同，有些通过工具调用实现，有些使用提供商原生的结构化输出接口。若对应集成不支持，可能在绑定或调用阶段报错。

### include\_raw 参数

部分模型集成支持：

```
with_structured_output(

    CourseInfo,

    include_raw=True

)
```

此时通常返回包含原始消息、解析结果和解析错误等字段的字典，方便调试。

在开发时，可以检查：

```
raw

parsed

parsing_error
```

这些字段的具体结构以对应模型集成的文档为准。

---

## 十七、bind()：预先绑定模型参数

`bind()` 是 Runnable 提供的通用方法。

可以为后续调用预先设置参数，例如停止序列。

### 实例

## 实例

model = init\_chat\_model("deepseek:deepseek-chat")

limited\_model = model.bind(stop=["END"])

response = limited\_model.invoke(
    "列出三个 Python 关键字，最后输出 END。"
)

print(response.content)

注意，停止序列是否按预期生效取决于模型服务商的支持情况。

`bind()` 与 `configurable_fields` 的区别在于：

- `bind()`：为 Runnable 预先绑定调用参数。
- `configurable_fields`：允许通过 RunnableConfig 在运行时修改指定模型字段。

---

## 十八、模型提供商速查表

LangChain 提供多个模型服务商集成。下面列出常见的 Provider 名称和对应安装包。

表中的模型名称属于示例，实际是否可用应以服务商模型目录及账号权限为准。

| Provider | 安装包 | 模型标识示例 |
| --- | --- | --- |
| `openai` | langchain-openai | openai:gpt-4o |
| `anthropic` | langchain-anthropic | anthropic:claude-sonnet-4-5-20250929 |
| `deepseek` | langchain-deepseek | deepseek:deepseek-chat |
| `google_genai` | langchain-google-genai | google\_genai:gemini-2.5-flash |
| `google_vertexai` | langchain-google-vertexai | google\_vertexai:gemini-2.5-flash |
| `ollama` | langchain-ollama | ollama:llama3.2 |
| `groq` | langchain-groq | groq:llama-3.3-70b-versatile |
| `xai` | langchain-xai | xai:grok-3 |
| `mistralai` | langchain-mistralai | mistralai:mistral-large-latest |
| `openrouter` | langchain-openrouter | openrouter:openai/gpt-4o |
| `perplexity` | langchain-perplexity | perplexity:sonar-pro |
| `bedrock_converse` | langchain-aws | 通过 model\_provider 与服务商模型 ID 指定 |
| `cohere` | langchain-cohere | 通过 model\_provider 与服务商模型 ID 指定 |
| `together` | langchain-together | 通过 model\_provider 与服务商模型 ID 指定 |

例如安装 DeepSeek 集成：

```
pip install -U langchain-deepseek
```

使用 Ollama：

```
pip install -U langchain-ollama
```

注意，Ollama 还需要本地 Ollama 服务正常运行，并且已下载对应模型。

对于 Google Gemini，建议显式使用 `google_genai` 或 `google_vertexai`，因为仅根据 gemini 模型名称自动推断 Provider 的默认行为可能随 LangChain 版本变化。

---

## 十九、Rate Limiter：模型调用限速

如果程序需要并发处理大量请求，可能触发模型服务商的速率限制。

LangChain 支持向模型传入 Rate Limiter。

### 实例：InMemoryRateLimiter

## 实例

from langchain.chat\_models import init\_chat\_model
from langchain\_core.rate\_limiters import InMemoryRateLimiter

limiter = InMemoryRateLimiter(
    requests\_per\_second=1,
    check\_every\_n\_seconds=0.1,
    max\_bucket\_size=2
)

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    rate\_limiter=limiter
)

response = model.invoke("你好")
print(response.content)

这个例子将请求发起速率限制在指定范围内。

**注意：**InMemoryRateLimiter 只适合单进程内的速率管理，不能自动协调多个服务器实例的全局限流。

如果使用分布式部署，通常需要通过 Redis、API 网关或其他集中式限流机制进行控制。

---

## 二十、Chat Model 与 Agent 的区别

初学者容易把 Chat Model 与 Agent 混为一谈。

实际上，两者负责的工作不同。

| 能力 | Chat Model | Agent |
| --- | --- | --- |
| 调用模型生成文本 | 支持 | 支持 |
| 绑定工具 Schema | 支持 | 通过工具配置支持 |
| 自动执行工具 | 不负责 | 支持 |
| 多次模型与工具循环 | 需要自行编排 | 内置执行流程 |
| Checkpointer 记忆 | 需额外编排 | 可直接配置 |
| Middleware | 不是主要使用入口 | 直接支持 |

例如普通模型调用：

```
model.invoke("查询 Python 课程")
```

只是一次聊天模型请求。

而 Agent：

```
agent.invoke({

    "messages": [{"role": "user", "content": "查询 Python 课程"}]

})
```

可以根据模型产生的工具调用请求，执行工具并继续生成最终回答。

---

## 二十一、完整 API 综合示例

最后，通过一个简洁的程序演示固定模型、流式输出、批量调用、工具绑定和结构化输出。

文件名：`chat_model_api_demo.py`

安装依赖：

```
pip install -U langchain langchain-deepseek pydantic python-dotenv
```

配置 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dotenv import load\_dotenv
from pydantic import BaseModel, Field
from langchain.chat\_models import init\_chat\_model
from langchain.tools import tool

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-chat",
    temperature=0,
    timeout=30
)

# ========== 1. 同步调用 ==========
response = model.invoke("用一句话介绍 Python。")
print("同步回答：", response.content)

# ========== 2. 流式输出 ==========
print("\n流式回答：", end="", flush=True)
for chunk in model.stream("介绍 Python 的三个优点。"):
    if isinstance(chunk.content, str):
        print(chunk.content, end="", flush=True)
print()

# ========== 3. 批量调用 ==========
results = model.batch(
    ["Python 是什么？", "Java 是什么？"],
    config={"max\_concurrency": 2}
)
for i, result in enumerate(results, 1):
    print(f"批量结果{i}：", result.content)

# ========== 4. 工具绑定 ==========
@tool
def get\_weather(city: str) -> str:
    """查询模拟天气。"""
    return f"{city}：模拟晴天，25°C"

tool\_model = model.bind\_tools([get\_weather])
response = tool\_model.invoke("请调用工具查询杭州天气。")

print("\n工具调用：", response.tool\_calls)

# ========== 5. 结构化输出 ==========
class CourseInfo(BaseModel):
    name: str = Field(description="课程名称")
    level: str = Field(description="课程难度")

structured\_model = model.with\_structured\_output(CourseInfo)

result = structured\_model.invoke(
    "提取课程信息：Python3 基础教程，入门级。"
)

print("\n结构化结果：", result)

# ========== 6. 查看 Token 用量 ==========
response = model.invoke("你好")
print("\nToken 用量：", response.usage\_metadata)

其中工具绑定和结构化输出取决于当前模型服务商的支持情况。

工具绑定不会自动执行工具；结构化输出如果不符合 Schema，也可能出现解析或校验错误。

---

## 二十二、常见问题

### 1、init\_chat\_model() 和 ChatDeepSeek 有什么区别？

init\_chat\_model() 是统一初始化入口；ChatDeepSeek 是 DeepSeek 提供商的专用模型类。两者都可以用于创建 DeepSeek Chat Model。

### 2、为什么 init\_chat\_model() 需要安装额外的提供商包？

因为 LangChain 将不同模型的集成实现拆分到独立 Python 包中。

### 3、model\_provider 可以不写吗？

可以。使用 provider:model\_name 格式时，通常不需要再单独指定。

### 4、为什么 deepseek-chat 不写前缀也可能工作？

因为 LangChain 可以尝试根据模型名称推断服务商。但推荐显式使用 deepseek:deepseek-chat。

### 5、temperature=0 是否代表完全确定性？

不一定。模型服务、采样实现和推理环境可能影响结果。

### 6、max\_tokens 是模型上下文窗口大小吗？

不是。max\_tokens 通常用于限制输出 Token 数，上下文窗口则限制模型能够处理的整体上下文规模。

### 7、invoke() 和 stream() 有什么区别？

invoke() 通常返回完整消息；stream() 用于逐步获取消息片段。

### 8、batch() 一定比循环 invoke() 更快吗？

不一定。batch() 可以提供并发处理，但实际速度受到模型服务、速率限制和并发配置影响。

### 9、bind\_tools() 会自动运行工具吗？

不会。它只是将工具定义提供给模型。自动工具执行通常交给 Agent。

### 10、with\_structured\_output() 一定返回 Pydantic 吗？

不一定。返回类型取决于 Schema 类型及相关配置。

### 11、如何查看模型消耗了多少 Token？

可以查看 AIMessage.usage\_metadata，但必须以服务商实际返回的统计为准。

### 12、如何动态切换模型？

创建模型时配置 configurable\_fields，再通过 config["configurable"] 传入允许运行时修改的字段。

### 13、configurable\_fields="any" 是否安全？

对于接收不可信用户配置的服务，不建议直接使用 any。应明确限制可修改字段，避免暴露 API Key 或 base\_url 等敏感配置。

### 14、如何限制 API 请求频率？

可以使用 rate\_limiter。多实例部署还应使用共享的限流服务。

### 15、是否所有模型都支持图片、工具和结构化输出？

不是。必须检查模型服务商当前支持的功能和 LangChain 对应集成能力。

---

## 总结

LangChain Chat Model API 提供了一套统一的模型调用方式。

本章节主要介绍了：

- `init_chat_model()`：统一初始化模型。
- `model_provider`：指定模型服务商。
- `configurable_fields`：运行时修改模型参数。
- `config_prefix`：区分动态模型配置。
- `invoke()` 和 `ainvoke()`：同步与异步调用。
- `stream()` 和 `astream()`：流式输出。
- `batch()`：批量调用。
- `bind_tools()`：绑定工具。
- `with_structured_output()`：结构化输出。
- `bind()`：预先绑定参数。
- `AIMessage`：读取模型消息和 Token 用量。
- `rate_limiter`：限制模型请求速率。

最常用的初始化方式：

```
from langchain.chat_models import init_chat_model



model = init_chat_model(

    "deepseek:deepseek-chat",

    temperature=0,

    timeout=30

)



response = model.invoke("你好")

print(response.content)
```

**核心原则：**init\_chat\_model() 解决的是不同模型的初始化问题，BaseChatModel 解决的是模型调用接口统一的问题。对于复杂的工具协作、执行循环和对话状态管理，则可以在 Chat Model 基础上使用 LangChain Agent。

## 参考文档

- [init\_chat\_model API Reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model)
- [BaseChatModel API Reference](https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel)
- [LangChain Models 官方文档](https://docs.langchain.com/oss/python/langchain/models)
- [LangChain Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Structured Output 官方文档](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangChain DeepSeek 模型集成](https://docs.langchain.com/oss/python/integrations/chat/deepseek)
