# LangChain 消息类型

> 来源: https://www.runoob.com/langchain/langchain-messages.html

---

# LangChain 消息类型

在 LangChain 中，聊天模型和 Agent 主要通过消息（Message）进行交互。

消息不仅可以包含普通文本，还可以包含系统指令、工具调用、工具执行结果，以及图片等多模态内容。

理解 LangChain 的消息类型，有助于开发多轮对话、Agent、工具调用和多模态 AI 应用。

本章节基于 LangChain 1.x 的消息 API，介绍常用消息类型、消息属性、流式消息以及消息历史管理。

---

## 一、LangChain 的核心消息类型

LangChain 常用的四种消息类型分别是：

| 消息类型 | 角色 | 主要用途 |
| --- | --- | --- |
| `HumanMessage` | 用户 | 表示用户发送的问题或内容 |
| `AIMessage` | AI 助手 | 表示模型回复，也可能包含工具调用 |
| `SystemMessage` | 系统 | 设置模型的角色、任务和行为规则 |
| `ToolMessage` | 工具 | 保存工具执行结果，关联对应的工具调用 |

一个简单的对话可以表示为：

```
SystemMessage：你是一名 Python 编程助手。



HumanMessage：什么是 Python？



AIMessage：Python 是一种通用编程语言。
```

如果使用工具，消息流程可能变成：

```
HumanMessage

    ↓

AIMessage（包含 tool_calls）

    ↓

ToolMessage（工具执行结果）

    ↓

AIMessage（最终回答）
```

这些消息会组成消息列表，用于向聊天模型提供对话上下文。

---

## 二、HumanMessage：用户消息

`HumanMessage` 用于表示用户发送给聊天模型的内容。

例如：

## 实例

from langchain.messages import HumanMessage

# 创建用户消息
message = HumanMessage(
    content="菜鸟教程 RUNOOB 是什么？"
)

print("消息类型：", message.type)
print("消息内容：", message.content)
print("消息 ID：", message.id)

可能输出：

```
消息类型： human

消息内容： 菜鸟教程 RUNOOB 是什么？

消息 ID： None
```

其中：

- `content`：消息内容。
- `type`：LangChain 消息类型，HumanMessage 为 `human`。
- `id`：消息标识，可能为 None，也可能由调用方或模型提供商设置。

**注意：**不能直接假设 `HumanMessage` 对象存在 `role` 属性。LangChain 消息对象通常使用 `type` 标识消息类型，而 OpenAI 风格的消息字典使用 `role` 字段。

### 使用 HumanMessage 调用模型

## 实例

from dotenv import load\_dotenv
from langchain.messages import HumanMessage
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

messages = [
    HumanMessage(
        content="用一句话介绍菜鸟教程 RUNOOB。"
    )
]

response = model.invoke(messages)

print(response.content)

模型通常返回一个 `AIMessage` 对象，其中包含生成的回复。

本例需要提前安装 `langchain`、`langchain-deepseek` 和 `python-dotenv`，并配置 `DEEPSEEK_API_KEY`。

### HumanMessage 的快捷写法

除了直接创建消息对象，LangChain 还支持通过元组和字典描述消息。

## 实例

from langchain.messages import HumanMessage

# 方式一：消息对象
msg1 = HumanMessage(content="你好")

# 方式二：元组
msg2 = ("user", "你好")

# 方式三：字典
msg3 = {
    "role": "user",
    "content": "你好"
}

print(type(msg1))
print(msg2)
print(msg3)

需要注意，元组和字典本身不会立即变成 `HumanMessage` 对象。

只有在传入接受 MessageLike 格式的 LangChain API 时，框架才会按需进行消息转换。

例如：

```
messages = [

    ("system", "你是一名编程助手"),

    ("user", "介绍 Python")

]



response = model.invoke(messages)
```

这种写法可以减少重复创建消息对象的代码。

---

## 三、AIMessage：AI 回复

`AIMessage` 表示语言模型生成的消息。

除了普通文本，AIMessage 还可以包含工具调用信息、Token 使用统计和响应元数据。

### 创建普通 AIMessage

## 实例

from langchain.messages import AIMessage

message = AIMessage(
    content="菜鸟教程是一个在线编程学习平台。"
)

print("类型：", message.type)
print("内容：", message.content)
print("工具调用：", message.tool\_calls)

执行结果：

```
类型： ai

内容： 菜鸟教程是一个在线编程学习平台。

工具调用： []
```

这里的 `tool_calls` 是空列表，说明这条消息没有提供标准化的工具调用请求。

### 包含工具调用的 AIMessage

当模型需要调用工具时，可以通过 `tool_calls` 返回工具名称与参数。

## 实例

from langchain.messages import AIMessage

message = AIMessage(
    content="",
    tool\_calls=[
        {
            "name": "get\_weather",
            "args": {
                "city": "杭州"
            },
            "id": "call\_001",
            "type": "tool\_call"
        }
    ]
)

print("消息内容：", message.content)

for call in message.tool\_calls:
    print("工具名称：", call["name"])
    print("工具参数：", call["args"])
    print("调用 ID：", call["id"])

执行结果：

```
消息内容：

工具名称： get_weather

工具参数： {'city': '杭州'}

调用 ID： call_001
```

这里手动构造了一条带有工具调用请求的消息，仅用于展示数据结构。

实际调用时，工具调用 ID 通常由模型服务生成。

需要特别注意：**AIMessage 中出现 tool\_calls，并不表示工具已经执行。**

工具需要由应用程序或 Agent 执行。

### AIMessage 的附加信息

模型返回的 AIMessage 还可能提供以下信息：

| 属性 | 说明 |
| --- | --- |
| `content` | 模型生成的内容 |
| `tool_calls` | 有效的工具调用请求 |
| `invalid_tool_calls` | 未能正常解析的工具调用 |
| `response_metadata` | 模型响应元数据 |
| `usage_metadata` | 标准化 Token 使用统计 |
| `id` | 消息标识 |

### 实例：查看模型响应信息

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

response = model.invoke(
    "介绍菜鸟教程 RUNOOB。"
)

print("内容：", response.content)
print("消息 ID：", response.id)

# 提供商返回的响应元数据
print("响应元数据：")
print(response.response\_metadata)

# 标准化 Token 使用情况
usage = response.usage\_metadata

if usage:
    print("输入 Token：", usage.get("input\_tokens"))
    print("输出 Token：", usage.get("output\_tokens"))
    print("总 Token：", usage.get("total\_tokens"))
else:
    print("没有返回标准化 Token 统计")

不同模型提供商返回的 `response_metadata` 字段可能不同。

例如，模型名称、结束原因等字段的名称并不保证完全一致，因此不应依赖固定的元数据结构。

`usage_metadata` 是 LangChain 提供的标准化使用量字段，但也取决于服务商是否返回相关数据。

---

## 四、SystemMessage：系统指令

`SystemMessage` 通常用于向模型提供系统级指令，例如角色设定、回答风格和任务要求。

它一般放在消息列表的开头。

### 实例：设置回答风格

## 实例

from langchain.messages import (
    SystemMessage,
    HumanMessage
)
from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

messages = [
    SystemMessage(
        content=(
            "你是一名 Python 编程教程作者。"
            "回答应该简洁、准确，"
            "尽量使用初学者容易理解的语言。"
        )
    ),
    HumanMessage(
        content="什么是 Python 装饰器？"
    )
]

response = model.invoke(messages)

print(response.content)

模型会参考系统指令来组织回答。

例如，可以通过 SystemMessage 要求：

- 使用中文回答。
- 限制回答长度。
- 采用指定的解释风格。
- 优先使用提供的资料。
- 对未知信息明确说明。

**注意：**系统指令能够影响模型行为，但不能保证模型严格遵守所有要求。

对于必须执行的业务规则，例如用户权限、文件访问范围和支付校验，仍应在应用程序中实现，不能只依赖 SystemMessage。

---

## 五、ToolMessage：工具执行结果

`ToolMessage` 用于保存工具执行后返回的信息。

每个工具结果需要通过 `tool_call_id` 与先前的工具调用请求关联。

### 实例：完整的工具调用消息

## 实例

from langchain.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)
from langchain.chat\_models import init\_chat\_model

# 手动构造一段工具调用历史
messages = [
    HumanMessage(
        content="根据模拟数据，杭州天气怎么样？"
    ),

    # 模拟模型请求调用工具
    AIMessage(
        content="",
        tool\_calls=[
            {
                "name": "get\_weather",
                "args": {
                    "city": "杭州"
                },
                "id": "call\_weather\_001",
                "type": "tool\_call"
            }
        ]
    ),

    # 模拟工具已执行并返回结果
    ToolMessage(
        content="晴，25°C，湿度60%",
        name="get\_weather",
        tool\_call\_id="call\_weather\_001"
    )
]

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

response = model.invoke(messages)

print(response.content)

模型可以基于工具结果生成类似回答：

```
根据提供的模拟数据，杭州天气晴，

气温25°C，湿度60%。
```

这里需要注意：

- `AIMessage.tool_calls[0]["id"]` 是工具调用 ID。
- `ToolMessage.tool_call_id` 必须与对应的调用 ID 匹配。
- 如果一条 AIMessage 包含多个工具调用，就需要提供对应的工具结果。
- 工具消息通常不能作为与上下文无关的独立用户输入。

不匹配或缺失的工具结果可能导致模型 API 拒绝请求，具体行为取决于提供商。

**注意：**本例中的天气数据是模拟数据，不是实时查询结果。

### ToolMessage 的其他属性

除了 `content` 和 `tool_call_id`，ToolMessage 还支持一些附加字段。

例如，部分场景可以使用 `artifact` 保存不需要直接发送给模型的完整工具结果：

## 实例

from langchain.messages import ToolMessage

message = ToolMessage(
    content="找到3条匹配记录",
    tool\_call\_id="call\_search\_001",
    name="search\_database",
    artifact={
        "records": [
            {"id": 1, "name": "Python"},
            {"id": 2, "name": "Java"},
            {"id": 3, "name": "Go"}
        ]
    }
)

print(message.content)
print(message.artifact)

`content` 可以保存供模型使用的简要结果，`artifact` 则可以保存应用需要的补充数据。

不过，`artifact` 不是模型自动理解或使用的内容，具体处理方式取决于应用程序。

---

## 六、AIMessageChunk：流式消息片段

使用 `stream()` 时，模型可以逐步返回消息片段。

对于常见聊天模型，流式响应通常包含 `AIMessageChunk`。

与完整的 `AIMessage` 不同，消息片段可能只包含部分文本或部分工具调用信息。

### 实例：流式输出

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

print("开始流式输出：")

for chunk in model.stream(
    "用一句话介绍菜鸟教程 RUNOOB。"
):
    print(
        chunk.text,
        end="",
        flush=True
    )

print()

`chunk.text` 用于获取消息片段中的文本内容。

需要注意，流式响应不一定只包含文本。部分模型可能返回思考信息、工具调用片段或其他内容。

### 合并消息片段

LangChain 支持使用 `+` 合并兼容的消息片段。

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

full\_chunk = None

for chunk in model.stream("介绍 Python。"):

    if full\_chunk is None:
        full\_chunk = chunk
    else:
        full\_chunk = full\_chunk + chunk

if full\_chunk is not None:
    print("完整文本：")
    print(full\_chunk.text)

    print("消息类型：")
    print(type(full\_chunk))

这种方式适合需要累积流式响应、工具调用参数片段或统计信息的应用。

**注意：**当流式输出包含工具调用时，不应将每个工具调用片段直接当作完整 JSON 参数使用，应等待并合并必要的片段。

---

## 七、LangChain 消息属性速查表

| 消息类型 | type | 常用属性 | 主要用途 |
| --- | --- | --- | --- |
| `HumanMessage` | human | content、id | 用户输入 |
| `AIMessage` | ai | content、tool\_calls、usage\_metadata | 模型回复 |
| `SystemMessage` | system | content | 系统指令 |
| `ToolMessage` | tool | content、tool\_call\_id、name、artifact | 工具结果 |
| `AIMessageChunk` | ai | content、tool\_call\_chunks | 流式模型响应 |
| `RemoveMessage` | remove | id | 状态更新中删除指定消息 |

`AIMessageChunk` 虽然同样可以显示 `type="ai"`，但它与完整的 `AIMessage` 是不同的 Python 类型。

另外，LangChain 消息类型在不断扩展，除了上面的常用类型，还有 `ChatMessage` 等用于自定义角色的消息类。

---

## 八、ContentBlock：结构化消息内容

前面的示例大多使用字符串作为 `content`。

但在实际应用中，消息还可能包含图片、音频、视频、文件以及其他结构化信息。

LangChain 1.x 提供标准化内容块（Content Blocks）机制，用于表示这些不同类型的内容。

### 常见 ContentBlock 类型

| 类型 | 说明 |
| --- | --- |
| `TextContentBlock` | 普通文本内容 |
| `ImageContentBlock` | 图片内容 |
| `AudioContentBlock` | 音频内容 |
| `VideoContentBlock` | 视频内容 |
| `FileContentBlock` | PDF、Word 等文件内容 |
| `PlainTextContentBlock` | 纯文本文件内容，例如 TXT 文件 |
| `ReasoningContentBlock` | 模型提供的标准化推理内容 |

需要特别区分：

- `TextContentBlock`：普通聊天文本。
- `PlainTextContentBlock`：主要用于表示文本文件或纯文本数据资源。

另外，`ToolCall` 通常保存在 AIMessage 的 `tool_calls` 中，而不是简单地当作图片或文本内容块处理。

### 实例：构建文本内容块

## 实例

from langchain.messages import HumanMessage

# 普通字符串消息
simple\_message = HumanMessage(
    content="你好，LangChain！"
)

# 使用标准化文本内容块
block\_message = HumanMessage(
    content\_blocks=[
        {
            "type": "text",
            "text": "你好，LangChain！"
        }
    ]
)

print(simple\_message.content)
print(block\_message.content\_blocks)

通过 `content_blocks` 可以访问标准化后的消息内容块。

与 `content` 不同，`content_blocks` 更适合用于处理不同提供商的多模态内容格式。

---

## 九、多模态消息：文本与图片

对于支持图片输入的模型，可以在同一条消息中同时提供文本和图片。

例如，让模型描述一张网页截图。

**注意：**不是所有聊天模型都支持图片输入。是否支持多模态，需要核对当前模型及其 API 的实际能力。

本节使用 OpenAI 的视觉模型展示消息结构，不使用 DeepSeek Flash 作为默认图片示例。

### 实例：通过图片 URL 提问

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model
from langchain.messages import HumanMessage

load\_dotenv()

# 使用支持图片输入的模型
model = init\_chat\_model(
    "openai:gpt-4o"
)

messages = [
    HumanMessage(
        content=[
            {
                "type": "text",
                "text": "请描述这张图片的主要内容。"
            },
            {
                "type": "image\_url",
                "image\_url": {
                    "url": "https://www.langchain.com/images/brand/langchain\_logo\_text\_w\_white.png"
                }
            }
        ]
    )
]

response = model.invoke(messages)

print(response.content)

上面使用的是 OpenAI 兼容的多模态消息格式。

如果使用其他提供商，其底层图片格式可能不同。LangChain 也提供标准化的 `ImageContentBlock` 来减少不同提供商之间的格式差异。

### 实例：标准化图片内容块

## 实例

from langchain.messages import HumanMessage

message = HumanMessage(
    content\_blocks=[
        {
            "type": "text",
            "text": "请描述这张图片。"
        },
        {
            "type": "image",
            "url": "https://www.langchain.com/images/brand/langchain\_logo\_text\_w\_white.png"
        }
    ]
)

for block in message.content\_blocks:
    print(block)

标准化图片块通常使用 `type="image"`。

图片可以通过 URL、Base64 数据或部分提供商支持的文件 ID 等方式提供。

### 实例：读取本地图片

如果需要使用本地图片，可以先将图片转换为 Base64 编码。

## 实例

import base64

from pathlib import Path
from langchain.messages import HumanMessage

# 读取本地 PNG 图片
image\_path = Path("screenshot.png")

image\_base64 = base64.b64encode(
    image\_path.read\_bytes()
).decode("utf-8")

message = HumanMessage(
    content\_blocks=[
        {
            "type": "text",
            "text": "请描述截图中的主要内容。"
        },
        {
            "type": "image",
            "base64": image\_base64,
            "mime\_type": "image/png"
        }
    ]
)

print("内容块数量：", len(message.content\_blocks))

以上代码展示如何创建包含本地图片数据的消息，不会自动调用模型。

后续可以将 `message` 传入支持图片输入的聊天模型。

**注意：**图片上传可能产生额外 Token 或费用，实际限制由模型提供商决定。

---

## 十、ToolCall：工具调用请求

在 Agent 和 Tool Calling 场景中，模型可能返回一个或多个工具调用请求。

每个工具调用通常包含：

- 工具名称。
- 调用参数。
- 工具调用 ID。

### 实例：手动构建 ToolCall

## 实例

from langchain.messages import AIMessage

tool\_call = {
    "name": "get\_weather",
    "args": {
        "city": "杭州"
    },
    "id": "call\_001",
    "type": "tool\_call"
}

message = AIMessage(
    content="",
    tool\_calls=[tool\_call]
)

print(message.tool\_calls[0]["name"])
print(message.tool\_calls[0]["args"])
print(message.tool\_calls[0]["id"])

执行结果：

```
get_weather

{'city': '杭州'}

call_001
```

LangChain 的 `ToolCall` 是结构化的数据类型。教学代码中可以直接使用符合格式的字典构建工具调用。

### 正确判断是否存在工具调用

应该检查：

```
if response.tool_calls:

    print("模型请求调用工具")
```

不建议使用：

```
if not response.content:

    print("模型请求调用工具")
```

因为 `content` 为空，并不一定意味着模型请求了工具。

如果需要检查解析失败的工具调用，还可以查看：

```
print(response.invalid_tool_calls)
```

---

## 十一、trim\_messages()：裁剪消息历史

当对话越来越长时，消息历史可能超过模型允许的上下文窗口。

`trim_messages()` 用于根据指定的数量或 Token 预算裁剪消息列表。

例如，可以优先保留最近的对话，同时保留最开始的系统指令。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `messages` | 需要裁剪的消息列表 |
| `max_tokens` | 消息长度预算，含义取决于计数器 |
| `token_counter` | Token 计数函数、模型或 approximate |
| `strategy` | first 保留最早内容，last 保留最近内容 |
| `include_system` | 在符合条件时保留开头的系统消息 |
| `start_on` | 指定裁剪后消息从哪种类型开始 |
| `allow_partial` | 是否允许保留消息的部分内容 |
| `end_on` | 指定裁剪结果结束的消息类型 |

### 实例：保留最近对话

## 实例

from langchain.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim\_messages
)

messages = [
    SystemMessage(
        content="你是菜鸟教程 RUNOOB 的 AI 助手。"
    ),

    HumanMessage(content="Python 怎么入门？"),
    AIMessage(content="建议先学习基础语法。"),

    HumanMessage(content="推荐什么 IDE？"),
    AIMessage(content="可以使用 VS Code 或 PyCharm。"),

    HumanMessage(content="如何安装第三方库？"),
    AIMessage(content="可以使用 pip install。"),

    HumanMessage(content="什么是 NumPy？"),
    AIMessage(content="NumPy 是 Python 数值计算库。"),

    HumanMessage(content="pandas 和 NumPy 有什么区别？")
]

# 通过消息数量演示裁剪，结果更直观
trimmed = trim\_messages(
    messages,
    max\_tokens=5,
    strategy="last",
    token\_counter=len,
    include\_system=True,
    start\_on="human"
)

print("裁剪前：", len(messages))
print("裁剪后：", len(trimmed))

for message in trimmed:
    print(message.type, message.content)

这里使用 `token_counter=len`，因此 `max_tokens=5` 代表以消息数量为预算，而不是实际 Token 数。

这样更适合展示裁剪机制。

实际使用中，应根据模型上下文窗口使用 Token 计数方式。

### 按近似 Token 数裁剪

## 实例

from langchain.messages import trim\_messages

trimmed = trim\_messages(
    messages,
    max\_tokens=100,
    strategy="last",
    token\_counter="approximate",
    include\_system=True,
    start\_on="human"
)

for message in trimmed:
    print(message.type, message.content)

`token_counter="approximate"` 使用近似计数，适合希望减少精确 Token 计算开销的场景。

但近似计数并不等于具体模型最终使用的真实 Token 数。

此外，`trim_messages()` 会根据预算和消息结构决定保留多少条消息，不应假设它始终返回固定数量的消息。

### 裁剪时需要注意工具消息

当历史记录包含工具调用时，需要尽量保持消息结构完整。

例如：

```
AIMessage(tool_calls=[...])

ToolMessage(tool_call_id="...")

AIMessage(content="最终回答")
```

如果只保留 ToolMessage，却删除了对应的工具调用消息，可能导致请求无效。

对于复杂 Agent，不应简单地按字符数删除消息；可以结合 LangChain 的消息工具、中间件或上下文压缩策略进行管理。

---

## 十二、RemoveMessage：删除状态中的消息

`RemoveMessage` 用于在支持消息状态归并的流程中表示删除某条消息的操作。

它通常与 LangGraph 的 `add_messages` reducer 或 Agent 状态管理机制配合使用。

需要注意：**创建 RemoveMessage 对象，不会自动从普通 Python 列表中删除消息。**

### 实例：创建删除指令

## 实例

from langchain.messages import (
    HumanMessage,
    AIMessage,
    RemoveMessage
)

messages = [
    HumanMessage(
        content="你好",
        id="msg\_1"
    ),
    AIMessage(
        content="你好！",
        id="msg\_2"
    ),
    HumanMessage(
        content="查询天气",
        id="msg\_3"
    )
]

# 构造删除指定消息的指令
removal = RemoveMessage(
    id="msg\_3"
)

print("删除目标：", removal.id)
print("消息类型：", removal.type)

# 普通列表不会自动变化
print("原始列表长度：", len(messages))

执行结果：

```
删除目标： msg_3

消息类型： remove

原始列表长度： 3
```

如果需要真正更新消息状态，应将 `RemoveMessage` 交给支持此操作的状态归并器处理。

例如，在 LangGraph 工作流中，可以结合对应的消息状态更新机制实现删除。

**注意：**RemoveMessage 主要用于修改工作流中的消息状态，不能代替数据库删除、日志清理或其他持久化数据删除操作。

---

## 十三、消息对象的常用属性与方法

LangChain 的消息类型通常继承自 `BaseMessage`，共享一些常见属性和方法。

### 实例

## 实例

from langchain.messages import HumanMessage

message = HumanMessage(
    content="你好，菜鸟教程 RUNOOB"
)

# 消息内容
print("content：", message.content)

# LangChain 消息类型
print("type：", message.type)

# 消息 ID（可能为空）
print("id：", message.id)

# 提取文本
print("text：", message.text)

# 标准化内容块
print("content\_blocks：", message.content\_blocks)

# 格式化输出
print(message.pretty\_repr())

其中：

| 属性或方法 | 作用 |
| --- | --- |
| `content` | 原始消息内容 |
| `content_blocks` | 标准化的内容块列表 |
| `type` | 消息类型 |
| `id` | 消息标识，可能为 None |
| `text` | 提取消息中的文本内容 |
| `pretty_repr()` | 生成便于调试的消息描述 |

例如，`pretty_repr()` 可以输出带有消息类型标题的文本，便于在终端查看对话记录。

不同 LangChain 版本可能对文本访问接口存在差异，使用旧版教程时应注意区分 `text` 属性与历史版本中采用的方法写法。

---

## 十四、完整实例：使用消息列表构建对话

下面通过一个完整的 Python 程序，将系统消息、用户消息、AI 回复以及对话历史组合起来。

**文件名：**`langchain_messages.py`

## 完整实例

from dotenv import load\_dotenv

from langchain.chat\_models import init\_chat\_model
from langchain.messages import (
    HumanMessage,
    SystemMessage
)

load\_dotenv()

# 初始化模型
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 初始化对话历史
messages = [
    SystemMessage(
        content=(
            "你是一名 Python 编程教程助手。"
            "请使用简洁、准确的中文回答。"
        )
    )
]

def chat(question: str):
    """向模型发送问题并保存对话历史。"""

    # 添加用户消息
    messages.append(
        HumanMessage(content=question)
    )

    # 调用模型
    response = model.invoke(messages)

    # 保存 AI 回复
    messages.append(response)

    print("用户：", question)
    print("AI：", response.text)
    print("-" \* 40)

# 第一轮
chat("什么是 Python？")

# 第二轮
chat("它适合初学者吗？")

# 第三轮
chat("给我一个简单的学习计划。")

print("\n当前消息历史：")

for message in messages:
    print(
        f"[{message.type}]",
        message.text
    )

在这个示例中，`messages` 列表会保存之前的用户消息和模型回复。

因此，第二轮和第三轮调用时，模型可以参考前面的对话内容。

需要注意：

- 这种方式只在当前 Python 进程中保存历史。
- 程序重新启动后，消息列表会重新初始化。
- 真实项目应注意上下文长度限制。
- 多个用户不能直接共享同一个全局消息列表，否则可能造成对话串扰。

如果需要支持多用户会话和持久化历史，可以使用独立的会话存储，以及 LangGraph 的状态检查点等机制。

---

## 十五、常见问题

### 1、HumanMessage 和 AIMessage 有什么区别？

`HumanMessage` 表示用户输入；`AIMessage` 表示模型输出。

AIMessage 还可能包含工具调用信息和模型响应元数据。

### 2、为什么 message.role 会报错？

因为 LangChain 的消息对象通常使用 `type` 属性区分消息类型。

例如：

```
message.type
```

而下面这种字典表示方式使用 `role`：

```
{

    "role": "user",

    "content": "你好"

}
```

不能将两种表示方式中的属性完全混用。

### 3、为什么 AIMessage 的 content 为空？

可能是模型返回了工具调用，也可能是其他非文本内容或特殊响应情况。

应检查：

```
response.tool_calls

response.invalid_tool_calls

response.content_blocks
```

不要只根据 `content` 是否为空判断是否调用工具。

### 4、ToolMessage 可以直接发送给模型吗？

ToolMessage 通常需要与前面的工具调用请求对应。

如果缺少相关 AIMessage 或工具调用 ID 不匹配，部分模型服务可能直接拒绝请求。

### 5、LangChain 可以处理图片、音频和视频吗？

LangChain 提供标准化的多模态消息结构。

但真正能够处理哪些内容，取决于具体模型提供商、模型版本与 API 的支持情况。

### 6、如何限制消息历史长度？

可以使用 `trim_messages()` 按消息数量或 Token 预算裁剪历史。

例如：

```
trimmed = trim_messages(

    messages,

    max_tokens=1000,

    token_counter="approximate",

    strategy="last",

    include_system=True,

    start_on="human"

)
```

对于包含工具调用的复杂历史，还应检查裁剪后消息之间的关联是否完整。

### 7、消息 ID 是否会自动生成？

不能假设所有消息都有自动生成的 ID。

手动创建的消息可能没有 ID，而模型响应或状态管理流程可能分配相应标识。

如果需要自行管理消息，可以显式指定：

```
message = HumanMessage(

    content="你好",

    id="user_message_001"

)
```

### 8、RemoveMessage 会删除数据库中的聊天记录吗？

不会。

它主要用于支持消息状态归并的工作流中，表示删除指定消息的状态更新操作。

数据库或其他持久化存储中的删除，需要通过对应的存储接口处理。

---

## 总结

LangChain 的消息系统是聊天模型、Agent 和多模态应用的重要基础。

需要重点掌握以下内容：

- `HumanMessage`：表示用户输入。
- `AIMessage`：表示模型回复，也可能包含工具调用。
- `SystemMessage`：提供系统级指令。
- `ToolMessage`：保存工具执行结果，并关联对应的工具调用。
- `AIMessageChunk`：用于流式消息输出。
- `ContentBlock`：用于表示文本、图片、音频等结构化内容。
- `trim_messages()`：根据预算裁剪聊天历史。
- `RemoveMessage`：在支持的工作流中删除指定消息。

在开发 LangChain 应用时，应特别区分消息对象与消息字典、工具调用请求与工具执行结果，以及消息内容和标准化内容块。

掌握这些概念后，再学习 Agent 工具调用、会话记忆、多模态模型和 LangGraph 状态管理会更加容易。
