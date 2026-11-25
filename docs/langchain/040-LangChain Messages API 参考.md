# LangChain Messages API 参考

> 来源: https://www.runoob.com/langchain/langchain-messages-api.html

---

# LangChain Messages API 参考

在 LangChain 中，无论是普通聊天、Agent 工具调用、多轮对话，还是图片、音频等多模态输入，都离不开 Messages（消息）。

Messages 是 LangChain 与聊天模型交互的基础数据结构，用于区分系统指令、用户输入、AI 回复和工具执行结果。

例如：

```
SystemMessage → 告诉模型应该如何回答

HumanMessage  → 用户提出问题

AIMessage     → 模型生成回答或工具调用

ToolMessage   → 工具返回执行结果
```

本章节基于 LangChain 1.x，整理常用消息类型、ContentBlock 多模态结构、ToolCall、消息转换、历史裁剪以及在 Agent 中的实际用法。

---

## 一、LangChain 消息类型速查

LangChain 的消息对象通常继承自 BaseMessage。

常用消息如下：

| 消息类型 | 对应角色 | type 属性 | 主要用途 |
| --- | --- | --- | --- |
| `HumanMessage` | user | human | 用户输入 |
| `AIMessage` | assistant | ai | 模型回复及工具调用请求 |
| `SystemMessage` | system | system | 系统指令 |
| `ToolMessage` | tool | tool | 工具执行结果 |
| `AIMessageChunk` | assistant | AIMessageChunk | 模型流式输出片段 |
| `RemoveMessage` | 无普通对话角色 | remove | 请求删除状态中的消息 |
| `ChatMessage` | 自定义角色 | chat | 自定义消息角色 |

这里需要区分两个概念：

- **role：**模型 API 通常使用的角色，例如 user、assistant、system、tool。
- **type：**LangChain 消息对象自身的类型标识，例如 human、ai、tool。

因此，HumanMessage 的 type 是 `human`，而不等于模型 API 中的 user。

另外，AIMessageChunk 虽然代表助手的流式回复，但其 Python 对象的 type 是 `AIMessageChunk`。

---

## 二、创建基本消息

可以从 `langchain.messages` 导入常见消息类。

### 实例：创建消息对象

## 实例

from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

system = SystemMessage(content="你是 Python 编程助手。")
human = HumanMessage(content="什么是 Python？")
ai = AIMessage(content="Python 是一种通用编程语言。")

tool = ToolMessage(
    content="计算结果：42",
    tool\_call\_id="call\_001",
    name="calculator"
)

for message in [system, human, ai, tool]:
    print(message.type, message.content)

输出：

```
system 你是 Python 编程助手。

human 什么是 Python？

ai Python 是一种通用编程语言。

tool 计算结果：42
```

注意，这里的 ToolMessage 是一个单独创建的示例对象。

真正发送给模型时，ToolMessage 应当与此前 AIMessage 中对应的工具调用请求配对，不能随意插入一个没有上下文的工具结果。

---

## 三、消息对象的常用属性

BaseMessage 提供了一些通用属性。

| 属性 | 说明 |
| --- | --- |
| `content` | 消息原始内容，可以是字符串或内容块列表 |
| `content_blocks` | 标准化后的多模态 ContentBlock 列表 |
| `type` | LangChain 消息类型 |
| `id` | 消息 ID，可能为空 |
| `name` | 可选的消息名称 |
| `additional_kwargs` | 服务商相关的额外数据 |
| `response_metadata` | 响应元数据，例如模型名称或其他统计信息 |

### 实例：读取消息属性

## 实例

from langchain.messages import HumanMessage

message = HumanMessage(
    content="介绍 Python 列表。",
    id="user-msg-001",
    name="student"
)

print("类型：", message.type)
print("内容：", message.content)
print("消息 ID：", message.id)
print("名称：", message.name)
print("标准内容块：", message.content\_blocks)

如果 content 是普通字符串，LangChain 也可以将其表示为标准文本内容块。

**注意：**消息 ID 不一定由模型服务商返回。实际需要稳定标识时，可以在应用层为消息分配 ID。

---

## 四、HumanMessage：用户消息

`HumanMessage` 用于表示用户输入。

既可以包含普通文字，也可以在模型支持的情况下包含图片、文件等多模态内容。

### 实例：普通用户消息

## 实例

from langchain.messages import HumanMessage

message = HumanMessage(content="Python 字典是什么？")

print(message.type)
print(message.content)

输出：

```
human

Python 字典是什么？
```

在调用 Chat Model 或 Agent 时，通常不需要手动创建 HumanMessage，也可以直接使用 role 字典。

---

## 五、SystemMessage：系统指令

`SystemMessage` 用于向模型提供角色设定、行为规则和其他高优先级指令。

### 实例

## 实例

from langchain.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是编程老师，请使用简洁中文回答。"),
    HumanMessage(content="解释 Python 循环语句。")
]

消息通常按照对话顺序排列。

如果使用 `create_agent()`，更常见的方式是通过 `system_prompt` 设置系统指令。

```
agent = create_agent(

    model="deepseek:deepseek-chat",

    tools=[],

    system_prompt="你是 Python 编程老师。"

)
```

不建议把不可信文档内容直接转换为 SystemMessage，因为这会提高外部数据在提示词中的权限。

---

## 六、AIMessage：模型回复

`AIMessage` 表示模型生成的响应。

除了文字，还可能包含工具调用、Token 使用量和模型元数据。

### 主要属性

| 属性 | 作用 |
| --- | --- |
| `content` | 模型内容 |
| `tool_calls` | 有效工具调用列表 |
| `invalid_tool_calls` | 无法正常解析的工具调用 |
| `usage_metadata` | Token 使用量 |
| `response_metadata` | 模型响应元数据 |
| `id` | 消息标识 |

### 实例：查看 AIMessage

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model("deepseek:deepseek-chat")
response = model.invoke("你好")

print("回答：", response.content)
print("类型：", response.type)
print("工具调用：", response.tool\_calls)
print("Token 使用量：", response.usage\_metadata)
print("响应元数据：", response.response\_metadata)

如果服务商返回 Token 数据，usage\_metadata 可能包含：

```
{

    "input_tokens": 12,

    "output_tokens": 20,

    "total_tokens": 32

}
```

这里是示例格式，并非真实调用结果。

不同提供商对 Token 用量、额外元数据和内容块的支持可能有所不同。

---

## 七、ToolCall：模型请求调用工具

ToolCall 用于表示模型希望调用某个工具。

通常可以通过：

```
AIMessage.tool_calls
```

读取。

一个工具调用常见的字段包括：

| 字段 | 说明 |
| --- | --- |
| `name` | 工具名称 |
| `args` | 工具参数字典 |
| `id` | 工具调用 ID |
| `type` | 工具调用类型标识 |

### 实例：创建工具调用消息

## 实例

from langchain.messages import AIMessage

ai = AIMessage(
    content="",
    tool\_calls=[
        {
            "name": "get\_weather",
            "args": {"city": "杭州"},
            "id": "call\_weather\_001",
            "type": "tool\_call"
        }
    ]
)

for call in ai.tool\_calls:
    print("工具：", call["name"])
    print("参数：", call["args"])
    print("调用 ID：", call["id"])

**重要：**创建 AIMessage.tool\_calls 不会自动执行工具。

如果直接使用 Chat Model，需要应用程序执行工具，再将结果作为 ToolMessage 返回给模型。

而使用 create\_agent() 时，LangChain 可以自动管理常见的工具调用循环。

---

## 八、ToolMessage：工具执行结果

`ToolMessage` 用于表示工具执行后返回的结果。

其中最重要的属性是：

```
tool_call_id
```

它必须对应先前 AIMessage 中某个 ToolCall 的 ID。

### 常用属性

| 属性 | 说明 |
| --- | --- |
| `content` | 提供给模型的工具结果 |
| `tool_call_id` | 对应的工具调用 ID |
| `name` | 工具名称，可选 |
| `status` | 执行状态，如 success、error |
| `artifact` | 不必直接发送给模型的附加数据 |

### 实例：工具请求和结果配对

## 实例

from langchain.messages import AIMessage, ToolMessage

ai = AIMessage(
    content="",
    tool\_calls=[{
        "name": "get\_weather",
        "args": {"city": "杭州"},
        "id": "call\_001",
        "type": "tool\_call"
    }]
)

tool\_result = ToolMessage(
    content="杭州模拟天气：晴，25°C",
    tool\_call\_id="call\_001",
    name="get\_weather",
    status="success"
)

messages = [ai, tool\_result]

print(ai.tool\_calls[0]["id"])
print(tool\_result.tool\_call\_id)

两个 ID 必须一致：

```
call_001

call_001
```

如果模型一次提出多个工具调用，就需要根据提供商的协议为每个调用提供正确的工具响应。

不要在一个尚未响应完所有工具调用的消息序列中，随意插入无关用户消息。

---

## 九、完整工具调用消息序列

下面展示一次有效的模型工具调用交互。

```
HumanMessage

    ↓

AIMessage(tool_calls)

    ↓

ToolMessage

    ↓

AIMessage(最终回答)
```

### 实例：手动执行工具

## 完整实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

load\_dotenv()

@tool
def get\_weather(city: str) -> str:
    """查询城市模拟天气。"""
    return f"{city}：模拟晴，25°C"

model = init\_chat\_model("deepseek:deepseek-chat")
model\_with\_tools = model.bind\_tools([get\_weather])

messages = [HumanMessage(content="请用工具查询杭州的模拟天气。")]

ai = model\_with\_tools.invoke(messages)
messages.append(ai)

for call in ai.tool\_calls:
    if call["name"] == "get\_weather":
        result = get\_weather.invoke(call["args"])
        messages.append(ToolMessage(
            content=result,
            tool\_call\_id=call["id"],
            name="get\_weather"
        ))

if ai.tool\_calls:
    final = model\_with\_tools.invoke(messages)
    print("最终回答：", final.content)
else:
    print("模型未请求工具：", ai.content)

这里演示了工具调用的三个关键步骤：

1. 模型生成包含 tool\_calls 的 AIMessage。
2. 应用根据调用参数执行工具，并返回对应的 ToolMessage。
3. 将工具执行结果放回消息历史，再调用模型生成回答。

这个示例只处理一轮工具调用。如果模型再次返回 tool\_calls，还需要继续执行循环。

生产环境建议使用 create\_agent() 管理工具循环、执行次数限制和异常处理。

---

## 十、AIMessageChunk：流式消息

调用模型的 `stream()` 方法时，通常会逐步返回 `AIMessageChunk`。

它表示模型输出的一个增量片段。

例如：

```
Chunk 1：Python

Chunk 2： 是一种

Chunk 3： 编程语言。
```

实际 Chunk 不一定与单个 Token 一一对应。

### 实例：读取流式消息

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model("deepseek:deepseek-chat")

for chunk in model.stream("简单介绍 Python。"):
    if isinstance(chunk.content, str):
        print(chunk.content, end="", flush=True)

print()

AIMessageChunk 可以具有以下字段：

- `content`：文本或其他内容增量。
- `tool_call_chunks`：工具调用参数的流式片段。
- `usage_metadata`：可用时的 Token 信息。
- `chunk_position`：某些流中的片段位置标记。

需要注意，AIMessageChunk 的 type 通常是：

```
AIMessageChunk
```

并非 AIMessage 的 `ai`。

---

## 十一、合并流式消息 Chunk

如果不仅希望逐步打印文本，还想获得完整消息对象，可以将多个 AIMessageChunk 合并。

### 实例：合并消息增量

## 实例

from langchain.messages import AIMessageChunk

chunks = [
    AIMessageChunk(content="Python"),
    AIMessageChunk(content=" 是一种"),
    AIMessageChunk(content=" 编程语言。")
]

merged = chunks[0]

for chunk in chunks[1:]:
    merged = merged + chunk

print(merged.content)

输出：

```
Python 是一种 编程语言。
```

这里使用 `+` 合并消息片段。

实际流式工具调用中，工具名称和参数也可能分批返回，需要按照 LangChain 的 Chunk 合并规则进行处理，而不是直接把所有工具参数片段当成完整 JSON。

---

## 十二、ContentBlock：标准化多模态内容

LangChain 1.x 提供了标准化的 ContentBlock 结构，用于描述文本、图片、音频、视频和文件等不同内容。

模型消息不一定只有一个字符串，还可能包含多个内容块。

例如：

```
HumanMessage

    |

    ├── TextContentBlock

    └── ImageContentBlock
```

常见 ContentBlock 类型：

| 类型 | type 标识 | 用途 |
| --- | --- | --- |
| `TextContentBlock` | text | 普通消息文本 |
| `PlainTextContentBlock` | text-plain | TXT、Markdown 等纯文本文件数据 |
| `ImageContentBlock` | image | 图片 |
| `AudioContentBlock` | audio | 音频 |
| `VideoContentBlock` | video | 视频 |
| `FileContentBlock` | file | PDF、Word 等文件数据 |
| `ReasoningContentBlock` | reasoning | 模型接口公开返回的推理相关内容 |
| `NonStandardContentBlock` | non\_standard | 未标准化的服务商专用内容 |

此外，还有 ToolCall、ServerToolCall、ServerToolResult 等与工具调用相关的标准结构。

**注意：**ContentBlock 是统一的数据表示格式，并不代表所有模型都支持所有模态。是否可以发送图片、音频、PDF 或视频，仍需检查具体模型和提供商集成。

---

## 十三、TextContentBlock 与 PlainTextContentBlock 的区别

这两个名字比较相似，但用途不同。

### TextContentBlock

表示普通消息正文，例如：

```
{

    "type": "text",

    "text": "请解释 Python 列表"

}
```

### PlainTextContentBlock

用于表示纯文本文件数据，例如 Markdown、TXT 文件内容。

```
{

    "type": "text-plain",

    "mime_type": "text/plain",

    "text": "# Python 教程\n变量用于保存数据"

}
```

一般聊天使用普通字符串或 TextContentBlock 就足够了，不必把每段普通文字都封装成 PlainTextContentBlock。

### 实例：创建标准文本内容块

## 实例

from langchain.messages import HumanMessage
from langchain\_core.messages.content import create\_text\_block

message = HumanMessage(
    content=[create\_text\_block("什么是 Python？")]
)

print(message.content\_blocks)

create\_text\_block() 可以自动创建符合标准格式的文本内容块。

---

## 十四、图片输入 ImageContentBlock

如果使用支持图片理解的模型，可以在 HumanMessage 中添加图片内容块。

例如：

## 实例：图片消息结构

from langchain.messages import HumanMessage
from langchain\_core.messages.content import create\_text\_block, create\_image\_block

message = HumanMessage(content=[
    create\_text\_block("请描述这张图片。"),
    create\_image\_block(
        url="https://example.com/sample.png",
        mime\_type="image/png"
    )
])

for block in message.content\_blocks:
    print(block["type"])

输出：

```
text

image
```

这里的 URL 仅用于演示数据结构，不保证存在对应图片。

图片内容可以根据模型接口要求使用 URL、Base64 数据或其他受支持的文件引用方式。

**注意：**使用 `deepseek-chat` 等聊天模型时，不应假定其支持图片输入。需要选择明确支持视觉能力的模型。

---

## 十五、音频、视频与文件内容块

标准化 ContentBlock 还覆盖以下类型：

| 内容类型 | 常见数据形式 | 应用场景 |
| --- | --- | --- |
| AudioContentBlock | 音频数据、文件引用等 | 语音转写、语音理解 |
| VideoContentBlock | 视频数据或帧信息 | 视频内容分析 |
| FileContentBlock | URL、Base64、文件 ID | PDF 和其他文档输入 |
| PlainTextContentBlock | 文本内容或文件引用 | TXT、Markdown 文档 |

对于不属于图片、音频或普通文本的文件，可以使用 FileContentBlock 表示。

但文件输入功能在不同模型间差异较大。

例如，某些模型支持直接读取 PDF；另一些模型则需要开发者先用 Document Loader 提取文本，再将文本放入消息中。

因此，不能只因为 LangChain 支持 FileContentBlock，就认为任意聊天模型都能直接分析 PDF。

---

## 十六、content 与 content\_blocks

在 LangChain 消息对象中，两个字段容易混淆：

| 属性 | 用途 |
| --- | --- |
| `content` | 消息的原始或服务商相关内容表示 |
| `content_blocks` | LangChain 标准化后的内容块 |

例如：

## 实例

from langchain.messages import AIMessage

message = AIMessage(content="你好，我是 Python 助手。")

print("原始内容：", message.content)
print("标准内容块：", message.content\_blocks)

如果只是显示普通文本，通常直接读取 content 即可。

如果需要统一处理来自不同服务商的多模态数据，可以考虑使用 content\_blocks。

**注意：**不要假设所有 content 都是字符串。多模态模型可能返回内容块列表。

---

## 十七、RemoveMessage：删除消息

`RemoveMessage` 用于在支持消息删除的状态管理中移除指定消息。

它并不是普通用户聊天消息。

例如，LangGraph 的 messages 状态字段通常使用 `add_messages` Reducer。

如果需要删除某条已有消息，可以返回对应的 RemoveMessage。

### 实例：删除指定消息

## 实例

from langchain.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add\_messages

messages = [
    HumanMessage(content="旧问题", id="msg-001"),
    AIMessage(content="旧回答", id="msg-002")
]

updated = add\_messages(
    messages,
    [RemoveMessage(id="msg-001")]
)

for msg in updated:
    print(msg.id, msg.content)

输出：

```
msg-002 旧回答
```

这里是通过 add\_messages Reducer 删除消息，而不是通过普通 Python 列表自动实现删除。

**注意：**RemoveMessage 应与支持该语义的 Reducer 配合使用。对于不包含对应 ID 的消息，删除操作可能失败。

正式 Agent 还要注意消息依赖关系，例如不要只删除工具调用的 AIMessage，却保留依赖它的 ToolMessage。

---

## 十八、消息历史为什么不能直接截取？

很多初学者会这样截取历史：

```
messages = messages[-10:]
```

对于普通文本对话，这种方式有时可以工作。

但对包含工具调用的 Agent，可能产生无效的消息序列。

例如：

```
AIMessage(tool_calls=[...])

ToolMessage(tool_call_id="call_001")
```

如果恰好把 AIMessage 删除，只保留 ToolMessage，那么工具结果就失去了对应的调用请求。

因此，裁剪 Agent 历史时，应特别注意消息的合法顺序。

对于复杂的多轮工具调用，通常更适合使用 LangChain Middleware 或专门的对话压缩策略。

---

## 十九、trim\_messages()：裁剪消息历史

LangChain 提供 `trim_messages()`，可以根据指定预算裁剪消息列表。

它支持按 Token 数量或使用自定义计数方式控制长度。

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `messages` | 必填 | 待裁剪的消息列表 |
| `max_tokens` | 必填 | Token 或指定计数预算 |
| `token_counter` | 必填 | 模型、计数函数或 approximate |
| `strategy` | last | 保留最前面或最后面的消息 |
| `allow_partial` | False | 是否允许保留部分消息 |
| `start_on` | None | 限制裁剪后起始消息类型 |
| `end_on` | None | 限制裁剪后结束消息类型 |
| `include_system` | False | 是否保留第一条 SystemMessage |
| `text_splitter` | None | 部分裁剪时使用的文本切分器 |

### 实例：保留最近对话

## 实例

from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim\_messages
)

messages = [
    SystemMessage(content="你是 Python 助手。"),
    HumanMessage(content="什么是列表？"),
    AIMessage(content="列表可以存储多个元素。"),
    HumanMessage(content="什么是字典？"),
    AIMessage(content="字典使用键值对存储数据。"),
    HumanMessage(content="列表和字典有什么区别？")
]

trimmed = trim\_messages(
    messages,
    max\_tokens=80,
    strategy="last",
    token\_counter="approximate",
    include\_system=True,
    start\_on="human",
    end\_on=("human", "tool")
)

for msg in trimmed:
    print(msg.type, msg.content)

这里的 `token_counter="approximate"` 使用近似 Token 计数，适合需要快速估算上下文长度的场景。

它不保证与具体模型服务商的实际 Token 计费数量完全一致。

示例中的最终保留消息数量取决于实际计数结果。

---

## 二十、按消息数量裁剪

如果只是想保留固定数量的消息，也可以传入：

```
token_counter=len
```

这时 max\_tokens 实际上代表消息数量预算，而不是模型 Token 数。

### 实例

## 实例

trimmed = trim\_messages(
    messages,
    max\_tokens=4,
    strategy="last",
    token\_counter=len,
    include\_system=True,
    start\_on="human"
)

for msg in trimmed:
    print(msg.type, msg.content)

这种方式比较简单，但无法反映不同消息的实际 Token 长度。

例如一条 5000 字的消息与一条只有"你好"的消息，按消息数量计数时都算一条。

**建议：**生产环境中的上下文控制，应优先结合实际 Token 预算，而不是只依赖消息数量。

---

## 二十一、filter\_messages()：筛选消息

如果只想提取某类消息，可以使用 `filter_messages()`。

例如只查看用户消息。

### 实例：筛选 HumanMessage

## 实例

from langchain.messages import filter\_messages

messages = [
    HumanMessage(content="你好"),
    AIMessage(content="你好！"),
    HumanMessage(content="介绍 Python"),
    AIMessage(content="Python 是一种编程语言")
]

users = filter\_messages(
    messages,
    include\_types="human"
)

for msg in users:
    print(msg.content)

输出：

```
你好

介绍 Python
```

还可以按照消息 ID、名称或消息类型进行筛选。

例如：

```
filter_messages(messages, exclude_types="system")
```

用于排除 SystemMessage。

---

## 二十二、convert\_to\_messages()：转换消息格式

LangChain 可以使用多种方式表达聊天消息。

例如：

```
("user", "你好")



{"role": "user", "content": "你好"}



HumanMessage(content="你好")
```

`convert_to_messages()` 可以将支持的消息表示转换成消息对象。

### 实例

## 实例

from langchain.messages import convert\_to\_messages

raw\_messages = [
    ("system", "你是 Python 助手。"),
    {"role": "user", "content": "你好"},
    ("assistant", "你好！")
]

messages = convert\_to\_messages(raw\_messages)

for msg in messages:
    print(type(msg).\_\_name\_\_, msg.content)

输出：

```
SystemMessage 你是 Python 助手。

HumanMessage 你好

AIMessage 你好！
```

这在需要兼容不同历史消息格式时非常实用。

---

## 二十三、merge\_message\_runs()：合并连续消息

如果消息列表中连续存在多条相同类型的消息，可以使用 `merge_message_runs()` 合并。

### 实例

## 实例

from langchain.messages import merge\_message\_runs

messages = [
    HumanMessage(content="请介绍 Python。"),
    HumanMessage(content="要求适合零基础。"),
    AIMessage(content="Python 是一种编程语言。")
]

merged = merge\_message\_runs(messages)

for msg in merged:
    print(msg.type, msg.content)

这个函数可以合并相邻的同类消息。

但需要注意：

- 只处理可合并的连续消息。
- 并不是所有类型的消息都适合合并。
- ToolMessage 通常不会被合并，因为它具有独立的 tool\_call\_id。

因此，不要用合并函数随意修改尚未完成的工具调用消息序列。

---

## 二十四、MessageLikeRepresentation：快捷消息格式

LangChain 在许多模型和 Agent API 中支持多种消息输入格式。

常见写法：

| 格式 | 示例 |
| --- | --- |
| 字符串 | `"你好"` |
| 消息对象 | `HumanMessage(content="你好")` |
| 元组 | `("user", "你好")` |
| 字典 | `{"role": "user", "content": "你好"}` |

例如，直接调用 Chat Model：

```
model.invoke("你好")
```

或者：

```
model.invoke([

    ("system", "你是助手"),

    ("user", "你好")

])
```

调用 Agent：

```
agent.invoke({

    "messages": [

        {"role": "user", "content": "你好"}

    ]

})
```

对于简单开发，使用字典或元组比较方便。

如果需要直接读取和处理 message.id、tool\_calls、metadata 等属性，使用明确的消息对象通常更方便。

---

## 二十五、LangChain Messages 与对话记忆

Agent 多轮对话中的消息历史通常保存在 State 中。

例如：

```
{

    "messages": [

        HumanMessage(content="我正在学 Python"),

        AIMessage(content="好的"),

        HumanMessage(content="推荐学习路线")

    ]

}
```

LangGraph 默认的 messages 状态通常通过 `add_messages` Reducer 管理更新。

这意味着：

- 新的消息通常会被追加。
- 具有相同 ID 的消息可以替换已有消息。
- RemoveMessage 可以请求删除指定消息。

但仅仅创建 messages 列表，并不意味着 Agent 可以跨程序重启自动记住历史。

要实现多轮记忆，需要根据应用情况使用 Checkpointer。

如果需要长期保存用户偏好，还可以进一步配合 Store。

---

## 二十六、完整实例：Messages API 综合使用

下面通过一个可以直接运行的示例，演示消息创建、属性读取、工具调用格式、消息筛选以及消息裁剪。

文件名：`messages_api_demo.py`

安装：

```
pip install -U langchain langchain-core
```

本示例全部使用本地模拟消息，不需要 API Key。

## 完整实例

from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    RemoveMessage,
    trim\_messages,
    filter\_messages,
    convert\_to\_messages
)
from langgraph.graph.message import add\_messages

# ========== 1. 基本消息 ==========

messages = [
    SystemMessage(content="你是 Python 助手。", id="sys-001"),
    HumanMessage(content="你好", id="user-001"),
    AIMessage(content="你好！", id="ai-001"),
    HumanMessage(content="介绍 Python 列表。", id="user-002"),
    AIMessage(content="列表可以保存多个元素。", id="ai-002")
]

print("=== 基本消息 ===")
for msg in messages:
    print(msg.type, msg.content)

# ========== 2. ToolCall + ToolMessage ==========

tool\_request = AIMessage(
    content="",
    id="ai-tool-001",
    tool\_calls=[{
        "name": "get\_weather",
        "args": {"city": "杭州"},
        "id": "call-001",
        "type": "tool\_call"
    }]
)

tool\_result = ToolMessage(
    content="杭州模拟天气：晴，25°C",
    tool\_call\_id="call-001",
    name="get\_weather",
    id="tool-msg-001"
)

print("\n=== 工具调用 ===")
print(tool\_request.tool\_calls)
print(tool\_result.tool\_call\_id)

# ========== 3. 流式消息合并 ==========

chunk1 = AIMessageChunk(content="Python")
chunk2 = AIMessageChunk(content=" 很容易入门。")

merged = chunk1 + chunk2
print("\n=== Chunk 合并 ===")
print(merged.content)

# ========== 4. 消息裁剪 ==========

trimmed = trim\_messages(
    messages,
    max\_tokens=4,
    strategy="last",
    token\_counter=len,
    include\_system=True,
    start\_on="human"
)

print("\n=== 裁剪结果 ===")
for msg in trimmed:
    print(msg.type, msg.content)

# ========== 5. 消息筛选 ==========

users = filter\_messages(messages, include\_types="human")
print("\n=== 用户消息 ===")
for msg in users:
    print(msg.content)

# ========== 6. 消息转换 ==========

converted = convert\_to\_messages([
    ("user", "什么是字典？"),
    ("assistant", "字典通过键值对保存数据。")
])

print("\n=== 格式转换 ===")
for msg in converted:
    print(type(msg).\_\_name\_\_, msg.content)

# ========== 7. 消息删除 ==========

updated = add\_messages(
    messages,
    [RemoveMessage(id="user-001")]
)

print("\n=== 删除消息后 ===")
for msg in updated:
    print(msg.id, msg.content)

此程序不调用模型，因此无需联网即可测试上述消息对象和辅助函数。

对于具体的裁剪结果，应以当前安装的 LangChain 版本和配置为准。

---

## 二十七、常见问题

### 1、HumanMessage 和 role=user 有什么区别？

HumanMessage 是 LangChain 消息对象，user 是聊天模型 API 中的角色名称。

### 2、AIMessage 的 type 是 assistant 吗？

不是。AIMessage 的 type 通常是 ai，assistant 是对应的对话角色。

### 3、AIMessageChunk 的 type 是 ai 吗？

不是。Python 的 AIMessageChunk 使用自己的类型标识 AIMessageChunk。

### 4、ToolCall 和 ToolMessage 有什么区别？

ToolCall 是模型的工具调用请求；ToolMessage 是工具实际执行后的返回消息。

### 5、为什么 ToolMessage 需要 tool\_call\_id？

因为模型可能同时请求多个工具，需要通过 ID 将调用结果与原始请求对应。

### 6、bind\_tools() 会自动创建 ToolMessage 吗？

不会。bind\_tools() 主要让模型具备生成工具调用请求的能力，实际工具执行通常由应用代码或 Agent 处理。

### 7、content 一定是字符串吗？

不一定。多模态消息可能包含内容块列表。

### 8、TextContentBlock 和 PlainTextContentBlock 有什么区别？

前者用于普通文本，后者主要用于 TXT、Markdown 等纯文本文件数据。

### 9、ImageContentBlock 能用于任何 Chat Model 吗？

不能。模型必须支持图片输入，并且提供商集成能够处理对应内容格式。

### 10、AIMessageChunk 一定是一个 Token 吗？

不是。Chunk 表示流式增量，可能包含一个或多个 Token 的内容。

### 11、如何合并多个 AIMessageChunk？

可以使用 + 操作符，将同一条消息的多个 Chunk 聚合。

### 12、为什么 trim\_messages() 之后工具调用报错？

可能因为裁剪破坏了 AIMessage.tool\_calls 与对应 ToolMessage 的关系，需要保留有效消息序列。

### 13、token\_counter=len 表示按 Token 计数吗？

不是。它表示按消息数量计数。

### 14、RemoveMessage 会直接删除 Python 列表元素吗？

不会。它需要通过支持删除语义的消息 Reducer 处理。

### 15、如何只获取用户消息？

可以使用 filter\_messages(messages, include\_types="human")。

### 16、如何把字典转换为消息对象？

可以使用 convert\_to\_messages()。

### 17、Chat Model 和 Agent 使用的消息类型相同吗？

通常共享 LangChain Messages 体系，但 Agent 还会结合工具消息、状态更新和 Checkpointer 管理完整执行流程。

### 18、Messages 可以保存用户长期偏好吗？

消息历史可以提供会话上下文，但跨会话长期偏好更适合使用 Store 或业务数据库保存。

---

## 总结

Messages API 是 LangChain 1.x 的核心基础之一。

本章节介绍了：

- `HumanMessage`：用户消息。
- `AIMessage`：模型回复。
- `SystemMessage`：系统指令。
- `ToolMessage`：工具执行结果。
- `AIMessageChunk`：流式消息增量。
- `ToolCall`：模型工具调用请求。
- `RemoveMessage`：删除状态消息。
- `TextContentBlock`：普通文本。
- `ImageContentBlock`、`AudioContentBlock` 等：多模态数据。
- `content_blocks`：标准化内容块。
- `trim_messages()`：裁剪消息历史。
- `filter_messages()`：筛选消息。
- `convert_to_messages()`：转换消息格式。
- `merge_message_runs()`：合并连续消息。

最基本的聊天消息结构：

```
messages = [

    SystemMessage(content="你是助手"),

    HumanMessage(content="你好"),

    AIMessage(content="你好！")

]
```

工具调用的基本结构：

```
HumanMessage

      ↓

AIMessage(tool_calls)

      ↓

ToolMessage(tool_call_id)

      ↓

AIMessage(最终回答)
```

**核心原则：**LangChain Messages 不只是聊天文本容器，还承担角色区分、工具调用关联、流式增量、多模态内容表示和状态更新等任务。

理解消息结构后，才能更准确地使用 Agent、Middleware、Checkpointer、Streaming 和 Human-in-the-Loop。

## 参考文档

- [LangChain Messages 官方文档](https://docs.langchain.com/oss/python/langchain/messages)
- [LangChain Messages API Reference](https://reference.langchain.com/python/langchain/messages)
- [LangChain Core Messages](https://reference.langchain.com/python/langchain-core/messages)
- [ContentBlock API Reference](https://reference.langchain.com/python/langchain-core/messages/content)
- [trim\_messages API](https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain 对话记忆](https://docs.langchain.com/oss/python/langchain/short-term-memory)
