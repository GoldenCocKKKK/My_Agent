# LangChain 流式输出 Streaming

> 来源: https://www.runoob.com/langchain/langchain-streaming.html

---

# LangChain 流式输出 Streaming

流式输出（Streaming）可以让 AI 在生成回复的过程中逐步返回内容，而不是等整个任务完成后才一次性显示结果。

例如，ChatGPT 的打字效果就是常见的流式交互形式。

在 LangChain 中，不仅可以流式获取模型文本，还可以实时观察 Agent 的工具调用、节点更新和自定义任务进度。

本章节介绍 LangChain 1.x 的 `stream()`、`astream()` 和各种 `stream_mode`，最后使用 FastAPI 实现一个可供前端调用的 SSE 流式接口。

---

## 一、为什么需要流式输出？

使用 `invoke()` 时，程序通常需要等待模型或 Agent 完成执行后，才能获取完整结果。

如果 Agent 需要调用多个工具，用户在等待期间可能看不到任何进展。

流式输出可以在任务执行过程中持续发送事件，让用户更早看到模型文本、工具调用或任务状态。

| 方式 | 返回方式 | 适用场景 |
| --- | --- | --- |
| `invoke()` | 等待执行完成后返回结果 | 脚本、批处理、普通 API |
| `stream()` | 同步迭代流式事件 | 终端展示、执行追踪 |
| `ainvoke()` | 异步等待完整结果 | 异步 Web 应用 |
| `astream()` | 异步迭代流式事件 | SSE、WebSocket、实时聊天 |

**注意：**流式输出主要改善响应内容的可见性，不保证整个任务的总执行时间更短。

另外，Streaming 不一定每次返回一个 Token，实际返回的是消息片段或执行事件。

---

## 二、安装与环境准备

本教程使用 LangChain 1.x、LangGraph 1.1 或更新版本，并以 DeepSeek Flash 作为演示模型。

安装依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

在项目目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

初始化模型及 Agent：

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的中文编程助手。"
)

接下来的示例统一使用 `version="v2"`。

与旧版流式格式不同，v2 事件采用统一结构：

```
{

    "type": "messages",

    "ns": [],

    "data": (...)

}
```

其中：

- `type`：事件对应的流模式。
- `ns`：事件所属图或子图的命名空间。
- `data`：具体事件内容。

---

## 三、stream\_mode="messages"：实时输出模型文本

`stream_mode="messages"` 用于获取语言模型生成过程中的消息片段。

它非常适合实现 AI 聊天的打字效果。

在 v2 格式中，事件的 `data` 通常包含：

```
(message_chunk, metadata)
```

`message_chunk` 是消息片段，`metadata` 则包含节点来源等相关信息。

### 实例：终端流式输出

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的中文编程助手。"
)

print("实时流式输出：")

for event in agent.stream(
    {"messages": [{"role": "user", "content": "用一句话介绍菜鸟教程 RUNOOB。"}]},
    stream\_mode="messages",
    version="v2"
):
    if event["type"] != "messages":
        continue

    chunk, metadata = event["data"]
    if metadata.get("langgraph\_node") == "model" and chunk.text:
        print(chunk.text, end="", flush=True)

print()

如果模型正常返回内容，终端会逐步显示类似文本：

```
实时流式输出：

菜鸟教程 RUNOOB 是一个提供编程语言、开发工具和相关技术教程的在线学习网站。
```

与 `invoke()` 不同，这里通过 `for` 循环逐步接收消息。

`chunk.text` 用于提取消息片段中的文本，而不是直接打印整个 `content`。

**注意：**流式响应可能包含工具调用片段、推理内容或其他结构化内容，不能假设所有片段都是普通字符串。

---

## 四、理解 message\_chunk 和 metadata

除了显示文本，开发者还可以查看流式消息的类型和来源。

例如：

- 当前消息来自哪个节点。
- 消息片段属于哪种 Python 类型。
- 模型是否正在生成工具调用。

### 实例：查看消息元数据

## 实例

first\_info = None

for event in agent.stream(
    {"messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}]},
    stream\_mode="messages",
    version="v2"
):
    if event["type"] != "messages":
        continue

    chunk, metadata = event["data"]
    if first\_info is None and chunk.text:
        first\_info = {
            "text": chunk.text,
            "node": metadata.get("langgraph\_node"),
            "type": type(chunk).\_\_name\_\_
        }

if first\_info:
    print("文本片段：", first\_info["text"])
    print("来源节点：", first\_info["node"])
    print("消息类型：", first\_info["type"])

可能输出：

```
文本片段： 你好

来源节点： model

消息类型： AIMessageChunk
```

这里先完整消费事件流，再打印收集到的第一个非空文本片段。

不建议为了查看一个片段，就直接在流式循环中随意 `break`，因为这可能导致当前流没有被完整消费。

### AIMessageChunk 不等于一个 Token

`AIMessageChunk` 表示流式模型消息的一部分。

一次返回可能包含几个字符、一个词、一段文本，甚至包含工具调用参数片段。

因此，不应该按照"一个 Chunk 等于一个 Token"计算 Token 消耗。

---

## 五、stream\_mode="updates"：查看 Agent 执行过程

`updates` 模式用于查看 Agent 执行图中各节点产生的状态更新。

例如，一个课程查询 Agent 可能按照以下顺序运行：

```
用户提问

   ↓

model：请求调用 search_course

   ↓

tools：执行搜索

   ↓

model：生成最终回答
```

这非常适合在开发阶段观察 Agent 的工具执行过程。

**注意：**节点更新展示的是执行步骤，不等于模型内部的私有推理过程。

### 实例：追踪工具调用

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent

@tool
def search\_course(keyword: str) -> str:
    """按关键词查询模拟编程课程。"""
    courses = {"python": "Python3 基础教程", "html": "HTML 基础教程"}
    return courses.get(keyword.lower(), "未找到相关模拟课程")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course],
    system\_prompt="你是课程顾问。用户要求查询课程时使用工具，工具数据仅供演示。"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": "请用工具搜索 python 课程。"}]},
    stream\_mode="updates",
    version="v2"
):
    if event["type"] != "updates":
        continue

    for node, update in event["data"].items():
        print("节点：", node)
        if not isinstance(update, dict):
            continue

        for msg in update.get("messages", []):
            if msg.type == "ai" and msg.tool\_calls:
                print("工具请求：", [(c["name"], c["args"]) for c in msg.tool\_calls])
            elif msg.type == "tool":
                print("工具结果：", msg.name, msg.content)
            elif msg.type == "ai" and msg.text:
                print("AI 回答：", msg.text)

如果模型选择调用工具，执行日志可能类似：

```
节点： model

工具请求： [('search_course', {'keyword': 'python'})]



节点： tools

工具结果： search_course Python3 基础教程



节点： model

AI 回答：可以先学习 Python3 基础教程。
```

实际工具调用次数和输出顺序由模型决定，并不是固定执行三步。

此外，一个 `tools` 节点更新可能包含多个工具执行结果。

---

## 六、stream\_mode="values"：查看完整状态

`updates` 返回节点更新，而 `values` 用于观察运行过程中的完整 State。

例如，可以查看消息历史如何逐步增加。

### 实例

## 实例

for event in agent.stream(
    {"messages": [{"role": "user", "content": "请搜索 python 课程。"}]},
    stream\_mode="values",
    version="v2"
):
    if event["type"] != "values":
        continue

    messages = event["data"].get("messages", [])
    print("当前消息数：", len(messages))
    print("消息类型：", [msg.type for msg in messages])

如果模型进行了一次工具调用，消息历史可能依次包含：

```
['human']

['human', 'ai']

['human', 'ai', 'tool']

['human', 'ai', 'tool', 'ai']
```

但状态快照数量和消息数量并不保证固定。

对于需要观察全部对话内容或调试自定义 State 的应用，`values` 比 `updates` 更方便。

不过，完整 State 可能包含较多数据，因此生产环境不建议无条件将所有状态返回前端。

---

## 七、stream\_mode="custom"：自定义进度事件

除了模型文本和节点更新，LangChain 还允许开发者发送自定义事件。

例如，处理多个文件时，可以实时报告当前进度：

```
正在处理文件 1/3

正在处理文件 2/3

正在处理文件 3/3
```

可以通过 `get_stream_writer()` 在支持的图节点或工具执行上下文中发送自定义数据。

### 实例：工具发送模拟进度

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent
from langgraph.config import get\_stream\_writer

@tool
def process\_files(task: str) -> str:
    """模拟处理三个文件，并报告处理进度。"""
    writer = get\_stream\_writer()
    for i in range(1, 4):
        writer({"task": task, "current": i, "total": 3, "message": f"已处理 {i}/3"})
    return "三个模拟文件处理完成"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[process\_files],
    system\_prompt="你是文件处理助手。用户要求执行模拟处理时，请使用 process\_files。"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": "请使用工具模拟处理测试文件。"}]},
    stream\_mode="custom",
    version="v2"
):
    if event["type"] == "custom":
        print(event["data"])

如果模型调用了该工具，可能收到：

```
{'task': '测试文件', 'current': 1, 'total': 3, 'message': '已处理 1/3'}

{'task': '测试文件', 'current': 2, 'total': 3, 'message': '已处理 2/3'}

{'task': '测试文件', 'current': 3, 'total': 3, 'message': '已处理 3/3'}
```

这里没有处理真实文件，只是模拟进度。

自定义事件不仅可以从 Middleware 发送，工具和 LangGraph 节点也可以发送。

**注意：**`get_stream_writer()` 需要在受支持的运行时上下文中使用，不能假设直接调用普通 Python 函数时也存在流式写入器。

---

## 八、组合多个 stream\_mode

在实际 Agent 应用中，通常需要同时展示：

- 模型正在生成的文本。
- 当前执行到哪个节点。
- 调用了哪个工具。
- 工具是否完成。

可以组合多个流模式：

```
stream_mode=["messages", "updates", "custom"]
```

使用 v2 格式时，可以根据 `event["type"]` 区分不同事件。

### 实例：同时展示模型文本与工具结果

## 实例

for event in agent.stream(
    {"messages": [{"role": "user", "content": "请模拟处理测试文件。"}]},
    stream\_mode=["messages", "updates", "custom"],
    version="v2"
):
    mode, data = event["type"], event["data"]

    if mode == "messages":
        chunk, metadata = data
        if metadata.get("langgraph\_node") == "model" and chunk.text:
            print(chunk.text, end="", flush=True)

    elif mode == "updates":
        for node in data:
            print(f"\n[节点] {node}")

    elif mode == "custom":
        print(f"\n[进度] {data.get('message', data)}")

print()

这个例子复用了上一节的 `process_files` 工具及其 Agent。

可能观察到：

```
[节点] model

[进度] 已处理 1/3

[进度] 已处理 2/3

[进度] 已处理 3/3

[节点] tools

三个模拟文件处理完成

[节点] model
```

实际事件的数量、顺序和模型文本可能有所不同。

组合模式适合用于完整的 Agent 前端界面，但如果事件过多，也需要考虑前端渲染和网络传输成本。

---

## 九、异步流式输出 astream()

对于 FastAPI、WebSocket 等异步应用，可以使用 `astream()`。

它通过 `async for` 逐步接收事件，不需要阻塞等待全部结果。

### 实例：异步流式聊天

## 实例

import asyncio
from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程助手。"
)

async def main():
    full\_text = []

    async for event in agent.astream(
        {"messages": [{"role": "user", "content": "介绍 Python 的三个优点。"}]},
        stream\_mode="messages",
        version="v2"
    ):
        if event["type"] != "messages":
            continue

        chunk, metadata = event["data"]
        if metadata.get("langgraph\_node") == "model" and chunk.text:
            full\_text.append(chunk.text)
            print(chunk.text, end="", flush=True)

    print("\n\n完整回复字符数：", len("".join(full\_text)))

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(main())

这里使用列表收集片段，最后通过 `"".join()` 合并完整文本。

相比在循环中不断执行字符串拼接，这种方式更适合较长的流式响应。

**注意：**字符数量不等于 Token 数量。如果需要统计 API Token 消耗，应读取模型返回的使用量信息。

---

## 十、FastAPI 集成：实现 SSE 流式接口

SSE（Server-Sent Events）是一种基于 HTTP 的单向事件推送技术。

它非常适合将 AI 的流式文本持续推送到浏览器。

下面使用 FastAPI 创建一个简单的聊天接口：

```
GET /chat?message=你好
```

服务端通过 `StreamingResponse` 返回 `text/event-stream` 数据。

### 安装依赖

```
pip install -U fastapi uvicorn langchain langgraph langchain-deepseek python-dotenv
```

### 完整服务端代码

文件名：`main.py`

## 实例

import json
from dotenv import load\_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain.agents import create\_agent

load\_dotenv()

app = FastAPI()
agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的中文编程助手。"
)

def sse(event: str, data: dict) -> str:
    """生成合法的 SSE JSON 事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure\_ascii=False)}\n\n"

@app.get("/chat")
async def chat(message: str = Query(min\_length=1, max\_length=2000)):
    async def generate():
        try:
            async for item in agent.astream(
                {"messages": [{"role": "user", "content": message}]},
                stream\_mode="messages",
                version="v2"
            ):
                if item["type"] != "messages":
                    continue

                chunk, metadata = item["data"]
                if metadata.get("langgraph\_node") == "model" and chunk.text:
                    yield sse("token", {"text": chunk.text})

            yield sse("done", {"finished": True})
        except Exception:
            yield sse("error", {"message": "模型请求失败，请稍后重试"})

    return StreamingResponse(
        generate(),
        media\_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

启动服务：

```
uvicorn main:app --reload
```

然后访问：

```
http://127.0.0.1:8000/chat?message=你好
```

服务端可能持续发送：

```
event: token

data: {"text": "你好"}



event: token

data: {"text": "！欢迎"}



event: token

data: {"text": "学习 Python。"}



event: done

data: {"finished": true}
```

这里没有直接将文本拼接到 `data:` 后面，而是先编码为 JSON。

这样可以避免模型输出包含换行、特殊字符时破坏 SSE 消息格式。

**注意：**示例适用于本地开发。生产环境还需要加入 API 鉴权、请求限流、并发控制和错误日志，避免公开接口被滥用。

### 浏览器接收 SSE

前端可以使用原生 `EventSource`，不需要额外安装 JavaScript 库。

## 实例

<div id="output"></div>
<script>
const output = document.getElementById("output");
const url = "http://127.0.0.1:8000/chat?message=" + encodeURIComponent("介绍 Python");
const source = new EventSource(url);

source.addEventListener("token", (event) => {
    const data = JSON.parse(event.data);
    output.textContent += data.text;
});

source.addEventListener("done", () => source.close());
source.addEventListener("error", () => {
    source.close();
    console.error("SSE 连接结束或发生错误");
});
</script>

这段代码通过 `textContent` 追加文本，而不是将模型输出直接写入 `innerHTML`，可以避免将模型内容当作 HTML 执行。

如果前端与 FastAPI 服务使用不同的域名或端口，还需要正确配置 CORS。

对于生产应用，建议通过受控的 HTTPS API 提供 SSE 服务，并考虑反向代理的响应缓冲设置。

---

## 十一、stream\_mode 速查表

| 模式 | 返回内容 | 典型用途 |
| --- | --- | --- |
| `messages` | 模型消息片段与元数据 | 实时聊天、打字效果 |
| `updates` | 节点执行后的状态更新 | 工具调用追踪 |
| `values` | 完整 State 快照 | 状态调试 |
| `custom` | 工具或节点发送的自定义数据 | 处理进度、状态提示 |
| `debug` | 更详细的图执行信息 | 开发排障 |
| `checkpoints` | 检查点创建事件 | 检查状态保存过程 |
| `tasks` | 任务开始、结束及相关结果 | 深入观察图执行 |

对于大多数 Agent 应用，首先掌握以下三个模式即可：

```
messages：显示模型生成的文本

updates：显示 Agent 执行步骤

custom：显示业务处理进度
```

这些模式也可以组合使用。

---

## 十二、常见问题

### 1、Streaming 是否意味着模型生成得更快？

不一定。Streaming 主要让用户在模型继续生成时，就能看到已产生的内容。

它可以缩短首次可见响应的等待时间，但不一定缩短整个任务的执行时间。

### 2、为什么 messages 返回的不是完整句子？

因为模型正在分段输出。每个消息片段可能只包含部分文本。

如果需要完整内容，可以收集片段，并在流结束后合并。

### 3、AIMessageChunk 是否等于一个 Token？

不是。Chunk 与 Token 不一定一一对应，不能据此精确计算 Token 消耗。

### 4、为什么 chunk.content 不是字符串？

因为多模态和结构化响应可能使用内容块列表。

对于普通文本展示，可以使用当前版本支持的 `chunk.text` 属性提取文本内容。

### 5、updates 与 messages 有什么区别？

`updates` 主要用于观察节点执行和工具结果，`messages` 主要用于接收模型生成过程中的消息片段。

### 6、如何同时显示工具调用和 AI 打字效果？

可以使用：

```
stream_mode=["messages", "updates"]
```

通过事件类型分别处理模型文本与工具执行信息。

### 7、custom 是否只能在 Middleware 中使用？

不是。工具和 LangGraph 节点也可以通过 StreamWriter 发送自定义事件。

### 8、为什么 EventSource 收不到消息？

应检查服务端是否返回 `text/event-stream`，事件格式是否正确，以及是否存在 CORS、反向代理缓冲或网络连接问题。

### 9、为什么 SSE 不能直接使用字符串拼接？

因为模型文本可能包含换行等特殊内容。

推荐将每个文本片段编码为 JSON，再放入 SSE 的 `data:` 字段。

### 10、FastAPI StreamingResponse 是否支持 POST？

支持。但浏览器原生 `EventSource` 主要使用 GET 请求。

如果需要通过 POST 发送较长提示词或复杂消息，可以使用 `fetch()` 读取流式响应，或者使用支持相关能力的客户端。

### 11、流式输出可以查看模型完整思考过程吗？

不能简单这样理解。

`updates` 可以显示工具请求和节点执行信息，但不代表能够获取模型的内部私有推理过程。

部分模型可能提供公开的推理摘要或特定推理内容块，是否可见取决于模型及服务接口。

### 12、生产环境怎样处理用户断开连接？

需要考虑请求取消、后端资源释放、超时和错误处理。

不能假设浏览器关闭页面后，所有正在运行的工具都会立即停止。

对于具有副作用的工具，还需要设计幂等性和任务状态管理，防止断线重试造成重复操作。

---

## 总结

Streaming 是 LangChain Agent 中非常重要的交互能力。

本章节学习了：

- `stream()`：同步接收流式事件。
- `astream()`：异步接收流式事件。
- `messages`：流式接收模型文本。
- `updates`：查看 Agent 的执行过程。
- `values`：查看完整状态快照。
- `custom`：发送业务进度事件。
- `version="v2"`：统一流式事件格式。
- FastAPI + SSE：将流式内容推送到浏览器。

最常用的流式代码如下：

```
for event in agent.stream(

    inputs,

    stream_mode="messages",

    version="v2"

):

    if event["type"] == "messages":

        chunk, metadata = event["data"]

        print(chunk.text, end="", flush=True)
```

如果需要实现完整的 Agent 前端，推荐组合 `messages`、`updates` 和 `custom`，分别展示模型文本、工具执行以及业务处理进度。

> **核心原则：**流式输出不仅是让 AI 的文字逐步显示，更重要的是让 Agent 的执行进展能够被实时观察和展示。
