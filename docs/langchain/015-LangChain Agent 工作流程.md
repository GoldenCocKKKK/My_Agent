# LangChain Agent 工作流程

> 来源: https://www.runoob.com/langchain/langchain-agent-workflow.html

---

# LangChain Agent 工作流程

上一节我们学习了 [create\_agent()](https://www.runoob.com/langchain/langchain-create_agent.html) 函数的参数及基本用法。

本章节将进一步了解 Agent 的内部执行机制：模型如何选择工具、工具结果如何返回模型、Agent 在什么条件下结束，以及如何通过流式接口观察整个执行过程。

在 LangChain 1.x 中，`create_agent()` 基于 LangGraph 构建 Agent 执行图，能够自动管理模型调用、工具执行和状态更新。

本教程继续使用 DeepSeek Flash 模型，并使用模拟天气、模拟时间和课程查询工具进行演示。

**注意：**本章使用 LangGraph 1.1 及以上版本支持的 `version="v2"` 流式事件格式。旧版 LangGraph 的流式返回结构可能不同。

---

## 一、Agent 的基本执行循环

Agent 的核心机制可以概括为：

```
调用模型 → 判断工具调用 → 执行工具 → 再次调用模型
```

当模型返回工具调用请求时，Agent 执行对应工具，并将结果加入消息历史。

随后，模型可以根据工具结果继续处理任务。

整个过程通常持续到模型不再请求工具，或者遇到其他停止、暂停或错误条件。

Agent 的基本流程如下：

![LangChain Agent 模型与工具之间的执行循环](https://www.runoob.com/wp-content/uploads/2026/05/cae926b2-024e-4c62-b83b-1c90ac07075f.png)

### Model 节点与 Tools 节点

一个典型的 Agent 可以抽象成两个核心节点：

| 节点 | 作用 |
| --- | --- |
| `model` | 调用语言模型，生成回复或工具调用请求 |
| `tools` | 执行模型请求的客户端工具，并返回工具结果 |

例如，用户提问：

```
杭州的模拟天气和模拟时间分别是什么？
```

模型可能生成两个工具调用：

```
get_weather(city="杭州")

get_time(city="杭州")
```

工具执行完成后，结果会作为 `ToolMessage` 返回，模型再生成最终回答。

**注意：**模型可能一次请求多个工具，也可能分多轮调用工具。因此，执行图的节点顺序和次数并不固定。

---

## 二、安装与环境准备

本教程使用 LangChain 1.x，并采用 LangGraph v2 流式格式。

安装或更新依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

使用 `version="v2"` 需要 LangGraph 1.1 或更新版本。

项目目录中的 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

创建模型时使用：

```
model = init_chat_model(

    "deepseek:deepseek-flash"

)
```

实际调用前需要确认已正确配置 API Key，并且对应模型支持工具调用。

---

## 三、追踪 Agent 的完整执行过程

接下来定义两个工具：

- `get_weather()`：返回模拟天气。
- `get_time()`：返回模拟时间。

为了方便观察，两个工具都使用固定字典数据，不会调用真实的天气或时间接口。

### 实例：使用 updates 追踪执行节点

## 实例

# 文件：agent\_workflow.py

from dotenv import load\_dotenv

from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 工具一：模拟天气
@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。"""

    weather\_data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C"
    }

    return weather\_data.get(
        city,
        f"未找到{city}的模拟天气数据"
    )

# 工具二：模拟时间
@tool
def get\_time(city: str) -> str:
    """查询指定城市的模拟时间。"""

    time\_data = {
        "杭州": "14:30",
        "北京": "14:30",
        "纽约": "02:30"
    }

    return time\_data.get(
        city,
        f"未找到{city}的模拟时间数据"
    )

# 初始化模型
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 创建 Agent
agent = create\_agent(
    model=model,
    tools=[get\_weather, get\_time],
    system\_prompt=(
        "你是一名中文助手。"
        "天气和时间工具中的数据都是模拟数据。"
        "用户查询这类数据时应使用工具。"
        "不要把模拟数据当作实时信息。"
    )
)

# 用户消息
inputs = {
    "messages": [
        {
            "role": "user",
            "content": (
                "请分别查询杭州的模拟天气"
                "和模拟时间。"
            )
        }
    ]
}

# 追踪节点更新
print("=== Agent 执行过程 ===")

event\_count = 0

for event in agent.stream(
    inputs,
    stream\_mode="updates",
    version="v2"
):
    if event["type"] != "updates":
        continue

    event\_count += 1

    print(f"\n--- 更新事件 {event\_count} ---")

    # v2 事件数据
    updates = event["data"]

    for node\_name, update in updates.items():

        print("节点：", node\_name)

        if not isinstance(update, dict):
            print("节点数据：", update)
            continue

        messages = update.get("messages", [])

        for message in messages:

            if message.type == "ai" and message.tool\_calls:

                for call in message.tool\_calls:
                    print(
                        "[请求工具]",
                        call["name"],
                        call["args"]
                    )

            elif message.type == "tool":
                print(
                    "[工具结果]",
                    message.name,
                    message.content
                )

            elif message.type == "ai":
                print(
                    "[模型回复]",
                    message.text
                )

### 可能的执行过程

如果模型在同一轮请求两个工具，日志可能类似：

```
=== Agent 执行过程 ===



--- 更新事件 1 ---

节点： model

[请求工具] get_weather {'city': '杭州'}

[请求工具] get_time {'city': '杭州'}



--- 更新事件 2 ---

节点： tools

[工具结果] get_weather 晴，25°C

[工具结果] get_time 14:30



--- 更新事件 3 ---

节点： model

[模型回复] 杭州的模拟天气为晴天、25°C，

模拟时间为14:30。
```

这里包含三次节点更新：

1. **Model：**模型生成两个工具调用请求。
2. **Tools：**工具执行，返回模拟天气和时间。
3. **Model：**模型读取工具结果并生成回答。

但是，**三次更新只是其中一种可能的执行路径**。

模型也可能先调用天气工具，下一轮再调用时间工具，这样就会产生更多节点更新。

如果多个工具在同一轮执行，工具结果也不一定按照模型请求的顺序完成。

---

## 四、理解 Agent 的消息变化

除了观察节点，还可以查看执行过程中产生的消息。

一次典型的工具调用包含以下消息：

```
HumanMessage

    ↓

AIMessage(tool_calls=[...])

    ↓

ToolMessage

    ↓

AIMessage(最终回答)
```

如果模型一次请求两个工具，消息可能为：

```
HumanMessage

    ↓

AIMessage（两个 tool_calls）

    ↓

ToolMessage（天气）

ToolMessage（时间）

    ↓

AIMessage（最终回答）
```

其中：

| 消息类型 | 内容 |
| --- | --- |
| `HumanMessage` | 用户输入 |
| `AIMessage` | 模型回复或工具调用请求 |
| `ToolMessage` | 工具执行结果 |

注意，包含工具调用的 AIMessage，其普通文本内容可能为空，但 `tool_calls` 中仍然有工具请求。

因此，观察 Agent 工作流程时，应同时检查：

```
message.content

message.tool_calls

message.type
```

不要简单根据 `content` 是否为空，判断模型是否调用了工具。

---

## 五、stream\_mode 参数详解

`agent.stream()` 支持不同的流式模式，用于获取不同粒度的数据。

| 模式 | 返回内容 | 适用场景 |
| --- | --- | --- |
| `updates` | 节点执行后的状态更新 | 观察 Agent 执行步骤 |
| `values` | 图运行过程中的完整状态快照 | 检查消息历史和状态字段 |
| `messages` | 语言模型的消息片段及元数据 | 聊天流式输出 |
| `custom` | 工具或图节点发送的自定义数据 | 进度提示和任务状态 |
| `debug` | 图执行相关的调试信息 | 深入排查执行过程 |

`updates`、`messages` 和 `custom` 是 LangChain Agent 官方入门文档重点介绍的模式；`values` 和 `debug` 等也可参考 LangGraph 流式文档。

使用 `version="v2"` 时，流事件采用统一结构：

```
{

    "type": "updates",

    "ns": [],

    "data": {

        "model": {

            "messages": [...]

        }

    }

}
```

其中：

- `type`：当前流式模式。
- `ns`：事件所属的图或子图命名空间。
- `data`：具体事件内容。

不同 stream\_mode 的 `data` 内容不同。

---

## 六、stream\_mode="updates"：查看节点更新

`updates` 用于观察 Agent 的执行步骤。

每当相关节点产生状态更新时，就可以收到相应事件。

### 实例

## 实例

for event in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请查询杭州的模拟天气。"
            }
        ]
    },
    stream\_mode="updates",
    version="v2"
):

    if event["type"] != "updates":
        continue

    for node\_name, update in event["data"].items():

        print("节点：", node\_name)

        if isinstance(update, dict):
            for message in update.get("messages", []):
                print(
                    message.type,
                    message.content
                )

如果模型调用天气工具，可能观察到：

```
节点： model

ai



节点： tools

tool 晴，25°C



节点： model

ai 杭州的模拟天气为晴天，气温25°C。
```

这种模式适合调试 Agent 工具调用，以及在后台展示当前执行到哪个节点。

**注意：**一个节点更新事件不一定等于一次工具调用。一个 Tools 节点更新中可能包含多条工具结果。

---

## 七、stream\_mode="values"：查看完整状态

`updates` 主要显示节点产生的变化，而 `values` 用于观察运行过程中的完整状态。

例如，可以查看每次更新后的消息数量。

### 实例

## 实例

inputs = {
    "messages": [
        {
            "role": "user",
            "content": "查询杭州的模拟天气。"
        }
    ]
}

snapshot\_count = 0

for event in agent.stream(
    inputs,
    stream\_mode="values",
    version="v2"
):

    if event["type"] != "values":
        continue

    snapshot\_count += 1

    state = event["data"]

    messages = state.get("messages", [])

    print(
        f"\n状态快照 {snapshot\_count}："
        f"{len(messages)} 条消息"
    )

    for message in messages:

        print(
            f"[{message.type}]",
            str(message.content)[:80]
        )

如果工具调用流程是：

```
用户 → 模型请求工具 → 工具返回 → 模型回答
```

那么观察到的消息数量可能依次增长：

```
状态快照 1：1 条消息

状态快照 2：2 条消息

状态快照 3：3 条消息

状态快照 4：4 条消息
```

但具体快照数量和消息数量会受到实际执行路径影响，不能将其视为固定结果。

**注意：**不要在尚未执行完成时，仅仅为了减少打印次数就随意 `break` 退出流式循环。

提前停止消费事件可能使执行过程、资源清理或后续结果处理变得不完整。教学示例建议消费完整个事件流。

---

## 八、stream\_mode="messages"：流式生成文本

`messages` 用于获取语言模型生成过程中的消息片段。

它适合制作类似 ChatGPT 的流式打字效果。

需要注意，流式返回的并不一定是一个完整 Token，也不一定只包含普通文本。

模型可能产生：

- 文本消息片段。
- 工具调用参数片段。
- 推理内容块。
- 响应元数据。

### 实例：逐步输出模型文本

## 实例

from langchain.agents import create\_agent

# 不注册工具，便于观察普通文本输出
text\_agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[]
)

print("=== 流式输出 ===")

for event in text\_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "用一句话介绍菜鸟教程 RUNOOB。"
                )
            }
        ]
    },
    stream\_mode="messages",
    version="v2"
):

    if event["type"] != "messages":
        continue

    message\_chunk, metadata = event["data"]

    # 只显示模型节点输出的文本
    if metadata.get("langgraph\_node") == "model":
        print(
            message\_chunk.text,
            end="",
            flush=True
        )

print()

代码中的 `message_chunk` 通常是 `AIMessageChunk`。

`metadata` 中可以包含当前节点名称等信息。

由于不同模型的流式响应格式可能不同，`message_chunk.text` 可能为空。

如果模型返回工具调用、思考内容等非普通文本块，还需要根据对应的数据结构进行处理。

### 流式工具调用

如果 Agent 注册了工具，模型在流式输出期间还可能逐步返回工具调用参数。

例如：

```
工具名称：get_weather

参数片段：{"city":

参数片段："杭州"

参数片段：}
```

这些片段不一定是完整的 JSON。

需要等待必要的消息片段合并后，才能将工具调用作为完整请求理解。

因此，在展示 Agent 过程时，通常建议：

- 使用 `messages` 显示模型生成中的文本。
- 使用 `updates` 显示已经完成的工具调用和工具结果。

---

## 九、stream\_mode="custom"：自定义进度事件

除了模型和图节点产生的事件，开发者还可以从工具内部发送自定义进度。

例如，一个需要处理大量文件的工具，可以持续报告当前处理数量。

LangChain 可以通过 `get_stream_writer()` 获取流式写入器。

### 实例：工具发送进度提示

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent

from langgraph.config import get\_stream\_writer

@tool
def process\_documents(task\_name: str) -> str:
    """模拟处理文档并发送进度更新。"""

    writer = get\_stream\_writer()

    for i in range(1, 4):

        writer({
            "task": task\_name,
            "current": i,
            "total": 3,
            "message": f"已处理 {i}/3 个文档"
        })

    return "三个模拟文档处理完成"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[process\_documents],
    system\_prompt=(
        "你是一名文档处理助手。"
        "用户要求执行模拟文档处理时，"
        "调用 process\_documents。"
    )
)

for event in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "请使用工具模拟处理测试文档。"
                )
            }
        ]
    },
    stream\_mode="custom",
    version="v2"
):

    if event["type"] == "custom":
        print(event["data"])

如果模型调用工具，可以观察到类似进度：

```
{'task': '测试文档', 'current': 1, 'total': 3, 'message': '已处理 1/3 个文档'}

{'task': '测试文档', 'current': 2, 'total': 3, 'message': '已处理 2/3 个文档'}

{'task': '测试文档', 'current': 3, 'total': 3, 'message': '已处理 3/3 个文档'}
```

这里没有真正读取文件，只是通过循环模拟进度。

**注意：**`get_stream_writer()` 需要适当的 LangGraph 执行上下文。直接调用包含它的工具函数时，可能无法获得对应的流式写入器。

真实项目可以将自定义事件发送给前端，用于展示上传进度、数据处理进度或任务阶段。

---

## 十、同时使用多个 stream\_mode

如果需要同时观察节点更新和模型文本，可以传入多个模式。

例如：

```
stream_mode=["messages", "updates"]
```

在 v2 格式中，不同事件都使用同样的外层结构，可以通过 `event["type"]` 区分。

### 实例：同时观察工具调用与文本输出

## 实例

inputs = {
    "messages": [
        {
            "role": "user",
            "content": "查询杭州的模拟天气。"
        }
    ]
}

for event in agent.stream(
    inputs,
    stream\_mode=[
        "messages",
        "updates"
    ],
    version="v2"
):

    mode = event["type"]

    if mode == "messages":

        message\_chunk, metadata = event["data"]

        if metadata.get("langgraph\_node") == "model":

            text = message\_chunk.text

            if text:
                print(
                    text,
                    end="",
                    flush=True
                )

    elif mode == "updates":

        updates = event["data"]

        for node\_name, update in updates.items():

            if not isinstance(update, dict):
                continue

            for message in update.get("messages", []):

                if message.type == "tool":
                    print(
                        f"\n[工具 {message.name}] "
                        f"{message.content}"
                    )

print()

此示例假设当前程序中的 `agent` 是前面定义的天气 Agent。

如果使用不包含天气工具的其他 Agent，应替换为对应的工具及输入。

另外，使用多个流式模式时，不要混淆事件数据与完整执行状态。

`messages` 提供模型生成过程中的片段，`updates` 提供节点完成后的更新。

---

## 十一、Agent 在什么情况下停止？

通常情况下，当模型不再返回工具调用请求时，Agent 会结束当前工具调用循环。

但这不是唯一的停止条件。

| 条件 | 行为 |
| --- | --- |
| 没有工具调用 | 模型生成最终回答，正常结束 |
| `return_direct` | 满足直接返回条件时，在工具执行后结束 |
| 结构化输出完成 | 满足结构化输出策略的完成条件后结束 |
| 中间件结束执行 | 中间件可根据受支持的执行控制机制终止循环 |
| 达到图执行限制 | 可能触发相应异常或限制处理流程 |
| 执行中断 | 暂停并等待恢复，不等于正常完成 |
| 模型或工具错误 | 可能抛出异常或进入错误处理流程 |

### 1、模型不再调用工具

这是最常见的正常结束条件。

例如，模型收到问题后直接生成回答：

```
AIMessage(

    content="Python 是一种通用编程语言。",

    tool_calls=[]

)
```

由于没有工具调用请求，Agent 不需要再进入 Tools 节点。

### 2、return\_direct=True

在支持该行为的 Agent 中，工具设置 `return_direct=True` 后，可以在满足直接返回条件时结束循环。

需要注意，当前 `create_agent()` 对同一批客户端工具调用会综合判断其直接返回设置。

不能简单认为一批工具里只要有一个设置了 `return_direct=True`，就一定立即结束。

此外，直接返回时最后一条消息可能是 `ToolMessage`。

### 3、结构化输出完成

如果 `create_agent()` 配置了 `response_format`，Agent 可以在满足对应结构化输出策略时结束。

结构化数据通常通过：

```
result["structured_response"]
```

获取。

不能认为只要某条模型回复看起来像 JSON，Agent 就一定已经完成结构化输出流程。

### 4、中间件控制结束

LangChain 中间件可以在相应的执行钩子中控制 Agent 后续流程。

例如，中间件可以根据执行状态决定结束当前运行。

但这需要使用当前 LangChain 版本支持的中间件状态更新或路由机制，不应通过修改普通变量实现。

### 5、执行限制

如果 Agent 一直请求工具，可能造成循环次数过多。

可以通过 `recursion_limit` 设置图的执行步数限制。

但达到限制时不代表任务正常完成，可能抛出 `GraphRecursionError`。

### 6、中断与暂停

Agent 还可以通过 LangGraph 的中断机制暂停执行。

例如，在工具执行前等待用户批准。

**暂停并不等于结束。**如果需要继续完成任务，还需要按照对应的恢复流程执行。

---

## 十二、实例：比较无工具与有工具的执行流程

下面通过两个 Agent 对比消息历史的变化。

### 实例

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent

# 无工具 Agent
agent\_without\_tools = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[]
)

@tool
def search\_course(keyword: str) -> str:
    """查询指定关键词的模拟课程。"""

    data = {
        "python": "Python 基础、Python 数据分析",
        "java": "Java 基础、Java 面向对象"
    }

    return data.get(
        keyword.lower(),
        "没有找到相关模拟课程"
    )

# 有工具 Agent
agent\_with\_tools = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course],
    system\_prompt=(
        "你是编程课程助手。"
        "查询课程时使用工具。"
        "工具结果是模拟数据。"
    )
)

def show\_messages(agent, question):

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    print("\n问题：", question)
    print("消息总数：", len(result["messages"]))

    for message in result["messages"]:

        if message.type == "ai" and message.tool\_calls:

            print(
                "[AI 工具调用]",
                [
                    call["name"]
                    for call in message.tool\_calls
                ]
            )

        elif message.type == "tool":

            print(
                "[工具结果]",
                message.name,
                message.content
            )

        else:

            print(
                f"[{message.type}]",
                str(message.content)[:100]
            )

# 不使用工具
show\_messages(
    agent\_without\_tools,
    "用一句话介绍 Python。"
)

# 使用工具
show\_messages(
    agent\_with\_tools,
    "请使用工具搜索 python 课程。"
)

无工具场景一般包含用户消息与模型回复。

有工具场景如果模型发起一次工具调用，通常会包含：

```
human

ai（包含 tool_calls）

tool

ai（最终回答）
```

但是，如果模型调用多个工具或没有选择工具，消息数量也会发生变化。

因此，不能将消息总数作为判断工具执行是否成功的唯一标准。

更可靠的方式是检查对应的 `tool_calls`、`ToolMessage` 和最终状态。

---

## 十三、限制 Agent 的执行步数

Agent 可能连续调用多个工具，如果工具结果不满足任务需要，就可能不断循环。

LangGraph 提供 `recursion_limit`，用于限制图执行步数。

### 实例

## 实例

from langgraph.errors import GraphRecursionError

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请查询杭州的模拟天气和模拟时间。"
                    )
                }
            ]
        },
        config={
            "recursion\_limit": 15
        }
    )

    print(result["messages"][-1].content)

except GraphRecursionError:
    print(
        "Agent 达到执行步数限制，"
        "任务未正常完成。"
    )

这里的 `recursion_limit=15` 表示图执行的步数预算，并不是严格限制最多调用 15 次工具。

实际应用中，还应根据业务情况限制：

- 模型调用次数。
- 工具调用次数。
- 请求超时时间。
- API Token 消耗。
- 最大任务执行时长。

如果任务达到了限制，应明确向用户说明没有完成，而不是将部分结果伪装为成功结果。

---

## 十四、invoke() 与 stream() 的区别

LangChain Agent 同时支持同步和异步调用，以及完整结果和流式输出。

| 方法 | 返回方式 | 主要场景 |
| --- | --- | --- |
| `invoke()` | 同步等待执行结果 | 普通脚本、简单后端接口 |
| `ainvoke()` | 异步等待执行结果 | 异步 Web 服务 |
| `stream()` | 同步迭代流式事件 | 执行追踪、实时输出 |
| `astream()` | 异步迭代流式事件 | SSE、WebSocket、异步服务 |

选择哪一种方式，取决于应用是否需要实时展示执行过程，以及程序是否使用异步架构。

### 实例：异步流式调用

## 实例

import asyncio

from langchain.agents import create\_agent

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[]
)

async def main():

    async for event in agent.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "介绍 Python 的三个优点。"
                }
            ]
        },
        stream\_mode="messages",
        version="v2"
    ):

        if event["type"] != "messages":
            continue

        message\_chunk, metadata = event["data"]

        if metadata.get("langgraph\_node") == "model":
            print(
                message\_chunk.text,
                end="",
                flush=True
            )

    print()

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(main())

这种方式适合与 FastAPI 等异步 Web 框架配合，实现流式聊天接口。

但 `astream()` 本身只是异步事件迭代器，并不会自动创建 SSE 或 WebSocket 服务。还需要由 Web 框架完成事件传输。

---

## 十五、thread\_id：管理不同对话线程

如果希望 Agent 在多次调用之间保留对话历史，可以结合 Checkpointer 使用 `thread_id`。

`thread_id` 用于区分不同会话。

需要注意，\*\*仅传入 thread\_id 并不能自动保存对话历史\*\*，还需要配置 Checkpointer。

### 实例：保存同一会话的历史

## 实例

from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    checkpointer=InMemorySaver(),
    system\_prompt="你是一名编程学习助手。"
)

config = {
    "configurable": {
        "thread\_id": "conversation-001"
    }
}

# 第一轮对话
agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我正在学习 Python。"
            }
        ]
    },
    config=config
)

# 第二轮对话
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我刚才说在学习什么？"
            }
        ]
    },
    config=config
)

print(result["messages"][-1].content)

由于两次调用使用相同 `thread_id`，第二轮可以访问 Checkpointer 保存的此前状态。

`InMemorySaver` 只在当前 Python 进程中保存数据，程序退出后通常会丢失。

真实项目如果需要长期保存对话，应使用数据库支持的 Checkpointer。

### stream() 同样支持 thread\_id

## 实例

# 使用同一个已配置 Checkpointer 的 Agent

for event in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "再给我推荐一个 Python 学习方向。"
            }
        ]
    },
    config=config,
    stream\_mode="updates",
    version="v2"
):

    if event["type"] != "updates":
        continue

    print(event["data"])

使用 `invoke()` 和 `stream()` 都可以传入 `config`。

如果希望预先绑定配置，也可以使用：

```
session_agent = agent.with_config(

    configurable={

        "thread_id": "conversation-001"

    }

)
```

之后通过 `session_agent.invoke()` 或 `session_agent.stream()` 执行调用。

**注意：**在 Web 应用中，不同用户或会话必须使用经过应用层管理的线程标识，并实现权限隔离，避免一个用户读取另一个用户的聊天记录。

---

## 十六、完整实例：同时追踪工具与最终回答

下面将前面内容整合为一个完整的 Agent 工作流程观察程序。

文件名：`agent_workflow_demo.py`

## 完整实例

# 文件：agent\_workflow\_demo.py

from dotenv import load\_dotenv

from langchain.tools import tool
from langchain.agents import create\_agent
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# -------------------------
# 模拟工具
# -------------------------

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气。"""

    data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C"
    }

    return data.get(
        city,
        "没有该城市的模拟天气"
    )

@tool
def get\_time(city: str) -> str:
    """查询指定城市的模拟时间。"""

    data = {
        "杭州": "14:30",
        "北京": "14:30"
    }

    return data.get(
        city,
        "没有该城市的模拟时间"
    )

# -------------------------
# 创建 Agent
# -------------------------

model = init\_chat\_model(
    "deepseek:deepseek-flash",
    timeout=30,
    max\_retries=2
)

agent = create\_agent(
    model=model,
    tools=[get\_weather, get\_time],
    system\_prompt=(
        "你是一名中文助手。"
        "工具提供的数据都是模拟数据。"
        "遇到天气或时间查询时使用工具。"
        "不要声称模拟数据是实时数据。"
    )
)

# -------------------------
# 追踪执行过程
# -------------------------

def trace\_agent(question: str):

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }

    final\_ai\_message = None

    print("\n问题：", question)
    print("=" \* 50)

    for event in agent.stream(
        inputs,
        stream\_mode="updates",
        version="v2",
        config={
            "recursion\_limit": 15
        }
    ):

        if event["type"] != "updates":
            continue

        for node\_name, update in event["data"].items():

            print("\n执行节点：", node\_name)

            if not isinstance(update, dict):
                continue

            for message in update.get("messages", []):

                if message.type == "ai":

                    if message.tool\_calls:

                        for call in message.tool\_calls:
                            print(
                                "[工具请求]",
                                call["name"],
                                call["args"]
                            )

                    else:
                        final\_ai\_message = message

                elif message.type == "tool":

                    print(
                        "[工具结果]",
                        message.name,
                        message.content
                    )

    print("\n=== 最终回答 ===")

    if final\_ai\_message is not None:
        print(final\_ai\_message.content)
    else:
        print(
            "没有观察到完整的最终 AI 回复，"
            "请检查执行过程。"
        )

if \_\_name\_\_ == "\_\_main\_\_":

    trace\_agent(
        "请查询杭州的模拟天气和模拟时间。"
    )

运行：

```
python agent_workflow_demo.py
```

如果 Agent 依次完成模型工具请求、工具执行和最终回复，就能观察整个工作流程。

这类追踪方法适合：

- 检查 Agent 是否正确选择工具。
- 查看工具实际使用了什么参数。
- 观察工具是否返回预期数据。
- 判断是否发生重复调用。
- 分析 Agent 执行路径。

由于模型行为是动态的，工具调用顺序、事件数量和最终措辞可能有所不同。

---

## 十七、常见问题

### 1、Agent 一定会调用工具吗？

不一定。

即使注册了工具，模型也可能认为不需要使用工具，直接生成回答。

如果业务要求必须执行工具，应该在程序或工作流中明确控制。

### 2、为什么一次 Model 节点会请求多个工具？

因为部分支持 Tool Calling 的模型可以在同一轮响应中产生多个工具调用请求。

例如，同时查询两个城市的天气。

但是否使用多个工具调用，仍取决于模型和相关设置。

### 3、stream\_mode="messages" 返回的是一个个 Token 吗？

不一定。

返回的是消息片段和元数据。一个片段可能包含一段文本、工具参数的一部分或其他结构化内容。

### 4、为什么 stream\_mode="updates" 的事件数量不固定？

因为 Agent 的执行路径是动态的。

模型可能直接回复，也可能调用一个或多个工具，还可能进行多轮工具调用。

### 5、为什么 Agent 的最后一条消息不是 AIMessage？

可能是因为工具使用了 `return_direct=True`，Agent 在工具执行后直接结束。

也可能是因为图处于暂停或异常状态，尚未产生最终回复。

应根据具体执行状态进行判断。

### 6、为什么设置 thread\_id 后没有记住对话？

因为 `thread_id` 只是会话标识。

还需要配置 Checkpointer，才能保存和恢复会话状态。

### 7、recursion\_limit 是否等于最大工具调用次数？

不是。

它限制的是图执行步数，不是严格的模型请求次数或工具调用次数。

### 8、怎样查看完整 Agent 状态？

可以使用 `stream_mode="values"` 观察执行过程中的状态快照。

如果配置了 Checkpointer，还可以在执行后使用 `get_state(config)` 查看已保存状态。

### 9、如何在前端展示 Agent 执行进度？

可以使用 `stream_mode="updates"` 显示节点执行步骤，通过 `messages` 显示生成中的文本，通过 `custom` 显示业务进度。

然后使用 FastAPI 等后端框架将事件转换为 SSE 或 WebSocket 消息。

### 10、什么是新的 Event Streaming API？

LangChain v1.3 引入了新的 Event Streaming API，例如 `stream_events()`，提供不同类型事件的独立访问方式。

对于新开发的复杂流式应用，官方推荐考虑这种方式。

但本教程先以 `stream()` 和 `astream()` 为主，便于理解 Agent 的执行节点、消息片段和基本循环机制。

---

## 总结

LangChain Agent 的核心工作机制是：

```
模型 → 工具调用 → 工具结果 → 模型
```

使用 `create_agent()` 后，这个过程由 LangGraph 执行图负责管理。

本章节介绍了：

- Model 节点与 Tools 节点的执行关系。
- `stream_mode="updates"`：观察节点更新。
- `stream_mode="values"`：观察完整状态。
- `stream_mode="messages"`：流式接收模型消息。
- `stream_mode="custom"`：发送自定义进度。
- 使用多个流式模式组合观察 Agent。
- Agent 的正常结束、暂停和异常条件。
- `recursion_limit`：限制图执行步数。
- `thread_id` 与 Checkpointer：保存多轮对话。

对于初学者，建议首先掌握：

```
for event in agent.stream(

    inputs,

    stream_mode="updates",

    version="v2"

):

    print(event["data"])
```

通过观察模型节点与工具节点的更新，可以更直观地理解 Agent 每一步发生了什么。

**最后需要记住：**Agent 的执行路径并不是固定脚本。模型负责提出下一步操作，工具负责执行，而 LangGraph 则负责管理整个执行流程及其状态。
