# LangChain 多工具个人助手

> 来源: https://www.runoob.com/langchain/langchain-project-personal-assistant.html

---

# LangChain 多工具个人助手

前面我们已经介绍了 LangChain 的 Tools、Agent、Middleware、Structured Output、Streaming 和 Checkpointer。

本章节将这些能力组合起来，构建一个 多工具个人助手 Agent。

这个助手能够查询天气、查看日程、处理邮件请求，还可以根据天气和日程生成结构化的每日摘要，并记住同一会话中的聊天内容。

本项目基于 LangChain 1.x，使用 DeepSeek 聊天模型，并通过 SQLite 保存对话状态。

**说明：**天气、日程和邮件均为本地模拟功能。项目不会访问真实天气服务、Google Calendar 或邮件服务器，也不会实际发送邮件。

---

## 一、系统功能与架构

我们希望个人助手具备以下能力：

| 功能 | 实现方式 | 说明 |
| --- | --- | --- |
| 天气查询 | get\_weather Tool | 查询模拟城市天气 |
| 日程管理 | query\_schedule Tool | 查看指定日期的模拟日程 |
| 邮件处理 | send\_email Tool + HITL | 发送前请求人工确认 |
| 结构化摘要 | Pydantic + Structured Output | 生成 DailySummary |
| Markdown 展示 | to\_markdown() | 渲染每日摘要 |
| 执行过程展示 | agent.stream() | 展示模型和工具执行步骤 |
| 对话记忆 | SqliteSaver | 保存多轮会话状态 |
| 动态日期 | @dynamic\_prompt | 注入当前日期和时区 |

### 系统执行流程

```
用户发送自然语言指令

            ↓

         LangChain Agent

            ↓

       Dynamic Prompt

            ↓

          模型判断

            |

    +-------+--------+

    |       |        |

    ↓       ↓        ↓

 天气工具  日程工具  邮件工具

    |       |        |

    |       |        ↓

    |       |     人工确认

    |       |        ↓

    +-------+--------+

            ↓

         模型回复

            ↓

      SqliteSaver 保存状态



每日摘要请求：

天气数据 + 日程数据

            ↓

      结构化摘要 Agent

            ↓

       DailySummary

            ↓

         Markdown
```

需要注意，主 Agent 可以自行决定调用哪些工具，但每日摘要使用单独的结构化生成步骤，不要求普通邮件查询也返回摘要对象。

---

## 二、环境准备

建议使用 Python 3.11 或更新的兼容版本。

### 创建虚拟环境

```
python -m venv venv
```

Windows：

```
venv\Scripts\activate
```

macOS / Linux：

```
source venv/bin/activate
```

### 安装依赖

```
python -m pip install -U langchain langchain-deepseek langgraph langgraph-checkpoint-sqlite pydantic python-dotenv
```

| Python 包 | 作用 |
| --- | --- |
| `langchain` | Agent、Tools、Middleware |
| `langchain-deepseek` | DeepSeek 模型接口 |
| `langgraph` | 状态与中断恢复 |
| `langgraph-checkpoint-sqlite` | SQLite 对话持久化 |
| `pydantic` | 结构化数据校验 |
| `python-dotenv` | 加载环境变量 |

### 配置 API Key

创建 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

本教程只需要 DeepSeek API Key，不需要额外配置 Embedding 服务，因为没有使用 RAG 向量检索。

---

## 三、准备模拟天气和日程

为了让读者不接入第三方天气和日历 API 也能测试，我们使用 Python 字典保存模拟数据。

与固定写死 2024 年日期不同，本例根据程序运行日期生成日程。

### 实例：模拟数据

## 实例

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TIMEZONE = ZoneInfo("Asia/Shanghai")

def today\_str() -> str:
    return datetime.now(TIMEZONE).date().isoformat()

today = today\_str()
tomorrow = (datetime.now(TIMEZONE).date() + timedelta(days=1)).isoformat()

weather\_db = {
    "杭州": {"condition": "晴", "temp": 25, "humidity": 60},
    "北京": {"condition": "多云", "temp": 18, "humidity": 45},
    "上海": {"condition": "小雨", "temp": 22, "humidity": 80}
}

calendar\_events = [
    {"id": 1, "date": today, "time": "10:00", "title": "团队周会", "duration": "1小时"},
    {"id": 2, "date": today, "time": "14:00", "title": "Python 学习", "duration": "2小时"},
    {"id": 3, "date": tomorrow, "time": "15:00", "title": "代码审查", "duration": "1.5小时"}
]

print("今天：", today)
print("明天：", tomorrow)

**注意：**这些天气数值只是固定测试数据，不能用于真实的出行、天气预警等决策。

正式项目需要接入实际天气 API，并保留数据时间、城市及来源。

---

## 四、工具一：天气查询

使用 `@tool` 定义天气工具。

工具接受城市名称，从模拟数据库中查询天气。

### 实例：get\_weather

## 实例

from langchain.tools import tool

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气，并非实时天气。"""
    data = weather\_db.get(city.strip())
    if not data:
        return f"没有 {city} 的模拟天气数据。支持：{', '.join(weather\_db)}"

    return (
        f"{city}模拟天气：{data['condition']}，"
        f"温度 {data['temp']}°C，"
        f"湿度 {data['humidity']}%。"
    )

例如：

```
get_weather("杭州")
```

工具返回：

```
杭州模拟天气：晴，温度 25°C，湿度 60%。
```

Agent 可以将这个结果转换成更自然的回答，但不能将模拟天气描述成真实的实时天气。

---

## 五、工具二：查询日程

第二个工具负责查询指定日期的日程。

如果用户没有提供日期，则默认查询今天。

日期使用标准格式：

```
YYYY-MM-DD
```

### 实例：query\_schedule

## 实例

from datetime import date
from langchain.tools import tool

@tool
def query\_schedule(date\_str: str = "") -> str:
    """查询模拟日程。date\_str 使用 YYYY-MM-DD，留空表示今天。"""
    if not date\_str:
        date\_str = today\_str()

    try:
        date.fromisoformat(date\_str)
    except ValueError:
        return "日期格式错误，请使用 YYYY-MM-DD。"

    events = sorted(
        [e for e in calendar\_events if e["date"] == date\_str],
        key=lambda e: e["time"]
    )

    if not events:
        return f"{date\_str} 没有模拟日程。"

    lines = [f"{date\_str} 的模拟日程："]
    for event in events:
        lines.append(f"- {event['time']} {event['title']}（{event['duration']}）")
    return "\n".join(lines)

如果查询今天，可能输出：

```
2026-10-13 的模拟日程：

- 10:00 团队周会（1小时）

- 14:00 Python 学习（2小时）
```

这里的日期仅用于说明格式，实际程序会读取运行当天的日期。

**注意：**目前的日程工具只支持读取，不支持新增、删除和修改日程。后续可以继续添加 `create_event`、`update_event` 等工具。

---

## 六、工具三：模拟邮件发送

邮件发送属于有外部副作用的操作。

如果直接允许模型调用真实 SMTP 或邮件 API，可能出现误发邮件、收件人错误或敏感信息泄露等问题。

因此，本教程增加人工确认环节。

执行流程：

```
用户要求发送邮件

        ↓

Agent 调用 send_email

        ↓

展示收件人、主题和正文

        ↓

interrupt() 暂停

        ↓

人工批准或拒绝

        ↓

模拟执行结果
```

### 实例：send\_email

## 实例

from langchain.tools import tool
from langgraph.types import interrupt

@tool
def send\_email(to: str, subject: str, body: str) -> str:
    """模拟发送邮件，必须先获得人工确认。"""
    decision = interrupt({
        "action": "send\_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "请确认这封模拟邮件。"
    })

    if not isinstance(decision, dict) or decision.get("approved") is not True:
        return "邮件请求已取消，没有发送邮件。"

    return (
        f"模拟邮件操作已批准：收件人 {to}，"
        f"主题《{subject}》。本程序没有实际发送邮件。"
    )

这里有两个需要注意的地方。

**第一：**只有审批通过后，工具才返回批准结果。

**第二：**本示例不调用真实邮件 API，因此不能返回"邮件发送成功"或虚构邮件 ID。

如果要接入真实服务，应该在审批后调用邮件服务，并根据其实际响应确认发送状态。

---

## 七、动态提示词：注入日期和时区

用户可能会问：

```
今天有什么安排？

明天下午有什么日程？
```

为了让模型正确理解"今天""明天"，我们可以通过 `@dynamic_prompt` 动态注入当前日期。

与在 messages 中反复添加 SystemMessage 不同，dynamic\_prompt 在模型请求阶段构造系统提示词，不需要修改聊天历史。

### 实例：inject\_date\_context

## 实例

from datetime import datetime
from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def inject\_date\_context(request: ModelRequest) -> str:
    """在系统提示词中加入当前日期与时区。"""
    now = datetime.now(TIMEZONE)
    weekday = "一二三四五六日"[now.weekday()]

    return (
        "你是个人助手，负责模拟天气查询、日程查询和邮件处理。\n"
        f"当前日期：{now.date().isoformat()}，星期{weekday}。\n"
        "时区：Asia/Shanghai。\n"
        "用户未指定日程日期时，默认今天。\n"
        "天气是模拟数据，不得声称实时。\n"
        "邮件仅模拟处理，不得声称实际发送。\n"
        "不要虚构工具调用结果。"
    )

这里显式设置了时区，避免服务器使用不同时区时导致日期解释不一致。

正式应用可以通过可信 Runtime Context 传入用户时区，实现不同地区用户的个性化日期处理。

---

## 八、结构化输出：DailySummary

用户除了查询天气和日程，还可能希望得到一份每日摘要。

例如：

```
帮我整理今天的天气和日程，给我一份今日安排。
```

如果只让模型输出自由文本，格式可能每次不同。

可以通过 Pydantic 定义统一的数据结构。

### 实例：DailySummary

## 实例

from pydantic import BaseModel, Field

class DailySummary(BaseModel):
    """每日摘要结构。"""
    date: str = Field(description="日期，格式 YYYY-MM-DD")
    city: str = Field(description="天气所在城市")
    weather\_summary: str = Field(description="天气概述")
    event\_count: int = Field(ge=0, description="日程数量")
    key\_events: list[str] = Field(description="日程事项")
    suggestion: str = Field(description="基于已知数据给出的建议")

模型需要按照这些字段生成结构化数据。

例如：

```
{

    "date": "2026-10-13",

    "city": "杭州",

    "weather_summary": "模拟天气：晴，25°C",

    "event_count": 2,

    "key_events": ["10:00 团队周会", "14:00 Python 学习"],

    "suggestion": "按计划参加会议和学习"

}
```

这比直接解析自由文本更容易用于前端展示、保存和程序处理。

---

## 九、将结构化数据渲染为 Markdown

结构化输出适合程序处理，而 Markdown 适合展示给用户。

我们可以单独编写一个格式化函数。

### 实例：to\_markdown()

## 实例

def to\_markdown(summary: DailySummary) -> str:
    """将 DailySummary 转换为 Markdown。"""
    lines = [
        f"## {summary.date} 每日摘要",
        "",
        f"\*\*城市：\*\* {summary.city}",
        f"\*\*天气：\*\* {summary.weather\_summary}",
        f"\*\*日程数量：\*\* {summary.event\_count}",
        "",
        "### 日程安排"
    ]

    lines.extend(f"- {event}" for event in summary.key\_events)
    if not summary.key\_events:
        lines.append("- 暂无日程")

    lines.extend(["", f"\*\*建议：\*\* {summary.suggestion}"])
    return "\n".join(lines)

生成结果类似：

```
## 2026-10-13 每日摘要



**城市：** 杭州

**天气：** 模拟天气：晴，25°C

**日程数量：** 2



### 日程安排

- 10:00 团队周会

- 14:00 Python 学习



**建议：** 按计划参加会议和学习
```

---

## 十、为什么不在主 Agent 上设置全局 response\_format？

原始代码中使用：

```
agent = create_agent(

    model=model,

    tools=[get_weather, query_schedule, send_email],

    response_format=DailySummary

)
```

这种方式存在一个问题：每次调用主 Agent，包括普通天气查询、邮件请求和历史追问，都会尝试生成 DailySummary。

例如用户只是问：

```
我刚才那封邮件的主题是什么？
```

程序却可能继续尝试生成包含天气和日程的完整每日摘要。

这不是我们希望的行为。

更合理的设计是：

- 主 Agent：负责工具调用和自然语言对话。
- 摘要 Agent：仅在需要每日摘要时生成 DailySummary。

这里我们使用 `ToolStrategy`，创建专门的结构化摘要 Agent。

### 实例：创建摘要 Agent

## 实例

from langchain.agents import create\_agent
from langchain.agents.structured\_output import ToolStrategy

summary\_agent = create\_agent(
    model=model,
    tools=[],
    response\_format=ToolStrategy(DailySummary),
    system\_prompt=(
        "你是每日摘要生成器。只能根据提供的数据生成 DailySummary。"
        "必须准确计算日程数量，不得补充不存在的天气或日程。"
        "天气为模拟数据，必须明确标记。"
    )
)

调用后，结构化结果可以通过：

```
result["structured_response"]
```

获取。

**注意：**结构化输出依赖模型的相关能力。即使使用 Pydantic 校验，也应检查业务字段是否与原始工具数据一致。

---

## 十一、生成每日摘要

为了保证摘要中的天气和日程都来自工具，我们先执行对应查询，再把结果交给摘要 Agent。

这是一种确定性的处理方式。

### 实例：generate\_daily\_summary()

## 实例

def generate\_daily\_summary(city: str = "杭州") -> str:
    """获取模拟数据并生成结构化每日摘要。"""
    date\_str = today\_str()

    weather = get\_weather.invoke({"city": city})
    schedule = query\_schedule.invoke({"date\_str": date\_str})

    result = summary\_agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                f"日期：{date\_str}\n城市：{city}\n"
                f"天气资料：\n{weather}\n"
                f"日程资料：\n{schedule}\n"
                "请只根据上述资料生成结构化摘要。"
            )
        }]
    })

    summary = result["structured\_response"]
    return to\_markdown(summary)

这个函数不依赖主 Agent 是否主动调用工具，而是明确执行：

```
天气查询

   ↓

日程查询

   ↓

结构化摘要

   ↓

Markdown
```

因此，如果业务要求每日摘要必须包含两个数据来源，这种方式比单纯依赖提示词更可控。

对于正式项目，还应校验结构化摘要中的日程数量是否与真实工具返回数量一致，而不是完全信任模型生成的数字。

---

## 十二、使用 SqliteSaver 保存对话

个人助手需要记住用户在同一会话中说过什么。

例如：

```
用户：帮我准备一封邮件，主题是今日总结。



用户：刚才那封邮件的主题是什么？
```

如果没有 Checkpointer，第二轮问题可能无法关联此前对话。

这里使用 SQLite 保存 Agent State。

### 实例：创建 Checkpointer

## 实例

from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from\_conn\_string("personal\_assistant.db") as checkpointer:
    agent = create\_agent(
        model=model,
        tools=[get\_weather, query\_schedule, send\_email],
        checkpointer=checkpointer,
        middleware=[inject\_date\_context]
    )

需要使用稳定的：

```
config = {

    "configurable": {

        "thread_id": "personal-chat-001"

    }

}
```

同一个 thread\_id 下，后续调用可以恢复之前保存的消息历史。

**注意：**所有使用该 SqliteSaver 连接的 Agent 调用都应放在 with 代码块内部。

SQLite 更适合本地同步应用。正式的多用户 Web 服务推荐使用 PostgreSQL 等适合并发访问的后端。

---

## 十三、流式输出：展示 Agent 执行过程

LangChain 支持不同的 Streaming 模式。

其中：

| 模式 | 用途 |
| --- | --- |
| `updates` | 展示 Agent 节点执行后的状态更新 |
| `messages` | 接收模型生成的文本和工具调用增量 |
| `custom` | 接收自定义事件 |

本项目希望展示：

```
正在调用 get_weather

get_weather 返回结果

正在调用 query_schedule

query_schedule 返回结果

生成最终回复
```

因此可以使用 `stream_mode="updates"`。

**注意：**这种输出展示的是可观察到的执行步骤和工具结果，而不是模型隐藏的内部推理过程。

### 实例：展示节点更新

## 实例

def chat\_stream(agent, message: str, config: dict):
    """按执行步骤展示 Agent 更新。"""
    for event in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream\_mode="updates"
    ):
        for node, update in event.items():
            for msg in update.get("messages", []):
                if msg.type == "ai" and getattr(msg, "tool\_calls", []):
                    for call in msg.tool\_calls:
                        print("[调用工具]", call["name"])
                elif msg.type == "tool":
                    print("[工具返回]", msg.name, str(msg.content)[:120])
                elif msg.type == "ai" and msg.content:
                    print("[助手]", msg.content)

使用时，需要注意 Agent 可能在工具执行过程中触发 HITL 中断。

正式实现还应该检查当前线程是否有待处理的中断，再决定是否继续执行或返回审批信息。

---

## 十四、完整代码：多工具个人助手

下面将天气、日程、邮件审批、结构化摘要和对话记忆整合为一个文件。

文件名：`personal_assistant.py`

这个完整示例将邮件操作设计为模拟审批，不会连接真实 SMTP，也不会自动批准邮件发送。

## 完整实例

import os
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from dotenv import load\_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create\_agent
from langchain.agents.middleware import dynamic\_prompt, ModelRequest
from langchain.agents.structured\_output import ToolStrategy
from langchain.tools import tool
from langchain\_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

load\_dotenv()
assert os.getenv("DEEPSEEK\_API\_KEY"), "请先配置 DEEPSEEK\_API\_KEY"

TZ = ZoneInfo("Asia/Shanghai")

def today\_str():
    return datetime.now(TZ).date().isoformat()

today = today\_str()
tomorrow = (datetime.now(TZ).date() + timedelta(days=1)).isoformat()

weather\_db = {
    "杭州": {"condition": "晴", "temp": 25, "humidity": 60},
    "北京": {"condition": "多云", "temp": 18, "humidity": 45},
    "上海": {"condition": "小雨", "temp": 22, "humidity": 80}
}

calendar\_events = [
    {"date": today, "time": "10:00", "title": "团队周会", "duration": "1小时"},
    {"date": today, "time": "14:00", "title": "Python 学习", "duration": "2小时"},
    {"date": tomorrow, "time": "15:00", "title": "代码审查", "duration": "1.5小时"}
]

# ========== 1. 工具 ==========

@tool
def get\_weather(city: str) -> str:
    """查询指定城市的模拟天气，不是实时天气。"""
    data = weather\_db.get(city.strip())
    if not data:
        return f"没有 {city} 的模拟天气数据"
    return f"{city}模拟天气：{data['condition']}，{data['temp']}°C，湿度 {data['humidity']}%"

@tool
def query\_schedule(date\_str: str = "") -> str:
    """查询模拟日程。日期格式 YYYY-MM-DD，留空表示今天。"""
    date\_str = date\_str or today\_str()
    try:
        date.fromisoformat(date\_str)
    except ValueError:
        return "日期格式错误"

    events = sorted(
        [e for e in calendar\_events if e["date"] == date\_str],
        key=lambda e: e["time"]
    )
    if not events:
        return f"{date\_str} 没有模拟日程"
    return "\n".join(
        [f"{date\_str} 的模拟日程："] +
        [f"- {e['time']} {e['title']}（{e['duration']}）" for e in events]
    )

@tool
def send\_email(to: str, subject: str, body: str) -> str:
    """模拟发送邮件，必须先人工批准。"""
    decision = interrupt({
        "action": "send\_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "请确认模拟邮件"
    })
    if not isinstance(decision, dict) or decision.get("approved") is not True:
        return "邮件操作已取消，未发送邮件"
    return f"模拟邮件操作已批准：{to}，主题《{subject}》。未实际发送。"

# ========== 2. 动态提示词 ==========

@dynamic\_prompt
def inject\_date\_context(request: ModelRequest) -> str:
    now = datetime.now(TZ)
    return (
        "你是中文个人助手，负责模拟天气、日程和邮件处理。\n"
        f"当前日期：{now.date().isoformat()}，时区：Asia/Shanghai。\n"
        "日期不明确时可以询问用户。\n"
        "天气与日程均为模拟数据。\n"
        "邮件必须通过 send\_email 工具请求审批，不能声称实际发送。\n"
        "不知道的信息不要编造。"
    )

# ========== 3. 结构化摘要 ==========

class DailySummary(BaseModel):
    date: str = Field(description="日期")
    city: str = Field(description="城市")
    weather\_summary: str = Field(description="模拟天气")
    event\_count: int = Field(ge=0, description="日程数量")
    key\_events: list[str] = Field(description="日程列表")
    suggestion: str = Field(description="今日建议")

def to\_markdown(summary: DailySummary) -> str:
    lines = [
        f"## {summary.date} 每日摘要",
        "",
        f"\*\*城市：\*\* {summary.city}",
        f"\*\*天气：\*\* {summary.weather\_summary}",
        f"\*\*日程数量：\*\* {summary.event\_count}",
        "",
        "### 日程安排"
    ]
    lines.extend(f"- {e}" for e in summary.key\_events)
    if not summary.key\_events:
        lines.append("- 暂无安排")
    lines.extend(["", f"\*\*今日建议：\*\* {summary.suggestion}"])
    return "\n".join(lines)

model = ChatDeepSeek(model="deepseek-chat", temperature=0)

summary\_agent = create\_agent(
    model=model,
    tools=[],
    response\_format=ToolStrategy(DailySummary),
    system\_prompt=(
        "你是每日摘要生成器。只能根据输入的数据生成摘要。"
        "不得增加不存在的日程或天气事实。"
        "必须注明天气为模拟数据。"
    )
)

def daily\_summary(city="杭州"):
    weather = get\_weather.invoke({"city": city})
    schedule = query\_schedule.invoke({"date\_str": today\_str()})

    result = summary\_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"日期：{today\_str()}\n城市：{city}\n天气：{weather}\n日程：{schedule}"
        }]
    })
    return to\_markdown(result["structured\_response"])

# ========== 4. 主 Agent ==========

def make\_agent(checkpointer):
    return create\_agent(
        model=model,
        tools=[get\_weather, query\_schedule, send\_email],
        middleware=[inject\_date\_context],
        checkpointer=checkpointer
    )

# ========== 5. 对话函数 ==========

def chat(agent, message: str, thread\_id: str):
    config = {"configurable": {"thread\_id": thread\_id}}

    print("\n用户：", message)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )

    pending = result.get("\_\_interrupt\_\_", ())
    if pending:
        print("[等待人工确认]")
        for item in pending:
            print(item.value)
        return {"status": "pending", "config": config}

    print("助手：", result["messages"][-1].content)
    return {"status": "completed", "config": config}

def resume\_email(agent, config: dict, approved: bool):
    state = agent.get\_state(config)
    pending = [i for task in state.tasks for i in task.interrupts]
    if not pending:
        print("没有待处理的邮件审批")
        return

    result = agent.invoke(
        Command(resume={"approved": approved}),
        config=config
    )

    if result.get("\_\_interrupt\_\_"):
        print("还有新的待处理请求")
    else:
        print("助手：", result["messages"][-1].content)

def chat\_updates(agent, message: str, thread\_id: str):
    config = {"configurable": {"thread\_id": thread\_id}}
    print("\n用户：", message)

    for event in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream\_mode="updates"
    ):
        for node, update in event.items():
            for msg in update.get("messages", []):
                if msg.type == "ai" and getattr(msg, "tool\_calls", []):
                    for call in msg.tool\_calls:
                        print("[调用工具]", call["name"])
                elif msg.type == "tool":
                    print("[工具返回]", msg.name, str(msg.content)[:120])
                elif msg.type == "ai" and msg.content:
                    print("[助手]", msg.content)

# ========== 6. 运行测试 ==========

if \_\_name\_\_ == "\_\_main\_\_":
    with SqliteSaver.from\_conn\_string("personal\_assistant.db") as checkpointer:
        agent = make\_agent(checkpointer)
        thread\_id = "demo-personal-assistant"

        print("=== 测试 1：多工具协作 ===")
        chat\_updates(agent, "查询杭州的模拟天气和今天的模拟日程。", thread\_id)

        print("\n=== 测试 2：结构化每日摘要 ===")
        print(daily\_summary("杭州"))

        print("\n=== 测试 3：邮件审批 ===")
        response = chat(
            agent,
            "请模拟给 team@example.com 发送邮件，主题是今日总结，正文是今天的日程已确认。",
            thread\_id
        )
        if response["status"] == "pending":
            # 本地教学演示：明确拒绝，不会实际发送
            resume\_email(agent, response["config"], approved=False)

        print("\n=== 测试 4：对话记忆 ===")
        chat(agent, "我刚才让你准备的邮件主题是什么？", thread\_id)

### 运行方式

保存文件后执行：

```
python personal_assistant.py
```

程序会创建：

```
personal_assistant.db
```

用于保存主 Agent 的消息和执行状态。

如果要从全新对话开始测试，可以更换 thread\_id。

**注意：**如果使用上一次遗留的 SQLite 数据库和相同 thread\_id，Agent 会恢复旧历史，可能影响本次演示。可以为每次测试使用不同线程，避免混入旧会话。

---

## 十五、运行结果示例

以下输出用于说明功能，实际回复取决于模型调用结果。

### 测试一：天气与日程

```
=== 测试 1：多工具协作 ===



用户：查询杭州的模拟天气和今天的模拟日程。



[调用工具] get_weather

[调用工具] query_schedule



[工具返回] get_weather 杭州模拟天气：晴，25°C，湿度60%

[工具返回] query_schedule 今天的模拟日程：...



[助手] 根据模拟数据，杭州天气为晴，25°C。

今天有团队周会和 Python 学习两项安排。
```

Agent 根据问题选择两个工具并汇总结果。

模型可能调整工具调用顺序，甚至没有按照预期调用全部工具。因此，以上仅为可能的执行日志。

### 测试二：结构化每日摘要

```
=== 测试 2：结构化每日摘要 ===



## 2026-10-13 每日摘要



**城市：** 杭州

**天气：** 模拟天气：晴，25°C

**日程数量：** 2



### 日程安排

- 10:00 团队周会

- 14:00 Python 学习



**今日建议：** 按计划安排会议和学习。
```

结构化摘要只在调用 `daily_summary()` 时生成，不会影响其他普通对话。

### 测试三：邮件审批

```
=== 测试 3：邮件审批 ===



用户：请模拟给 team@example.com 发送邮件...



[等待人工确认]

{

    "action": "send_email",

    "to": "team@example.com",

    "subject": "今日总结",

    "body": "今天的日程已确认"

}



本地演示拒绝审批



助手：邮件请求已取消，没有发送邮件。
```

这里不会连接真实邮件服务，也不会产生发送记录。

### 测试四：对话记忆

```
=== 测试 4：对话记忆 ===



用户：我刚才让你准备的邮件主题是什么？



助手：你刚才准备的模拟邮件主题是“今日总结”，

但这封邮件并没有实际发送。
```

只要同一会话线程保存了此前的消息，Agent 就可以结合历史进行回答。

但模型的回答仍可能出现错误，因此需要可靠状态时，应直接读取保存的业务数据，而不应完全依赖模型回忆。

---

## 十六、如何实现真正的邮件发送？

当前邮件工具只是模拟。

如果要接入真实邮件服务，可以选择 SMTP、Gmail API 或其他邮件服务提供商。

但必须把以下步骤分开：

```
AI 生成邮件草稿

       ↓

展示收件人、标题、正文

       ↓

用户或管理员确认

       ↓

后端校验身份与权限

       ↓

调用邮件服务

       ↓

获得真实发送结果

       ↓

返回消息 ID 或失败原因
```

不建议让模型直接持有 SMTP 密码或邮件服务的高权限凭据。

真正的发送动作应该由可信后端完成，并考虑重复发送保护、收件人验证、日志审计和发送限流。

尤其是 Agent 发生网络重试或中断恢复时，需要防止同一封邮件被重复发送。

---

## 十七、生产环境注意事项

### 1、天气数据必须注明时间和来源

真实天气接口需要返回城市、天气时间和观测或预报类型。不能把历史天气或模拟数据当成实时天气。

### 2、日历工具需要用户授权

接入真实日历时，应由后端根据认证用户访问对应日历，不能让模型任意指定其他用户的账号。

### 3、邮件发送必须获得确认

特别是包含收件人、附件或敏感内容的邮件，应在执行前展示完整信息。

### 4、结构化输出不等于事实验证

Pydantic 可以验证字段类型，但无法自动验证天气是否真实、日程数量是否正确。

### 5、明确时区

"今天""明天""下午三点"都依赖用户所在时区，不能完全依赖服务器默认时区。

### 6、控制工具调用次数

模型可能反复调用工具，实际应用应设置合理的执行限制、超时和调用监控。

### 7、保护对话记录

Checkpointer 可能保存用户邮件正文、日程和私人信息，应限制访问并设置合理的数据保留策略。

### 8、避免泄露模型内部推理

前端应该展示可观察的工具步骤和必要的进度信息，而不是宣称可以直接展示模型的全部内部思考。

### 9、正式部署使用适合并发的 Checkpointer

SQLite 适合本地和轻量同步环境。多用户 Web 服务通常应使用 PostgreSQL 等支持并发管理的持久化方案。

---

## 十八、常见问题

### 1、一个 Agent 可以调用多个工具吗？

可以。通过 create\_agent() 的 tools 参数注册多个工具，模型可以根据任务选择调用。

### 2、工具调用顺序是固定的吗？

不一定。Agent 可以根据模型判断调整顺序。需要严格执行顺序时，应该使用确定性的工作流。

### 3、模拟天气可以直接用于真实天气应用吗？

不能。需要替换为真实的天气 API，并保留数据来源及时间。

### 4、为什么日程查询要设置时区？

因为不同地区的日期和时间可能不同，时区可以避免对"今天"等相对日期的错误解释。

### 5、dynamic\_prompt 与 before\_model 有什么区别？

dynamic\_prompt 适合为当前模型请求动态构造 System Prompt；before\_model 适合执行模型前的状态检查和更新。

### 6、为什么 DailySummary 不应该全局设置？

因为邮件处理、普通问答等任务不需要每日摘要，统一强制生成会导致无关结构化结果。

### 7、ToolStrategy 有什么作用？

它可以利用工具调用机制让模型生成符合指定 Schema 的结构化数据。

### 8、流式输出可以展示哪些内容？

可以展示模型生成片段、工具调用、工具结果和自定义事件，具体取决于所选 Streaming 模式。

### 9、stream\_mode="updates" 是逐 Token 输出吗？

不是。它主要展示图节点的状态更新。逐步输出模型文本可以使用 messages 模式。

### 10、为什么 SqliteSaver 需要 thread\_id？

因为 Checkpointer 通过 thread\_id 识别并恢复对应的会话状态。

### 11、邮件批准后就代表发送成功吗？

不代表。人工批准只允许后续执行，是否成功还需要查看真实邮件服务的返回结果。

### 12、如何防止同一封邮件重复发送？

可以在业务后端为邮件请求设置唯一 ID，并记录执行状态，避免网络重试或恢复导致重复发送。

### 13、可以接入 Google Calendar 或 Gmail 吗？

可以，但需要对应的授权流程、API 凭据和访问权限控制。

### 14、个人助手如何记住用户偏好？

Checkpointer 可以保存同一线程的对话历史；需要跨会话共享长期偏好时，可以使用 Store 或业务数据库。

---

## 项目总结

本章节构建了一个基于 LangChain 的多工具个人助手。

主要使用以下技术：

| LangChain 功能 | 本项目实现 |
| --- | --- |
| Tools | get\_weather、query\_schedule、send\_email |
| Agent | 根据用户意图选择多个工具 |
| Dynamic Prompt | 注入日期和时区信息 |
| Structured Output | DailySummary + ToolStrategy |
| Markdown | to\_markdown() 格式化摘要 |
| Streaming | 通过 updates 展示执行步骤 |
| Checkpointer | SqliteSaver 保存对话状态 |
| HITL | 邮件操作前等待人工确认 |

整个系统可以概括为：

```
自然语言输入

      ↓

 LangChain Agent

      ↓

  动态提示词

      ↓

   工具选择

      |

      ├── 天气查询

      ├── 日程查询

      └── 邮件审批

      ↓

   结果汇总

      ↓

   保存对话



独立摘要流程：

天气 + 日程 → DailySummary → Markdown
```

**核心原则：**让 Agent 负责理解用户意图和协调工具，让工具负责查询和执行，让结构化输出负责生成稳定的数据格式，让 Checkpointer 负责会话记忆。涉及发送邮件等真实外部操作时，还需要独立的权限验证与人工确认。

在此基础上，可以进一步接入真实天气 API、日历服务、邮件系统和用户管理功能，逐步构建一个能够完成日常事务的个人 AI 助手。
