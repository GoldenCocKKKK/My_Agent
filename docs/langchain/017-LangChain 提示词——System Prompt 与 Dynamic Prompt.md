# LangChain 提示词——System Prompt 与 Dynamic Prompt

> 来源: https://www.runoob.com/langchain/langchain-system-prompt.html

---

# LangChain 提示词——System Prompt 与 Dynamic Prompt

提示词（Prompt）是控制大语言模型行为的重要方式。

在 LangChain Agent 中，System Prompt 用于定义模型的角色、职责、回答风格和工具使用规则。

如果需要针对不同用户、不同任务或不同对话阶段调整提示词，还可以通过 @dynamic\_prompt 动态生成系统提示。

本章节介绍 LangChain 1.x 的静态与动态提示词，并使用 DeepSeek Flash 模型进行演示。

---

## 一、System Prompt 基本用法

`create_agent()` 通过 `system_prompt` 参数接收系统提示词，支持字符串和 `SystemMessage` 两种常见形式。

### 方式一：使用字符串

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent

load\_dotenv()

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是菜鸟教程 RUNOOB 的学习顾问，回答要简洁、准确、通俗易懂。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Python 适合零基础学习吗？"}]
})
print(result["messages"][-1].content)

这种方式最简单，适合系统提示词相对固定的场景。

### 方式二：使用 SystemMessage

## 实例

from langchain.agents import create\_agent
from langchain.messages import SystemMessage

system\_msg = SystemMessage(content="你是菜鸟教程 RUNOOB 的学习顾问，回答要简洁专业。")

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt=system\_msg
)

`SystemMessage` 是 LangChain 的消息对象，可以在多个 Agent 或模型配置之间复用。

两种方式都用于设置系统提示词，通常选择字符串即可。

**注意：**System Prompt 可以影响模型行为，但不能保证模型始终严格遵守。涉及权限、支付和数据访问等操作，应由应用程序进行校验。

---

## 二、如何设计有效的 System Prompt？

好的系统提示词不一定很长，关键是让模型清楚自己的职责、能够使用哪些工具，以及遇到未知信息时如何处理。

一般可以包含五个部分：

| 要素 | 作用 | 示例 |
| --- | --- | --- |
| 角色定义 | 说明 AI 的身份 | 你是一名编程学习顾问 |
| 任务职责 | 说明需要完成什么 | 推荐课程、解答学习问题 |
| 工具规则 | 说明何时使用工具 | 查询课程时使用 search\_course |
| 回答规范 | 控制语言和格式 | 中文回答，尽量简洁 |
| 边界约束 | 处理未知和异常情况 | 没有数据时不编造 |

### 实例：课程推荐助手

## 实例

from langchain.tools import tool
from langchain.agents import create\_agent

@tool
def search\_course(keyword: str) -> str:
    """根据编程语言搜索模拟课程。"""
    courses = {"python": "Python3 基础教程", "html": "HTML 基础教程"}
    return courses.get(keyword.lower(), "未找到相关模拟课程")

GOOD\_PROMPT = """你是菜鸟教程 RUNOOB 的编程学习顾问。

## 职责
- 解答编程学习问题
- 根据用户水平推荐学习方向
- 查询课程时使用 search\_course 工具

## 回答规则
- 使用简洁、准确的中文
- 零基础用户优先推荐基础知识
- 没有课程数据时明确说明
- 工具提供的是模拟课程资料，不代表网站实时课程目录
- 不编造课程价格、优惠或学习时长"""

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course],
    system\_prompt=GOOD\_PROMPT
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我是零基础，想学 Python，有什么课程？"}]
})
print(result["messages"][-1].content)

如果模型正确调用工具，可能回答：

```
可以先学习 Python3 基础教程，从变量、数据类型和基本语法开始。



这是演示课程数据，实际课程信息请以网站页面为准。
```

模型是否调用工具、最终回答的具体措辞并不固定。

建议让提示词尽量明确，避免同时提出相互矛盾的要求，例如既要求"详细解释"，又要求"只能回答一句话"。

---

## 三、@dynamic\_prompt：动态生成系统提示词

静态 System Prompt 在不同调用之间保持相对固定，但实际应用可能需要根据当前任务动态修改。

例如：

- 初学者使用通俗易懂的解释。
- 高级开发者使用更专业的术语。
- 根据对话进度调整回答长度。
- 根据当前用户的学习目标提供建议。

LangChain 提供 `@dynamic_prompt` 装饰器，它是一种 Middleware，可以在每次模型调用前生成当前使用的系统提示。

### 基本语法

```
from langchain.agents.middleware import dynamic_prompt, ModelRequest



@dynamic_prompt

def my_prompt(request: ModelRequest) -> str:

    return "你是一名编程助手。"
```

随后将其加入 `create_agent()` 的 `middleware` 参数：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    middleware=[my_prompt]

)
```

`@dynamic_prompt` 接收 `ModelRequest`，返回 `str` 或 `SystemMessage`，作为当前模型请求的系统提示。

---

## 四、根据对话阶段动态调整提示词

可以通过 `request.state` 读取当前 Agent 状态，根据消息数量调整提示词。

### 实例：短对话与长对话

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def personalized\_prompt(request: ModelRequest) -> str:
    """根据消息数量调整回答风格。"""
    count = len(request.state.get("messages", []))
    base = "你是菜鸟教程 RUNOOB 的学习顾问。"

    if count <= 2:
        return base + "用户刚开始对话，请耐心介绍，并了解学习目标。"
    if count > 10:
        return base + "对话内容已经较多，请优先回答当前问题，尽量简洁。"
    return base + "结合已有对话，给出实用的编程学习建议。"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[personalized\_prompt]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "你好，我想学习编程。"}]
})
print(result["messages"][-1].content)

首次调用时，消息较少，模型可能采用引导式回答：

```
你好！你想学习 Python、Java 还是前端开发？

目前是否有编程基础？
```

当消息历史较长时，动态提示词会要求模型更加简洁。

**注意：**消息数量不等于对话轮数。一轮 Agent 执行可能产生 AIMessage、ToolMessage 等多条消息。

另外，如果没有配置 Checkpointer，多次独立 `invoke()` 不会自动共享上一轮的完整对话历史。

---

## 五、ModelRequest 提供哪些信息？

`@dynamic_prompt` 的参数是 `ModelRequest`，其中包含当前模型请求的相关信息。

| 属性 | 说明 |
| --- | --- |
| `request.state` | 当前 Agent State，可读取消息历史和自定义字段 |
| `request.runtime` | 当前运行时对象，包含 Context 等信息 |
| `request.system_prompt` | 当前请求的系统提示词文本 |
| `request.model` | 当前使用的模型对象 |
| `request.messages` | 当前模型请求的消息 |
| `request.tools` | 当前模型请求可用的工具定义 |

其中，最常用的是 `request.state` 和 `request.runtime.context`。

**注意：**`request.runtime.context` 的数据结构由应用程序的 `context_schema` 决定，不能假设它始终是字典。

---

## 六、根据用户级别动态生成提示词

相比根据消息数量调整提示词，实际业务更常见的需求是根据用户信息进行调整。

例如，编程学习网站可以按照用户的学习等级提供不同的解释方式。

这里推荐使用 `context_schema` 定义运行时上下文，并由应用程序传入用户等级。

### 完整实例：不同等级的学习助手

## 实例

from dataclasses import dataclass
from langchain.agents import create\_agent
from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dataclass
class UserContext:
    level: str = "入门"
    language: str = "中文"

@dynamic\_prompt
def learning\_prompt(request: ModelRequest[UserContext]) -> str:
    """根据学习等级生成系统提示。"""
    context = request.runtime.context
    level = context.level

    base = f"你是菜鸟教程 RUNOOB 的编程导师，请使用{context.language}回答。"

    if level == "入门":
        return base + "用户是初学者，请少用专业术语，结合简单例子解释。"
    if level == "进阶":
        return base + "用户有编程基础，可以介绍原理、最佳实践和注意事项。"
    return base + "用户是高级开发者，重点分析实现原理、性能和设计取舍。"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    context\_schema=UserContext,
    middleware=[learning\_prompt]
)

question = {"messages": [{"role": "user", "content": "什么是 Python 装饰器？"}]}

# 初学者
result = agent.invoke(question, context=UserContext(level="入门"))
print("入门版：", result["messages"][-1].content)

# 高级开发者
result = agent.invoke(question, context=UserContext(level="高级"))
print("高级版：", result["messages"][-1].content)

两次调用的问题相同，但提供了不同的用户等级。

模型因此可能采用不同的解释方式：

| 用户级别 | 回答重点 |
| --- | --- |
| 入门 | 通俗解释、简单示例、减少术语 |
| 进阶 | 实现原理、使用技巧、常见问题 |
| 高级 | 设计机制、性能影响、工程取舍 |

这里的 Context 由应用程序传入，不应该直接信任模型自己声称的用户身份或权限。

---

## 七、根据时间动态生成提示词

动态提示词也可以读取时间信息，例如用于不同时间段的问候语。

不过，服务器时间不一定与用户所在地时间相同，因此应明确使用哪个时区。

### 实例：按指定时区生成问候语

## 实例

from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def time\_aware\_prompt(request: ModelRequest) -> str:
    """根据北京时间生成问候语。"""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    hour = now.hour
    greeting = "早上好" if hour < 12 else "下午好" if hour < 18 else "晚上好"

    return (
        f"你是编程学习助手。当前北京时间：{now:%Y-%m-%d %H:%M}。"
        f"可以自然地使用"{greeting}"问候，回答保持简洁。"
    )

这里显式使用 `Asia/Shanghai` 时区，避免依赖服务器操作系统的本地时区。

但它仍然只是北京时间，不一定是用户所在地时间。如果网站支持多时区，应从可信的用户设置中读取时区，并验证其有效性。

另外，`@dynamic_prompt` 在每次模型请求前执行，建议避免在函数中直接进行耗时的数据库扫描、大文件处理或复杂网络请求。

---

## 八、静态与动态 System Prompt 的关系

如果同时设置：

```
create_agent(system_prompt=...)



middleware=[dynamic_prompt_function]
```

那么 `@dynamic_prompt` 返回的提示词会被用于当前模型请求。

因此，它可能替换原来的静态 System Prompt，而不是自动拼接在后面。

### 实例：动态提示词覆盖静态提示词

## 实例

from langchain.agents import create\_agent
from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def override\_prompt(request: ModelRequest) -> str:
    """替换当前模型请求的系统提示。"""
    return "你是一名 Python 教师，请用一句话解释概念。"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是通用编程助手，请提供详细解释。",
    middleware=[override\_prompt]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "什么是变量？"}]
})
print(result["messages"][-1].content)

此时模型请求使用的是动态函数返回的系统提示。

**注意：**如果存在多个修改系统提示词的 Middleware，最终效果还受到中间件的执行顺序及组合逻辑影响，不能简单套用一个固定的全局优先级。

---

## 九、将静态与动态提示词组合

实际项目中，通常希望保留一组固定规则，同时根据用户信息追加个性化内容。

推荐把公共规则单独定义，然后在动态提示词中组合。

### 实例：公共规则 + 用户级别

## 实例

from langchain.agents.middleware import dynamic\_prompt, ModelRequest

BASE\_PROMPT = """你是菜鸟教程 RUNOOB 的编程学习顾问。
- 回答准确、简洁
- 不知道的信息应明确说明
- 不编造不存在的教程、价格或学习资料"""

@dynamic\_prompt
def combined\_prompt(request: ModelRequest[UserContext]) -> str:
    """组合通用规则和用户级别。"""
    level = request.runtime.context.level
    extra = "请使用简单例子解释。" if level == "入门" else "可以深入解释技术原理。"
    return BASE\_PROMPT + "\n" + extra

这里引用的 `UserContext` 与前面的运行时上下文示例相同。

如果单独运行这个示例，需要同时定义 UserContext，并将 `combined_prompt` 注册到 Agent。

### 读取当前 system\_prompt

在当前 LangChain API 中，还可以使用 `request.system_prompt` 访问当前模型请求的系统提示文本。

## 实例

from langchain.agents.middleware import dynamic\_prompt, ModelRequest

@dynamic\_prompt
def extend\_prompt(request: ModelRequest) -> str:
    """在当前提示词基础上增加规则。"""
    base = request.system\_prompt or "你是一名编程学习助手。"
    return base + "\n回答时优先给出简单、可执行的示例。"

这种方式适合保留当前静态系统提示，再追加动态规则。

但是，如果配置了多个动态提示词中间件，应该检查是否可能重复追加内容。不要假设每个中间件都一定读取到最初未经修改的系统提示。

---

## 十、System Prompt 与工具调用

System Prompt 可以指导 Agent 什么时候使用工具。

例如：

```
当用户查询课程目录时，使用 search_course 工具。

工具没有返回相关课程时，明确告知用户。

不要编造课程价格或会员信息。
```

但即使这样设置，也不能保证模型每次都选择调用工具。

如果某项业务操作必须执行，例如扣款、保存订单、检查权限，应该由程序或明确的工作流负责，而不是只依赖自然语言提示词。

建议将职责分开：

| 功能 | 负责机制 |
| --- | --- |
| 指导模型回答风格 | System Prompt |
| 根据用户调整提示词 | Dynamic Prompt |
| 定义可调用的能力 | @tool |
| 执行工具与循环调度 | create\_agent() |
| 验证身份与权限 | 可信后端业务逻辑 |

---

## 十一、完整实例：个性化编程学习 Agent

下面将静态公共规则、运行时上下文、动态提示词和课程搜索工具组合起来。

文件名：`dynamic_prompt_demo.py`

安装依赖：

```
pip install -U langchain langchain-deepseek python-dotenv
```

在 `.env` 文件中配置：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dataclasses import dataclass
from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import dynamic\_prompt, ModelRequest
from langchain.tools import tool

load\_dotenv()

@dataclass
class UserContext:
    level: str = "入门"
    goal: str = "学习编程"

COURSES = {
    "python": "Python3 基础教程",
    "html": "HTML 基础教程",
    "java": "Java 基础教程"
}

@tool
def search\_course(keyword: str) -> str:
    """根据编程语言查询模拟课程。"""
    return COURSES.get(keyword.lower(), "没有找到对应的模拟课程")

BASE\_PROMPT = """你是菜鸟教程 RUNOOB 的编程学习顾问。
- 使用简洁、准确的中文回答
- 查询课程时优先使用 search\_course
- 课程数据为教学模拟数据
- 不编造价格、会员信息或不存在的课程"""

@dynamic\_prompt
def personalized\_prompt(request: ModelRequest[UserContext]) -> str:
    """根据用户等级和学习目标生成提示词。"""
    ctx = request.runtime.context
    if ctx.level == "入门":
        extra = "用户是初学者，少用术语，先解释基础概念。"
    elif ctx.level == "进阶":
        extra = "用户有编程基础，可以介绍项目实践与常见问题。"
    else:
        extra = "用户是高级开发者，重点解释技术原理和工程设计。"
    return f"{BASE\_PROMPT}\n学习目标：{ctx.goal}\n{extra}"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course],
    context\_schema=UserContext,
    middleware=[personalized\_prompt]
)

def ask(question: str, level: str):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        context=UserContext(level=level, goal="学习 Python")
    )
    print(f"\n[{level}] {result['messages'][-1].content}")

if \_\_name\_\_ == "\_\_main\_\_":
    ask("我想学 Python，有什么课程推荐？", "入门")
    ask("我想学 Python，有什么课程推荐？", "高级")

这个示例中，用户提出相同的问题，但 Context 中的等级不同。

Agent 可以根据不同用户动态调整系统提示词。

初学者可能得到更简单的学习建议，高级开发者可能得到更深入的技术方向。

课程搜索工具仍然使用固定模拟数据，不会连接真实网站。

**注意：**不同用户的学习等级应由应用程序提供可信上下文，不应直接把模型生成的角色信息当作权限依据。

---

## 十二、常见问题

### 1、System Prompt 与 HumanMessage 有什么区别？

System Prompt 主要设置模型的角色和行为规则，HumanMessage 用于表达用户当前的问题或要求。

### 2、@dynamic\_prompt 每轮都会执行吗？

它在 Agent 每次准备调用模型时执行。一次 Agent 运行可能因为工具调用而进行多次模型请求，因此动态提示词也可能执行多次。

### 3、动态提示词可以返回 SystemMessage 吗？

可以。当前 `@dynamic_prompt` 支持返回字符串或 SystemMessage。

### 4、如何读取用户等级？

推荐通过 `context_schema` 定义运行时 Context，再使用 `request.runtime.context` 获取。

### 5、Context 是字典还是对象？

取决于应用程序的定义。可以使用字典、TypedDict 对应的数据结构，也可以使用 dataclass 等对象，读取方式需保持一致。

### 6、动态提示词会自动保留静态提示词吗？

不会自动保证拼接。动态函数的返回值会用于当前请求，建议明确组合固定和动态部分。

### 7、可以获取当前 System Prompt 吗？

可以通过 `request.system_prompt` 读取当前模型请求的系统提示文本。

### 8、为什么对话消息数量比实际轮数多？

因为 Agent 还会生成工具调用请求、工具结果和其他消息。统计用户轮数时，应单独筛选 HumanMessage。

### 9、Dynamic Prompt 是否适合查询数据库？

可以根据业务需要读取数据，但不建议在每次模型调用前执行复杂、耗时的数据库查询。通常更适合提前在应用层获取必要信息，并通过 Context 传入。

### 10、System Prompt 可以保证模型一定调用工具吗？

不能。它可以指导工具使用，但需要强制执行的任务应该交给应用程序或确定性的工作流控制。

---

## 总结

LangChain 提供两种主要的系统提示词配置方式：

- `system_prompt`：适合角色和规则相对固定的 Agent。
- `@dynamic_prompt`：适合需要根据用户、上下文或执行阶段动态调整行为的 Agent。

对于简单应用，可以直接使用：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    system_prompt="你是一名编程学习助手。"

)
```

对于需要动态控制的应用，可以使用：

```
@dynamic_prompt

def my_prompt(request: ModelRequest) -> str:

    return "根据当前任务生成的系统提示词"



agent = create_agent(

    model="deepseek:deepseek-flash",

    middleware=[my_prompt]

)
```

**核心原则：**固定的角色、规则放在基础 System Prompt 中，用户相关的变化通过 Dynamic Prompt 与可信 Context 提供。提示词负责指导模型，而关键业务约束应由程序保证。
