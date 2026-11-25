# LangChain 中间件钩子——@before\_model 与 @after\_model

> 来源: https://www.runoob.com/langchain/langchain-before-after-model.html

---

# LangChain 中间件钩子——@before\_model 与 @after\_model

在 LangChain Agent 中，@before\_model 和 @after\_model 是两个非常常用的 Middleware 钩子。

它们分别在模型调用之前和之后执行，可以用于消息预处理、上下文管理、输入检查、响应审核和流程控制。

与普通工具不同，Middleware 不需要由模型主动选择，而是由 Agent 执行框架在对应阶段运行。

本章节基于 LangChain 1.x，介绍两个钩子的工作机制、状态更新、消息裁剪、回复替换及 `can_jump_to` 流程控制。

---

## 一、before\_model 与 after\_model 的区别

一个典型的 Agent 执行循环如下：

```
用户输入

    ↓

before_model

    ↓

模型调用

    ↓

after_model

    ↓

是否需要调用工具？

    ├── 是 → 执行工具 → 回到 before_model

    └── 否 → 返回最终结果
```

两者的主要区别：

| 对比项 | before\_model | after\_model |
| --- | --- | --- |
| 执行时机 | 模型调用前 | 模型产生响应后 |
| 主要对象 | 输入消息与当前状态 | 模型响应与当前状态 |
| 典型用途 | 消息管理、输入检查、提前结束 | 响应审核、消息修正、结果统计 |
| 运行次数 | 可能多次 | 可能多次 |
| 流程控制 | 支持 can\_jump\_to | 支持 can\_jump\_to |

如果 Agent 需要多轮工具调用，模型也可能调用多次，因此这两个钩子不一定只执行一次。

---

## 二、钩子的定义方式

可以通过装饰器快速创建 Middleware。

基本语法：

## 实例

from langchain.agents.middleware import before\_model, after\_model, AgentState
from langgraph.runtime import Runtime

@before\_model
def pre\_model(state: AgentState, runtime: Runtime) -> None:
    """模型调用前执行。"""
    print("准备调用模型")

@after\_model
def post\_model(state: AgentState, runtime: Runtime) -> None:
    """模型调用后执行。"""
    print("模型已经返回结果")

随后将两个中间件传给 `create_agent()`：

```
agent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[],

    middleware=[pre_model, post_model]

)
```

这两个钩子接收两个重要参数：

- `state`：当前 Agent State，包括 messages 等字段。
- `runtime`：当前执行上下文，可以访问运行时 Context 等信息。

如果只是记录日志，可以返回 `None`，不需要修改 State。

---

## 三、@before\_model：模型调用前处理

`@before_model` 在每次模型调用前运行，可以读取当前消息历史，并根据需要提交状态更新。

例如，检查消息数量：

## 实例

from langchain.agents.middleware import before\_model, AgentState
from langgraph.runtime import Runtime

@before\_model
def inspect\_messages(state: AgentState, runtime: Runtime) -> None:
    """检查本次模型调用前的消息数量。"""
    messages = state.get("messages", [])
    human\_count = sum(msg.type == "human" for msg in messages)
    print(f"消息总数：{len(messages)}，用户消息：{human\_count}")

需要注意，消息数量并不等于对话轮数。

一次 Agent 工具调用可能增加 AIMessage 和 ToolMessage 等多条消息。

---

## 四、场景一：限制对话历史长度

随着对话增加，Agent State 中的消息数量可能不断增长，从而占用更多上下文 Token。

可以使用 before\_model 管理历史消息，但必须理解 `add_messages` Reducer 的工作方式。

### 错误写法

```
return {"messages": messages[-6:]}
```

这种写法并不能保证只保留最近六条消息。

因为 `messages` 使用 `add_messages` 合并更新。原有的其他消息不会仅仅因为没有出现在新列表中就自动删除。

### 正确方式：通过 RemoveMessage 删除消息

如果需要真正修改已保存的消息历史，可以使用 `RemoveMessage`。

下面先给出一个简单示例，删除早期的用户与 AI 普通文本消息。

## 实例

from langchain.agents.middleware import before\_model, AgentState
from langchain.messages import RemoveMessage
from langgraph.runtime import Runtime

@before\_model
def trim\_simple\_history(state: AgentState, runtime: Runtime) -> dict | None:
    """裁剪不含工具调用的演示对话。"""
    messages = state.get("messages", [])
    if len(messages) <= 6:
        return None

    # 仅用于没有工具调用的简单对话
    if any(getattr(msg, "tool\_calls", []) or msg.type == "tool" for msg in messages):
        return None

    old = [msg for msg in messages[:-6] if msg.id]
    if not old:
        return None

    return {"messages": [RemoveMessage(id=msg.id) for msg in old]}

这段代码与原文不同，它会明确向 Reducer 提交删除指令。

但为了避免误删工具消息，示例主动跳过包含 Tool Calling 的对话历史。

**注意：**正式项目不能简单按消息条数随意删除，尤其不能让 AIMessage 中的工具调用与对应 ToolMessage 脱离。

### 推荐方案：使用 SummarizationMiddleware

对于长对话，更推荐使用 LangChain 内置的 `SummarizationMiddleware`，在上下文达到阈值时生成摘要，而不是盲目保留最后几条消息。

如果需要直接清理旧的工具输出，也可以考虑 `ContextEditingMiddleware`。

因此，建议：

| 需求 | 建议方案 |
| --- | --- |
| 简单演示删除历史消息 | RemoveMessage |
| 长会话压缩 | SummarizationMiddleware |
| 清理旧工具结果 | ContextEditingMiddleware |
| 只改变本次模型请求的上下文 | 使用适合的模型请求包装或上下文处理机制 |

---

## 五、场景二：输入内容检查

可以通过 before\_model 在模型调用前检查用户输入，并在满足规则时提前结束 Agent。

例如，对于某些演示专用关键词，直接返回提示，不调用模型。

需要注意，不建议直接将"密码""身份证"等词作为全部禁止关键词，因为用户可能只是正常询问密码安全、身份认证等技术知识。

### 实例：演示关键词拦截

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import before\_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

load\_dotenv()

@before\_model(can\_jump\_to=["end"])
def content\_filter(state: AgentState, runtime: Runtime) -> dict | None:
    """拦截约定的演示关键词。"""
    users = [msg for msg in state["messages"] if msg.type == "human"]
    if not users:
        return None

    content = users[-1].content
    if isinstance(content, str) and "BLOCK\_DEMO" in content:
        return {
            "jump\_to": "end",
            "messages": [AIMessage(content="该请求已被演示规则拦截。")]
        }
    return None

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[content\_filter],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程助手。"
)

result = agent.invoke({"messages": [{"role": "user", "content": "BLOCK\_DEMO"}]})
print(result["messages"][-1].content)

输出：

```
该请求已被演示规则拦截。
```

这里有两个关键点：

- 通过 `can_jump_to=["end"]` 声明允许跳转到结束流程。
- 通过 `AIMessage` 添加助手回复，而不是错误地用 HumanMessage 返回提示。

**注意：**这个示例只是演示输入拦截，不能当作完整的敏感信息识别或安全防护系统。

如果需要保护用户个人信息，应使用更可靠的规则、数据检测机制以及服务端权限控制。

---

## 六、@after\_model：模型响应后处理

`@after_model` 会在模型返回响应后执行，适合对输出内容进行检查或记录。

它既可能在最终回复后运行，也可能在模型请求工具调用后运行。

因此，处理最终文本之前，应先检查模型是否产生了 `tool_calls`。

### 实例：检查模型是否请求工具

## 实例

from langchain.agents.middleware import after\_model, AgentState
from langgraph.runtime import Runtime

@after\_model
def inspect\_output(state: AgentState, runtime: Runtime) -> None:
    """检查模型响应类型。"""
    last = state["messages"][-1]
    if getattr(last, "tool\_calls", []):
        print("模型请求工具：", [c["name"] for c in last.tool\_calls])
    else:
        print("模型文本：", str(last.content)[:100])

如果模型请求工具，Agent 随后可能继续执行工具节点。

如果模型没有产生工具调用，并且不存在其他流程控制，通常会结束当前循环。

---

## 七、场景三：审核并替换模型回复

对于某些应用，需要在模型返回回复后检查内容。

例如，某个系统禁止向用户展示约定的内部测试标记。

这里使用 `INTERNAL_DEMO_ONLY` 作为教学用拦截字符串。

### 为什么不能直接新增 AIMessage？

原文使用下面的方式：

```
return {

    "messages": [AIMessage(content="替换后的回复")]

}
```

这并不能保证覆盖旧回复，因为新 AIMessage 通常具有不同的消息 ID。

如果要替换已经存在的消息，应使用相同的消息 ID，让 `add_messages` Reducer 执行替换。

### 正确实例：保留原消息 ID

## 实例

from langchain.agents.middleware import after\_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

@after\_model
def response\_audit(state: AgentState, runtime: Runtime) -> dict | None:
    """替换包含演示内部标记的最终回复。"""
    last = state["messages"][-1]

    if last.type != "ai" or last.tool\_calls:
        return None
    if not isinstance(last.content, str) or "INTERNAL\_DEMO\_ONLY" not in last.content:
        return None

    return {"messages": [AIMessage(
        id=last.id,
        content="该回复包含演示用内部标记，已被替换。"
    )]}

这里的关键代码是：

```
id=last.id
```

由于新旧消息使用相同的 ID，Reducer 可以替换已有的模型回复，而不是在消息历史中额外追加一条回答。

**注意：**如果已经将未经审核的消息片段通过 Streaming 发送给前端，那么 after\_model 的替换无法撤回已经传输的内容。

对于需要严格审核后才能展示的内容，应先缓冲响应并完成审核，再发送给用户，或采用其他可靠的输出防护机制。

---

## 八、场景四：为最终回答添加提示信息

如果需要自动在模型最终回答后添加说明，也可以通过 after\_model 更新最后一条消息。

例如，为 AI 生成内容添加固定的说明。

### 实例：追加内容声明

## 实例

from langchain.agents.middleware import after\_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

DISCLAIMER = "\n\n---\n以上内容由 AI 生成，仅供参考。"

@after\_model
def append\_disclaimer(state: AgentState, runtime: Runtime) -> dict | None:
    """在最终文本回复后添加说明。"""
    last = state["messages"][-1]

    if last.type != "ai" or last.tool\_calls:
        return None
    if not isinstance(last.content, str) or not last.content or DISCLAIMER in last.content:
        return None

    return {"messages": [AIMessage(id=last.id, content=last.content + DISCLAIMER)]}

如果模型返回：

```
Python 是一门适合初学者的编程语言。
```

更新后的消息内容为：

```
Python 是一门适合初学者的编程语言。



---

以上内容由 AI 生成，仅供参考。
```

这里再次使用 `id=last.id` 替换原有回复。

同时还检查了声明是否已经存在，避免重复追加。

**注意：**这里为了便于教学，构造了新的 AIMessage。生产环境如果需要保留原消息的响应元数据、使用量信息等，可以优先考虑复制原消息并只更新必要字段。

---

## 九、can\_jump\_to：控制 Agent 执行流程

`can_jump_to` 用于声明钩子允许将 Agent 路由到哪些目标。

可选目标包括：

| 目标 | 含义 | 常见用途 |
| --- | --- | --- |
| `"end"` | 结束 Agent 主循环 | 提前退出、输入拦截 |
| `"model"` | 转向模型执行路径 | 需要重新进行模型处理 |
| `"tools"` | 转向工具节点 | 已有合法工具调用请求时的流程控制 |

例如：

```
@before_model(can_jump_to=["end"])

def check_exit(state, runtime):

    ...
```

只有明确设计了路由目标，才应该返回对应的 `jump_to` 值。

### 实例：用户输入再见时提前结束

## 实例

from langchain.agents.middleware import before\_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

@before\_model(can\_jump\_to=["end"])
def conditional\_exit(state: AgentState, runtime: Runtime) -> dict | None:
    """用户发送结束语时直接返回固定回复。"""
    users = [m for m in state["messages"] if m.type == "human"]
    if not users or not isinstance(users[-1].content, str):
        return None

    if users[-1].content.strip().lower() in {"再见", "拜拜", "bye"}:
        return {
            "jump\_to": "end",
            "messages": [AIMessage(content="再见！欢迎下次继续学习编程。")]
        }
    return None

这个示例可以让 Agent 对简单结束语直接返回固定回复，不需要调用模型。

需要特别注意：

- `jump_to="end"` 不是 Python 的 `return`，而是 LangGraph 的执行路由控制。
- `jump_to="tools"` 不会凭空创建工具调用。
- `jump_to="model"` 需要避免形成无限模型调用循环。
- 应明确设置 `can_jump_to`，不要依赖未声明目标时的实现细节。

---

## 十、完整实例：输入检查与回复处理

下面将 before\_model 与 after\_model 组合为一个完整示例。

这个 Agent 支持：

- 记录每次模型调用前的消息数量。
- 遇到特定演示关键词时直接结束。
- 检查模型响应是否包含工具调用。
- 在最终文本回答中添加固定的 AI 内容说明。

文件名：`middleware_hooks_demo.py`

安装依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

配置 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 完整实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import before\_model, after\_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

load\_dotenv()

DISCLAIMER = "\n\n---\n以上内容由 AI 生成，仅供参考。"

@before\_model(can\_jump\_to=["end"])
def check\_input(state: AgentState, runtime: Runtime) -> dict | None:
    """检查输入并记录消息数量。"""
    messages = state.get("messages", [])
    print("[before\_model] 消息数量：", len(messages))

    users = [m for m in messages if m.type == "human"]
    if users and isinstance(users[-1].content, str):
        if "BLOCK\_DEMO" in users[-1].content:
            return {
                "jump\_to": "end",
                "messages": [AIMessage(content="请求已被演示规则拦截。")]
            }
    return None

@after\_model
def process\_output(state: AgentState, runtime: Runtime) -> dict | None:
    """检查最终文本回复并追加说明。"""
    last = state["messages"][-1]
    if last.type != "ai":
        return None

    if last.tool\_calls:
        print("[after\_model] 模型请求了工具")
        return None

    print("[after\_model] 模型已生成文本回复")
    if not isinstance(last.content, str) or not last.content or DISCLAIMER in last.content:
        return None

    return {"messages": [AIMessage(id=last.id, content=last.content + DISCLAIMER)]}

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    middleware=[check\_input, process\_output],
    system\_prompt="你是菜鸟教程 RUNOOB 的编程学习助手，使用简洁中文回答。"
)

def ask(question: str):
    print("\n用户：", question)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("AI：", result["messages"][-1].content)

if \_\_name\_\_ == "\_\_main\_\_":
    ask("Python 适合初学者吗？")
    ask("BLOCK\_DEMO")

### 可能的运行结果

```
用户：Python 适合初学者吗？

[before_model] 消息数量： 1

[after_model] 模型已生成文本回复

AI：Python 语法相对简洁，适合初学者入门。



---

以上内容由 AI 生成，仅供参考。



用户：BLOCK_DEMO

[before_model] 消息数量： 1

AI：请求已被演示规则拦截。
```

这里两个钩子的职责非常明确：

- `check_input`：处理模型调用之前的输入检查和流程控制。
- `process_output`：处理模型调用完成后的结果。

当 before\_model 直接结束 Agent 时，模型不会按正常路径运行，因此 after\_model 通常不会处理这次拦截产生的固定回复。

---

## 十一、before\_model 与 wrap\_model\_call 的区别

有时需要修改消息，有时需要真正拦截模型请求，这两种需求应区分处理。

| 需求 | 推荐方式 |
| --- | --- |
| 检查当前 Agent State | before\_model |
| 更新自定义 State 字段 | before\_model |
| 提前结束 Agent | before\_model + can\_jump\_to |
| 修改本次模型请求的消息 | wrap\_model\_call |
| 切换模型或重试 API 请求 | wrap\_model\_call |
| 检查模型输出 | after\_model |
| 替换已生成的最终消息 | after\_model + 相同消息 ID |

一个重要区别是：

`before_model` 返回的 `messages` 会作为状态更新交给 Reducer 处理；而 `wrap_model_call` 可以修改当前模型请求，而不必通过删除历史消息来改变本次输入。

因此，对于只想临时调整模型上下文、而不希望删除保存的消息历史的场景，通常更适合使用 wrap\_model\_call。

---

## 十二、常见问题

### 1、before\_model 每轮都会执行吗？

如果 Agent 多次调用模型，before\_model 通常会在每次相应模型调用前执行。

### 2、after\_model 能拦截工具调用吗？

可以检查模型响应中的 `tool_calls`，并根据当前支持的流程控制机制决定后续执行。

但它本身不等于工具权限控制。需要控制真实工具操作时，还应在工具执行层完成校验。

### 3、为什么返回最近六条消息没有删除旧消息？

因为 messages 使用 `add_messages` Reducer。只返回较短的消息列表不等于删除旧消息。

需要删除时应使用 RemoveMessage 等受支持的机制。

### 4、为什么 after\_model 返回 AIMessage 后出现两条 AI 回复？

因为新消息使用了不同的 ID，被当作新消息追加。

要替换旧消息，应复用原来的消息 ID。

### 5、为什么内容过滤使用 AIMessage？

因为拦截提示是助手向用户返回的内容，不是用户新发送的消息。

因此不应该使用 HumanMessage。

### 6、can\_jump\_to=["tools"] 能直接执行任意工具吗？

不能。工具节点仍然需要合法的工具调用请求和参数。

### 7、可以同时注册多个 before\_model 吗？

可以，但要注意多个 Middleware 的执行顺序、状态更新和跳转行为之间的相互影响。

### 8、after\_model 适合做完整的内容安全审核吗？

可以作为审核流程中的一个环节，但不能代替专门的安全检测与权限控制。

尤其在流式响应场景中，已经发送的内容不能依靠 after\_model 自动撤回。

### 9、怎样在中间件中发送自定义进度？

可以在支持流式执行的运行环境中使用 Runtime 的 StreamWriter，并通过 `stream_mode="custom"` 消费事件。

### 10、如何限制长对话上下文？

建议优先考虑 LangChain 内置的 SummarizationMiddleware，而不是简单截取固定数量的消息。

### 11、after\_model 一定是 Agent 的最后一步吗？

不是。如果模型请求了工具，after\_model 执行后可能继续调用工具，然后再次进入模型执行循环。

### 12、如何测试中间件是否真的生效？

可以检查返回的 `messages` 和 State，也可以使用 `stream_mode="updates"` 观察节点执行过程。

对于消息替换逻辑，最好测试消息 ID 是否保持不变，以及历史消息数量是否符合预期。

---

## 总结

`@before_model` 和 `@after_model` 是 LangChain Agent 中两个重要的 Middleware 钩子。

主要区别：

```
before_model：

    模型执行前

    检查输入、管理 State、控制执行流程



after_model：

    模型执行后

    审核结果、修改消息、记录输出信息
```

本章节重点学习了：

- 两个钩子的执行时机。
- 通过 AgentState 读取消息历史。
- 使用 RemoveMessage 删除旧消息。
- 使用 can\_jump\_to 提前结束 Agent。
- 通过 AIMessage 返回拦截提示。
- 保留消息 ID，正确替换模型回答。
- 在最终回答中添加固定说明。
- 区分 before\_model 与 wrap\_model\_call 的使用场景。

> **核心原则：**需要读取或更新 Agent State 时使用节点钩子；需要修改当前模型请求时考虑包装钩子。处理消息历史时，必须遵守 add\_messages Reducer 的合并规则。
