# LangChain LangSmith——可观测性与调试

> 来源: https://www.runoob.com/langchain/langchain-langsmith.html

---

# LangChain LangSmith——可观测性与调试

前面的教程中，我们已经使用 LangChain 构建了 RAG Agent、智能客服机器人和多工具个人助手。

但随着 Agent 变得复杂，一个新的问题也出现了：**当 AI 回答错误、工具调用失败或响应速度变慢时，我们应该如何定位问题？**

例如，一个 RAG Agent 没有正确回答用户的问题，可能是因为模型没有调用检索工具、检索结果不相关，或者模型没有正确理解资料。

仅仅查看最终回答，通常无法快速判断问题发生在哪一步。

这时就可以使用 LangSmith。

LangSmith 是 LangChain 提供的 AI 应用可观测性、调试和评估平台，能够追踪 Agent 的模型调用、工具执行、输入输出、耗时以及部分 Token 使用情况。

本章节基于 LangChain 1.x，介绍 LangSmith 的基本使用、自动追踪、自定义 Trace、项目管理、性能分析、追踪采样和数据隐私。

---

## 一、什么是 LangSmith？

LangSmith 可以帮助开发者了解 AI 应用的实际运行情况。

例如，用户询问：

```
帮我查一下 Python 入门课程。
```

Agent 可能经历以下过程：

```
用户问题

    ↓

Agent 开始执行

    ↓

调用 DeepSeek 模型

    ↓

模型选择 search_course 工具

    ↓

执行 search_course("Python")

    ↓

工具返回课程资料

    ↓

再次调用 DeepSeek

    ↓

生成最终回答

    ↓

Agent 执行完成
```

如果没有追踪工具，开发者通常只能依靠 print() 和程序日志查看部分过程。

启用 LangSmith 后，可以在控制台中查看这些操作的调用关系、执行时间以及可记录的输入输出。

### LangSmith 的主要功能

| 功能 | 说明 | 典型用途 |
| --- | --- | --- |
| Tracing | 追踪模型与工具的执行过程 | 排查错误调用 |
| Observability | 观察延迟、调用量及错误情况 | 性能分析 |
| Token Usage | 记录可获取的 Token 使用数据 | 成本分析 |
| Debugging | 查看历史输入、输出和执行步骤 | 问题定位 |
| Evaluation | 使用测试数据集评估应用效果 | 模型版本比较 |
| Feedback | 收集和分析反馈 | 持续改进 Agent |

**注意：**LangSmith 展示的是应用能够记录的执行轨迹，并不意味着能够读取模型未公开的内部推理过程。

---

## 二、几个重要概念

在使用 LangSmith 之前，需要了解几个基本术语。

| 概念 | 说明 |
| --- | --- |
| Trace | 一次请求的完整追踪记录 |
| Run / Span | 执行过程中的一个操作，例如模型调用或工具执行 |
| Project | 用于分组管理 Trace 的项目 |
| Metadata | 附加的环境、版本等信息 |
| Tags | 用于筛选追踪记录的标签 |
| Dataset | 用于评估的测试数据集 |

例如，一个智能客服的 Trace 可能包含多个子 Run：

```
Trace：客服问题处理

    |

    ├── Run：Agent

    |

    ├── Run：ChatDeepSeek

    |

    ├── Run：search_knowledge

    |

    └── Run：ChatDeepSeek
```

通过这个执行结构，可以看到每一步花费了多少时间，以及问题可能发生在哪里。

---

## 三、注册 LangSmith

首先访问 LangSmith 官方网站：

<https://smith.langchain.com/>

登录后，在 Workspace 的相关设置中创建或获取 API Key。

API Key 用于让本地 Python 程序把追踪数据发送到对应的 LangSmith 工作空间。

**注意：**LangSmith API Key 与 DeepSeek API Key 是两套独立的凭据，不能互相替代。

---

## 四、安装依赖

如果已经安装了 LangChain，可以额外安装 LangSmith SDK：

```
pip install -U langsmith
```

为了运行本文的完整 Agent 示例，建议安装：

```
pip install -U langchain langchain-deepseek langsmith python-dotenv
```

其中：

| 包 | 用途 |
| --- | --- |
| `langchain` | 创建 Agent 和 Tools |
| `langchain-deepseek` | 调用 DeepSeek 模型 |
| `langsmith` | LangSmith SDK 与自定义追踪 |
| `python-dotenv` | 加载环境变量 |

LangChain 本身具有 LangSmith 追踪集成。对于一般的 Agent 自动追踪，设置好环境变量后通常不需要再为每一个工具手动编写追踪代码。

---

## 五、配置 LangSmith 环境变量

在项目目录中创建 `.env` 文件：

## 实例：.env

# 开启 LangSmith 追踪
LANGSMITH\_TRACING=true

# LangSmith API Key
LANGSMITH\_API\_KEY=your\_langsmith\_api\_key

# 追踪项目名称
LANGSMITH\_PROJECT=runoob-agent-demo

# DeepSeek 模型密钥
DEEPSEEK\_API\_KEY=your\_deepseek\_api\_key

主要环境变量：

| 环境变量 | 作用 | 示例 |
| --- | --- | --- |
| `LANGSMITH_TRACING` | 是否启用追踪 | true |
| `LANGSMITH_API_KEY` | LangSmith API 凭据 | API Key |
| `LANGSMITH_PROJECT` | 追踪项目名称 | runoob-agent-demo |
| `LANGSMITH_ENDPOINT` | LangSmith API 地址 | 按部署区域配置 |
| `LANGSMITH_WORKSPACE_ID` | 指定工作空间 | Workspace ID |

其中 `LANGSMITH_ENDPOINT` 通常可以省略，使用默认的服务地址。

如果账号所在区域不是默认的美国区域，需要按 LangSmith 官方文档配置相应的区域 Endpoint。

当 API Key 与多个 Workspace 关联，或者使用需要明确指定 Workspace 的凭据时，应配置 `LANGSMITH_WORKSPACE_ID`。

**版本说明：**旧教程可能使用 `LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY` 和 `LANGCHAIN_PROJECT`。在新项目中，建议统一使用 LANGSMITH 前缀，避免混淆。

---

## 六、自动追踪 Agent

LangSmith 最方便的地方是自动追踪。

配置环境变量后，正常创建并调用 LangChain Agent，模型和工具等支持追踪的组件通常会自动记录执行信息。

下面创建一个简单的课程查询 Agent。

### 完整实例：自动追踪

## 实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.tools import tool
from langchain\_deepseek import ChatDeepSeek

load\_dotenv()

@tool
def search\_course(keyword: str) -> str:
    """查询模拟编程课程。"""
    print(f"[工具执行] search\_course: {keyword}")
    courses = {
        "Python": "Python 入门、Python 函数、Python 面向对象",
        "Java": "Java 基础、Java 面向对象"
    }
    return courses.get(keyword, "没有对应的模拟课程资料")

agent = create\_agent(
    model=ChatDeepSeek(model="deepseek-chat", temperature=0),
    tools=[search\_course],
    system\_prompt="你是编程学习助手。用户查询课程时应先调用 search\_course。课程资料均为演示数据。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请用工具查询 Python 有哪些课程。"}]
})

print("最终回答：", result["messages"][-1].content)
print("执行完成，请在 LangSmith 项目中查看 Trace。")

运行：

```
python trace_agent.py
```

如果配置正确，LangSmith 会接收这次 Agent 执行的追踪数据。

在控制台中打开：

```
runoob-agent-demo
```

就可以查看相关执行记录。

实际 Trace 是否出现，取决于 API Key、工作空间配置、网络连接以及追踪数据是否成功上传。

---

## 七、查看 Agent 的执行轨迹

在 LangSmith 中选择对应 Project，然后打开某条 Trace。

你通常可以看到以下信息：

### 1、执行时间线

例如：

```
Agent

  ├── ChatDeepSeek

  │     └── 请求 search_course

  ├── search_course

  │     └── 返回模拟课程

  └── ChatDeepSeek

        └── 生成最终回答
```

这样可以判断模型实际调用了哪些工具，而不是仅依靠最终回答推测。

### 2、输入和输出

可以检查模型收到的消息、工具参数以及模型返回的内容。

例如：

```
Tool：search_course



Input：

{"keyword": "Python"}



Output：

"Python 入门、Python 函数、Python 面向对象"
```

如果工具参数不正确，就可以进一步检查模型的工具选择逻辑与 Prompt。

### 3、执行耗时

可以分析耗时主要发生在：

- 模型生成。
- 工具执行。
- 外部 API 请求。
- 检索或其他应用处理步骤。

例如一个 Agent 总共耗时 8 秒，其中 6 秒来自模型请求，就应优先检查模型调用，而不是盲目优化 Python 字符串处理。

### 4、错误信息

如果某个被追踪步骤抛出异常，LangSmith 可以记录相应的错误信息。

错误细节的完整程度取决于 SDK、异常处理以及实际运行环境。

---

## 八、Token 使用量与费用分析

LangSmith 可以展示可获取的模型 Token 用量和费用信息，帮助开发者分析 Agent 的调用成本。

例如一次 Agent 请求可能包含：

| 步骤 | 输入 Token | 输出 Token |
| --- | --- | --- |
| 第一次模型调用 | 400 | 80 |
| 第二次模型调用 | 650 | 120 |
| 合计 | 1050 | 200 |

以上数字仅用于演示，不代表真实模型消耗。

需要注意，Token 用量与成本不是同一个概念。

即使模型接口提供了准确的 Token 数量，如果 LangSmith 尚未识别该模型的价格，也不一定能够正确计算费用。

**建议：**正式成本统计应以模型服务商的实际计费记录为准，并结合 LangSmith 分析每次请求的成本结构。

---

## 九、查看实际工具调用次数

除了使用 LangSmith 界面，还可以在代码中检查 Agent 是否真正调用了工具。

### 实例：打印工具调用记录

## 实例

result = agent.invoke({
    "messages": [{"role": "user", "content": "请查询 Python 课程。"}]
})

for msg in result["messages"]:
    if msg.type == "ai" and getattr(msg, "tool\_calls", []):
        for call in msg.tool\_calls:
            print("模型请求工具：", call["name"])
            print("调用参数：", call["args"])

    elif msg.type == "tool":
        print("实际工具：", msg.name)
        print("返回内容：", str(msg.content)[:200])

这适合本地调试。

LangSmith 则可以进一步将每次模型和工具调用以可视化方式关联起来，方便查看完整执行流程。

---

## 十、使用 @traceable 手动追踪函数

除了自动追踪 LangChain Agent，还可以通过 LangSmith 的 `@traceable` 装饰器，为普通 Python 函数创建独立的追踪记录。

例如，我们希望追踪：

```
用户问题预处理

        ↓

    调用 Agent

        ↓

    格式化最终结果
```

可以将整个业务函数标记为一个 Trace 节点。

### 实例：追踪业务函数

## 实例

from langsmith import traceable

@traceable(name="process\_user\_query", run\_type="chain")
def process\_user\_query(query: str) -> dict:
    """预处理用户问题并调用 Agent。"""
    cleaned = query.strip()

    result = agent.invoke({
        "messages": [{"role": "user", "content": cleaned}]
    })

    return {
        "query": cleaned,
        "answer": result["messages"][-1].content
    }

result = process\_user\_query("  查询 Python 课程  ")
print(result["answer"])

如果追踪成功，LangSmith 中可以看到：

```
process_user_query

    |

    └── Agent

         |

         ├── ChatDeepSeek

         ├── search_course

         └── ChatDeepSeek
```

`@traceable` 很适合追踪自己编写的业务函数，例如输入清理、权限检查、RAG 查询和结果转换。

**注意：**追踪普通 Python 函数时，只有被显式追踪的函数及其支持上下文传播的子调用，才会形成相应的可观测执行结构。并不是任意第三方代码都会自动变成独立 Trace 节点。

---

## 十一、给追踪添加 Tags 和 Metadata

实际项目中，你可能需要区分不同环境、模型版本和业务场景。

例如：

```
开发环境：development

生产环境：production



业务类型：course-search

模型：deepseek-chat

版本：v1.0
```

可以在调用 Agent 时使用 `config` 添加 Tags 和 Metadata。

### 实例

## 实例

result = agent.invoke(
    {"messages": [{"role": "user", "content": "请查询 Java 入门课程。"}]},
    config={
        "run\_name": "runoob\_course\_query",
        "tags": ["development", "course-search"],
        "metadata": {
            "app\_version": "1.0.0",
            "environment": "development",
            "feature": "course\_search"
        }
    }
)

print(result["messages"][-1].content)

这样可以在 LangSmith 中根据不同标签筛选追踪记录。

例如，单独查看：

```
environment = production
```

或者：

```
feature = course_search
```

**注意：**Metadata 不应包含密码、API Key、访问令牌或不必要的用户隐私信息。

---

## 十二、为不同环境设置 Project

实际项目建议将开发和生产环境分开。

例如：

```
runoob-agent-dev

runoob-agent-staging

runoob-agent-production
```

可以通过环境变量设置默认项目：

```
LANGSMITH_PROJECT=runoob-agent-dev
```

也可以针对单个调用动态指定 Project。

### 实例：tracing\_context

## 实例

import langsmith as ls

with ls.tracing\_context(
    project\_name="runoob-agent-test",
    enabled=True
):
    result = agent.invoke({
        "messages": [{"role": "user", "content": "介绍 Python 入门课程。"}]
    })

print(result["messages"][-1].content)

这次调用会尝试记录到指定的项目中。

`tracing_context()` 也适合在某些需要临时启用或关闭追踪的代码块中使用。

---

## 十三、如何关闭 LangSmith 追踪？

如果不希望应用上传追踪信息，可以在环境变量中设置：

```
LANGSMITH_TRACING=false
```

或者对指定代码块关闭追踪：

## 实例

import langsmith as ls

with ls.tracing\_context(enabled=False):
    result = agent.invoke({
        "messages": [{"role": "user", "content": "你好"}]
    })

print(result["messages"][-1].content)

这里仅关闭指定范围的 LangSmith 追踪，不影响 Agent 的正常模型调用。

需要注意，其他独立的应用日志、监控工具或服务商日志不会因为关闭 LangSmith 而自动停止记录。

---

## 十四、生产环境设置采样率

在开发阶段，可以开启全部追踪，方便排查问题。

但在高流量生产系统中，记录所有请求可能带来额外的数据存储、传输和处理成本。

LangSmith SDK 提供追踪采样配置。

例如：

## 实例：采样配置

LANGSMITH\_TRACING=true
LANGSMITH\_API\_KEY=your\_langsmith\_api\_key
LANGSMITH\_PROJECT=runoob-agent-production

# 对符合 SDK 采样机制的追踪采用 10% 采样率
LANGSMITH\_TRACING\_SAMPLING\_RATE=0.1

采样率通常使用 0 到 1 之间的数值：

| 设置 | 含义 |
| --- | --- |
| 1.0 | 全部采样 |
| 0.5 | 约 50% 采样 |
| 0.1 | 约 10% 采样 |
| 0.01 | 约 1% 采样 |
| 0 | 不采样 |

具体生效范围取决于 LangSmith SDK 和追踪客户端的实现方式。

采样适合控制正常请求的记录量，但并不保证所有重要错误都会被采样到。

对于关键异常，建议结合服务端错误日志、告警和其他监控系统，而不是完全依赖随机采样。

---

## 十五、保护用户隐私与敏感数据

LangSmith 可以记录模型请求与工具输入输出。这些信息可能包含用户聊天内容、内部文档和其他业务数据。

因此，在生产环境中应认真考虑数据保护。

### 隐藏输入和输出

如果不希望追踪内容中记录原始输入输出，可以设置：

```
LANGSMITH_HIDE_INPUTS=true

LANGSMITH_HIDE_OUTPUTS=true
```

这可以减少向追踪系统暴露原始内容。

但代价是，开发者也无法直接在 LangSmith 中查看被隐藏的输入或输出。

### 生产安全建议

- 不要在追踪 Metadata 中存储密码、Token、支付凭据等敏感数据。
- 根据业务情况隐藏或脱敏用户消息和工具结果。
- 控制 LangSmith 工作空间的成员权限。
- 为追踪数据设置合理的数据保留策略。
- 不要未经评估就上传企业机密文档或个人隐私内容。
- 根据部署地区和业务要求评估数据传输与合规问题。

**重要：**隐藏 LangSmith 的输入输出，并不等于数据不再发送给底层模型服务商。模型调用本身仍需要根据实际服务商的数据处理政策单独评估。

---

## 十六、LangSmith 与普通日志有什么区别？

普通 Python 日志也可以记录工具执行信息，例如：

```
print("开始调用模型")

print("调用 search_course")

print("执行完成")
```

但当 Agent 发生复杂的多轮模型调用时，手动日志通常很难清晰展示完整的调用关系。

| 能力 | 普通 Python 日志 | LangSmith |
| --- | --- | --- |
| 记录文字消息 | 支持 | 支持 |
| 调用关系可视化 | 需要自行实现 | 提供追踪界面 |
| 模型和工具步骤 | 手动记录 | 支持自动追踪 |
| 调用耗时 | 手动统计 | 提供相关指标 |
| Token 使用量 | 需要自行获取 | 可展示可用数据 |
| 测试集评估 | 自行开发 | 提供评估工作流 |

实际项目中，LangSmith 不一定需要替代原有日志系统。

更常见的做法是：

```
应用日志 → 记录业务错误、HTTP 请求和安全事件



LangSmith → 追踪模型、工具和 Agent 执行



监控平台 → 监控服务器、数据库和系统资源
```

三者相互补充。

---

## 十七、使用 LangSmith 评估 Agent

LangSmith 不仅可以用于调试，还提供数据集和评估功能。

例如，开发一个课程查询 Agent，可以准备以下测试问题：

| 测试问题 | 期望行为 |
| --- | --- |
| Python 有哪些入门课程？ | 调用课程搜索工具 |
| Java 有什么基础课程？ | 查询 Java 课程 |
| 知识库没有的课程价格是多少？ | 不编造价格 |
| 查询不存在的课程 | 说明没有结果 |

这些问题可以作为测试集，用于比较不同模型、Prompt 和工具实现的效果。

评估方式可以包括：

- **规则评估：**检查是否调用了正确工具、是否包含必要字段。
- **参考答案比较：**将输出与人工准备的标准答案比较。
- **LLM-as-a-Judge：**使用模型按照评价标准打分。
- **人工评估：**由开发者或业务人员检查回答质量。

需要注意，LLM-as-a-Judge 的评分也可能出错，不能直接等同于真实准确率。

建议把规则测试与人工抽样结合起来使用。

---

## 十八、如何分析一个 RAG Agent 的问题？

假设用户询问：

```
Python 的装饰器有什么作用？
```

Agent 最终回答错误。

在 LangSmith 中，可以按照以下顺序排查：

1. 模型是否实际调用了知识库检索工具。
2. 检索工具是否收到了正确的查询参数。
3. 向量数据库返回的文档是否包含装饰器知识。
4. 工具结果是否被完整传给模型。
5. 模型的最终回答是否正确使用检索资料。
6. 是否存在异常、超时或者上下文截断。

例如：

```
用户问题：Python 装饰器有什么作用？

             ↓

模型调用 search_knowledge

             ↓

检索参数：Python 装饰器

             ↓

返回资料：Python 列表、Python 字典

             ↓

模型生成错误回答
```

这种情况下，问题可能主要出在知识库检索，而不是模型生成能力。

相反，如果检索结果已经包含准确的装饰器说明，但模型仍然回答错误，就需要检查 Prompt、上下文处理或模型表现。

**这正是可观测性的价值：帮助开发者定位问题发生在哪个环节，而不是一遇到错误就直接更换模型。**

---

## 十九、常见问题

### 1、LangSmith 必须与 LangChain 一起使用吗？

不必须。LangSmith SDK 可以追踪自定义 Python 函数以及其他框架中的应用逻辑，但与 LangChain 配合时通常更方便。

### 2、LangSmith 会修改 Agent 的回答吗？

一般的追踪配置用于记录执行信息，不负责改变模型业务逻辑。但追踪本身可能带来一定的运行开销。

### 3、为什么控制台没有出现 Trace？

应检查 LANGSMITH\_TRACING、API Key、Project、Workspace、区域 Endpoint 和网络连接。

### 4、为什么配置了 DeepSeek Key，还需要 LangSmith Key？

DeepSeek Key 用于调用模型，LangSmith Key 用于上传和管理追踪数据。

### 5、LANGCHAIN\_TRACING\_V2 还能使用吗？

它属于旧配置命名。新教程建议统一采用 LANGSMITH\_TRACING。

### 6、LangSmith 能看到模型全部内部思考吗？

不能。它展示的是应用和模型接口实际返回、并被记录的执行数据，不等于模型隐藏的推理过程。

### 7、Token 和费用为什么没有完整显示？

可能因为模型接口未提供完整用量信息，或者当前模型的价格映射与计费数据不完整。

### 8、@traceable 和自动追踪有什么区别？

自动追踪适合 LangChain 模型和工具调用；@traceable 可以为自定义 Python 函数建立追踪节点。

### 9、如何区分开发和生产环境？

可以使用不同的 LANGSMITH\_PROJECT，并通过 Tags 和 Metadata 标记环境。

### 10、如何临时关闭追踪？

可以设置 LANGSMITH\_TRACING=false，或者使用 tracing\_context(enabled=False)。

### 11、采样率设置为 0.1 是什么意思？

表示在相应 SDK 采样机制下，对追踪进行约 10% 的采样，而不是保证每连续十次请求恰好记录一次。

### 12、LangSmith 会记录用户输入吗？

启用追踪后，默认可能记录输入输出及工具参数等信息。敏感业务应配置脱敏或隐藏，并评估数据保护要求。

### 13、历史 Trace 可以直接当成 Agent 回放执行吗？

Trace 首先是历史执行记录。重新运行 Agent 是另一种操作，可能产生新的模型结果或真实工具副作用。

### 14、LangSmith 可以用来评估 RAG 吗？

可以。可以评估检索工具是否调用、回答是否有依据，以及不同版本的问答表现。

---

## 总结

LangSmith 是 LangChain Agent 开发中的重要可观测性工具。

本章节介绍了：

- Trace、Run、Project 的基本概念。
- 通过环境变量启用 LangSmith。
- 自动追踪 Agent 的模型和工具调用。
- 在控制台中查看执行时间线。
- 分析 Token、耗时和错误。
- 通过 `@traceable` 追踪自定义函数。
- 使用 Tags 和 Metadata 管理追踪记录。
- 通过 tracing\_context 选择性启用追踪。
- 使用采样率控制生产环境追踪量。
- 隐藏输入输出并保护敏感数据。
- 通过 Dataset 和 Evaluation 评估 Agent。

最常用的 LangSmith 配置：

```
LANGSMITH_TRACING=true

LANGSMITH_API_KEY=your_langsmith_api_key

LANGSMITH_PROJECT=my-agent-project
```

配置后正常执行：

```
result = agent.invoke({

    "messages": [{"role": "user", "content": "查询 Python 课程"}]

})
```

然后访问 [LangSmith 控制台](https://smith.langchain.com/) 查看执行追踪。

**核心原则：**开发 Agent 不仅要关心最终回答是否正确，还要能够解释应用实际调用了哪些工具、在哪一步失败、花费了多少资源。LangSmith 的作用就是让这些执行过程变得可观察、可分析和可评估。
