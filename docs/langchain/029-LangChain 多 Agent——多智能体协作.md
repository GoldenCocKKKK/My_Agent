# LangChain 多 Agent——多智能体协作

> 来源: https://www.runoob.com/langchain/langchain-multi-agent.html

---

# LangChain 多 Agent——多智能体协作

当任务涉及多个不同领域，或者单个 Agent 需要处理大量工具和复杂流程时，可以考虑使用 Multi-agent（多智能体） 架构。

多 Agent 的核心思想，是将复杂任务拆分成多个相对独立的子任务，让不同 Agent 负责不同领域，再由协调者汇总结果。

例如，一个编程学习助手可以分成课程检索 Agent、代码分析 Agent 和学习规划 Agent。每个 Agent 使用自己的 System Prompt 和工具，不需要掌握所有业务细节。

本章节基于 LangChain 1.x，介绍 Subagents、Supervisor、Router、Handoffs、顺序工作流及多 Agent 的上下文管理。

---

## 一、为什么需要多 Agent？

单个 Agent 可以完成很多工作，但当任务复杂度持续增加时，可能遇到以下问题：

- **上下文过多：**大量系统提示词、文档和工具结果占用上下文窗口。
- **工具选择困难：**可用工具较多时，模型可能选择错误的工具或参数。
- **专业能力混杂：**不同任务需要不同的处理规则和数据来源。
- **维护复杂：**所有业务逻辑集中在一个 Agent 中，修改和测试比较困难。

多 Agent 可以把任务拆分为独立模块，每个 Agent 只接收完成当前任务需要的信息。

但需要注意，多 Agent 并不一定提高回答准确率，也不一定降低成本。

因为每个子 Agent 都可能产生额外的模型调用、Token 消耗和网络延迟。

### 单 Agent 与多 Agent 对比

| 对比项 | 单 Agent | 多 Agent |
| --- | --- | --- |
| 结构 | 一个 Agent 负责全部任务 | 多个专业 Agent 协作 |
| 工具管理 | 集中管理 | 按领域分组 |
| 上下文 | 通常由一个 Agent 处理 | 可以为子任务隔离上下文 |
| 开发成本 | 相对简单 | 需要协调和路由逻辑 |
| 模型调用 | 通常较少 | 可能增加 |
| 适用场景 | 简单任务、少量工具 | 多领域、复杂工作流 |

**建议：**如果使用一个 Agent 加上合理的工具分组就能解决问题，没有必要为了使用多 Agent 而增加架构复杂度。

---

## 二、LangChain 多 Agent 架构模式

LangChain 官方主要介绍以下几种协作模式：

| 模式 | 原理 | 适用场景 |
| --- | --- | --- |
| `Subagents` | 主 Agent 通过工具调用专业子 Agent | 多领域任务、统一协调 |
| `Router` | 先分类任务，再分发给专业 Agent | 客服分类、知识库检索 |
| `Handoffs` | 根据当前状态切换专业处理阶段 | 多阶段客服、人工接管 |
| `Skills` | 按需加载专业知识与操作说明 | 工具和知识很多，但不需要多个独立 Agent |
| `Custom Workflow` | 使用 LangGraph 定义执行节点和边 | 固定顺序、复杂条件与并行任务 |

其中，最容易入门的方式是 Subagents（子 Agent 作为工具）。

---

## 三、方式一：子 Agent 作为工具

Subagents 模式也称为 Supervisor 模式。

它包含一个主 Agent（协调者），以及多个具备专业能力的子 Agent。

执行流程：

```
                 用户问题

                    |

                    v

              主 Agent

              Supervisor

                    |

          +---------+---------+

          |                   |

          v                   v

      课程 Agent          计算 Agent

          |                   |

      查询课程目录          数学计算

          |                   |

          +---------+---------+

                    |

                    v

              主 Agent 汇总

                    |

                    v

                 最终回答
```

主 Agent 不需要掌握每个子 Agent 的全部内部工具，只需要知道什么时候调用对应的专家。

### 基本实现方式

先创建子 Agent：

```
subagent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[...],

    system_prompt="你是专业助手。"

)
```

然后将它包装成工具：

```
@tool

def ask_expert(question: str) -> str:

    """向专业 Agent 咨询。"""

    result = subagent.invoke({

        "messages": [{"role": "user", "content": question}]

    })

    return result["messages"][-1].content
```

最后让主 Agent 调用该工具：

```
coordinator = create_agent(

    model="deepseek:deepseek-flash",

    tools=[ask_expert]

)
```

这里的子 Agent 虽然是独立的 CompiledStateGraph，但父 Agent 是通过 Python 工具函数调用它，而不是自动将子图节点合并到父图中。

---

## 四、完整实例：课程专家与数学专家协作

下面创建三个 Agent：

- 课程专家：查询模拟编程课程。
- 数学专家：进行数值计算。
- 主 Agent：分析用户问题，调用专家并整理答案。

本示例不使用 `eval()` 执行模型生成的表达式，而是定义具有明确参数的数学工具。

文件名：`multi_agent_demo.py`

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
from langchain.tools import tool

load\_dotenv()
MODEL = "deepseek:deepseek-flash"

@tool
def search\_course(keyword: str) -> str:
    """按语言关键词查询模拟编程课程。"""
    courses = {"python": "Python3 基础教程，入门", "java": "Java 基础教程，入门"}
    return courses.get(keyword.strip().lower(), "没有找到匹配的模拟课程")

@tool
def celsius\_to\_fahrenheit(celsius: float) -> float:
    """将摄氏温度转换为华氏温度。"""
    return round(celsius \* 9 / 5 + 32, 2)

course\_agent = create\_agent(
    model=MODEL,
    tools=[search\_course],
    name="course\_expert",
    system\_prompt="你是课程专家。查询课程时使用工具，课程数据仅为教学模拟。"
)

math\_agent = create\_agent(
    model=MODEL,
    tools=[celsius\_to\_fahrenheit],
    name="math\_expert",
    system\_prompt="你是数值计算专家。温度换算必须使用工具，不要猜测计算结果。"
)

@tool
def ask\_course\_expert(question: str) -> str:
    """委派编程课程查询给课程专家。"""
    print("[协调者] 调用课程专家")
    result = course\_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)

@tool
def ask\_math\_expert(question: str) -> str:
    """委派数值计算与温度换算给数学专家。"""
    print("[协调者] 调用数学专家")
    result = math\_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)

coordinator = create\_agent(
    model=MODEL,
    tools=[ask\_course\_expert, ask\_math\_expert],
    name="coordinator",
    system\_prompt="你是协调者。课程问题调用课程专家，计算问题调用数学专家。复合任务应分别委派，再合并结果。"
)

result = coordinator.invoke({
    "messages": [{"role": "user", "content": "推荐一门 Python 入门课程，并计算 25 摄氏度等于多少华氏度。"}]
})
print("最终回答：", result["messages"][-1].content)

### 可能的运行结果

```
[协调者] 调用课程专家

[协调者] 调用数学专家



最终回答：

可以学习 Python3 基础教程，适合编程初学者。



另外，25 摄氏度等于 77 华氏度。
```

这是多 Agent 最典型的用法：主 Agent 不负责具体计算和课程查询，而是将任务委派给对应的专业 Agent。

**注意：**模型不保证按照固定顺序调用专家，也可能只调用其中一个。因此以上日志和回复仅作为可能的运行结果。

---

## 五、子 Agent 的上下文隔离

Subagents 模式的重要价值之一，是减少不相关的上下文进入子任务。

例如，用户在主 Agent 中连续讨论了多个主题：

```
1. Python 课程

2. Java 基础

3. 数据库优化

4. 温度计算
```

当主 Agent 调用数学专家时，可以只发送当前数学问题：

```
25 摄氏度等于多少华氏度？
```

数学专家不需要处理前面所有课程和数据库相关的聊天记录。

这样可以降低子任务的上下文干扰。

### 默认无状态

如果子 Agent 没有配置 Checkpointer，并且每次都使用独立的消息输入，那么不同调用之间不会自动保留子 Agent 的历史对话。

例如：

```
result = math_agent.invoke({

    "messages": [{"role": "user", "content": question}]

})
```

每次只发送当前问题，形成独立的子任务上下文。

如果子 Agent 需要多轮记忆，可以额外配置 Checkpointer，但必须仔细设计 thread\_id、用户隔离和任务作用域。

**注意：**主 Agent 不会自动获得子 Agent 的全部执行消息。上面的工具只把子 Agent 最终回答返回给了主 Agent。

---

## 六、name 参数：为 Agent 命名

可以通过 `create_agent(name=...)` 为 Agent 指定名称。

### 实例

## 实例

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    name="customer\_service",
    system\_prompt="你是客服助手。"
)

`name` 主要用于标识 Agent 图，有助于调试、追踪和嵌入更复杂的图结构。

建议使用有意义的名称：

```
course_expert

math_expert

code_reviewer

learning_planner

coordinator
```

但需要注意，不能假设设置 name 后，所有 AIMessage 都会自动带上该名称。

在前面的工具包装模式中，如果需要区分执行来源，可以通过工具名称、日志和追踪信息进行标记。

---

## 七、方式二：Router 任务路由

如果业务只需要判断用户的问题属于哪个领域，再交给对应 Agent 处理，那么未必需要一个拥有多个工具的主 Agent。

可以直接创建 Router（路由器）。

流程：

```
用户问题

    |

    v

任务分类 Router

    |

    +---- 课程问题 ---- 课程 Agent

    |

    +---- 数学问题 ---- 数学 Agent

    |

    +---- 其他问题 ---- 通用 Agent
```

Router 主要负责判断任务类型，专业 Agent 负责实际处理。

对于任务类型明确的系统，这种结构通常更容易维护和测试。

### 实例：根据任务类型选择 Agent

## 实例

from typing import Literal
from pydantic import BaseModel
from langchain.chat\_models import init\_chat\_model

class RouteDecision(BaseModel):
    category: Literal["course", "math", "general"]

router\_model = init\_chat\_model("deepseek:deepseek-flash")
router = router\_model.with\_structured\_output(RouteDecision)

general\_agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是通用中文助手。"
)

def route\_question(question: str) -> str:
    decision = router.invoke(f"判断问题属于课程查询、数学计算还是其他：{question}")
    agents = {"course": course\_agent, "math": math\_agent, "general": general\_agent}
    result = agents[decision.category].invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return str(result["messages"][-1].content)

print(route\_question("Python 有哪些入门课程？"))

这里使用 Pydantic 将路由结果限制为三个值：

```
course

math

general
```

不同于协调者模式，Router 在这个例子中只选择一个 Agent。

如果需要将问题拆分成多个领域并行处理，应使用更完整的 Router 工作流或 Supervisor 模式。

**注意：**Router 的分类结果也可能出错，需要根据实际业务测试分类准确率。对于简单、固定的规则，也可以直接使用 Python 条件判断，不必额外调用模型。

---

## 八、方式三：顺序工作流

有些任务具有明确的执行顺序，例如：

```
生成文章

    ↓

检查技术事实

    ↓

润色文章

    ↓

输出最终内容
```

这种流程不一定需要由主 Agent 动态选择下一步。

如果执行顺序固定，可以直接通过 Python 或 LangGraph 构建确定性工作流。

### 实例：文章生成与审核

## 实例

from langchain.agents import create\_agent

writer = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是技术写作者。写简洁易懂的 Python 教程。"
)

reviewer = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[],
    system\_prompt="你是技术审稿助手。指出内容中的疑点、错误和需要核对的事实，不要臆测已验证。"
)

def write\_and\_review(topic: str):
    draft = writer.invoke({"messages": [{"role": "user", "content": f"写一段关于{topic}的教程，200字以内。"}]})
    article = str(draft["messages"][-1].content)

    checked = reviewer.invoke({"messages": [{
        "role": "user", "content": f"请审核以下文章，列出需要修改或验证的内容：\n{article}"
    }]})
    return article, str(checked["messages"][-1].content)

article, review = write\_and\_review("Python 装饰器")
print("文章：", article)
print("审核建议：", review)

这个例子创建了两个 Agent，但并没有创建协调者。

因为任务顺序已经确定：

```
Writer → Reviewer
```

这种方式通常比让协调者反复决定下一步更直接。

**注意：**第二个 Agent 只能根据自身能力识别潜在问题，并不能自动保证技术事实正确。如果需要事实核查，应提供可信文档、测试工具或其他数据来源。

---

## 九、方式四：Handoffs 控制权交接

Handoffs 与 Supervisor 不同。

Supervisor 由主 Agent 统一调用子 Agent，而 Handoffs 允许执行流程根据当前状态切换到另一个专业处理阶段。

例如，在线客服：

```
普通客服

    ↓

判断用户需要退款

    ↓

切换到退款处理阶段

    ↓

退款助手处理相关流程

    ↓

必要时进入人工审批

    ↓

完成处理
```

这种模式适合具有明确阶段切换的业务，例如：

- 普通客服转订单客服。
- 技术支持转售后处理。
- 问题收集转人工审核。
- 资料审核完成后进入执行阶段。

Handoffs 通常需要维护当前活动阶段，例如：

```
active_agent = "support"



active_agent = "refund"
```

在 LangChain 中，可以通过工具调用更新状态，再根据状态调整当前使用的 Prompt 和工具。

如果需要多个独立图节点相互交接控制权，也可以使用 LangGraph 构建更明确的 Handoff 工作流。

**注意：**Handoff 不代表自动获得更高权限。进入退款、支付或删除等阶段后，仍然必须执行后端权限校验和必要的人工审批。

---

## 十、Middleware 与多 Agent 路由

Middleware 可以动态修改模型请求的工具列表或系统提示词，但它本身并不等于多 Agent 系统。

例如，根据用户角色调整当前 Agent 可使用的工具：

```
普通用户 → 查询工具

管理员 → 查询工具 + 管理工具
```

这属于 动态工具选择，不一定需要创建两个独立 Agent。

### 实例：使用 wrap\_model\_call 过滤工具

## 实例

from dataclasses import dataclass
from langchain.agents.middleware import wrap\_model\_call, ModelRequest
from langchain.tools import tool

@dataclass
class UserContext:
    is\_admin: bool = False

@tool
def search\_course(keyword: str) -> str:
    """查询模拟课程。"""
    return f"正在查询：{keyword}"

@tool
def delete\_course(course\_id: str) -> str:
    """模拟删除课程。"""
    return f"模拟删除课程：{course\_id}"

@wrap\_model\_call
def select\_tools(request: ModelRequest[UserContext], handler):
    """根据可信角色筛选当前模型可见工具。"""
    if request.runtime.context.is\_admin:
        return handler(request)

    allowed = [t for t in request.tools if t.name != "delete\_course"]
    return handler(request.override(tools=allowed))

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course, delete\_course],
    context\_schema=UserContext,
    middleware=[select\_tools]
)

这里普通用户不会在当前模型请求中看到 `delete_course` 的工具定义。

但需要特别强调，**动态隐藏工具不等于真正的权限隔离**。

如果工具具有真实删除、支付或管理能力，工具执行层仍必须验证可信用户身份和权限。

如果需要完全确定性的安全隔离，更稳妥的方式是根据后端权限创建不同工具集合，或在受信任的工具执行入口再次强制校验。

---

## 十一、多 Agent 的消息与记忆管理

多 Agent 系统不仅需要管理任务，还需要明确各 Agent 能够访问哪些上下文。

常见有三种方式：

| 方式 | 说明 | 适用场景 |
| --- | --- | --- |
| 仅传当前问题 | 子 Agent 不读取完整主会话历史 | 独立计算、简单查询 |
| 传递必要上下文 | 提取完成子任务需要的历史信息 | 依赖用户偏好的专业任务 |
| 共享长期 Store | 根据授权读取相同长期数据 | 学习偏好、用户长期记录 |

例如，课程专家可以获得：

```
用户当前学习等级：入门

用户想学习：Python

任务：推荐适合的课程
```

而不需要获得用户全部历史消息。

这就是多 Agent 中非常重要的 Context Engineering（上下文工程）。

通过合理分配上下文，可以减少无关数据，提高各专业 Agent 的任务聚焦程度。

**注意：**不要把其他用户的完整对话历史、密钥或不相关的敏感信息传入子 Agent。

---

## 十二、并行执行多个子 Agent

如果多个子任务之间没有依赖关系，可以并行执行。

例如：

```
课程分析 ────┐

              ├── 汇总结果

代码检查 ────┘
```

对于相互独立的两个 Agent，可以使用 Python 的 `asyncio.gather()` 并发调用。

### 实例：并发获取两个专家的回答

## 实例

import asyncio

async def consult\_experts():
    questions = [
        course\_agent.ainvoke({"messages": [{"role": "user", "content": "介绍 Python 入门课程。"}]}),
        math\_agent.ainvoke({"messages": [{"role": "user", "content": "计算25摄氏度对应的华氏温度。"}]})
    ]

    results = await asyncio.gather(\*questions)
    for name, result in zip(["课程专家", "数学专家"], results):
        print(name, "：", result["messages"][-1].content)

if \_\_name\_\_ == "\_\_main\_\_":
    asyncio.run(consult\_experts())

这里复用了前面定义的两个 Agent。

两个独立任务可以并发等待模型和工具执行。

但并发不保证整体一定更快，还需要考虑 API 限流、网络状况、工具资源以及服务端并发能力。

如果任务之间存在依赖，例如先获取课程数据再计算统计值，则应按依赖顺序执行。

---

## 十三、如何选择多 Agent 架构？

| 业务需求 | 建议方式 | 原因 |
| --- | --- | --- |
| 只有少量工具 | 单 Agent | 结构简单、开销低 |
| 需要按领域委派任务 | Subagents | 由协调者调用专家 |
| 根据输入分类处理 | Router | 路由职责清晰 |
| 任务步骤固定 | 顺序工作流 | 执行过程可预测 |
| 需要切换当前专业处理阶段 | Handoffs | 适合多阶段交互 |
| 知识多但工具不复杂 | Skills | 按需加载知识，避免创建多余 Agent |
| 需要精确控制并行和状态 | LangGraph | 能够明确定义节点和路由 |

对于多数初学者，建议按照以下顺序学习：

```
单 Agent + Tools

       ↓

子 Agent 作为工具

       ↓

Router 任务分类

       ↓

Handoffs 状态切换

       ↓

LangGraph 自定义工作流
```

不要一开始就设计复杂的多 Agent 网络。

---

## 十四、生产环境的注意事项

### 1、控制模型调用成本

一个用户请求可能先调用协调者，再调用多个子 Agent，最后再次调用协调者汇总结果。

这会增加 Token 消耗和 API 请求次数。

建议记录每个 Agent 的调用次数、响应时间和成本。

### 2、限制任务委派层级

如果子 Agent 可以继续调用其他 Agent，可能出现循环委派或者重复执行。

应限制最大执行深度、模型调用次数和超时时间。

### 3、不要使用 eval() 执行模型生成的代码

模型生成的字符串不是可信代码。

对于数学计算，应使用明确参数的计算工具，或者具有受限语法的安全表达式解析器。

### 4、子 Agent 应保持最小权限

课程专家不需要数据库管理权限，数学专家也不需要发送邮件的能力。

为每个 Agent 配置完成任务所需的最少工具。

### 5、明确输入输出协议

复杂系统中，建议使用 Pydantic 结构化输出定义子 Agent 的返回字段，避免父 Agent 依赖不稳定的自然语言格式解析。

### 6、合理处理错误

某个子 Agent 失败时，可以根据业务决定重试、使用备用服务、返回部分结果或终止任务。

但不能将没有完成的子任务描述为执行成功。

### 7、监控调用链

多 Agent 排查问题时，不仅要知道最终回答，还应该知道哪个 Agent、哪个工具、在哪一步出现问题。

可以使用 LangSmith 等追踪工具观察执行路径。

---

## 十五、常见问题

### 1、多 Agent 一定比单 Agent 更强吗？

不一定。多 Agent 适合需要领域隔离、独立上下文或复杂执行流程的任务，但会增加开发和调用成本。

### 2、子 Agent 可以使用不同模型吗？

可以。不同 Agent 可以根据任务需要配置不同模型，只要对应集成和工具能力满足要求。

### 3、子 Agent 必须作为工具注册吗？

不是。可以将子 Agent 包装为工具，也可以通过 LangGraph 直接编排子图或自定义节点。

### 4、父 Agent 会自动获得子 Agent 的全部历史吗？

不会。工具包装通常只返回子 Agent 的指定结果，需要更多信息时应显式传递。

### 5、子 Agent 默认记住之前的任务吗？

不一定。在没有 Checkpointer 且每次传入新消息的情况下，子 Agent 通常独立执行。

### 6、name 参数有什么作用？

主要用于标识 Agent 图、辅助追踪和图组合，不保证所有生成消息都带相同名称。

### 7、Router 和 Supervisor 有什么区别？

Router 主要负责分类与分发；Supervisor 是可以在多轮执行中反复选择子 Agent 的协调者。

### 8、Middleware 是否可以实现多 Agent？

Middleware 可以动态选择工具、模型和提示词，但如果没有真正调用多个 Agent，它通常只是动态单 Agent 配置。

### 9、可以并行调用多个子 Agent 吗？

可以。当子任务彼此独立时，可以通过异步并发或 LangGraph 工作流并行处理。

### 10、多个 Agent 如何共享长期记忆？

可以通过 Store 或业务数据库，在经过授权的情况下访问相同的长期数据。

### 11、如何避免 Agent 之间相互循环调用？

可以限制委派深度、调用次数和执行时间，并对路由结果进行检查。

### 12、什么时候使用 LangGraph？

当需要明确控制执行顺序、并行节点、条件分支、状态共享和中断恢复时，适合使用 LangGraph。

---

## 总结

多 Agent 是 LangChain 实现复杂任务协作的重要架构模式。

本章节介绍了：

- `Subagents`：将专业 Agent 包装成工具。
- `Supervisor`：由主 Agent 协调多个子 Agent。
- `Router`：按任务类别分发给不同 Agent。
- `Handoffs`：根据状态转移专业处理阶段。
- `Skills`：按需加载专业知识。
- 顺序工作流与并行任务。
- Agent 命名及上下文隔离。
- 使用 Middleware 动态选择工具。
- 多 Agent 的成本、权限和追踪管理。

最简单的多 Agent 写法：

```
subagent = create_agent(

    model="deepseek:deepseek-flash",

    tools=[...]

)



@tool

def ask_subagent(question: str) -> str:

    """向子 Agent 委派任务。"""

    result = subagent.invoke({

        "messages": [{"role": "user", "content": question}]

    })

    return result["messages"][-1].content



coordinator = create_agent(

    model="deepseek:deepseek-flash",

    tools=[ask_subagent]

)
```

> **核心原则：**多 Agent 的价值不是数量，而是职责分工与上下文管理。只有当独立的专业能力、任务分发或执行流程确实带来收益时，才应该增加 Agent。
