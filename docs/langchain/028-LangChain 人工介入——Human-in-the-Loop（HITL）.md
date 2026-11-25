# LangChain 人工介入——Human-in-the-Loop（HITL）

> 来源: https://www.runoob.com/langchain/langchain-human-in-the-loop.html

---

# LangChain 人工介入——Human-in-the-Loop（HITL）

在实际的 AI Agent 应用中，并不是所有操作都适合交给模型自动执行。

例如，发送邮件、删除数据、修改数据库、执行支付等操作，可能对真实业务产生影响，因此通常需要人工确认。

LangChain 提供 Human-in-the-Loop（HITL，人工介入） 机制，让 Agent 在执行关键操作前暂停，等待人工审批，再决定继续、修改或取消。

本章节基于 LangChain 1.x 和 LangGraph，介绍 `interrupt()`、`Command(resume=...)`、`HumanInTheLoopMiddleware`、静态中断点以及 Web 审批架构。

---

## 一、什么是人工介入 HITL？

Human-in-the-Loop 指的是在 AI 自动化流程中保留人工决策环节。

例如，一个管理员要求 Agent 删除课程：

```
管理员：删除《过时的 Java 教程》

           ↓

       Agent 理解任务

           ↓

       准备执行删除工具

           ↓

        暂停执行

           ↓

    人工查看操作内容

           ↓

       ┌───────────────┐

       │               │

     批准             拒绝

       │               │

    执行删除         取消删除

       │               │

       └───────┬───────┘

               ↓

          返回最终结果
```

HITL 的作用不是让模型自己判断"是否获得了授权"，而是让外部可信系统真正参与决策。

常见场景包括：

| 场景 | 需要审批的操作 | 主要风险 |
| --- | --- | --- |
| 邮件助手 | 发送邮件 | 误发、敏感信息泄露 |
| 课程管理 | 删除或发布课程 | 误删、错误发布 |
| 数据库助手 | 执行 UPDATE、DELETE | 数据损坏 |
| 财务系统 | 支付、退款、转账 | 资金损失 |
| 运维 Agent | 修改配置、执行命令 | 服务中断 |

---

## 二、LangChain 的三种中断机制

LangChain 和 LangGraph 主要提供三种人工介入方式。

| 方式 | 工作原理 | 适用场景 |
| --- | --- | --- |
| `interrupt()` | 在代码内部动态暂停 | 自定义审批、表单输入 |
| `HumanInTheLoopMiddleware` | 拦截指定工具调用并请求审批 | 通用 Agent 工具审核 |
| `interrupt_before / interrupt_after` | 在图节点前后暂停 | 调试、执行检查点 |

对于大多数需要审核工具调用的 Agent，推荐优先使用 `HumanInTheLoopMiddleware`。

如果需要更复杂的自定义业务流程，可以直接使用 `interrupt()`。

---

## 三、interrupt()：动态暂停执行

`interrupt()` 是 LangGraph 提供的动态中断函数。

执行到该函数时，图会暂停，并向外部调用方返回审批或输入请求。

基本写法：

```
from langgraph.types import interrupt



decision = interrupt({

    "action": "delete_course",

    "message": "确认删除课程？"

})
```

外部系统收到中断信息后，由人工做出决定。

随后通过：

```
Command(resume={"approved": True})
```

恢复执行。

### interrupt() 的必要条件

正常使用需要三个关键要素：

- 一个 Checkpointer，用于保存暂停时的执行状态。
- 一个稳定的 `thread_id`，用于恢复正确的会话。
- 一个可序列化的中断数据，用于向外部显示审批内容。

**重要：**恢复时会重新执行包含 interrupt() 的节点，直到再次执行到对应的 interrupt()，才会得到之前通过 resume 传入的值。

因此，不应在 interrupt() 之前执行有不可重复副作用的操作。

---

## 四、在工具中使用 interrupt()

下面创建一个模拟课程删除工具。

工具会先暂停并申请批准，只有审批通过才执行模拟删除逻辑。

### 实例：模拟删除课程审批

## 实例

from langchain.tools import tool
from langgraph.types import interrupt

@tool
def delete\_course(course\_name: str) -> str:
    """模拟删除指定课程，执行前必须获得人工审批。"""
    decision = interrupt({
        "action": "delete\_course",
        "course": course\_name,
        "message": f"确认删除《{course\_name}》？"
    })

    if not isinstance(decision, dict) or decision.get("approved") is not True:
        return f"已取消删除《{course\_name}》"

    # 教学演示，不执行真实数据库删除
    return f"模拟删除成功：《{course\_name}》"

这里的中断请求包含操作类型、课程名称和提示信息。

恢复时传入：

```
{"approved": True}
```

表示人工批准。

传入：

```
{"approved": False}
```

则取消操作。

**注意：**以上仅是教学模拟。真实删除工具必须在执行时再次检查操作者权限，并使用可信后端完成数据库操作。

---

## 五、完整实例：暂停、审批、恢复

接下来将工具接入 Agent，完成一次完整的 HITL 流程。

安装依赖：

```
pip install -U langchain langgraph langchain-deepseek python-dotenv
```

配置 `.env`：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

文件名：`hitl_interrupt_demo.py`

## 完整实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.tools import tool
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load\_dotenv()

@tool
def delete\_course(course\_name: str) -> str:
    """模拟删除课程，必须先审批。"""
    decision = interrupt({
        "action": "delete\_course",
        "course": course\_name,
        "message": f"是否删除《{course\_name}》？"
    })
    if not isinstance(decision, dict) or decision.get("approved") is not True:
        return "管理员拒绝删除，未执行操作"
    return f"已模拟删除《{course\_name}》"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[delete\_course],
    checkpointer=InMemorySaver(),
    system\_prompt="你是课程管理助手。用户要求删除课程时使用 delete\_course，不得声称进行了真实数据库操作。"
)

config = {"configurable": {"thread\_id": "approval-001"}}

result = agent.invoke({
    "messages": [{"role": "user", "content": "请使用工具模拟删除《过时的 Java 教程》。"}]
}, config=config)

pending = result.get("\_\_interrupt\_\_", ())
if pending:
    print("等待人工审批：", pending[0].value)

    # 演示人工批准；真实应用必须从已认证的审批接口取得决定
    result = agent.invoke(
        Command(resume={"approved": True}),
        config=config
    )

if result.get("\_\_interrupt\_\_"):
    print("仍有待处理审批")
else:
    print("执行结果：", result["messages"][-1].content)

### 可能的运行结果

```
等待人工审批： {'action': 'delete_course', 'course': '过时的 Java 教程', ...}



执行结果： 已模拟删除《过时的 Java 教程》。
```

这里分为两个阶段：

**第一阶段：**Agent 执行到 interrupt()，返回待审批信息。

**第二阶段：**外部程序使用相同的 thread\_id，并通过 `Command(resume=...)` 恢复执行。

示例使用 `result["__interrupt__"]` 检查中断，这是普通 `invoke()` 返回结构中的常见方式。

需要注意，工具是否被模型调用由模型决定。即使用户明确提出删除任务，实际运行也可能没有触发工具。

---

## 六、查看 Agent 的中断状态

当 Agent 暂停后，可以通过 `get_state()` 查看线程当前状态。

### 实例

## 实例

state = agent.get\_state(config)

print("待执行节点：", state.next)
for task in state.tasks:
    for item in task.interrupts:
        print("中断 ID：", item.id)
        print("审批内容：", item.value)

如果工具节点触发了中断，`state.next` 可能包含：

```
('tools',)
```

表示图仍有待继续执行的工具节点。

**注意：**不应该假设 `state.tasks[0]` 必然对应当前审批，也不应该将所有暂停一律认定为工具审批。

真实应用应遍历待处理任务，结合中断 ID、线程及业务记录管理审批状态。

---

## 七、HumanInTheLoopMiddleware：内置工具审批

LangChain 1.x 提供 HumanInTheLoopMiddleware，可以在模型请求执行敏感工具时自动暂停，并等待人工审核。

它比在每个工具中手动调用 interrupt() 更方便，特别适合大量工具的统一审批管理。

主要支持四种决策：

| 决策 | 说明 |
| --- | --- |
| `approve` | 按原始参数批准执行 |
| `edit` | 修改工具参数后执行 |
| `reject` | 拒绝该工具调用 |
| `respond` | 人工直接提供工具返回内容，不执行工具 |

其中 `respond` 主要适合等待用户回答的工具，例如 `ask_user`。

**注意：**不要使用 respond 拒绝删除或支付工具，因为它会生成成功的工具结果，可能让模型错误认为操作已经完成。

---

## 八、完整实例：HumanInTheLoopMiddleware

下面创建一个模拟课程删除 Agent，通过内置 Middleware 进行审批。

## 完整实例

from dotenv import load\_dotenv
from langchain.agents import create\_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load\_dotenv()

@tool
def delete\_course(course\_name: str) -> str:
    """模拟删除课程。"""
    return f"已模拟删除《{course\_name}》"

@tool
def search\_course(keyword: str) -> str:
    """查询模拟课程。"""
    return "Python3 基础教程、Java 基础教程"

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[delete\_course, search\_course],
    middleware=[HumanInTheLoopMiddleware(interrupt\_on={
        "delete\_course": {"allowed\_decisions": ["approve", "reject"]},
        "search\_course": False
    })],
    checkpointer=InMemorySaver(),
    system\_prompt="你是课程管理助手。删除课程时调用 delete\_course，工具仅模拟执行。"
)

config = {"configurable": {"thread\_id": "hitl-002"}}

result = agent.invoke({
    "messages": [{"role": "user", "content": "请调用工具模拟删除《旧版 Java 教程》。"}]
}, config=config, version="v2")

if result.interrupts:
    print("待审批：", result.interrupts[0].value)
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        version="v2"
    )

if result.interrupts:
    print("仍存在待审批操作")
else:
    print("最终回复：", result.value["messages"][-1].content)

这里使用 `version="v2"`，因此返回的是包含 `value` 和 `interrupts` 的执行结果对象。

它与不指定 v2 时通过 `result["__interrupt__"]` 读取中断信息的方式不同。

这一示例中：

- `delete_course`：必须审批。
- `search_course`：无需审批。
- `approve`：批准执行原工具调用。
- `reject`：拒绝执行并把拒绝反馈交给 Agent。

示例为简化演示，假定只有一个待审批动作。如果同时存在多个动作，需要针对每个动作分别提供决策。

---

## 九、批准、拒绝与修改工具参数

HumanInTheLoopMiddleware 的审批结果通过 `Command(resume=...)` 传递。

### 1、批准

```
Command(resume={

    "decisions": [{"type": "approve"}]

})
```

表示按原始参数执行工具。

### 2、拒绝

```
Command(resume={

    "decisions": [{

        "type": "reject",

        "message": "管理员不同意删除该课程，请勿再次尝试。"

    }]

})
```

被拒绝的工具不会执行，Agent 会获得对应的拒绝反馈。

### 3、编辑参数

如果允许人工修改工具调用，可以配置：

```
"delete_course": {

    "allowed_decisions": ["approve", "edit", "reject"]

}
```

恢复时：

```
Command(resume={

    "decisions": [{

        "type": "edit",

        "edited_action": {

            "name": "delete_course",

            "args": {"course_name": "指定的新课程名称"}

        }

    }]

})
```

这允许审批人修改工具参数。

**注意：**修改工具名称或参数可能改变实际业务操作，应由可信后端再次检查编辑后的工具、目标对象和审批人权限。

对于删除、支付等高风险操作，通常不建议允许审批人任意切换工具类型。

---

## 十、多个工具同时需要审批

Agent 可能在一次模型响应中请求调用多个工具。

例如：

```
delete_course("旧版 Java 教程")

delete_course("旧版 PHP 教程")
```

如果两个工具调用都需要审批，HITL Middleware 可以返回多个待审核动作。

此时必须为每个动作提供对应决定。

### 实例：批量审批

## 实例

decision = Command(resume={
    "decisions": [
        {"type": "approve"},
        {"type": "reject", "message": "第二门课程需要保留。"}
    ]
})

result = agent.invoke(decision, config=config, version="v2")

其中 decisions 的排列顺序必须与中断信息中 `action_requests` 的顺序一致。

不能只根据工具名称假设只有一个需要审批的请求。

实际 Web 系统还应记录每个审批动作的唯一 ID，确保审批人不会误批其他工具调用。

---

## 十一、interrupt\_before 与 interrupt\_after

除了动态中断，还可以通过 `create_agent()` 的参数在指定图节点前后设置静态中断点。

例如：

## 实例

from langchain.agents import create\_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create\_agent(
    model="deepseek:deepseek-flash",
    tools=[search\_course],
    checkpointer=InMemorySaver(),
    interrupt\_before=["tools"]
)

这表示图执行到工具节点之前暂停。

| 设置 | 暂停时机 | 典型用途 |
| --- | --- | --- |
| `interrupt_before=["model"]` | 模型节点执行前 | 调试输入状态 |
| `interrupt_after=["model"]` | 模型节点执行后 | 检查模型输出 |
| `interrupt_before=["tools"]` | 工具节点执行前 | 查看工具调用请求 |
| `interrupt_after=["tools"]` | 工具节点执行后 | 检查工具结果 |

需要注意，静态断点与 `interrupt()` 有重要区别。

静态断点只是让图在指定节点前后暂停，并不会自动构造包含批准、编辑和拒绝选项的工具审批请求。

如果需要按工具分别审核，推荐使用 HumanInTheLoopMiddleware。

### 静态断点如何继续执行？

检查状态后，可以使用：

```
result = agent.invoke(None, config=config)
```

继续执行暂停的图。

如果是静态工具节点断点，应先检查待执行工具及参数，并在业务后端确认授权。

**注意：**直接继续执行并不等于完成审批；是否允许继续，是应用层必须自行保证的。

---

## 十二、动态中断与静态中断的区别

| 对比项 | 动态 interrupt() | 静态 interrupt\_before/after |
| --- | --- | --- |
| 中断位置 | 代码中的任意适当位置 | 指定图节点前后 |
| 是否可以携带审批内容 | 可以 | 需要从图状态自行读取 |
| 恢复方式 | Command(resume=...) | 通常通过 invoke(None, config=...) 继续 |
| 能否输入人工决定 | 可以 | 需应用自行管理决策及状态 |
| 典型用途 | 审批、人工填写数据 | 节点调试、执行暂停 |

对于需要用户填写表单、补充参数或审批操作的场景，动态中断通常更合适。

---

## 十三、恢复执行时的重要注意事项

HITL 最容易出现的问题之一，是误以为 interrupt() 暂停后，会从 Python 函数中的下一行原地继续。

实际上，LangGraph 恢复中断节点时，会从该节点的开头重新执行代码。

### 错误示例

```
def delete_data():

    database.delete("course-001")

    approved = interrupt("确认删除？")

    return approved
```

这里在人工确认之前就执行了删除操作。

而且恢复执行时，还可能再次执行此前的代码。

### 正确思路

```
def delete_data():

    approved = interrupt("确认删除？")

    if approved:

        database.delete("course-001")

    return approved
```

对于真实业务，还需要保证数据库操作具有幂等性，避免网络重试、并发审批或执行恢复导致重复操作。

**注意：**将副作用放在 interrupt() 之后可以避免"尚未审批就执行"，但不能单独保证整个业务实现严格只执行一次。

---

## 十四、Web 应用中的 HITL 架构

真实 Web 应用通常将 HITL 拆成两类 API：

- 用户提交任务，Agent 执行到待审批节点。
- 授权审批人提交决定，后端恢复对应的 Agent 线程。

典型流程：

```
浏览器

  ↓ 提交任务

应用后端

  ↓

Agent 执行

  ↓

触发 interrupt

  ↓

Checkpointer 保存状态

  ↓

后端创建待审批记录

  ↓

审批界面展示操作详情

  ↓

管理员登录并批准或拒绝

  ↓

后端验证权限与审批状态

  ↓

Command(resume=...)

  ↓

Agent 恢复执行

  ↓

返回完成状态或新的审批请求
```

下面给出与前面自定义 interrupt() 示例配套的简单后端逻辑。

### 实例：提交任务与读取审批状态

## 实例

def handle\_user\_message(thread\_id: str, message: str):
    config = {"configurable": {"thread\_id": thread\_id}}
    result = agent.invoke({
        "messages": [{"role": "user", "content": message}]
    }, config=config)

    pending = result.get("\_\_interrupt\_\_", ())
    if pending:
        return {
            "status": "pending\_approval",
            "actions": [item.value for item in pending],
            "thread\_id": thread\_id
        }

    return {
        "status": "completed",
        "reply": result["messages"][-1].content
    }

这里返回待审批内容，而不是自动执行批准。

### 实例：审批后恢复

## 实例

from langgraph.types import Command

def handle\_approval(thread\_id: str, approved: bool, reason: str = ""):
    # 调用此函数前，后端必须完成身份、权限、会话归属及审批状态验证
    config = {"configurable": {"thread\_id": thread\_id}}
    result = agent.invoke(
        Command(resume={"approved": approved, "reason": reason}),
        config=config
    )

    pending = result.get("\_\_interrupt\_\_", ())
    if pending:
        return {"status": "pending\_approval", "actions": [i.value for i in pending]}

    return {
        "status": "completed",
        "reply": result["messages"][-1].content
    }

这两个函数展示的是核心执行逻辑，不是完整可直接公开部署的 HTTP API。

真实后端还必须负责认证、授权、审批日志、状态锁定、超时处理及异常恢复。

**重要：**这段自定义审批流程使用的是 `interrupt()` 的 resume 数据结构。如果改用 HumanInTheLoopMiddleware，则必须按照该 Middleware 的 `decisions` 格式恢复，二者不能混用。

---

## 十五、HITL 与 Checkpointer

HITL 需要 Checkpointer 保存中断状态，以便在外部审批完成后恢复执行。

开发环境通常使用：

```
from langgraph.checkpoint.memory import InMemorySaver



checkpointer = InMemorySaver()
```

它简单易用，但数据仅保存在当前进程内存中。

一旦程序重启，待审批任务的执行状态可能丢失。

生产环境建议使用支持持久化的 Checkpointer，例如 PostgreSQL：

```
PostgresSaver

AsyncPostgresSaver
```

对于异步 Web 服务，通常使用异步 Checkpointer，并在应用生命周期中管理数据库连接。

需要注意：

- 恢复必须使用原来的 thread\_id。
- Checkpointer 保存的是图执行状态，不是完整的审批业务数据库。
- 审批记录通常需要独立存储，便于审计和权限管理。
- 同一个线程的并发恢复请求需要进行协调，防止重复处理。

---

## 十六、HITL 的生产环境安全设计

### 1、审批操作必须由可信后端完成

不能仅因为请求中包含：

```
{"approved": true}
```

就认定该操作已经获得管理员批准。

后端必须确认审批人身份、权限及对应的业务对象。

### 2、审批对象必须绑定具体操作

审批记录应该包含：

```
审批 ID

thread_id

工具名称

工具参数摘要

操作对象

审批人

审批状态

创建时间

处理时间
```

避免出现"用户批准了 A 操作，但 Agent 最终执行 B 操作"的问题。

### 3、审批后仍要再次验证权限

用户权限可能在审批等待期间发生变化。

因此，真正执行删除、支付等敏感操作时，仍需要后端再次进行权限与业务校验。

### 4、保证操作幂等

对于支付、发送邮件、创建订单等操作，应使用幂等键或唯一业务请求 ID，防止重复执行。

### 5、审批需要超时与取消机制

用户可能长期不处理审批请求。

生产环境应该明确等待期限、取消方式和超时后的处理规则。

### 6、不要将中断信息直接当作可信指令

中断详情可能包含模型生成的参数，审批界面应正确转义显示，并由后端校验参数。

### 7、审计真实执行结果

人工批准不代表业务操作一定成功。

应分别记录"审批通过""工具开始执行"和"业务操作完成"等状态。

---

## 十七、常见问题

### 1、HITL 是什么？

HITL 是 Human-in-the-Loop，让人工参与 AI Agent 的关键决策与执行流程。

### 2、interrupt() 会阻塞整个 Python 服务吗？

不会以持续阻塞等待用户输入的方式工作。它通过图中断和状态持久化，让本次执行返回待处理信息，后续再恢复。

### 3、没有 Checkpointer 可以使用 HITL 吗？

正常的持久化中断恢复需要 Checkpointer，并使用稳定的 thread\_id。

### 4、恢复时为什么必须使用相同 thread\_id？

因为 Checkpointer 需要通过线程标识找到之前保存的执行状态。

### 5、Command(resume=...) 的作用是什么？

将外部输入交给暂停的 interrupt()，让图继续执行。

### 6、恢复后会从中断位置继续吗？

恢复会重新执行包含 interrupt() 的节点，在对应中断调用处接收 resume 数据。

### 7、interrupt\_before=["tools"] 是否等于工具审批？

不是。它只是工具节点级的静态断点，不会自动提供逐工具的批准、编辑和拒绝机制。

### 8、什么时候使用 HumanInTheLoopMiddleware？

当主要需求是审核 Agent 提出的工具调用时，通常优先使用内置 Middleware。

### 9、HITL 可以修改工具参数吗？

可以。HumanInTheLoopMiddleware 支持 edit 决策，但必须明确配置允许编辑。

### 10、多个工具同时审批怎么办？

需要对每个待审核动作提供决策，并保证决策顺序与中断请求中的动作顺序一致。

### 11、reject 和 respond 有什么区别？

reject 表示拒绝工具调用；respond 表示人工直接提供工具响应，不执行工具。不能用 respond 代替敏感操作的拒绝。

### 12、人工批准后一定执行成功吗？

不一定。后续工具仍可能因为权限变化、服务故障或业务校验失败而执行失败。

### 13、程序重启后还能恢复审批吗？

如果使用持久化 Checkpointer，并正确保存线程及审批信息，就可以支持跨进程恢复。

### 14、可以在 FastAPI 中使用 HITL 吗？

可以。通常由一个接口启动任务，另一个经过授权的接口提交审批决定，再恢复对应线程。

---

## 总结

Human-in-the-Loop 是 LangChain Agent 实现安全、可控自动化的重要能力。

本章节学习了：

- `interrupt()`：动态暂停执行。
- `Command(resume=...)`：恢复暂停的执行。
- `HumanInTheLoopMiddleware`：对工具调用进行人工审核。
- `approve`：批准操作。
- `edit`：修改工具参数。
- `reject`：拒绝操作。
- `respond`：人工直接提供工具结果。
- `interrupt_before / interrupt_after`：静态图节点中断。
- `Checkpointer`：保存中断状态。
- Web 应用中的审批权限、状态管理和幂等控制。

最基本的动态中断写法：

```
decision = interrupt({"message": "请确认操作"})

if decision.get("approved"):

    # 执行通过后允许的操作

    pass
```

恢复执行：

```
agent.invoke(

    Command(resume={"approved": True}),

    config={"configurable": {"thread_id": "approval-001"}}

)
```

对于常见的工具审批，推荐：

```
HumanInTheLoopMiddleware(

    interrupt_on={

        "delete_course": {"allowed_decisions": ["approve", "reject"]}

    }

)
```

> **核心原则：**Agent 可以提出操作建议，但高风险操作应由可信系统验证、由有权限的人审批，并由业务后端负责真正执行。人工批准、工具执行成功和业务结果完成，是三个需要分别确认的状态。
