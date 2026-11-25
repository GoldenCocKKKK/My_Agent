# LangChain 集成 DeepSeek

> 来源: https://www.runoob.com/langchain/langchain-deepseek.html

---

# LangChain 集成 DeepSeek

本章节介绍如何在 LangChain 中集成和使用 DeepSeek 聊天模型。

通过 **langchain-deepseek** 扩展包，开发者可以快速接入 DeepSeek 提供的大语言模型服务。

DeepSeek 模型既支持官方托管 API，也支持通过 Ollama、Fireworks、Together 等平台进行本地或第三方推理部署。

> 除了官方API接口外，我们也可以通过[Coding Plan/Token Plan](https://www.runoob.com/claude-code/coding-plan.html)套餐直接接入DeepSeek、Kimi、GLM、Doubao、MiniMax等主流大模型，无需单独购买各家API。

### DeepSeek 介绍

DeepSeek 是一个开源大语言模型系列，支持聊天、推理、代码生成等能力。

在 LangChain 中，DeepSeek 主要通过 **ChatDeepSeek** 类进行调用。

集成信息：

| 项目 | 说明 |
| --- | --- |
| 类名 | ChatDeepSeek |
| 安装包 | langchain-deepseek |
| 状态 | Beta（测试版） |
| 支持 JavaScript | 是 |
| 支持 Python | 是 |

模型能力支持：

| 功能 | 是否支持 |
| --- | --- |
| 工具调用（Tool Calling） | ✅ 支持 |
| 结构化输出（Structured Output） | ✅ 支持 |
| 图片输入 | ❌ 不支持 |
| 音频输入 | ❌ 不支持 |
| 视频输入 | ❌ 不支持 |
| Token 流式输出 | ✅ 支持 |
| 原生异步调用 | ✅ 支持 |
| Token 使用统计 | ✅ 支持 |
| Logprobs | ❌ 不支持 |

DeepSeek API 使用与 OpenAI/Anthropic 兼容的 API 格式，通过修改配置，您可以使用 OpenAI/Anthropic SDK 来访问 DeepSeek API，或使用与 OpenAI/Anthropic API 兼容的软件。

| 参数 | 值 |
| --- | --- |
| base\_url (OpenAI) | `https://api.deepseek.com` |
| base\_url (Anthropic) | `https://api.deepseek.com/anthropic` |
| api\_key | 点击链接申请 [API key](https://platform.deepseek.com/api_keys) |
| model\* | `deepseek-v4-flash` `deepseek-v4-pro` `deepseek-chat` (将于 2026/07/24 弃用) `deepseek-reasoner` (将于 2026/07/24 弃用) |

---

## 安装 langchain-deepseek

在开始使用前，需要先安装 DeepSeek 的 LangChain 集成包：

```


pip install -qU langchain-deepseek
```

---

## 配置 DeepSeek API Key

你需要先注册 DeepSeek 账号，并创建 API Key，如果还没有需要先去 <https://platform.deepseek.com/api_keys> 创建一个 API key。

通常推荐使用 **python-dotenv** 来读取当前目录中的 .env 配置文件。

```
pip install python-dotenv
```

在当前项目目录创建 **.env** 文件：

## .env 文件配置：

DEEPSEEK\_API\_KEY="sk-xxxxxxxxxxxxxxxx"
OPENAI\_API\_KEY="sk-xxxxxxxxxxxxxxxx"

![](https://www.runoob.com/wp-content/uploads/2026/05/runoob_1778729604970.png)

Python 读取 .env 文件：

## 完整实例

import os

from dotenv import load\_dotenv

# 加载当前目录 .env 文件
load\_dotenv()

# 获取 API Key
api\_key = os.getenv("DEEPSEEK\_API\_KEY")

print(api\_key)

**指定 .env 文件路径:** 如果 .env 文件不在当前目录，可以手动指定路径：

## 实例

from dotenv import load\_dotenv

load\_dotenv(dotenv\_path="./config/.env")

### 配置 LangSmith（可选）

如果需要开启 LangChain 调用链追踪与调试功能，
可以配置 LangSmith API Key：

## 实例

os.environ["LANGSMITH\_TRACING"] = "true"

os.environ["LANGSMITH\_API\_KEY"] = getpass.getpass(
    "Enter your LangSmith API key: "
)

### 创建 ChatDeepSeek 模型

安装完成后，可以通过 **ChatDeepSeek** 初始化模型：

## 实例

import os

from dotenv import load\_dotenv
from langchain\_deepseek import ChatDeepSeek

# 加载 .env
load\_dotenv()

# 获取 API KEY
api\_key = os.getenv("DEEPSEEK\_API\_KEY")

# 创建模型
llm = ChatDeepSeek(
    api\_key=api\_key,
    model="deepseek-v4-flash",
    temperature=0,
    max\_tokens=None,
    timeout=None,
    max\_retries=2
)

# 调用模型
response = llm.invoke("你好，请介绍 LangChain")

print(response.content)

参数说明:

| 参数 | 说明 |
| --- | --- |
| api\_key | 设置你申请的 API key，例如 "sk-xxx" |
| model | 指定模型名称，例如 deepseek-v4-flash |
| temperature | 控制随机性，越低结果越稳定 |
| max\_tokens | 限制生成最大 Token 数量 |
| timeout | 请求超时时间 |
| max\_retries | 失败后的最大重试次数 |

也可以使用 [init\_chat\_model()](https://www.runoob.com/langchain/langchain-init_chat_mode.html) 函数来调用：

## 实例

import os

from dotenv import load\_dotenv

# 加载当前目录 .env 文件
load\_dotenv()
from langchain.chat\_models import init\_chat\_model

# 指定了 model，返回固定模型
model = init\_chat\_model("deepseek:deepseek-v4-flash", temperature=0.7)
response = model.invoke("介绍菜鸟教程 RUNOOB")
print(response.content)

---

## 调用 DeepSeek 模型

下面示例演示如何向 DeepSeek 模型发送聊天消息：

> 注意我这里测试，把 API Key 写在了测试文件中：
>
> ```
> 

> llm = ChatDeepSeek(

>     api_key="sk-xxx",  # 设置你的 DeepSeek API Key

>     model="deepseek-v4-flash"

> )
> ```
>
> 实际的生产环境请设置在 .env 文件中。

## 实例

from langchain\_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    api\_key="sk-xxx",    # 设置你的 DeepSeek API Key
    model="deepseek-v4-flash",
    temperature=0,
    max\_tokens=None,
    timeout=None,
    max\_retries=2
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Chinese."
    ),
    (
        "human",
        "I love programming."
    ),
]

ai\_msg = llm.invoke(messages)

print(ai\_msg.content)

以上代码将英文翻译成中文，运行后，模型会返回翻译后的结果：

```
我喜欢编程。
```

支持的 DeepSeek 模型:

| 模型 | 用途 | 特点 |
| --- | --- | --- |
| deepseek-v4-flash | 通用聊天模型 | 支持 Tool Calling 与结构化输出 |
| deepseek-v4-pro | 推理模型（deepseek-v4-pro） | 更强推理能力，但不支持 Tool Calling |

## LangChain + DeepSeek 完整测试实例

下面演示一个完整可运行的 LangChain + DeepSeek 示例。

### 创建测试文件

创建 test.py 文件：

## 完整实例

from langchain\_deepseek import ChatDeepSeek
from langchain\_core.messages import HumanMessage, SystemMessage

# =========================
# 配置 DeepSeek API Key
# =========================

apiKey = "sk-xxx" # 设置你的 DeepSeek API Key

# =========================
# 创建 DeepSeek 模型
# =========================

llm = ChatDeepSeek(
    api\_key=apiKey,
    model="deepseek-v4-flash",
    temperature=0.7,
    max\_tokens=1024,
    timeout=60,
    max\_retries=3
)

# =========================
# 构造聊天消息
# =========================

messages = [
    SystemMessage(
        content="你是一名专业 Python 教师。"
    ),

    HumanMessage(
        content="请解释什么是 LangChain，并给出简单示例。"
    )
]

# =========================
# 调用模型
# =========================

response = llm.invoke(messages)

# =========================
# 输出结果
# =========================

print("AI 回复：")
print(response.content)

在终端执行：

```


python test.py
```

输出如下：

![](https://www.runoob.com/wp-content/uploads/2026/05/runoob_1778727460016.png)

代码说明：

| 代码 | 作用 |
| --- | --- |
| ChatDeepSeek | LangChain 的 DeepSeek 聊天模型类 |
| SystemMessage | 系统提示词，用于设定 AI 身份 |
| HumanMessage | 用户输入消息 |
| llm.invoke() | 调用 DeepSeek 模型 |
| response.content | 获取 AI 返回内容 |

### 使用 deepseek-v4-pro 推理模型

LangChain 支持 DeepSeek 流式输出，需要使用 deepseek-v4-pro 推理模型：

## 实例

from langchain\_deepseek import ChatDeepSeek
from langchain\_core.messages import HumanMessage, SystemMessage

# =========================
# 配置 DeepSeek API Key
# =========================

apiKey = "sk-xxx" # 设置你的 DeepSeek API Key

# =========================
# 创建 DeepSeek 模型
# =========================

llm = ChatDeepSeek(
    api\_key=apiKey,
    model="deepseek-v4-pro",
    temperature=0.7,
    max\_tokens=1024,
    timeout=60,
    max\_retries=3
)

for chunk in llm.stream("请介绍 Python"):
    print(chunk.content, end="", flush=True)

### 使用 PromptTemplate

结合 PromptTemplate 动态生成 Prompt：

## 实例

from langchain\_deepseek import ChatDeepSeek
from langchain\_core.prompts import ChatPromptTemplate

# =========================
# 配置 DeepSeek API Key
# =========================

apiKey = "sk-xxx" # 设置你的 DeepSeek API Key

# =========================
# 创建 DeepSeek 模型
# =========================

llm = ChatDeepSeek(
    api\_key=apiKey,
    model="deepseek-v4-pro",
    temperature=0.7,
    max\_tokens=1024,
    timeout=60,
    max\_retries=3
)

prompt = ChatPromptTemplate.from\_template(
    "请详细解释：{topic}"
)

chain = prompt | llm

response = chain.invoke({
    "topic": "Transformer"
})

print(response.content)

### 常见错误

| 错误 | 原因 | 解决方法 |
| --- | --- | --- |
| 401 Unauthorized | API Key 错误 | 检查 DEEPSEEK\_API\_KEY |
| ModuleNotFoundError | 未安装依赖 | 重新 pip install |
| Rate Limit | 请求频率过高 | 降低请求频率 |
| Timeout Error | 请求超时 | 增加 timeout 参数 |

### 推荐项目结构

## 项目结构

project/
│
├── app.py
├── requirements.txt
├── .env
├── prompts/
├── data/
└── vector\_db/

### requirements.txt 示例

## 实例

langchain
langchain-deepseek
python-dotenv

---

## 参考文档

- LangChain DeepSeek 文档：
  <https://python.langchain.com/docs/integrations/chat/deepseek/>
- DeepSeek 官网：
  <https://www.deepseek.com/>
- LangChain 官网：
  <https://www.langchain.com/>
- LangSmith：
  <https://smith.langchain.com/>
