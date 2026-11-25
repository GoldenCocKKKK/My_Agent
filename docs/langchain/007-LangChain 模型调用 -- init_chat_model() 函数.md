# LangChain 模型调用 -- init\_chat\_model() 函数

> 来源: https://www.runoob.com/langchain/langchain-init_chat_mode.html

---

# LangChain 模型调用 -- init\_chat\_model() 函数

`init_chat_model()` 是 LangChain 提供的统一聊天模型初始化函数，可以通过相同的接口连接不同模型提供商。

例如，使用 DeepSeek、OpenAI、Anthropic、Google Gemini 或 Ollama 时，开发者不必分别记忆所有模型集成类的名称，只需要指定模型提供商和模型名称。

`init_chat_model()` 主要提供两种使用模式：

- **固定模型：**创建模型时指定名称，后续直接调用。
- **可配置模型：**允许运行时动态切换模型，或者修改指定的模型初始化参数。

此外，初始化后的聊天模型还可以通过统一接口实现同步调用、异步调用、流式输出等功能。

本章节将以 DeepSeek 为主要示例，同时介绍其他模型提供商的接入方法。

**注意：**LangChain 统一的是模型初始化和调用接口，不代表不同模型支持完全相同的功能、参数或上下文长度。

## 一、安装 LangChain

首先安装 LangChain：

```
pip install -U langchain
```

`init_chat_model()` 不会自动安装所有模型提供商的集成包，因此还需要安装对应的依赖。

例如，使用 DeepSeek：

```
pip install -U langchain-deepseek
```

使用 OpenAI：

```
pip install -U langchain-openai
```

使用 Anthropic：

```
pip install -U langchain-anthropic
```

使用 Ollama 本地模型：

```
pip install -U langchain-ollama
```

如果需要通过 `.env` 文件加载 API Key，还可以安装：

```
pip install -U python-dotenv
```

### 配置 API Key

调用 DeepSeek API 需要设置 `DEEPSEEK_API_KEY` 环境变量。

可以在项目目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

在 Python 程序中加载该文件：

```
from dotenv import load_dotenv



load_dotenv()
```

如果使用其他提供商，可以配置对应的环境变量：

```
OPENAI_API_KEY=your_openai_api_key

ANTHROPIC_API_KEY=your_anthropic_api_key
```

**注意：**这里只需要配置实际使用的提供商对应的密钥。不要将包含真实 API Key 的 `.env` 文件提交到公开代码仓库。

---

## 二、init\_chat\_model() 函数语法

函数的基本语法如下：

```
from langchain.chat_models import init_chat_model



model = init_chat_model(

    model=None,

    *,

    model_provider=None,

    configurable_fields=None,

    config_prefix=None,

    **kwargs

)
```

参数说明：

| 参数 | 类型 | 说明 | 默认值 |
| --- | --- | --- | --- |
| `model` | str 或 None | 模型名称，推荐使用 provider:model 格式；为 None 时创建可配置模型 | None |
| `model_provider` | str 或 None | 单独指定模型提供商 | None |
| `configurable_fields` | "any"、list、tuple 或 None | 指定允许运行时修改的模型字段 | None |
| `config_prefix` | str 或 None | 运行时配置键的前缀 | None |
| `**kwargs` | 关键字参数 | 传递给底层模型的初始化参数 | 无 |

`init_chat_model()` 通常有两种返回结果：

- 固定模型模式：返回实现 `BaseChatModel` 接口的聊天模型实例。
- 可配置模型模式：返回支持通过运行时配置选择模型或修改字段的包装对象。

下面分别介绍各参数的使用方式。

---

## 三、model 参数详解

`model` 用于指定需要调用的模型。

LangChain 推荐使用 `provider:model` 格式，明确指定提供商。

### 1、provider:model 格式

基本格式为：

```
提供商名称:模型名称
```

例如：

## 实例

from langchain.chat\_models import init\_chat\_model

# DeepSeek Flash
deepseek\_flash = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# DeepSeek V4 Pro
deepseek\_pro = init\_chat\_model(
    "deepseek:deepseek-v4-pro"
)

# OpenAI
openai\_model = init\_chat\_model(
    "openai:gpt-4o-mini"
)

# Anthropic
anthropic\_model = init\_chat\_model(
    "anthropic:claude-sonnet-4-5"
)

# Ollama 本地模型
ollama\_model = init\_chat\_model(
    "ollama:llama3.2"
)

上述代码用于展示初始化方式，不会因为创建模型对象就自动完成一次聊天请求。

实际使用时，需要提前安装对应的 LangChain 集成包，并确认所指定的模型 ID 当前可用。

**推荐：**新项目优先使用 `provider:model` 格式，以减少模型提供商自动推断错误。

### 2、DeepSeek 当前推荐模型

根据 DeepSeek 官方 API 文档，目前推荐使用以下两个模型名称：

| API 模型名称 | 模型版本 | LangChain 调用名称 |
| --- | --- | --- |
| `deepseek-flash` | DeepSeek-V4.1-Flash | `deepseek:deepseek-flash` |
| `deepseek-v4-pro` | DeepSeek-V4-Pro-0813 | `deepseek:deepseek-v4-pro` |

其中，`deepseek-flash` 是当前官方推荐的 Flash 模型名称。

旧的 `deepseek-v4-flash` 和 `deepseek-v4-flash-vision-exp` 名称仍保留兼容调用，但对应的旧模型已经下线，请求实际由 DeepSeek-V4.1-Flash 提供服务。

因此，新项目应优先使用：

```
model = init_chat_model("deepseek:deepseek-flash")
```

需要使用 Pro 模型时：

```
model = init_chat_model("deepseek:deepseek-v4-pro")
```

模型的功能、思考模式、价格和上下文长度可能随服务商调整，建议以 [DeepSeek 官方模型文档](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) 为准。

### 3、自动推断模型提供商

如果省略提供商前缀，LangChain 会尝试根据模型名称自动推断。

## 实例

from langchain.chat\_models import init\_chat\_model

# 自动推断 DeepSeek
model = init\_chat\_model("deepseek-flash")

# 自动推断 OpenAI
model = init\_chat\_model("gpt-4o-mini")

# 自动推断 Anthropic
model = init\_chat\_model("claude-sonnet-4-5")

# 自动推断 xAI
model = init\_chat\_model("grok-3")

# 自动推断 Perplexity
model = init\_chat\_model("sonar")

常见的模型名称推断规则如下：

| 模型名称前缀 | 自动推断的 provider |
| --- | --- |
| gpt-、o1、o3、chatgpt、text-davinci | openai |
| claude | anthropic |
| deepseek | deepseek |
| gemini | google\_vertexai |
| command | cohere |
| mistral、mixtral | mistralai |
| grok | xai |
| sonar | perplexity |
| solar | upstage |
| accounts/fireworks | fireworks |
| amazon.、anthropic.、meta. | bedrock |

**注意：**自动推断不是通用的模型识别服务，只是根据预设的名称规则判断提供商。

例如，`deepseek-flash` 会被推断为 `deepseek`，而不是 `openai`。

Google Gemini 的默认推断规则也可能随 LangChain 版本变化，建议显式指定 `google_genai` 或 `google_vertexai`。

### 4、model\_provider 参数

如果需要分别传递模型名称和提供商，可以使用 `model_provider`。

例如，下面两种写法在提供商和模型选择上等价：

## 实例

from langchain.chat\_models import init\_chat\_model

# 方式一：单独指定提供商
model1 = init\_chat\_model(
    "deepseek-flash",
    model\_provider="deepseek"
)

# 方式二：provider:model 格式
model2 = init\_chat\_model(
    "deepseek:deepseek-flash"
)

当模型配置从文件、数据库或环境变量中读取时，可以使用这种方式：

```
provider = "deepseek"

model_name = "deepseek-flash"



model = init_chat_model(

    model_name,

    model_provider=provider

)
```

这种方式比较适合需要动态读取配置的应用。

---

## 四、固定模型模式

固定模型是 `init_chat_model()` 最常见的使用方式。

创建模型时指定名称和参数，然后使用 `invoke()` 调用。

### 实例：调用 DeepSeek Flash

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

# 加载当前项目 .env 中的 API Key
load\_dotenv()

# 初始化 DeepSeek Flash
model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

# 调用模型
response = model.invoke(
    "用一句话介绍菜鸟教程 RUNOOB。"
)

# 输出结果
print(response.content)

调用时，LangChain 会通过对应的 DeepSeek 集成向 API 发送请求。

`invoke()` 通常返回一个 `AIMessage` 对象，其中 `content` 保存响应内容。

由于模型是动态生成内容，每次调用的实际输出可能不同。

### 实例：调用 DeepSeek V4 Pro

如果需要使用 Pro 模型，只需要修改模型名称：

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek:deepseek-v4-pro"
)

response = model.invoke(
    "解释 Python 装饰器的工作原理，并给出示例。"
)

print(response.content)

这种方式不需要手动创建不同的模型集成类，直接替换模型 ID 即可。

**注意：**以上示例需要真实 API Key 和可用的模型服务，实际调用会产生相应的 API 费用。

---

## 五、通过 OpenAI 兼容接口调用 DeepSeek

DeepSeek 官方除了提供自己的 API，还提供与 OpenAI、Anthropic 格式兼容的接口。

官方地址如下：

| 接口格式 | base\_url |
| --- | --- |
| OpenAI 兼容 | `https://api.deepseek.com` |
| Anthropic 兼容 | `https://api.deepseek.com/anthropic` |

因此，在 LangChain 中可以选择使用 DeepSeek 原生集成，也可以使用 OpenAI 兼容集成。

### 方式一：使用 DeepSeek 原生集成

需要安装：

```
pip install -U langchain-deepseek
```

代码如下：

```
model = init_chat_model(

    "deepseek:deepseek-flash"

)
```

这是使用 LangChain 接入 DeepSeek 时推荐的方式。

### 方式二：使用 OpenAI 兼容接口

如果希望通过 `langchain-openai` 调用 DeepSeek，可以单独指定模型提供商、API 地址和密钥。

## 实例

import os

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

model = init\_chat\_model(
    "deepseek-flash",

    # 使用 OpenAI 兼容集成
    model\_provider="openai",

    # DeepSeek 官方 API 地址
    base\_url="https://api.deepseek.com",

    # DeepSeek API Key
    api\_key=os.environ["DEEPSEEK\_API\_KEY"]
)

response = model.invoke(
    "介绍一下 LangChain。"
)

print(response.content)

上述代码需要安装：

```
pip install -U langchain-openai
```

这里虽然使用的是 `openai` 提供商，但实际请求发送到 DeepSeek API，而不是 OpenAI API。

这是通过自定义 `base_url` 实现的。

两种调用方式对比如下：

| 方式 | LangChain provider | 安装包 |
| --- | --- | --- |
| DeepSeek 原生集成 | deepseek | langchain-deepseek |
| OpenAI 兼容接口 | openai | langchain-openai |

**注意：**兼容 OpenAI API 不代表第三方模型的所有特殊响应字段都会被通用集成完整处理。如果需要 DeepSeek 特定功能，应优先查看对应的 LangChain DeepSeek 集成是否支持。

此外，DeepSeek 的 Anthropic 兼容接口是另一套请求格式，不应简单地把该地址直接当成 OpenAI 格式的 `base_url` 使用。

---

## 六、可配置模型模式

如果希望在运行时切换模型，而不是每次创建新的模型变量，可以使用可配置模型。

这在模型对比、A/B 测试、成本管理和多模型应用中比较常见。

### 1、不指定 model

如果调用 `init_chat_model()` 时没有指定模型，LangChain 会返回可配置模型。

这种情况下，默认允许在运行时配置 `model` 和 `model_provider`。

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 不指定默认模型
model = init\_chat\_model()

# 使用 DeepSeek Flash
response = model.invoke(
    "介绍菜鸟教程 RUNOOB。",
    config={
        "configurable": {
            "model": "deepseek:deepseek-flash"
        }
    }
)

print(response.content)

# 切换为 DeepSeek Pro
response = model.invoke(
    "介绍菜鸟教程 RUNOOB。",
    config={
        "configurable": {
            "model": "deepseek:deepseek-v4-pro"
        }
    }
)

print(response.content)

可以看到，同一个可配置模型对象，可以通过 `config` 指定不同的模型。

切换模型时不需要重新创建应用程序，但调用过程仍需要加载目标模型的集成并访问对应服务。

### 2、configurable\_fields 参数

`configurable_fields` 用于控制允许运行时修改哪些字段。

常见取值如下：

| 取值 | 说明 |
| --- | --- |
| `None` | 指定 model 时不开放可配置字段；没有指定 model 时默认开放 model 和 model\_provider |
| `"any"` | 开放所有相关配置字段，需注意安全风险 |
| `["model", "temperature"]` | 仅允许配置指定字段 |
| `("model", "model_provider")` | 使用元组形式指定可配置字段 |

例如，只允许修改模型名称：

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash",
    configurable\_fields=["model"]
)

# 使用默认的 Flash 模型
response = model.invoke("介绍 Python。")
print(response.content)

# 切换到 Pro 模型
response = model.invoke(
    "解释 Python 装饰器。",
    config={
        "configurable": {
            "model": "deepseek:deepseek-v4-pro"
        }
    }
)

print(response.content)

这里的 `model` 字段被声明为可配置，因此能够在运行时覆盖初始化时指定的模型名称。

如果希望单独修改提供商和模型名称，也可以将 `model_provider` 加入可配置字段列表。

---

## 七、config\_prefix 参数

`config_prefix` 用于为可配置字段添加前缀，避免多个模型对象使用相同的配置键。

例如，一个应用可能有两个模型：

- 一个负责用户对话。
- 一个负责生成摘要。

为了避免两个模型的配置发生冲突，可以分别使用 `chat` 和 `summary` 前缀。

### 实例

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash",

    configurable\_fields=[
        "model",
        "temperature"
    ],

    config\_prefix="my",

    temperature=0.3
)

# 使用默认配置
response = model.invoke(
    "介绍菜鸟教程。"
)

print(response.content)

# 运行时修改温度
response = model.invoke(
    "介绍菜鸟教程 RUNOOB。",
    config={
        "configurable": {
            "my\_temperature": 0.7
        }
    }
)

print(response.content)

设置 `config_prefix="my"` 后，配置字段需要使用 `my_` 前缀。

例如：

```
my_model

my_temperature
```

如果没有设置前缀，则直接使用：

```
model

temperature
```

需要注意，上面修改 `temperature` 的示例适用于接受该参数的模型配置；实际是否允许调整采样温度，应以所调用模型的接口文档为准。

---

## 八、configurable\_fields="any"

如果希望所有可传入模型初始化器的字段都能在运行时配置，可以使用：

```
configurable_fields="any"
```

例如：

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash",
    configurable\_fields="any",
    config\_prefix="my"
)

response = model.invoke(
    "介绍 Python。",
    config={
        "configurable": {
            "my\_model": "deepseek:deepseek-v4-pro"
        }
    }
)

print(response.content)

这种方式非常灵活，但也存在一定风险。

例如，如果允许不可信用户直接控制所有运行时配置，可能导致 `api_key`、`base_url` 等敏感字段被篡改。

恶意配置甚至可能把包含敏感信息的模型请求发送到非预期的服务器。

因此，生产环境建议明确列出允许修改的字段：

```
configurable_fields=[

    "model",

    "model_provider"

]
```

并在应用层对可选模型和提供商进行白名单校验。

---

## 九、支持的模型提供商

`init_chat_model()` 支持多个主流模型提供商，具体的 provider 名称和集成包由 LangChain 版本决定。

以下列出官方 API 支持的常见提供商：

| provider | 安装包 | 说明 |
| --- | --- | --- |
| openai | langchain-openai | OpenAI 模型 |
| anthropic | langchain-anthropic | Claude 模型 |
| deepseek | langchain-deepseek | DeepSeek 模型 |
| google\_genai | langchain-google-genai | Google Gemini API |
| google\_vertexai | langchain-google-vertexai | Google Vertex AI |
| azure\_openai | langchain-openai | Azure OpenAI |
| azure\_ai | langchain-azure-ai | Azure AI |
| mistralai | langchain-mistralai | Mistral AI |
| groq | langchain-groq | Groq 推理平台 |
| ollama | langchain-ollama | Ollama 本地模型 |
| fireworks | langchain-fireworks | Fireworks AI |
| together | langchain-together | Together AI |
| xai | langchain-xai | xAI Grok |
| perplexity | langchain-perplexity | Perplexity Sonar |
| cohere | langchain-cohere | Cohere 模型 |
| huggingface | langchain-huggingface | Hugging Face 模型 |
| openrouter | langchain-openrouter | OpenRouter 模型网关 |
| bedrock | langchain-aws | Amazon Bedrock |
| upstage | langchain-upstage | Upstage 模型 |
| nvidia | langchain-nvidia-ai-endpoints | NVIDIA AI Endpoints |
| ibm | langchain-ibm | IBM watsonx.ai |

不同集成包的功能和版本要求可能不同，实际使用前应核对 [LangChain 官方 API Reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model)。

特别注意：**某个提供商存在 LangChain 集成包，不代表它的所有模型都可以通过任意名称调用。**

模型 ID、认证方式和调用权限仍然由具体模型平台决定。

---

## 十、常用 kwargs 参数

`**kwargs` 用于向底层模型传递初始化参数。

常见参数包括：

| 参数 | 作用 | 注意事项 |
| --- | --- | --- |
| `temperature` | 控制模型生成的采样随机性 | 取值范围及是否支持由具体模型决定 |
| `max_tokens` | 限制最大输出 Token 数 | 具体字段映射依赖模型集成 |
| `timeout` | 设置请求超时时间 | 常见单位为秒 |
| `max_retries` | 设置失败重试次数 | 重试行为取决于模型集成 |
| `base_url` | 配置自定义 API 地址 | 通常用于兼容接口或网关 |
| `rate_limiter` | 配置速率限制器 | 需要传入兼容的 BaseRateLimiter 实例 |
| `top_p` | 控制核采样范围 | 并非所有模型都支持 |

例如：

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash",

    # 请求超时
    timeout=30,

    # 失败重试
    max\_retries=2
)

response = model.invoke(
    "介绍 Python 编程语言。"
)

print(response.content)

**注意：**`temperature`、`top_p` 和 `max_tokens` 等参数不能保证适用于所有模型。

尤其是不同厂商的推理模型，可能对采样参数、思考模式和输出长度有不同要求。

如果模型返回参数不支持的错误，应优先核对该模型的官方 API，而不是假设所有参数都可以通用。

---

## 十一、常见模型调用方法

初始化模型之后，可以通过 LangChain 的统一接口完成不同类型的调用。

### 1、invoke() 同步调用

`invoke()` 用于发送输入，并等待模型返回完整响应。

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

response = model.invoke("什么是 Python？")

print(response.content)

### 2、stream() 流式输出

`stream()` 可以逐步接收模型输出，适合聊天界面和实时生成内容的应用。

## 实例

from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

for chunk in model.stream(
    "介绍 Python 的三个优点"
):
    print(chunk.text, end="", flush=True)

流式接口返回消息块，可以通过 `text` 属性读取其中的文本。

需要注意，部分模型还可能返回思考内容、工具调用信息或其他非普通文本内容。

### 3、ainvoke() 异步调用

如果程序使用 `asyncio`，可以通过 `ainvoke()` 异步调用模型。

## 实例

import asyncio
from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

async def main():
    response = await model.ainvoke(
        "简单介绍一下 Python。"
    )

    print(response.content)

asyncio.run(main())

异步方式适合 Web 服务、Agent 应用和需要同时处理多个请求的场景。

### 4、使用消息列表

除了直接传入字符串，LangChain 还支持使用消息列表组织对话。

## 实例

from langchain.chat\_models import init\_chat\_model
from langchain\_core.messages import (
    SystemMessage,
    HumanMessage
)

model = init\_chat\_model(
    "deepseek:deepseek-flash"
)

messages = [
    SystemMessage(
        content="你是一名 Python 编程教程作者。"
    ),
    HumanMessage(
        content="用简单语言解释什么是 Python 装饰器。"
    )
]

response = model.invoke(messages)

print(response.content)

这种方式方便设置系统提示词，以及组织需要不同角色消息的对话。

---

## 十二、实际应用：运行时切换 Flash 和 Pro

假设开发一个 AI 工具，需要根据任务复杂度选择不同的 DeepSeek 模型。

例如：

- 一般问题默认使用 Flash。
- 需要使用 Pro 模型时，通过参数切换。

可以使用 `configurable_fields` 实现模型动态选择。

## 实例

from dotenv import load\_dotenv
from langchain.chat\_models import init\_chat\_model

load\_dotenv()

# 模型白名单
ALLOWED\_MODELS = {
    "flash": "deepseek:deepseek-flash",
    "pro": "deepseek:deepseek-v4-pro"
}

# 默认使用 Flash
model = init\_chat\_model(
    ALLOWED\_MODELS["flash"],
    configurable\_fields=["model"]
)

def ask(question, mode="flash"):
    if mode not in ALLOWED\_MODELS:
        raise ValueError("不支持的模型类型")

    selected\_model = ALLOWED\_MODELS[mode]

    response = model.invoke(
        question,
        config={
            "configurable": {
                "model": selected\_model
            }
        }
    )

    return response.content

# 使用 Flash
print(ask("介绍 Python", "flash"))

# 使用 Pro
print(ask("深入解释 Python 装饰器", "pro"))

这个示例使用同一个 LangChain 可配置模型包装对象，根据参数选择不同的模型。

在实际应用中，可以根据用户套餐、任务类型或成本预算决定使用哪一个模型。

**注意：**示例中的 Flash 和 Pro 只是两个可供选择的模型，不代表 LangChain 会自动判断问题复杂度或自动优化成本。

生产环境建议在应用层维护模型白名单，不要允许用户直接提交任意 `base_url`、`api_key` 或模型初始化参数。

---

## 十三、常见问题

### 1、为什么提示无法识别模型提供商？

如果模型名称不符合 LangChain 已知的自动推断规则，可能出现类似错误：

```
ValueError: Unable to infer model provider
```

建议显式指定提供商：

```
model = init_chat_model(

    "deepseek:deepseek-flash"

)
```

或者：

```
model = init_chat_model(

    "deepseek-flash",

    model_provider="deepseek"

)
```

### 2、为什么提示找不到 langchain-deepseek？

需要先安装 DeepSeek 集成包：

```
pip install -U langchain-deepseek
```

如果已经安装，仍然出现错误，应检查是否使用了不同的 Python 虚拟环境。

### 3、为什么无法调用 deepseek-chat？

这是旧版教程中常见的模型名称。

根据当前 DeepSeek 官方 API 文档，新项目应使用：

```
deepseek-flash

deepseek-v4-pro
```

例如：

```
model = init_chat_model(

    "deepseek:deepseek-flash"

)
```

如果需要使用 Pro：

```
model = init_chat_model(

    "deepseek:deepseek-v4-pro"

)
```

不要继续将旧模型名称作为新教程的默认示例。

### 4、为什么 deepseek-v4-flash 还能调用？

DeepSeek 官方保留了部分旧名称作为兼容别名。

根据官方说明，`deepseek-v4-flash` 和 `deepseek-v4-flash-vision-exp` 对应的旧模型已经下线，但旧名称的请求仍可以由 DeepSeek-V4.1-Flash 处理。

因此，新项目推荐使用 `deepseek-flash`，不要依赖旧模型名称。

### 5、为什么会出现 API Key 错误？

常见原因包括：

- 没有配置 `DEEPSEEK_API_KEY`。
- API Key 已失效或不具备调用权限。
- 没有正确调用 `load_dotenv()`。
- 使用了错误的模型提供商或 API 地址。

建议先检查环境变量是否存在，再核对 DeepSeek 控制台中的 API Key 状态。

### 6、使用 openai provider 可以调用 DeepSeek 吗？

可以，但必须正确设置 `base_url` 和 DeepSeek API Key。

例如：

```
model = init_chat_model(

    "deepseek-flash",

    model_provider="openai",

    base_url="https://api.deepseek.com",

    api_key="your_deepseek_api_key"

)
```

这个示例只是展示参数格式。实际项目应该从环境变量读取密钥，而不是将真实密钥写入代码。

### 7、模型切换是否需要重新启动程序？

如果使用可配置模型，可以在运行时通过 `config` 指定模型，不需要为了切换模型而重新启动应用。

不过，目标模型仍然需要对应的集成包、认证信息和服务端调用权限。

### 8、为什么设置 temperature 后报错？

不同模型对采样参数的支持存在差异。

如果具体模型不接受 `temperature` 或限制其取值，需要根据该模型的官方文档调整调用参数。

建议从最小配置开始，确认模型可以调用后，再添加必要的初始化参数。

### 9、configurable\_fields="any" 是否安全？

如果配置完全由可信的服务器代码控制，可以根据需要使用。

但如果配置来自 HTTP 请求或其他不可信来源，可能允许修改敏感参数。

因此，生产环境更推荐：

```
configurable_fields=[

    "model"

]
```

并通过模型白名单控制实际可选模型。

---

## 总结

`init_chat_model()` 是 LangChain 提供的统一聊天模型初始化入口，能够简化不同模型提供商之间的切换。

它主要包含以下功能：

- 通过 `provider:model` 指定模型与提供商。
- 在支持的情况下自动推断模型提供商。
- 使用固定模型直接执行调用。
- 通过 `configurable_fields` 控制运行时可修改的字段。
- 通过 `config_prefix` 避免多个模型对象的配置键冲突。
- 通过 `invoke()`、`stream()` 和 `ainvoke()` 统一执行模型调用。

对于刚开始使用 LangChain 的开发者，建议先采用最简单的方式：

```
from langchain.chat_models import init_chat_model



model = init_chat_model(

    "deepseek:deepseek-flash"

)



response = model.invoke("你好")



print(response.content)
```

当项目需要切换 Flash、Pro 或其他提供商模型时，再使用可配置模型功能。

需要记住的是：**LangChain 统一了模型的调用入口，但模型名称、功能支持、参数限制和计费规则仍然由具体提供商决定。**
