# LangChain 环境搭建

> 来源: https://www.runoob.com/langchain/langchain-env-setup.html

---

# LangChain 环境搭建

本篇将带你完成 LangChain 的安装和配置，确保你能顺利运行第一个 LangChain 程序。

---

## 环境要求

| 项目 | 最低版本 | 推荐版本 |
| --- | --- | --- |
| Python | 3.10 | 3.11 或 3.12 |
| pip | 22.0 | 最新版 |
| 操作系统 | macOS / Linux / Windows 均可 | |

检查 Python 版本：

```


$ python --version

Python 3.12.7

```

> 如果你还没有安装 Python，可以参考我们的 [Python3 环境搭建](https://www.python.org/downloads/)。
>
> Python 包与环境管理工具参考： [uv 入门教程](https://www.runoob.com/python3/uv-tutorial.html)。

---

## 安装 LangChain

LangChain 采用模块化设计，核心功能和第三方集成分开安装。

### 安装核心包

```


$ pip install langchain

```

这个命令会安装 langchain 主包，它包含了 init\_chat\_model()、create\_agent() 等核心 API，同时会自动安装 langchain-core 作为依赖。

![](https://www.runoob.com/wp-content/uploads/2026/05/runoob_1778725725196.png)

### 安装模型提供商包

LangChain 本身不包含具体的模型实现，你需要根据使用的模型安装对应的提供商包：

| 模型提供商 | 安装命令 | 使用的模型 |
| --- | --- | --- |
| OpenAI | pip install langchain-openai | GPT-4、GPT-5 等 |
| Anthropic | pip install langchain-anthropic | Claude 系列 |
| DeepSeek | pip install langchain-deepseek | DeepSeek-V3、R1 等 |
| Google | pip install langchain-google-genai | Gemini 系列 |
| Ollama（本地模型） | pip install langchain-ollama | Llama、Qwen 等本地模型 |
| xAI | pip install langchain-xai | Grok 系列 |
| Mistral | pip install langchain-mistralai | Mistral 系列 |

### 一键安装

建议初学者直接安装以下组合：

```


$ pip install langchain langchain-openai python-dotenv

```

> python-dotenv 用于从 .env 文件加载 API Key，这是管理敏感信息的最佳实践，可以避免将 Key 硬编码在代码中。

---

## 配置 API Key

### 获取 API Key

以 OpenAI 为例，你需要先注册账号并获取 API Key：

1. 访问 <https://platform.openai.com> 注册或登录
2. 进入 API Keys 页面，点击 "Create new secret key"
3. 复制生成的 Key（格式为 sk-xxxxxxxx）

> API Key 只会显示一次，请立即保存到安全的地方。不要将 Key 提交到 Git 仓库或分享给他人。

其他提供商的获取方式类似，请参考对应文档。

### 设置环境变量

推荐使用 .env 文件管理 API Key。在项目根目录创建 .env 文件：

## 实例

# 文件路径：.env
# 填入你的 API Key
OPENAI\_API\_KEY=sk-your-api-key-here

# 如果使用其他模型，也在这里配置
# ANTHROPIC\_API\_KEY=sk-ant-your-key
# DEEPSEEK\_API\_KEY=sk-your-key

然后创建 .gitignore 文件，确保 .env 不会被提交：

```


# .gitignore

.env

__pycache__/

*.pyc

```

### 在代码中加载

## 实例

# 文件路径：config.py
# 在程序开头加载 .env 文件
import os
from dotenv import load\_dotenv

# 加载 .env 文件中的环境变量
load\_dotenv()

# 验证 API Key 是否加载成功
api\_key = os.getenv("OPENAI\_API\_KEY")
if api\_key:
    # 只显示前8位和后4位，避免泄露完整 Key
    print(f"API Key 已加载: {api\_key[:8]}...{api\_key[-4:]}")
else:
    print("警告：未找到 OPENAI\_API\_KEY，请检查 .env 文件")

---

## 验证安装

运行以下脚本验证安装是否成功：

## 实例

# 文件路径：verify\_install.py
# 验证 LangChain 安装和 API Key 配置
from dotenv import load\_dotenv
load\_dotenv()

# 测试 1：验证 langchain 导入
try:
    import langchain
    print(f"langchain 版本: {langchain.\_\_version\_\_}")
except ImportError:
    print("错误：langchain 未安装，请运行 pip install langchain")

# 测试 2：验证 langchain-openai 导入
try:
    import langchain\_openai
    print("langchain-openai 已安装")
except ImportError:
    print("错误：langchain-openai 未安装，请运行 pip install langchain-openai")

# 测试 3：验证 API Key 配置
import os
api\_key = os.getenv("OPENAI\_API\_KEY")
if not api\_key:
    print("错误：OPENAI\_API\_KEY 未配置，请在 .env 文件中设置")
else:
    print(f"API Key 配置成功: {api\_key[:8]}...{api\_key[-4:]}")

# 测试 4：发送一条测试请求
from langchain.chat\_models import init\_chat\_model

model = init\_chat\_model("openai:gpt-4o-mini")
response = model.invoke("用一句话介绍菜鸟教程（RUNOOB）")
print(f"\n模型回复: {response.content}")

运行结果：

```


langchain 版本: 1.3.0

langchain-openai 已安装

API Key 配置成功: sk-proj-z...xxxx

模型回复: 菜鸟教程（RUNOOB）是一个面向编程初学者的中文技术学习平台。

```

> 如果最后一步报错，请检查：(1) API Key 是否正确；(2) 是否有网络代理影响连接；(3) 账户余额是否充足。

国内我们可以采用 DeepSeek 大模型来测试，如果还没有需要先去 <https://platform.deepseek.com/api_keys> 创建一个 API key。

DeepSeek 的 API 文档参考：<https://api-docs.deepseek.com/zh-cn/>。

如果你需要统一管理多个第三方模型，可以选择安装 LiteLLM 作为模型网关：

```


pip install -U litellm
```

不过本文的示例直接使用 `ChatOpenAI` 配合 `openai_api_base` 参数连接 DeepSeek，无需额外安装 LiteLLM。

## 实例

import os
from langchain\_openai import ChatOpenAI

# 建议将 API Key 放在环境变量中，或者直接在此处赋值（生产环境请勿硬编码）
# os.environ["DEEPSEEK\_API\_KEY"] = "sk-你的DeepSeek密钥"

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-v4-pro",             # DeepSeek V4 模型名称
    openai\_api\_key="sk-你的Key",        # 填入你的 DeepSeek API Key
    openai\_api\_base="https://api.deepseek.com", # DeepSeek 的 API 地址
    temperature=0.7,
    max\_tokens=1024
)

# 测试一下
response = llm.invoke("你好，DeepSeek！请做一个简短的自我介绍。")
print(response.content)

执行以上代码，就会输出内容：

```
你好！很高兴认识你！



我是DeepSeek，由深度求索公司创造的AI助手。我的特点包括：

。。。
```

---

## 项目目录建议

推荐的项目目录结构：

```


langchain-learning/

├── .env                  # API Key 配置（不提交到 Git）

├── .gitignore            # Git 忽略规则

├── config.py             # 公共配置加载

├── 01_hello_world.py     # 第一篇的示例

├── 02_first_agent.py     # Agent 示例

└── ...                   # 更多练习

```
