# LangChain 教程

> 来源: https://www.runoob.com/langchain/langchain-tutorial.html

---

# LangChain 教程

![](https://www.runoob.com/wp-content/uploads/2026/05/langchain-color.png)

LangChain 是一套用于构建 AI 智能体（AI Agent）和大语言模型（LLM）应用的开发框架。

LangChain 可以帮助开发者快速构建基于 GPT、Claude、Gemini 等大模型的复杂 AI 应用。

LangChain 由 Harrison Chase 于 2022 年 10 月推出，核心目标是：简化大语言模型应用开发流程。

LangChain提供统一接口，可连接：大模型、Prompt、向量数据库、工具调用、记忆系统以及 Agent 工作流。

目前 LangChain 已成为最热门的 LLM 应用开发框架之一，广泛应用于：智能聊天机器人、RAG 知识库、文档分析、代码生成、AI 自动化等场景。

---

## 谁适合阅读本教程？

本教程适合具备 Python 基础，并希望学习 AI 应用开发的开发者。

- 有 Python 基础，想学习 AI 与大语言模型开发的新手
- 想开发 AI 聊天机器人、知识库、Agent 应用的开发者
- 对 GPT、Claude、RAG、向量数据库感兴趣的学习者
- 希望使用 LangChain 快速构建 AI 项目的工程师
- 想从传统开发转向 AI 应用开发的程序员

---

## 学习本教程前你需要了解

学习本教程前，你需要具备一定 Python 基础，并了解基础 Web 与 API 概念。

- Python 基础：变量、函数、类、模块导入、异常处理
- HTTP 与 API 基础：GET/POST 请求、JSON 数据格式
- Prompt 基础：了解什么是 Prompt 与大语言模型
- 基础数据库知识：了解 SQLite / MySQL 基础操作
- 了解基本命令行操作与 pip 包管理

---

## LangChain 可以做什么？

- AI 聊天机器人（ChatBot）
- RAG 企业知识库
- PDF 文档问答系统
- AI Agent 自动任务执行
- 代码生成与代码分析
- 多轮对话与上下文记忆
- 联网搜索与工具调用
- 工作流自动化系统

---

## 第一个 LangChain 程序

以下代码使用 LangChain 调用 OpenAI 模型，并输出 AI 生成内容：

## 实例

from langchain\_openai import ChatOpenAI
from langchain\_core.prompts import ChatPromptTemplate

# 初始化模型
llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# Prompt 模板
prompt = ChatPromptTemplate.from\_template(
    "请解释：{topic}"
)

# 创建 Chain
chain = prompt | llm

# 调用
result = chain.invoke({
    "topic": "什么是 Transformer"
})

print(result.content)

运行后，大模型会自动生成关于 Transformer 的解释内容。

---

## LangChain 核心组件

- LLM：连接 OpenAI、Claude、Gemini 等大模型
- PromptTemplate：管理 Prompt 模板
- Chains：构建多步骤 AI 工作流
- Memory：实现多轮对话记忆
- Tools：调用搜索、数据库、API 等工具
- Agents：让 AI 自动决策与执行任务
- Vector Store：连接向量数据库实现 RAG

---

## 参考文档

- LangChain 官网：
  <https://www.langchain.com/>
- LangChain Python 文档：
  <https://python.langchain.com/>
- LangChain JavaScript 文档：
  <https://js.langchain.com/>
- LangChain GitHub：
  <https://github.com/langchain-ai/langchain>
