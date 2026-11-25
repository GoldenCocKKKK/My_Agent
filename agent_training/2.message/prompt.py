import os
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
from rich import print as rprint
from langchain_core.prompts import ChatPromptTemplate


def keep_recent_messages(messages, max_pairs=3):
    """
    保留最近的 N 轮对话

    max_pairs: 保留的对话轮数（每轮 = user + assistant）
    """
    # 分离 system 和对话（LangChain Message 使用 type 字段）
    system_msgs = [m for m in messages if getattr(m, "type", None) == "system"]
    conversation_msgs = [m for m in messages if getattr(m, "type", None) != "system"]

    # 只保留最近的
    recent_msgs = conversation_msgs[-(max_pairs * 2):]

    # 返回：system + 最近对话
    return system_msgs + recent_msgs

start_time = time.time()
load_dotenv(override=True)

# 从环境变量读取配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
LLM_NAME = os.getenv("LLM_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
AGENT_ROLE = os.getenv("AGENT_ROLE")
AGENT_HELLO = os.getenv("AGENT_HELLO")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "3"))
MAX_PAIRS_HISTORY = MAX_HISTORY
EXIT_WORD = "quit"


llm = init_chat_model(
    model = OLLAMA_MODEL,
    model_provider = "ollama",
    temperature = 0.2,
    model_kwargs={
        "reasoning":False,
        "stream":False
    }
)

"""
llm = init_chat_model(
    model = "deepseek-flash",
    model_provider = "openai",
    temperature = 0.7,
    api_key = API_KEY,
    base_url = BASE_URL, 
    model_kwargs={
        "reasoning": {
        "effort": "none"
        },
        "max_output_tokens": 1024,
        "stream": False
    }
)
"""

question = "你好，请你介绍一下你自己。"

chat_prompt =ChatPromptTemplate.from_messages(
    # 字符串 role + 字符串 content
    [("system", AGENT_ROLE),
    ("ai", AGENT_HELLO),
    ("human", "{input}")]
)

# 用时监测
response = llm.invoke(chat_prompt.invoke({"input": question}))
end_time = time.time()

rprint(response)
#print(response.content)
#print("id:", response.id)
#print("Token 消耗:", response.usage_metadata)
#print("底层元数据:", response.response_metadata)
#print("工具调用:", response.tool_calls)

elapsed_time = end_time - start_time
print(f"执行耗时: {elapsed_time:.2f} 秒")
print()

message = [
    SystemMessage(content=AGENT_ROLE),
    AIMessage(content=AGENT_HELLO),
    HumanMessage(content=question),
    AIMessage(content=response.content),
]

i=0
while True:
    i+=1

    # 优化历史记忆
    if i>=MAX_HISTORY:
        message = keep_recent_messages(message, max_pairs=MAX_HISTORY)
        print(f"已达到 {MAX_PAIRS_HISTORY} 次的历史留存上限，之前的记录将被清空")
        i=0

    start_time = time.time()
    user_input = input(AGENT_HELLO)
    if user_input.lower() == EXIT_WORD:
        break

    # 追加用户消息
    message.append(HumanMessage(user_input))

    response = llm.invoke(message)   
    rprint(response)

    # 用时监测
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"执行耗时: {elapsed_time:.2f} 秒")
    print()

    # 追加 AI 回复
    message.append(AIMessage(response.content))