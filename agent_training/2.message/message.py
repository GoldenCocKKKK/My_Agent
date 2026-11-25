import os
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
from rich import print as rprint

start_time = time.time()
load_dotenv(override=True)

# 从环境变量读取配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
LLM_NAME = os.getenv("LLM_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
AGENT_ROLE = os.getenv("AGENT_ROLE")
AGENT_HELLO = os.getenv("AGENT_HELLO")

llm = init_chat_model(
    model = OLLAMA_MODEL,
    model_provider = "ollama",
    temperature = 0.2,
    model_kwargs={
        "reasoning":False,
        "stream":False
    }
)

question = "你好，请你介绍一下你自己。"

message = [ 
    SystemMessage(AGENT_ROLE),
    AIMessage(AGENT_HELLO),
    HumanMessage(question),
]

response = llm.invoke(message)
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