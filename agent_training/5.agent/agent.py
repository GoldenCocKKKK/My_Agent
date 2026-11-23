import os
import time
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
from rich import print as rprint

start_time = time.time()
load_dotenv(override=True)

# 从环境变量读取配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
LLM_NAME = os.getenv("LLM_NAME")
AGENT_ROLE = os.getenv("AGENT_ROLE")
AGENT_HELLO = os.getenv("AGENT_HELLO")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "3"))
MAX_PAIRS_HISTORY = MAX_HISTORY
EXIT_WORD = "quit"


llm = init_chat_model(
    model = LLM_NAME,
    model_provider = "ollama",
    temperature = 0.2,
    model_kwargs={
        "reasoning":False,
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
    }
)
"""

question = "你好，用一句话请你介绍一下你自己。"

messages = [ 
    #SystemMessage(AGENT_ROLE),
    AIMessage(AGENT_HELLO),
    HumanMessage(question),
]

agent = create_agent(
    model = llm
)
response = agent.invoke({"messages": messages}) 
rprint(response)
messages.append(response)

# 用时监测
end_time = time.time()
elapsed_time = end_time - start_time
print(f"执行耗时: {elapsed_time:.2f} 秒")
print()
