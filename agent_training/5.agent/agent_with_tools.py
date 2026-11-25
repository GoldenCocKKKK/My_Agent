import os
import time
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.tools import tool
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
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "3"))
MAX_PAIRS_HISTORY = MAX_HISTORY
EXIT_WORD = "quit"

@tool(name_or_callable = "WeatherReport", description = "查询指定城市的当前天气。city 必须是中文城市名，例如北京")
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。city 必须是中文城市名，例如"北京"。"""
    fake_db = {"北京": "晴，25℃，东北风 2 级",
               "上海": "小雨，18℃，东风 3 级"}
    return fake_db.get(city, f"查不到{city}的天气")


llm = init_chat_model(
    model=LLM_NAME,
    model_provider="openai",
    api_key = API_KEY,
    base_url = BASE_URL, 
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个虚假的天气查询助手，可以帮我查找指定城市的虚假天气",
)

response = agent.invoke({
    "messages": [HumanMessage(content="今天北京的天气如何？")]
})

# 完整轨迹（调试必看）
for m in response["messages"]:
    m.pretty_print()

# 用时监测
end_time = time.time()
elapsed_time = end_time - start_time
print(f"执行耗时: {elapsed_time:.2f} 秒")
print()