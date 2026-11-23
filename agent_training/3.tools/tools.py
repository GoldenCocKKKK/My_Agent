import os
import time
import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage
from dotenv import load_dotenv
from rich import print as rprint
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool

start_time =time.time()
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
        "stream":False
    }
)

class WeatherSchema(BaseModel):
    city:str = Field(default = "北京", description = "城市名称")
    if_forecast:bool = Field(default = False, description = "是否包含明天的预测")
    unit:Literal["celsius", "fahrenheit"]

@tool(name_or_callable = "WeatherReport", description = "天气查询工具，可以包含明天的天气情况", args_schema = WeatherSchema)
def get_weather(city: str, if_forecast: bool):
    res = f"{city} 今天天气不错"
    if if_forecast:
        res += "\n明天要下雨"
    return res

llm_with_tools = llm.bind_tools([get_weather])

messages = [
    SystemMessage(content="你是一个天气助手，当用户询问天气时，必须优先调用 WeatherReport 工具来获取信息。"),
    HumanMessage(content="今天杭州的天气如何？明天呢？"),
]

result = llm_with_tools.invoke(messages)

rprint(result)
messages.append(result)

tool_calls = result.tool_calls
# 将工具调用的结果添加到消息列表中
for tool_call in tool_calls:
    if tool_call["name"] == "WeatherReport":
        # 返回值tool_msg类型是ToolMessage
        tool_msg = get_weather.invoke(tool_call)
        messages.append(tool_msg)

final_response = llm_with_tools.invoke(messages)

messages.append(final_response)

end_time =time.time()
elapsed_time = end_time - start_time
print(f"执行耗时: {elapsed_time:.2f} 秒")

for msg in messages:
    msg.pretty_print()

print()