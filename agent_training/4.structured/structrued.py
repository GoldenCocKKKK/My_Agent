import os
import time
import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage
from dotenv import load_dotenv
from rich import print as rprint
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List
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
"""
llm = init_chat_model(
    model = "deepseek-flash",
    model_provider = "openai",
    temperature = 0.7,
    api_key = API_KEY,
    base_url = BASE_URL, 
    reasoning = {
        "effort": "none"
    },
    model_kwargs={
        "max_output_tokens": 1024,
        "stream": False
    }
)
"""

class Actor(BaseModel):
    name:str = Field(description = "演员名称")
    role:str = Field(description = "扮演的角色")

class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")
    cast: List[Actor] = Field(description="演员列表")  # 定义列表字段

llm_with_structured_out = llm.with_structured_output(Movie)

messages = [
    HumanMessage(content="帮我介绍下电影盗梦空间"),
]

ffinal_movie = None

for chunk in llm_with_structured_out.stream(messages):
    if isinstance(chunk, Movie):
        final_movie = chunk
        # 如果想看流式动态更新，可以用清屏或覆盖打印，或者简单打印当前进度
        print(f"\r[正在解析] 电影: {getattr(chunk, 'title', '')}", end="", flush=True)

print("\n--- 解析完成 ---")
if final_movie:
    print(f"电影名: {final_movie.title}")
    print(f"上映年份: {final_movie.year}")
    print(f"导演: {final_movie.director}")
    print(f"演员列表: {final_movie.cast}")
    
    # 将最终解析出来的结构化对象转成字符串存入历史记录
    messages.append(AIMessage(content=str(final_movie.model_dump())))

end_time =time.time()
elapsed_time = end_time - start_time
print(f"执行耗时: {elapsed_time:.2f} 秒")