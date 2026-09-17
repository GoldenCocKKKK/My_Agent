# cad_agent_client_local.py
import asyncio
import json
import sys
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 1. 连接本地 1650 上运行的 Ollama 服务
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

LOCAL_MODEL = "qwen2.5-coder:3b"

# 定义系统提示词
SYSTEM_PROMPT = """你是一个专业的 CAD 交互绘图 Agent。你唯一的职责是根据用户的指示调用 CAD 工具完成图形绘制。

【重要约束】
1. 当需要绘图时，请严格使用系统提供的 Tool Calling 机制输出调用，绝对不要手动用 ```json 代码块文本输出 JSON 格式！
2. 绘制复合图形时，请主动拆解为基本图元（直线、圆等）进行工具调用。
3. 必须严格按照提供的工具 JSON Schema 格式输出参数，坐标数值类型保持为数字。
"""

async def run_local_cad_agent():
    # 2. 配置 MCP 服务端启动参数
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["cad_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 从 MCP 服务端拉取工具列表
            mcp_tools = await session.list_tools()
            
            # 转换为 OpenAI 工具格式
            openai_tools = []
            for tool in mcp_tools.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema  # MCP 2.x 使用 input_schema
                    }
                })

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            print(f"Native CAD Agent [本地模型: {LOCAL_MODEL}] 已就绪！")
            print("--------------------------------------------------")

            while True:
                user_input = input("\n工程师指令: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break

                messages.append({"role": "user", "content": user_input})

                while True:
                    response = client.chat.completions.create(
                        model=LOCAL_MODEL,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        temperature=0.1
                    )

                    msg = response.choices[0].message
                    messages.append(msg)

                    # 分支 A：原生触发了 Tool Call
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            fn_name = tool_call.function.name
                            
                            try:
                                if isinstance(tool_call.function.arguments, str):
                                    fn_args = json.loads(tool_call.function.arguments)
                                else:
                                    fn_args = tool_call.function.arguments
                            except json.JSONDecodeError:
                                err_msg = "工具参数 JSON 解析失败，请重新输出严格的 JSON 参数。"
                                print(f"⚠️ 格式解析警告: {err_msg}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": err_msg
                                })
                                continue

                            print(f"\nAgent 决定调用本地工具: [ {fn_name} ]")
                            print(f"   └─ 参数: {fn_args}")

                            # 请求 MCP 执行工具
                            result = await session.call_tool(fn_name, fn_args)
                            result_text = result.content[0].text if result.content else "OK"
                            
                            print(f"   └─ 本地反馈: {result_text}")

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_text
                            })

                    # 分支 B（文本拦截）：处理 3B 小模型直接输出纯 JSON 或 Markdown JSON 块的情况
                    elif msg.content and '"name":' in msg.content and ('"arguments":' in msg.content or '"x1":' in msg.content):
                        try:
                            content = msg.content.strip()
                            # 兼容 Markdown 格式与裸文本格式
                            if "```json" in content:
                                json_str = content.split("```json")[1].split("```")[0].strip()
                            elif "```" in content:
                                json_str = content.split("```")[1].split("```")[0].strip()
                            else:
                                json_str = content
                                
                            data = json.loads(json_str)
                            fn_name = data.get("name")
                            fn_args = data.get("arguments", {})

                            print(f"\nAgent (文本拦截) 调用本地工具: [ {fn_name} ]")
                            print(f"   └─ 参数: {fn_args}")

                            # 请求 MCP 执行工具
                            result = await session.call_tool(fn_name, fn_args)
                            result_text = result.content[0].text if result.content else "OK"
                            
                            print(f"   └─ 本地反馈: {result_text}")

                            messages.append({
                                "role": "user",
                                "content": f"工具 {fn_name} 已成功执行，返回结果: {result_text}。请总结并结束。"
                            })
                        except Exception as e:
                            print(f"\nAgent: {msg.content}")
                            break
                    else:
                        print(f"\nAgent: {msg.content}")
                        break

if __name__ == "__main__":
    asyncio.run(run_local_cad_agent())