# cad_mcp_server.py
import sys
import asyncio
from mcp.server.mcpserver import MCPServer   # MCP 2.x 规范使用 MCPServer

# 实例化 MCPServer 服务端对象
mcp = MCPServer("LocalCADServer")

# 注册画直线工具
@mcp.tool()
async def draw_line(x1: float, y1: float, x2: float, y2: float) -> str:
    """在 CAD 图纸中绘制一条直线。
    
    参数:
        x1: 起点 X 轴坐标 (单位: mm)
        y1: 起点 Y 轴坐标 (单位: mm)
        x2: 终点 X 轴坐标 (单位: mm)
        y2: 终点 Y 轴坐标 (单位: mm)
    """
    # 在 stdio 模式下，调试日志必须输出到 sys.stderr，避免污染 stdout 的 JSON-RPC 通信通道
    print(f"\n[CAD 底层执行] 绘制直线: 从 ({x1}, {y1}) 到 ({x2}, {y2})", file=sys.stderr)
    return f"成功在 CAD 中绘制直线，起点:({x1}, {y1}), 终点:({x2}, {y2})"

# 注册画圆工具
@mcp.tool()
async def draw_circle(cx: float, cy: float, radius: float) -> str:
    """在 CAD 图纸中绘制一个圆。
    
    参数:
        cx: 圆心 X 轴坐标 (单位: mm)
        cy: 圆心 Y 轴坐标 (单位: mm)
        radius: 圆的半径 (单位: mm)
    """
    print(f"\n[CAD 底层执行] 绘制圆: 圆心 ({cx}, {cy}), 半径: {radius}", file=sys.stderr)
    return f"成功在 CAD 中绘制圆，圆心:({cx}, {cy}), 半径:{radius}"

if __name__ == "__main__":
    # MCP 2.x 规范启动入口
    mcp.run(transport='stdio')