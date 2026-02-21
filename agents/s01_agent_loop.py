#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop (~70 LOC)

The entire secret of coding agents in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

That's it. The ENTIRE agent is a while loop that feeds tool
results back to the model until the model decides to stop.

命令行参数:
    python s01_agent_loop.py                    # 默认：终端详细日志 + 显示RAW
    python s01_agent_loop.py -q                 # 安静模式：不在终端显示日志
    python s01_agent_loop.py --no-show-raw      # 不显示原始API数据
    python s01_agent_loop.py -o session.md      # 输出到Markdown文件
    python s01_agent_loop.py -q -o logs/s01.md  # 只写文件，不在终端显示
    python s01_agent_loop.py --log-file session.md --no-file-show-raw  # 文件中不含RAW
"""

import os
import subprocess
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from logger import create_logger_from_args, parse_logger_args, get_logger_config_string

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)

        # 显示调用 LLM 前的消息状态
        logger.messages_snapshot(messages, "BEFORE LLM CALL")

        # ========== 显示原始 API 请求数据 ==========
        logger.request_raw(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000
        )

        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # ========== 显示原始 API 响应数据 ==========
        logger.response_raw(response)

        # 显示 LLM 响应摘要
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        logger.llm_response_summary(response.stop_reason, usage, len(response.content))

        # 显示响应内容详情
        logger.response_content_blocks(response.content)

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # 显示追加后的消息状态
        logger.messages_snapshot(messages, "AFTER APPEND ASSISTANT")

        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            logger.loop_end("stop_reason != 'tool_use'")
            return

        # Execute each tool call, collect results
        logger.section("Executing Tool Calls", "🔧")
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 显示工具调用
                input_data = dict(block.input)
                logger.tool_call(block.name, input_data, block.id)

                # 执行工具
                if block.name == "bash":
                    output = run_bash(block.input["command"])
                    print(f"\033[33m$ {block.input['command']}\033[0m")
                    print(output[:200] if len(output) > 200 else output)

                # 显示工具结果
                logger.tool_result(block.id, output)

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # 追加工具结果
        messages.append({"role": "user", "content": results})
        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")

        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("s01 Agent Loop - Interactive Mode", "s01")

    # 显示当前日志配置
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        logger.user_input(query)

        history.append({"role": "user", "content": query})
        agent_loop(history)

        logger.separator("FINAL RESPONSE")
        for block in history[-1]["content"] if isinstance(history[-1]["content"], list) else []:
            if hasattr(block, "text"):
                print(block.text)
        print()

    # 结束会话
    logger.session_end("用户退出")
