#!/usr/bin/env python3
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from logger import create_logger_from_args, parse_logger_args, get_logger_config_string

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


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
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"\033[33m> {block.name}:\033[0m {output[:200]}")

                # 显示工具结果
                is_error = output.startswith("Error:")
                logger.tool_result(block.id, output, is_error=is_error)

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
    logger.header("s02 Multi-Tool - Interactive Mode", "s02")

    # 显示当前日志配置
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
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
