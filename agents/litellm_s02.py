#!/usr/bin/env python3
"""
litellm_s02.py - Tools (LiteLLM/OpenAI Format)

基于 s02_tool_use.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
The agent loop didn't change. We just added tools to the array
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

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s02.py                    # 默认：终端详细日志 + 显示RAW
    python litellm_s02.py -q                 # 安静模式：不在终端显示日志
    python litellm_s02.py -o session.md      # 输出到Markdown文件
"""

import json
import os
import subprocess
from pathlib import Path

import litellm
from dotenv import load_dotenv

from logger_openai import create_logger_from_args, parse_logger_args, get_logger_config_string

load_dotenv(override=True)

# ============================================================================
# 配置
# ============================================================================
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5.2")

WORKDIR = Path.cwd()
MODEL = f"azure/{AZURE_DEPLOYMENT}"
SYSTEM_PROMPT = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


def safe_path(p: str) -> Path:
    """确保路径不逃逸工作目录"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令"""
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
    """读取文件内容"""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件（替换文本）"""
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

# -- OpenAI 格式的工具定义 --
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "description": "Optional line limit"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


# ============================================================================
# Agent Loop
# ============================================================================
def agent_loop(messages: list):
    """
    核心 Agent 循环

    OpenAI 响应结构:
    {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "...",
                "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
            },
            "finish_reason": "tool_calls" | "stop"
        }],
        "usage": {"prompt_tokens": N, "completion_tokens": N}
    }
    """
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)

        # 显示调用 LLM 前的消息状态
        logger.messages_snapshot(messages, "BEFORE LLM CALL")

        # ========== 显示原始 API 请求数据 ==========
        logger.request_raw(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000
        )

        # 调用 LiteLLM
        response = litellm.completion(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )

        # 转换为字典
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)

        # ========== 显示原始 API 响应数据 ==========
        logger.response_raw(response_dict)

        # 提取响应信息
        choice = (response_dict.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = message.get("tool_calls") or []
        usage = response_dict.get("usage") or {}

        # 显示 LLM 响应摘要
        logger.llm_response_summary(
            finish_reason,
            {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
            len(tool_calls)
        )

        # 追加 assistant 消息
        assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        logger.messages_snapshot(messages, "AFTER APPEND ASSISTANT")

        # 如果模型没有调用工具，循环结束
        if finish_reason != "tool_calls":
            logger.loop_end(f"finish_reason = '{finish_reason}'")
            return

        # 执行工具调用
        logger.section("Executing Tool Calls", "🔧")
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args_str = fn.get("arguments", "{}")

            try:
                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            except json.JSONDecodeError:
                fn_args = {}

            # 显示工具调用
            logger.tool_call(fn_name, fn_args, tc_id)

            # 执行工具
            handler = TOOL_HANDLERS.get(fn_name)
            output = handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
            print(f"\033[33m> {fn_name}:\033[0m {output[:200]}")

            # 显示工具结果
            is_error = output.startswith("Error:")
            logger.tool_result(tc_id, output, is_error=is_error)

            # 追加 tool 结果消息
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": output,
            })

        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    logger.header("LiteLLM Multi-Tool - Azure GPT-5.2", "litellm-s02")
    logger.config(
        model=MODEL,
        api_base=AZURE_API_BASE,
        api_version=AZURE_API_VERSION
    )

    # 显示当前日志配置
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    # OpenAI 格式: system 是第一条消息
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            query = input("\033[36mlitellm-s02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        logger.user_input(query)
        history.append({"role": "user", "content": query})
        agent_loop(history)

        logger.separator("FINAL RESPONSE")
        # 获取最后的 assistant 消息
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    print(content)
                break
        print()

    # 结束会话
    logger.session_end("用户退出")
