#!/usr/bin/env python3
"""
litellm_s04.py - Subagents (LiteLLM/OpenAI Format)

基于 s04_subagent.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s04.py -o session.md   # 输出到日志文件
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
SYSTEM_PROMPT = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)

# 子代理计数器
_subagent_counter = 0


# -- Tool implementations shared by parent and child --
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
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
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


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning) - OpenAI format
CHILD_TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
]


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str, description: str = "subtask") -> str:
    """
    启动子代理执行任务

    子代理特点:
    - fresh context: 独立的消息历史
    - filtered tools: 只有基础工具，不能递归启动子代理
    - summary-only return: 只返回最终摘要给父代理
    """
    global _subagent_counter
    _subagent_counter += 1
    subagent_id = _subagent_counter

    # 子代理日志标题
    print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "magenta"))
    print(logger._color(f"║  🤖 SUBAGENT #{subagent_id} SPAWNED{' ' * 58}║", "magenta"))
    print(logger._color(f"║  Description: {description[:60]}{' ' * (61 - min(len(description), 61))}║", "magenta"))
    print(logger._color(f"╚{'═' * 78 + '╝'}", "magenta"))

    sub_messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    iteration = 0
    last_content = ""

    for _ in range(30):  # safety limit
        iteration += 1
        indent = "  "
        print(logger._color(f"\n{indent}🔄 SUBAGENT #{subagent_id} ITERATION #{iteration}", "magenta"))

        response = litellm.completion(
            model=MODEL,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )

        response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        choice = (response_dict.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = message.get("tool_calls") or []
        last_content = message.get("content") or ""

        # 追加 assistant 消息
        assistant_msg = {"role": "assistant", "content": last_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        sub_messages.append(assistant_msg)

        if finish_reason != "tool_calls":
            print(logger._color(f"{indent}🏁 SUBAGENT #{subagent_id} DONE: {finish_reason}", "magenta"))
            break

        # 执行工具调用
        print(logger._color(f"{indent}🔧 Executing tools...", "magenta"))
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args_str = fn.get("arguments", "{}")

            try:
                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            except json.JSONDecodeError:
                fn_args = {}

            handler = TOOL_HANDLERS.get(fn_name)
            try:
                output = handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
            except Exception as e:
                output = f"Error: {e}"

            output_preview = str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
            print(logger._color(f"{indent}  ⚡ {fn_name}: {output_preview}", "dim"))

            sub_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": str(output)[:50000],
            })

    summary = last_content or "(no summary)"

    # 子代理完成日志
    print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "green"))
    print(logger._color(f"║  ✅ SUBAGENT #{subagent_id} COMPLETED{' ' * 54}║", "green"))
    summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
    print(logger._color(f"║  Summary: {summary_preview[:66]}{' ' * (67 - min(len(summary_preview), 67))}║", "green"))
    print(logger._color(f"╚{'═' * 78 + '╝'}", "green"))

    return summary


# -- Parent tools: base tools + task dispatcher - OpenAI format --
PARENT_TOOLS = CHILD_TOOLS + [
    {"type": "function", "function": {"name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}}},
]


def agent_loop(messages: list):
    """父代理循环"""
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)
        logger.messages_snapshot(messages, "BEFORE LLM CALL")

        # 显示原始请求
        logger.request_raw(
            model=MODEL,
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000
        )

        response = litellm.completion(
            model=MODEL,
            messages=messages,
            tools=PARENT_TOOLS,
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )

        response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        logger.response_raw(response_dict)

        choice = (response_dict.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = message.get("tool_calls") or []
        usage = response_dict.get("usage") or {}

        logger.llm_response_summary(
            finish_reason,
            {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
            len(tool_calls)
        )

        assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)
        logger.messages_snapshot(messages, "AFTER APPEND ASSISTANT")

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

            logger.tool_call(fn_name, fn_args, tc_id)

            if fn_name == "task":
                desc = fn_args.get("description", "subtask")
                task_prompt = fn_args["prompt"]
                output = run_subagent(task_prompt, desc)
            else:
                handler = TOOL_HANDLERS.get(fn_name)
                try:
                    output = handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
                except Exception as e:
                    output = f"Error: {e}"

            is_error = str(output).startswith("Error:")
            logger.tool_result(tc_id, str(output), is_error=is_error)
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)})

        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("LiteLLM Subagent - Azure GPT-5.2", "litellm-s04")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)

    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            query = input("\033[36mlitellm-s04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        logger.user_input(query)
        history.append({"role": "user", "content": query})
        agent_loop(history)

        logger.separator("FINAL RESPONSE")
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    print(content)
                break
        print()

    logger.session_end("用户退出")
