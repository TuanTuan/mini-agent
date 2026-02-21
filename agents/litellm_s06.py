#!/usr/bin/env python3
"""
litellm_s06.py - Compact (LiteLLM/OpenAI Format)

基于 s06_context_compact.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Three-layer compression pipeline so the agent can work forever:

    [Layer 1: micro_compact] Replace old tool results with placeholders
    [Layer 2: auto_compact] Save transcript, summarize, replace messages (tokens > threshold)
    [Layer 3: compact tool] Manual compression triggered by model

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s06.py -o session.md   # 输出到日志文件
"""

import json
import os
import subprocess
import time
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
SYSTEM_PROMPT = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


def print_compact_header(layer: str, trigger: str = ""):
    """打印压缩操作的标题"""
    print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "red"))
    title = f"🗜️ {layer.upper()} COMPACT TRIGGERED"
    if trigger:
        title += f" ({trigger})"
    print(logger._color(f"║  {title}" + " " * (77 - len(title)) + "│", "red"))
    print(logger._color(f"╚{'═' * 78 + '╝'}", "red"))


def print_compact_summary(before_tokens: int, after_tokens: int, transcript_path: str = ""):
    """打印压缩结果摘要"""
    saved = before_tokens - after_tokens
    ratio = (saved / before_tokens * 100) if before_tokens > 0 else 0
    print(logger._color(f"\n  📊 Compression Summary:", "cyan"))
    print(logger._color(f"      Before: {before_tokens:,} tokens", "dim"))
    print(logger._color(f"      After:  {after_tokens:,} tokens", "dim"))
    print(logger._color(f"      Saved:  {saved:,} tokens ({ratio:.1f}%)", "green"))
    if transcript_path:
        print(logger._color(f"      Transcript: {transcript_path}", "dim"))


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> int:
    """Layer 1: 静默压缩旧工具结果"""
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "tool":
            tool_results.append((msg_idx, msg))

    if len(tool_results) <= KEEP_RECENT:
        return 0

    # Find tool_name for each result
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_name_map[tc.get("tool_call_id", "")] = tc.get("function", {}).get("name", "unknown")

    to_clear = tool_results[:-KEEP_RECENT]
    cleared_count = 0
    for _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_call_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
            cleared_count += 1

    return cleared_count


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list, verbose: bool = True) -> list:
    """Layer 2: 自动压缩 - 保存完整对话，生成摘要，替换消息"""
    before_tokens = estimate_tokens(messages)

    if verbose:
        print_compact_header("AUTO", f"tokens > {THRESHOLD:,}")
        print(logger._color(f"  💾 Saving transcript...", "yellow"))

    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")

    if verbose:
        print(logger._color(f"     Saved to: {transcript_path}", "dim"))
        print(logger._color(f"  🤖 Generating summary...", "yellow"))

    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        api_key=AZURE_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION,
        max_tokens=2000,
    )
    response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
    summary = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Replace all messages with compressed summary
    new_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]

    after_tokens = estimate_tokens(new_messages)

    if verbose:
        print_compact_summary(before_tokens, after_tokens, str(transcript_path))
        print(logger._color(f"\n  📝 Summary Preview:", "cyan"))
        summary_lines = summary.split("\n")[:5]
        for line in summary_lines:
            preview = line[:70] + "..." if len(line) > 70 else line
            print(logger._color(f"      {preview}", "dim"))
        if len(summary.split("\n")) > 5:
            print(logger._color(f"      ... ({len(summary.split(chr(10))) - 5} more lines)", "dim"))

    return new_messages


# -- Tool implementations --
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
    "compact":    lambda **kw: "Manual compression requested.",
}

# -- OpenAI 格式的工具定义 --
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "compact", "description": "Trigger manual conversation compression.",
     "parameters": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}}},
]


def agent_loop(messages: list):
    """Agent 循环，包含三层压缩机制"""
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)

        current_tokens = estimate_tokens(messages)
        print(logger._color(f"  📊 Token estimate: {current_tokens:,} / {THRESHOLD:,}", "dim"))

        # Layer 1: micro_compact
        cleared = micro_compact(messages)
        if cleared > 0:
            print(logger._color(f"  🗹 Layer 1 micro_compact: cleared {cleared} old tool results", "yellow"))

        # Layer 2: auto_compact
        if estimate_tokens(messages) > THRESHOLD:
            print(logger._color(f"  ⚠️ Threshold exceeded ({current_tokens:,} > {THRESHOLD:,})", "red"))
            messages[:] = auto_compact(messages, verbose=True)

        logger.messages_snapshot(messages, "BEFORE LLM CALL")
        logger.request_raw(model=MODEL, messages=messages, tools=TOOLS, max_tokens=8000)

        response = litellm.completion(
            model=MODEL, messages=messages, tools=TOOLS,
            api_key=AZURE_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION,
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

        logger.section("Executing Tool Calls", "🔧")
        manual_compact = False
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

            if fn_name == "compact":
                manual_compact = True
                output = "Compressing..."
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

        # Layer 3: manual compact
        if manual_compact:
            print_compact_header("MANUAL", "compact tool called")
            messages[:] = auto_compact(messages, verbose=True)

        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("LiteLLM Context Compact - Azure GPT-5.2", "litellm-s06")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)

    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    print(logger._color(f"  ⚙️ Compact Configuration:", "cyan"))
    print(logger._color(f"      Token threshold: {THRESHOLD:,}", "dim"))
    print(logger._color(f"      Keep recent tool results: {KEEP_RECENT}", "dim"))
    print(logger._color(f"      Transcript directory: {TRANSCRIPT_DIR}", "dim"))

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            query = input("\033[36mlitellm-s06 >> \033[0m")
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
