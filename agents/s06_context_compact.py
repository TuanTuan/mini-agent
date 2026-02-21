#!/usr/bin/env python3
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from logger import create_logger_from_args, parse_logger_args, get_logger_config_string

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

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
    print(logger._color(f"║  {title}" + " " * (77 - len(title)) + "║", "red"))
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
    """
    Layer 1: 静默压缩旧工具结果

    Returns:
        int: 被压缩的结果数量
    """
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))

    if len(tool_results) <= KEEP_RECENT:
        return 0

    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name

    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    cleared_count = 0
    for _, _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
            cleared_count += 1

    return cleared_count


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list, verbose: bool = True) -> list:
    """
    Layer 2: 自动压缩 - 保存完整对话，生成摘要，替换消息
    """
    before_tokens = estimate_tokens(messages)

    if verbose:
        print_compact_header("AUTO", f"tokens > {THRESHOLD:,}")
        print(logger._color(f"  💾 Saving transcript...", "yellow"))

    # Save full transcript to disk
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
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.content[0].text

    # Replace all messages with compressed summary
    new_messages = [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]

    after_tokens = estimate_tokens(new_messages)

    if verbose:
        print_compact_summary(before_tokens, after_tokens, str(transcript_path))
        # 显示摘要预览
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

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    """Agent 循环，包含三层压缩机制"""
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)

        # 显示当前 token 估算
        current_tokens = estimate_tokens(messages)
        print(logger._color(f"  📊 Token estimate: {current_tokens:,} / {THRESHOLD:,}", "dim"))

        # Layer 1: micro_compact before each LLM call
        cleared = micro_compact(messages)
        if cleared > 0:
            print(logger._color(f"  🗹 Layer 1 micro_compact: cleared {cleared} old tool results", "yellow"))

        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print(logger._color(f"  ⚠️ Threshold exceeded ({current_tokens:,} > {THRESHOLD:,})", "red"))
            messages[:] = auto_compact(messages, verbose=True)

        logger.messages_snapshot(messages, "BEFORE LLM CALL")

        # 显示原始请求
        logger.request_raw(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000
        )

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 显示原始响应
        logger.response_raw(response)

        # 显示响应摘要
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        logger.llm_response_summary(response.stop_reason, usage, len(response.content))
        logger.response_content_blocks(response.content)

        messages.append({"role": "assistant", "content": response.content})
        logger.messages_snapshot(messages, "AFTER APPEND ASSISTANT")

        if response.stop_reason != "tool_use":
            logger.loop_end(f"stop_reason = '{response.stop_reason}'")
            return

        # 执行工具调用
        logger.section("Executing Tool Calls", "🔧")
        results = []
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                input_data = dict(block.input)
                logger.tool_call(block.name, input_data, block.id)

                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"

                is_error = str(output).startswith("Error:")
                logger.tool_result(block.id, str(output), is_error=is_error)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        messages.append({"role": "user", "content": results})
        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")

        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print_compact_header("MANUAL", "compact tool called")
            messages[:] = auto_compact(messages, verbose=True)

        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("s06 Context Compact - Interactive Mode", "s06")

    # 显示当前日志配置
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    # 显示配置信息
    print(logger._color(f"  ⚙️ Compact Configuration:", "cyan"))
    print(logger._color(f"      Token threshold: {THRESHOLD:,}", "dim"))
    print(logger._color(f"      Keep recent tool results: {KEEP_RECENT}", "dim"))
    print(logger._color(f"      Transcript directory: {TRANSCRIPT_DIR}", "dim"))

    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
