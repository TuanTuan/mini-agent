#!/usr/bin/env python3
"""
litellm_s03.py - TodoWrite (LiteLLM/OpenAI Format)

基于 s03_todo_write.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s03.py -o session.md   # 输出到日志文件
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
SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


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
    "todo":       lambda **kw: TODO.update(kw["items"]),
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
    {"type": "function", "function": {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}}},
]


# -- Agent loop with nag reminder injection --
def agent_loop(messages: list):
    rounds_since_todo = 0
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)
        logger.messages_snapshot(messages, "BEFORE LLM CALL")

        # Nag reminder: if 3+ rounds without a todo update, inject reminder
        if rounds_since_todo >= 3 and messages:
            last = messages[-1]
            if last["role"] == "user" and isinstance(last.get("content"), str):
                last["content"] = "<reminder>Update your todos.</reminder>\n" + last["content"]
                logger.section("Nag Reminder Injected", "⚠️")

        # 显示原始请求
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

        # 显示原始响应
        logger.response_raw(response_dict)

        # 提取响应信息
        choice = (response_dict.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = message.get("tool_calls") or []
        usage = response_dict.get("usage") or {}

        # 显示响应摘要
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

        if finish_reason != "tool_calls":
            logger.loop_end(f"finish_reason = '{finish_reason}'")
            return

        # 执行工具调用
        logger.section("Executing Tool Calls", "🔧")
        used_todo = False
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

            handler = TOOL_HANDLERS.get(fn_name)
            try:
                output = handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
            except Exception as e:
                output = f"Error: {e}"

            # 显示工具结果
            is_error = str(output).startswith("Error:")
            logger.tool_result(tc_id, str(output), is_error=is_error)

            # 特殊处理 todo 工具
            if fn_name == "todo":
                used_todo = True
                logger.section("Todo State Updated", "📋")
                print(TODO.render())

            # 追加 tool 结果消息
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": str(output),
            })

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    logger.header("LiteLLM TodoWrite Agent - Azure GPT-5.2", "litellm-s03")
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
            query = input("\033[36mlitellm-s03 >> \033[0m")
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

    # 结束会话
    logger.session_end("用户退出")
