#!/usr/bin/env python3
"""
litellm_s08.py - Background Tasks (LiteLLM/OpenAI Format)

基于 s08_background_tasks.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Run commands in background threads. A notification queue is drained
before each LLM call to deliver results.

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s08.py -o session.md   # 输出到日志文件
"""

import json
import os
import subprocess
import threading
import uuid
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
SYSTEM_PROMPT = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


# -- BackgroundManager: threaded execution + notification queue --
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(target=self._execute, args=(task_id, command), daemon=True)
        thread.start()
        self._print_task_started(task_id, command)
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=300)
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id, "status": status, "command": command[:80],
                "result": (output or "(no output)")[:500],
            })
        self._print_task_completed(task_id, status, output)

    def check(self, task_id: str = None) -> str:
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            self._print_task_detail(task_id, t)
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
        self._print_task_list()
        lines = [f"{tid}: [{t['status']}] {t['command'][:60]}" for tid, t in self.tasks.items()]
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs

    def _print_task_started(self, task_id: str, command: str):
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "magenta"))
        print(logger._color(f"│  🚀 BACKGROUND TASK STARTED" + " " * 50 + "│", "magenta"))
        print(logger._color(f"│  Task ID: {task_id}" + " " * (69 - len(task_id)) + "│", "magenta"))
        cmd_preview = command[:65] + "..." if len(command) > 65 else command
        print(logger._color(f"│  Command: {cmd_preview}" + " " * (69 - len(cmd_preview)) + "│", "dim"))
        print(logger._color(f"│  Status: 🔄 Running..." + " " * 55 + "│", "yellow"))
        print(logger._color(f"└" + "─" * 78 + "┘", "magenta"))

    def _print_task_completed(self, task_id: str, status: str, output: str):
        status_icons = {"completed": "✅", "timeout": "⏱️", "error": "❌"}
        icon = status_icons.get(status, "❓")
        color = "green" if status == "completed" else "red"
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", color))
        print(logger._color(f"│  {icon} BACKGROUND TASK COMPLETED" + " " * (48 if status == "completed" else 49) + "│", color))
        print(logger._color(f"│  Task ID: {task_id}" + " " * (69 - len(task_id)) + "│", color))
        print(logger._color(f"│  Status: {status}" + " " * (70 - len(status)) + "│", "dim"))
        output_preview = (output[:100] + "...") if len(output) > 100 else output
        output_line = output_preview.replace("\n", " ")[:68]
        print(logger._color(f"│  Output: {output_line}" + " " * (69 - len(output_line)) + "│", "dim"))
        print(logger._color(f"└" + "─" * 78 + "┘", color))

    def _print_task_detail(self, task_id: str, task: dict):
        status_icons = {"running": "🔄", "completed": "✅", "timeout": "⏱️", "error": "❌"}
        icon = status_icons.get(task["status"], "❓")
        print(logger._color(f"\n  📋 BACKGROUND TASK #{task_id}:", "cyan"))
        print(logger._color(f"      Status: {icon} {task['status']}", "dim"))
        print(logger._color(f"      Command: {task['command'][:70]}", "dim"))
        if task.get("result"):
            print(logger._color(f"      Result: {task['result'][:200]}...", "dim"))

    def _print_task_list(self):
        if not self.tasks:
            print(logger._color(f"\n  📋 No background tasks.", "dim"))
            return
        running = sum(1 for t in self.tasks.values() if t["status"] == "running")
        completed = sum(1 for t in self.tasks.values() if t["status"] == "completed")
        error = sum(1 for t in self.tasks.values() if t["status"] in ("error", "timeout"))
        print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "cyan"))
        print(logger._color(f"║  📋 BACKGROUND TASKS" + " " * 57 + "║", "cyan"))
        print(logger._color(f"║  Total: {len(self.tasks)} | 🔄 Running: {running} | ✅ Completed: {completed} | ❌ Error: {error}" + " " * (78 - 75 - len(str([len(self.tasks), running, completed, error]))) + "║", "dim"))
        print(logger._color(f"╠" + "═" * 78 + "╣", "cyan"))
        for tid, t in self.tasks.items():
            status_icons = {"running": "🔄", "completed": "✅", "timeout": "⏱️", "error": "❌"}
            icon = status_icons.get(t["status"], "❓")
            line = f"  {icon} {tid}: {t['command'][:55]}"
            print(logger._color(f"║{line}" + " " * (78 - len(line) - 1) + "║", "dim"))
        print(logger._color(f"╚" + "═" * 78 + "╝", "cyan"))

    def print_notifications(self, notifs: list):
        if not notifs:
            return
        print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "yellow"))
        print(logger._color(f"║  📬 BACKGROUND NOTIFICATIONS ({len(notifs)} pending)" + " " * (31 - len(str(len(notifs)))) + "║", "yellow"))
        print(logger._color(f"╠" + "═" * 78 + "╣", "yellow"))
        for n in notifs:
            status_icons = {"completed": "✅", "timeout": "⏱️", "error": "❌"}
            icon = status_icons.get(n["status"], "❓")
            print(logger._color(f"║  {icon} [{n['task_id']}] {n['status']}", "yellow"))
            print(logger._color(f"║      Command: {n['command'][:65]}", "dim"))
            result_line = n["result"][:65].replace("\n", " ")
            print(logger._color(f"║      Result: {result_line}", "dim"))
        print(logger._color(f"╚" + "═" * 78 + "╝", "yellow"))

    def print_summary(self):
        running = sum(1 for t in self.tasks.values() if t["status"] == "running")
        print(logger._color(f"\n  📊 Background Task System:", "cyan"))
        print(logger._color(f"      Total tasks: {len(self.tasks)}", "dim"))
        print(logger._color(f"      Currently running: {running}", "dim"))
        print(logger._color(f"      Pending notifications: {len(self._notification_queue)}", "dim"))


BG = BackgroundManager()


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
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
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
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}

# -- OpenAI 格式的工具定义 --
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command (blocking).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}}}},
]


def agent_loop(messages: list):
    """Agent 循环"""
    iteration = 0
    while True:
        iteration += 1
        logger.loop_iteration(iteration)

        # Drain background notifications
        notifs = BG.drain_notifications()
        if notifs and messages:
            BG.print_notifications(notifs)
            notif_text = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
            messages.append({"role": "assistant", "content": "Noted background results."})
            logger.messages_snapshot(messages, "AFTER INJECT NOTIFICATIONS")

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

        logger.llm_response_summary(finish_reason, {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}, len(tool_calls))

        assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)
        logger.messages_snapshot(messages, "AFTER APPEND ASSISTANT")

        if finish_reason != "tool_calls":
            logger.loop_end(f"finish_reason = '{finish_reason}'")
            return

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
    logger.header("LiteLLM Background Tasks - Azure GPT-5.2", "litellm-s08")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()
    BG.print_summary()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            query = input("\033[36mlitellm-s08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        logger.user_input(query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        logger.separator("FINAL RESPONSE")
        for msg in reversed(history):
            if msg.get("role") == "assistant" and msg.get("content"):
                print(msg["content"])
                break
        print()
    logger.session_end("用户退出")
