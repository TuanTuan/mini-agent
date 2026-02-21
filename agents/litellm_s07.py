#!/usr/bin/env python3
"""
litellm_s07.py - Tasks (LiteLLM/OpenAI Format)

基于 s07_task_system.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task has a dependency graph (blockedBy/blocks).

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s07.py -o session.md   # 输出到日志文件
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
TASKS_DIR = WORKDIR / ".tasks"
SYSTEM_PROMPT = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


# -- TaskManager: CRUD with dependency graph, persisted as JSON files --
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id, "subject": subject, "description": description,
            "status": "pending", "blockedBy": [], "blocks": [], "owner": "",
        }
        self._save(task)
        self._next_id += 1
        self._print_task_created(task)
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        task = self._load(task_id)
        self._print_task_detail(task)
        return json.dumps(task, indent=2)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(task_id)
        old_status = task.get("status", "pending")

        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                unblocked = self._clear_dependency(task_id)
                if unblocked:
                    self._print_dependency_cleared(task_id, unblocked)

        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass

        self._save(task)
        self._print_task_updated(task, old_status, add_blocked_by, add_blocks)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int) -> list:
        unblocked = []
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)
                if not task["blockedBy"] and task["status"] == "pending":
                    unblocked.append(task["id"])
        return unblocked

    def list_all(self) -> str:
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        self._print_task_list(tasks)
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)

    def _print_task_created(self, task: dict):
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "green"))
        print(logger._color(f"│  ✅ TASK CREATED: #{task['id']}" + " " * (62 - len(str(task['id']))) + "│", "green"))
        print(logger._color(f"│  Subject: {task['subject'][:65]}" + " " * (68 - min(len(task['subject']), 65)) + "│", "green"))
        if task.get("description"):
            desc = task["description"][:60]
            print(logger._color(f"│  Description: {desc}" + " " * (63 - len(desc)) + "│", "dim"))
        print(logger._color(f"│  Status: pending" + " " * 61 + "│", "dim"))
        print(logger._color(f"└" + "─" * 78 + "┘", "green"))

    def _print_task_updated(self, task: dict, old_status: str, add_blocked_by: list, add_blocks: list):
        status_icons = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        icon = status_icons.get(task["status"], "❓")
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "yellow"))
        print(logger._color(f"│  {icon} TASK UPDATED: #{task['id']}" + " " * (61 - len(str(task['id']))) + "│", "yellow"))
        print(logger._color(f"│  Subject: {task['subject'][:65]}" + " " * (68 - min(len(task['subject']), 65)) + "│", "yellow"))
        if old_status != task["status"]:
            print(logger._color(f"│  Status: {old_status} → {task['status']}" + " " * (60 - len(old_status) - len(task['status'])) + "│", "cyan"))
        if add_blocked_by:
            print(logger._color(f"│  Blocked by: {add_blocked_by}" + " " * (64 - len(str(add_blocked_by))) + "│", "red"))
        if add_blocks:
            print(logger._color(f"│  Blocks: {add_blocks}" + " " * (68 - len(str(add_blocks))) + "│", "magenta"))
        print(logger._color(f"└" + "─" * 78 + "┘", "yellow"))

    def _print_task_detail(self, task: dict):
        status_icons = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        icon = status_icons.get(task["status"], "❓")
        print(logger._color(f"\n  📋 TASK #{task['id']} DETAILS:", "cyan"))
        print(logger._color(f"      Subject: {task['subject']}", "dim"))
        print(logger._color(f"      Status: {icon} {task['status']}", "dim"))
        if task.get("description"):
            print(logger._color(f"      Description: {task['description'][:100]}", "dim"))
        if task.get("blockedBy"):
            print(logger._color(f"      Blocked by: {task['blockedBy']}", "red"))
        if task.get("blocks"):
            print(logger._color(f"      Blocks: {task['blocks']}", "magenta"))

    def _print_task_list(self, tasks: list):
        if not tasks:
            print(logger._color(f"\n  📋 No tasks found.", "dim"))
            return
        pending = sum(1 for t in tasks if t["status"] == "pending")
        in_progress = sum(1 for t in tasks if t["status"] == "in_progress")
        completed = sum(1 for t in tasks if t["status"] == "completed")
        print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "cyan"))
        print(logger._color(f"║  📋 TASK LIST" + " " * 64 + "║", "cyan"))
        print(logger._color(f"║  Total: {len(tasks)} | ⏳ Pending: {pending} | 🔄 In Progress: {in_progress} | ✅ Completed: {completed}" + " " * (78 - 73 - len(str([len(tasks), pending, in_progress, completed]))) + "║", "dim"))
        print(logger._color(f"╠" + "═" * 78 + "╣", "cyan"))
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            marker_color = {"pending": "white", "in_progress": "yellow", "completed": "green"}.get(t["status"], "white")
            blocked = f" 🔒{t['blockedBy']}" if t.get("blockedBy") else ""
            line = f"  {marker} #{t['id']}: {t['subject']}{blocked}"
            print(logger._color(f"║{line}" + " " * (78 - len(line) - 1) + "║", marker_color))
        print(logger._color(f"╚" + "═" * 78 + "╝", "cyan"))

    def _print_dependency_cleared(self, completed_id: int, unblocked: list):
        print(logger._color(f"\n  🔓 DEPENDENCY CLEARED:", "green"))
        print(logger._color(f"      Task #{completed_id} completed, unblocked tasks: {unblocked}", "dim"))

    def print_summary(self):
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        print(logger._color(f"\n  📊 Task System Summary:", "cyan"))
        print(logger._color(f"      Tasks directory: {self.dir}", "dim"))
        print(logger._color(f"      Total tasks: {len(tasks)}", "dim"))
        print(logger._color(f"      Next task ID: {self._next_id}", "dim"))


TASKS = TaskManager(TASKS_DIR)


# -- Base tool implementations --
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
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
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
    {"type": "function", "function": {"name": "task_create", "description": "Create a new task.",
     "parameters": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}}},
    {"type": "function", "function": {"name": "task_update", "description": "Update a task's status or dependencies.",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "task_list", "description": "List all tasks with status summary.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "function": {"name": "task_get", "description": "Get full details of a task by ID.",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}}},
]


def agent_loop(messages: list):
    """Agent 循环"""
    iteration = 0
    while True:
        iteration += 1
        logger.loop_iteration(iteration)
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
    logger.header("LiteLLM Task System - Azure GPT-5.2", "litellm-s07")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()
    TASKS.print_summary()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            query = input("\033[36mlitellm-s07 >> \033[0m")
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
