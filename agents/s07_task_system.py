#!/usr/bin/env python3
"""
s07_task_system.py - Tasks

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task has a dependency graph (blockedBy/blocks).

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], "blocks":[], ...}

    Dependency resolution:
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- completing task 1 removes it from task 2's blockedBy

Key insight: "State that survives compression -- because it's outside the conversation."
"""

import json
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
MODEL = os.environ["MODEL_ID"]
TASKS_DIR = WORKDIR / ".tasks"

SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."

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

        # 日志：显示任务创建
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
            # When a task is completed, remove it from all other tasks' blockedBy
            if status == "completed":
                unblocked = self._clear_dependency(task_id)
                if unblocked:
                    self._print_dependency_cleared(task_id, unblocked)

        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            # Bidirectional: also update the blocked tasks' blockedBy lists
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass

        self._save(task)

        # 日志：显示任务更新
        self._print_task_updated(task, old_status, add_blocked_by, add_blocks)

        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int) -> list:
        """Remove completed_id from all other tasks' blockedBy lists. Returns unblocked task IDs."""
        unblocked = []
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)
                # 如果这个任务不再被阻塞，记录下来
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

    # -- 日志输出方法 --
    def _print_task_created(self, task: dict):
        """打印任务创建日志"""
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "green"))
        print(logger._color(f"│  ✅ TASK CREATED: #{task['id']}" + " " * (62 - len(str(task['id']))) + "│", "green"))
        print(logger._color(f"│  Subject: {task['subject'][:65]}" + " " * (68 - min(len(task['subject']), 65)) + "│", "green"))
        if task.get("description"):
            desc = task["description"][:60]
            print(logger._color(f"│  Description: {desc}" + " " * (63 - len(desc)) + "│", "dim"))
        print(logger._color(f"│  Status: pending" + " " * 61 + "│", "dim"))
        print(logger._color(f"└" + "─" * 78 + "┘", "green"))

    def _print_task_updated(self, task: dict, old_status: str, add_blocked_by: list, add_blocks: list):
        """打印任务更新日志"""
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
        """打印任务详情"""
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
        """打印任务列表"""
        if not tasks:
            print(logger._color(f"\n  📋 No tasks found.", "dim"))
            return

        # 统计
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
        """打印依赖解除日志"""
        print(logger._color(f"\n  🔓 DEPENDENCY CLEARED:", "green"))
        print(logger._color(f"      Task #{completed_id} completed, unblocked tasks: {unblocked}", "dim"))

    def print_summary(self):
        """打印任务系统状态摘要"""
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

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


def agent_loop(messages: list):
    """Agent 循环"""
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)
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
        for block in response.content:
            if block.type == "tool_use":
                input_data = dict(block.input)
                logger.tool_call(block.name, input_data, block.id)

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
        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("s07 Task System - Interactive Mode", "s07")

    # 显示当前日志配置
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    # 显示任务系统状态
    TASKS.print_summary()

    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
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
