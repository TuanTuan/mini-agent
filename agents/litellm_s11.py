#!/usr/bin/env python3
"""
litellm_s11.py - Autonomous Agents (LiteLLM/OpenAI Format)

基于 s11_autonomous_agents.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression.

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s11.py -o session.md   # 输出到日志文件
"""

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import litellm
from dotenv import load_dotenv

from logger_openai import create_logger_from_args, parse_logger_args, get_logger_config_string

load_dotenv(override=True)

AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5.2")

WORKDIR = Path.cwd()
MODEL = f"azure/{AZURE_DEPLOYMENT}"
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60
SYSTEM_PROMPT = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

_args = parse_logger_args()
logger = create_logger_from_args(_args)

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


# -- MessageBus --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type"
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return []
        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        return msgs

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = sum(1 for n in teammates if n != sender and self.send(sender, n, content, "broadcast"))
        return f"Broadcast to {count}"

    def print_summary(self):
        print(logger._color(f"\n  📊 Message System:", "cyan"))
        print(logger._color(f"      Active inboxes: {len(list(self.dir.glob('*.jsonl')))}", "dim"))


BUS = MessageBus(INBOX_DIR)


# -- Task board --
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if task.get("status") == "pending" and not task.get("owner") and not task.get("blockedBy"):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    _print_task_claimed(task_id, owner, task.get("subject", ""))
    return f"Claimed task #{task_id} for {owner}"


def _print_task_claimed(task_id, owner, subject):
    print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "green"))
    print(logger._color(f"│  🎯 TASK CLAIMED" + " " * 60 + "│", "green"))
    print(logger._color(f"│  Task ID: #{task_id}" + " " * 63 + "│", "green"))
    print(logger._color(f"│  Owner: {owner}" + " " * 70 + "│", "dim"))
    print(logger._color(f"│  Subject: {subject[:60]}" + " " * (68 - min(len(subject), 60)) + "│", "dim"))
    print(logger._color(f"└" + "─" * 78 + "┘", "green"))


def _print_auto_claimed(task_id, owner, subject):
    print(logger._color(f"\n  🎯 AUTO-CLAIMED TASK #{task_id}", "magenta"))
    print(logger._color(f"      Owner: {owner}", "dim"))
    print(logger._color(f"      Subject: {subject[:50]}", "dim"))


def _print_task_board_summary():
    TASKS_DIR.mkdir(exist_ok=True)
    tasks = [json.loads(f.read_text()) for f in sorted(TASKS_DIR.glob("task_*.json"))]
    pending = sum(1 for t in tasks if t.get("status") == "pending")
    in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
    completed = sum(1 for t in tasks if t.get("status") == "completed")
    unclaimed = sum(1 for t in tasks if t.get("status") == "pending" and not t.get("owner"))
    print(logger._color(f"\n  📊 Task Board:", "cyan"))
    print(logger._color(f"      Total: {len(tasks)} | ⏳ Pending: {pending} (unclaimed: {unclaimed}) | 🔄 In Progress: {in_progress} | ✅ Completed: {completed}", "dim"))


def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {"role": "user", "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>"}


# -- Autonomous TeammateManager --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
        self._teammate_iterations = {}
        self._teammate_cycles = {}

    def _load_config(self):
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str):
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()
            self._print_status_change(name, status)

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        self._print_teammate_spawned(name, role, prompt)
        self._teammate_iterations[name] = 0
        self._teammate_cycles[name] = 0
        thread = threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. Use idle tool when you have no more work. You will auto-claim new tasks."
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            self._teammate_cycles[name] = self._teammate_cycles.get(name, 0) + 1
            cycle = self._teammate_cycles[name]
            self._print_work_cycle_start(name, cycle)

            # WORK PHASE
            for iteration in range(50):
                self._teammate_iterations[name] = self._teammate_iterations.get(name, 0) + 1
                self._print_teammate_iteration(name, iteration + 1, cycle)

                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        self._print_teammate_shutdown(name, "shutdown_request received")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})

                try:
                    response = litellm.completion(model=MODEL, messages=messages, tools=tools, api_key=AZURE_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
                    response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
                except Exception as e:
                    self._set_status(name, "idle")
                    self._print_teammate_error(name, str(e))
                    return

                choice = (response_dict.get("choices") or [{}])[0] or {}
                message = choice.get("message") or {}
                finish_reason = choice.get("finish_reason") or "stop"
                tool_calls = message.get("tool_calls") or []

                assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                if finish_reason != "tool_calls":
                    self._print_teammate_done(name, finish_reason)
                    break

                results = []
                idle_requested = False
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    fn = tc.get("function") or {}
                    fn_name = fn.get("name", "")
                    fn_args_str = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                    except:
                        fn_args = {}

                    if fn_name == "idle":
                        idle_requested = True
                        output = "Entering idle phase."
                        self._print_idle_requested(name)
                    else:
                        output = self._exec(name, fn_name, fn_args)
                    self._print_teammate_tool(name, fn_name, str(output))
                    results.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)})

                messages.extend(results)
                if idle_requested:
                    break

            # IDLE PHASE
            self._set_status(name, "idle")
            self._print_idle_phase_start(name)

            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for poll_num in range(polls):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            self._print_teammate_shutdown(name, "shutdown_request during idle")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    self._print_resume_reason(name, "inbox message received")
                    break

                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    claim_task(task["id"], name)
                    _print_auto_claimed(task["id"], name, task.get("subject", ""))
                    task_prompt = f"<auto-claimed>Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"
                    if len(messages) <= 3:
                        messages.insert(0, make_identity_block(name, role, team_name))
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    self._print_resume_reason(name, f"auto-claimed task #{task['id']}")
                    break

                if poll_num % 3 == 0:
                    self._print_idle_poll(name, poll_num + 1, polls)

            if not resume:
                self._set_status(name, "shutdown")
                self._print_teammate_shutdown(name, "idle timeout")
                return

            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash": return _run_bash(args["command"])
        if tool_name == "read_file": return _run_read(args["path"])
        if tool_name == "write_file": return _run_write(args["path"], args["content"])
        if tool_name == "edit_file": return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message": return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox": return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response", {"request_id": req_id, "approve": args["approve"]})
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(sender, "lead", plan_text, "plan_approval_response", {"request_id": req_id, "plan": plan_text})
            return f"Plan submitted (request_id={req_id})."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self):
        return [
            {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write content to file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
            {"type": "function", "function": {"name": "send_message", "description": "Send message to a teammate.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}}},
            {"type": "function", "function": {"name": "read_inbox", "description": "Read and drain your inbox.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "shutdown_response", "description": "Respond to a shutdown request.", "parameters": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}}},
            {"type": "function", "function": {"name": "plan_approval", "description": "Submit a plan for lead approval.", "parameters": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}}},
            {"type": "function", "function": {"name": "idle", "description": "Signal that you have no more work.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "claim_task", "description": "Claim a task from the task board by ID.", "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}}},
        ]

    def list_all(self) -> str:
        self._print_team_list()
        if not self.config["members"]:
            return "No teammates."
        return "\n".join([f"  {m['name']} ({m['role']}): {m['status']}" for m in self.config["members"]])

    def member_names(self): return [m["name"] for m in self.config["members"]]

    def _print_teammate_spawned(self, name, role, prompt):
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "green"))
        print(logger._color(f"│  🤖 AUTONOMOUS AGENT SPAWNED" + " " * 49 + "│", "green"))
        print(logger._color(f"│  Name: {name}" + " " * (71 - len(name)) + "│", "green"))
        print(logger._color(f"│  Mode: 🔄 Autonomous (auto-claim)" + " " * 43 + "│", "yellow"))
        print(logger._color(f"│  Idle timeout: {IDLE_TIMEOUT}s" + " " * (59 - len(str(IDLE_TIMEOUT))) + "│", "dim"))
        print(logger._color(f"└" + "─" * 78 + "┘", "green"))

    def _print_work_cycle_start(self, name, cycle):
        print(logger._color(f"\n  {'╔' + '═' * 76 + '╗'}", "green"))
        print(logger._color(f"  ║  🔄 WORK CYCLE #{cycle} [{name}]" + " " * (51 - len(str(cycle)) - len(name)) + "║", "green"))
        print(logger._color(f"  ╚{'═' * 76 + '╝'}", "green"))

    def _print_teammate_iteration(self, name, iteration, cycle):
        print(logger._color(f"\n  🔄 [{name}] CYCLE #{cycle} ITER #{iteration}", "magenta"))

    def _print_teammate_tool(self, name, tool_name, output):
        print(logger._color(f"    ⚡ {tool_name}: {output[:70]}", "dim"))

    def _print_teammate_done(self, name, stop_reason):
        print(logger._color(f"\n  🏁 [{name}] WORK PHASE DONE: {stop_reason}", "green"))

    def _print_teammate_error(self, name, error):
        print(logger._color(f"\n  ❌ [{name}] ERROR: {error}", "red"))

    def _print_idle_requested(self, name):
        print(logger._color(f"\n  💤 [{name}] IDLE REQUESTED", "yellow"))

    def _print_idle_phase_start(self, name):
        print(logger._color(f"\n  {'╔' + '═' * 76 + '╗'}", "cyan"))
        print(logger._color(f"  ║  💤 IDLE PHASE [{name}]" + " " * (53 - len(name)) + "║", "cyan"))
        print(logger._color(f"  ║  Polling for tasks... (timeout: {IDLE_TIMEOUT}s)" + " " * (26 - len(str(IDLE_TIMEOUT))) + "║", "dim"))
        print(logger._color(f"  ╚{'═' * 76 + '╝'}", "cyan"))

    def _print_idle_poll(self, name, poll_num, total_polls):
        print(logger._color(f"    ⏳ Poll #{poll_num}/{total_polls}", "dim"))

    def _print_resume_reason(self, name, reason):
        print(logger._color(f"\n  ▶️ [{name}] RESUMING: {reason}", "green"))

    def _print_teammate_shutdown(self, name, reason):
        print(logger._color(f"\n  {'┌' + '─' * 76 + '┐'}", "red"))
        print(logger._color(f"  │  🛑 SHUTDOWN [{name}]" + " " * (54 - len(name)) + "│", "red"))
        print(logger._color(f"  │  Reason: {reason[:60]}" + " " * (67 - min(len(reason), 60)) + "│", "dim"))
        print(logger._color(f"  └" + "─" * 76 + "┘", "red"))

    def _print_status_change(self, name, status):
        icon = {"working": "🔄", "idle": "💤", "shutdown": "🛑"}.get(status, "❓")
        print(logger._color(f"  📊 [{name}] STATUS: {icon} {status}", "cyan"))

    def _print_team_list(self):
        if not self.config["members"]:
            print(logger._color(f"\n  📋 No teammates.", "dim"))
            return
        print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "cyan"))
        print(logger._color(f"║  👥 AUTONOMOUS TEAM: {self.config['team_name']}" + " " * (55 - len(self.config['team_name'])) + "║", "cyan"))
        print(logger._color(f"╠" + "═' * 78 + '╣", "cyan"))
        status_icons = {"working": "🔄", "idle": "💤", "shutdown": "🛑"}
        for m in self.config["members"]:
            icon = status_icons.get(m["status"], "❓")
            line = f"  {icon} {m['name']} ({m['role']}): {m['status']}"
            print(logger._color(f"║{line}" + " " * (78 - len(line) - 1) + "║", "dim"))
        print(logger._color(f"╚" + "═' * 78 + '╝", "cyan"))

    def print_summary(self):
        print(logger._color(f"\n  📊 Autonomous Team:", "cyan"))
        print(logger._color(f"      Members: {len(self.config['members'])}", "dim"))
        print(logger._color(f"      Active: {len([t for t in self.threads.values() if t.is_alive()])}", "dim"))
        print(logger._color(f"      Poll: {POLL_INTERVAL}s, Timeout: {IDLE_TIMEOUT}s", "dim"))


TEAM = TeammateManager(TEAM_DIR)


# -- Base tools --
def _safe_path(p): path = (WORKDIR / p).resolve(); return path if path.is_relative_to(WORKDIR) else (_ for _ in ()).throw(ValueError(f"Path escapes: {p}"))
def _run_bash(cmd):
    if any(d in cmd for d in ["rm -rf /", "sudo", "shutdown", "reboot"]): return "Error: Dangerous"
    try: r = subprocess.run(cmd, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120); return (r.stdout + r.stderr).strip()[:50000] or "(no output)"
    except subprocess.TimeoutExpired: return "Error: Timeout"
def _run_read(path):
    try: return "\n".join(_safe_path(path).read_text().splitlines()[:1000])[:50000]
    except Exception as e: return f"Error: {e}"
def _run_write(path, content):
    try: fp = _safe_path(path); fp.parent.mkdir(parents=True, exist_ok=True); fp.write_text(content); return f"Wrote {len(content)} bytes"
    except Exception as e: return f"Error: {e}"
def _run_edit(path, old_text, new_text):
    try: fp = _safe_path(path); c = fp.read_text();
    if old_text not in c: return "Error: Not found"; fp.write_text(c.replace(old_text, new_text, 1)); return f"Edited {path}"
    except Exception as e: return f"Error: {e}"


# -- Protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock: shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent"

def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock: req = plan_requests.get(request_id)
    if not req: return f"Error: Unknown plan"
    with _tracker_lock: req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response", {"request_id": request_id, "approve": approve})
    return f"Plan {req['status']}"


# -- Lead tools --
TOOL_HANDLERS = {
    "bash": lambda **kw: _run_bash(kw["command"]),
    "read_file": lambda **kw: _run_read(kw["path"]),
    "write_file": lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox": lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast": lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: json.dumps(shutdown_requests.get(kw.get("request_id", ""), {"error": "not found"})),
    "plan_approval": lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle": lambda **kw: "Lead does not idle.",
    "claim_task": lambda **kw: claim_task(kw["task_id"], "lead"),
}

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Edit file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "spawn_teammate", "description": "Spawn autonomous teammate.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}}},
    {"type": "function", "function": {"name": "list_teammates", "description": "List teammates.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send message.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}}, "required": ["to", "content"]}}},
    {"type": "function", "function": {"name": "read_inbox", "description": "Read inbox.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "broadcast", "description": "Broadcast.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "shutdown_request", "description": "Request shutdown.", "parameters": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}}},
    {"type": "function", "function": {"name": "shutdown_response", "description": "Check shutdown status.", "parameters": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}}},
    {"type": "function", "function": {"name": "plan_approval", "description": "Approve plan.", "parameters": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}}},
    {"type": "function", "function": {"name": "idle", "description": "Idle.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "claim_task", "description": "Claim task.", "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}}},
]


def agent_loop(messages: list):
    iteration = 0
    while True:
        iteration += 1
        logger.loop_iteration(iteration)
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"})
            messages.append({"role": "assistant", "content": "Noted."})

        logger.messages_snapshot(messages, "BEFORE LLM CALL")
        logger.request_raw(model=MODEL, messages=messages, tools=TOOLS, max_tokens=8000)

        response = litellm.completion(model=MODEL, messages=messages, tools=TOOLS, api_key=AZURE_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        logger.response_raw(response_dict)

        choice = (response_dict.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = message.get("tool_calls") or []
        usage = response_dict.get("usage") or {}

        logger.llm_response_summary(finish_reason, {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}, len(tool_calls))

        assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls: assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if finish_reason != "tool_calls":
            logger.loop_end(f"finish_reason = '{finish_reason}'")
            return

        logger.section("Executing Tool Calls", "🔧")
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args_str = fn.get("arguments", "{}")
            try: fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            except: fn_args = {}

            logger.tool_call(fn_name, fn_args, tc_id)
            handler = TOOL_HANDLERS.get(fn_name)
            try: output = handler(**fn_args) if handler else f"Unknown: {fn_name}"
            except Exception as e: output = f"Error: {e}"

            logger.tool_result(tc_id, str(output), is_error=str(output).startswith("Error:"))
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)})

        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("LiteLLM Autonomous Agents - Azure GPT-5.2", "litellm-s11")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    print()

    TEAM.print_summary()
    BUS.print_summary()
    _print_task_board_summary()

    print(logger._color(f"\n  💡 Commands: /team, /inbox, /tasks", "dim"))
    print(logger._color(f"  🔄 Autonomous: Teammates auto-claim tasks", "cyan"))

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try: query = input("\033[36mlitellm-s11 >> \033[0m")
        except (EOFError, KeyboardInterrupt): break
        if query.strip().lower() in ("q", "exit", ""): break
        if query.strip() == "/team": print(TEAM.list_all()); continue
        if query.strip() == "/inbox": print(json.dumps(BUS.read_inbox("lead"), indent=2)); continue
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                m = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                o = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {m} #{t['id']}: {t['subject'][:50]}{o}")
            continue

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
