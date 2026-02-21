#!/usr/bin/env python3
"""
litellm_s10.py - Team Protocols (LiteLLM/OpenAI Format)

基于 s10_team_protocols.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Shutdown protocol and plan approval protocol, both using the same
request_id correlation pattern.

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s10.py -o session.md   # 输出到日志文件
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
SYSTEM_PROMPT = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."

_args = parse_logger_args()
logger = create_logger_from_args(_args)

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()


# -- MessageBus --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'"
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        self._print_message_sent(sender, to, msg_type, content, extra)
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return []
        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        if msgs:
            self._print_inbox_read(name, msgs)
        return msgs

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = sum(1 for n in teammates if n != sender and self.send(sender, n, content, "broadcast"))
        self._print_broadcast(sender, content, count)
        return f"Broadcast to {count} teammates"

    def _print_message_sent(self, sender, to, msg_type, content, extra):
        print(logger._color(f"  📨 MESSAGE SENT", "green"))
        print(logger._color(f"      From: {sender} → To: {to}", "dim"))
        print(logger._color(f"      Type: {msg_type}", "dim"))
        if extra:
            print(logger._color(f"      Extra: {list(extra.keys())[:3]}", "dim"))
        print(logger._color(f"      Content: {content[:40]}", "dim"))

    def _print_inbox_read(self, name, messages):
        print(logger._color(f"  📬 INBOX READ: {name}", "yellow"))
        print(logger._color(f"      Messages: {len(messages)}", "dim"))

    def _print_broadcast(self, sender, content, count):
        print(logger._color(f"  📢 BROADCAST from {sender}", "cyan"))
        print(logger._color(f"      Recipients: {count}", "dim"))

    def print_summary(self):
        print(logger._color(f"\n  📊 Message System:", "cyan"))
        print(logger._color(f"      Inbox directory: {self.dir}", "dim"))
        print(logger._color(f"      Active inboxes: {len(list(self.dir.glob('*.jsonl')))}", "dim"))


BUS = MessageBus(INBOX_DIR)


# -- TeammateManager with shutdown + plan approval --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
        self._teammate_iterations = {}

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
        thread = threading.Thread(target=self._teammate_loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        sys_prompt = f"You are '{name}', role: {role}, at {WORKDIR}. Submit plans via plan_approval before major work. Respond to shutdown_request with shutdown_response."
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        tools = self._teammate_tools()
        should_exit = False

        for iteration in range(50):
            self._teammate_iterations[name] = iteration + 1
            self._print_teammate_iteration(name, iteration + 1)

            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            if should_exit:
                break

            try:
                response = litellm.completion(model=MODEL, messages=messages, tools=tools, api_key=AZURE_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
            except Exception as e:
                self._print_teammate_error(name, str(e))
                break

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
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                fn = tc.get("function") or {}
                fn_name = fn.get("name", "")
                fn_args_str = fn.get("arguments", "{}")
                try:
                    fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                except:
                    fn_args = {}
                output = self._exec(name, fn_name, fn_args)
                self._print_teammate_tool(name, fn_name, str(output))
                results.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)})
                if fn_name == "shutdown_response" and fn_args.get("approve"):
                    should_exit = True
                    self._print_shutdown_approved(name)
            messages.extend(results)

        member = self._find_member(name)
        if member:
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()
            self._print_teammate_status_change(name, member["status"])

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash": return _run_bash(args["command"])
        if tool_name == "read_file": return _run_read(args["path"])
        if tool_name == "write_file": return _run_write(args["path"], args["content"])
        if tool_name == "edit_file": return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message": return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox": return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response", {"request_id": req_id, "approve": approve})
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(sender, "lead", plan_text, "plan_approval_response", {"request_id": req_id, "plan": plan_text})
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."
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
        ]

    def list_all(self) -> str:
        self._print_team_list()
        if not self.config["members"]:
            return "No teammates."
        return "\n".join([f"  {m['name']} ({m['role']}): {m['status']}" for m in self.config["members"]])

    def member_names(self): return [m["name"] for m in self.config["members"]]

    def _print_teammate_spawned(self, name, role, prompt):
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "green"))
        print(logger._color(f"│  🤖 TEAMMATE SPAWNED" + " " * 56 + "│", "green"))
        print(logger._color(f"│  Name: {name}" + " " * (71 - len(name)) + "│", "green"))
        print(logger._color(f"│  Role: {role}" + " " * (71 - len(role)) + "│", "dim"))
        print(logger._color(f"│  Protocols: 🔌 shutdown, 📋 plan_approval" + " " * 33 + "│", "yellow"))
        print(logger._color(f"└" + "─" * 78 + "┘", "green"))

    def _print_teammate_iteration(self, name, iteration):
        print(logger._color(f"\n  🔄 TEAMMATE [{name}] ITERATION #{iteration}", "magenta"))

    def _print_teammate_tool(self, name, tool_name, output):
        print(logger._color(f"    ⚡ [{name}] {tool_name}: {output[:80]}", "dim"))

    def _print_teammate_done(self, name, stop_reason):
        print(logger._color(f"\n  🏁 TEAMMATE [{name}] DONE: {stop_reason}", "green"))

    def _print_teammate_error(self, name, error):
        print(logger._color(f"\n  ❌ TEAMMATE [{name}] ERROR: {error}", "red"))

    def _print_shutdown_approved(self, name):
        print(logger._color(f"\n  🔌 TEAMMATE [{name}] SHUTDOWN APPROVED - exiting...", "yellow"))

    def _print_teammate_status_change(self, name, status):
        icon = {"shutdown": "🛑", "idle": "💤", "working": "🔄"}.get(status, "❓")
        print(logger._color(f"\n  {icon} TEAMMATE [{name}] STATUS: {status.upper()}", "cyan" if status == "idle" else "red"))

    def _print_team_list(self):
        if not self.config["members"]:
            print(logger._color(f"\n  📋 No teammates found.", "dim"))
            return
        print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "cyan"))
        print(logger._color(f"║  👥 TEAM: {self.config['team_name']}" + " " * (66 - len(self.config['team_name'])) + "║", "cyan"))
        print(logger._color(f"╠" + "═" * 78 + "╣", "cyan"))
        status_icons = {"working": "🔄", "idle": "💤", "shutdown": "🛑"}
        for m in self.config["members"]:
            icon = status_icons.get(m["status"], "❓")
            line = f"  {icon} {m['name']} ({m['role']}): {m['status']}"
            print(logger._color(f"║{line}" + " " * (78 - len(line) - 1) + "║", "dim"))
        print(logger._color(f"╚" + "═" * 78 + "╝", "cyan"))

    def print_summary(self):
        print(logger._color(f"\n  📊 Team System:", "cyan"))
        print(logger._color(f"      Members: {len(self.config['members'])}", "dim"))
        print(logger._color(f"      Active threads: {len([t for t in self.threads.values() if t.is_alive()])}", "dim"))


TEAM = TeammateManager(TEAM_DIR)


# -- Base tools --
def _safe_path(p): path = (WORKDIR / p).resolve(); return path if path.is_relative_to(WORKDIR) else (_ for _ in ()).throw(ValueError(f"Path escapes: {p}"))
def _run_bash(cmd):
    if any(d in cmd for d in ["rm -rf /", "sudo", "shutdown", "reboot"]): return "Error: Dangerous command blocked"
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
    if old_text not in c: return f"Error: Text not found"; fp.write_text(c.replace(old_text, new_text, 1)); return f"Edited {path}"
    except Exception as e: return f"Error: {e}"


# -- Protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock: shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    _print_shutdown_request_initiated(req_id, teammate)
    BUS.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"

def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock: req = plan_requests.get(request_id)
    if not req: return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock: req["status"] = "approved" if approve else "rejected"
    _print_plan_review(request_id, req["from"], approve, feedback)
    BUS.send("lead", req["from"], feedback, "plan_approval_response", {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"

def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock: result = shutdown_requests.get(request_id, {"error": "not found"})
    return json.dumps(result)

def _print_shutdown_request_initiated(req_id, teammate):
    print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "yellow"))
    print(logger._color(f"│  🔌 SHUTDOWN REQUEST INITIATED" + " " * 46 + "│", "yellow"))
    print(logger._color(f"│  Request ID: {req_id}" + " " * (64 - len(req_id)) + "│", "yellow"))
    print(logger._color(f"│  Target: {teammate}" + " " * (69 - len(teammate)) + "│", "dim"))
    print(logger._color(f"└" + "─" * 78 + "┘", "yellow"))

def _print_plan_review(req_id, from_teammate, approve, feedback):
    icon = "✅" if approve else "❌"
    color = "green" if approve else "red"
    print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", color))
    print(logger._color(f"│  {icon} PLAN {'APPROVED' if approve else 'REJECTED'}" + " " * (64 - len('APPROVED' if approve else 'REJECTED')) + "│", color))
    print(logger._color(f"│  Request ID: {req_id}" + " " * (64 - len(req_id)) + "│", color))
    print(logger._color(f"│  From: {from_teammate}" + " " * (71 - len(from_teammate)) + "│", "dim"))
    print(logger._color(f"└" + "─" * 78 + "┘", color))

def _print_protocol_summary():
    with _tracker_lock:
        print(logger._color(f"\n  📊 Protocol System:", "cyan"))
        print(logger._color(f"      Shutdown requests: {len(shutdown_requests)}", "dim"))
        print(logger._color(f"      Plan requests: {len(plan_requests)}", "dim"))


# -- Lead tool dispatch --
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
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval": lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
}

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "spawn_teammate", "description": "Spawn a persistent teammate.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}}},
    {"type": "function", "function": {"name": "list_teammates", "description": "List all teammates.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send a message to a teammate.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}}},
    {"type": "function", "function": {"name": "read_inbox", "description": "Read and drain the lead's inbox.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "broadcast", "description": "Send a message to all teammates.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "shutdown_request", "description": "Request a teammate to shut down.", "parameters": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}}},
    {"type": "function", "function": {"name": "shutdown_response", "description": "Check shutdown request status.", "parameters": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}}},
    {"type": "function", "function": {"name": "plan_approval", "description": "Approve or reject a teammate's plan.", "parameters": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}}},
]


def agent_loop(messages: list):
    iteration = 0
    while True:
        iteration += 1
        logger.loop_iteration(iteration)
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"})
            messages.append({"role": "assistant", "content": "Noted inbox messages."})

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
            try: output = handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
            except Exception as e: output = f"Error: {e}"

            logger.tool_result(tc_id, str(output), is_error=str(output).startswith("Error:"))
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)})

        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("LiteLLM Team Protocols - Azure GPT-5.2", "litellm-s10")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    print()

    TEAM.print_summary()
    BUS.print_summary()
    _print_protocol_summary()
    print(logger._color(f"\n  💡 Commands: /team, /inbox", "dim"))
    print(logger._color(f"  🔌 Protocols: shutdown_request, plan_approval", "dim"))

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try: query = input("\033[36mlitellm-s10 >> \033[0m")
        except (EOFError, KeyboardInterrupt): break
        if query.strip().lower() in ("q", "exit", ""): break
        if query.strip() == "/team": print(TEAM.list_all()); continue
        if query.strip() == "/inbox": print(json.dumps(BUS.read_inbox("lead"), indent=2)); continue

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
