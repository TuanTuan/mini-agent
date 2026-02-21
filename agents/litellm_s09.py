#!/usr/bin/env python3
"""
litellm_s09.py - Agent Teams (LiteLLM/OpenAI Format)

基于 s09_agent_teams.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Persistent named agents with file-based JSONL inboxes. Each teammate runs
its own agent loop in a separate thread. Communication via append-only inboxes.

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s09.py -o session.md   # 输出到日志文件
"""

import json
import os
import subprocess
import threading
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
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
SYSTEM_PROMPT = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        self._print_message_sent(sender, to, msg_type, content)
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = [json.loads(line) for line in inbox_path.read_text().strip().splitlines() if line]
        inbox_path.write_text("")
        if messages:
            self._print_inbox_read(name, messages)
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = sum(1 for name in teammates if name != sender and self.send(sender, name, content, "broadcast"))
        self._print_broadcast(sender, content, count)
        return f"Broadcast to {count} teammates"

    def _print_message_sent(self, sender: str, to: str, msg_type: str, content: str):
        content_preview = content[:50] + "..." if len(content) > 50 else content
        print(logger._color(f"  📨 MESSAGE SENT", "green"))
        print(logger._color(f"      From: {sender} → To: {to}", "dim"))
        print(logger._color(f"      Type: {msg_type}", "dim"))
        print(logger._color(f"      Content: {content_preview}", "dim"))

    def _print_inbox_read(self, name: str, messages: list):
        print(logger._color(f"  📬 INBOX READ: {name}", "yellow"))
        print(logger._color(f"      Messages: {len(messages)}", "dim"))
        for i, msg in enumerate(messages[:3]):
            print(logger._color(f"      [{i+1}] {msg.get('type', 'message')} from {msg.get('from', 'unknown')}: {msg.get('content', '')[:40]}", "dim"))
        if len(messages) > 3:
            print(logger._color(f"      ... ({len(messages) - 3} more)", "dim"))

    def _print_broadcast(self, sender: str, content: str, count: int):
        print(logger._color(f"  📢 BROADCAST from {sender}", "cyan"))
        print(logger._color(f"      Recipients: {count}", "dim"))
        print(logger._color(f"      Content: {content[:50]}", "dim"))

    def print_summary(self):
        inbox_files = list(self.dir.glob("*.jsonl"))
        print(logger._color(f"\n  📊 Message System Summary:", "cyan"))
        print(logger._color(f"      Inbox directory: {self.dir}", "dim"))
        print(logger._color(f"      Active inboxes: {len(inbox_files)}", "dim"))


BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: persistent named agents with config.json --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
        self._teammate_iterations = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
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
        sys_prompt = f"You are '{name}', role: {role}, at {WORKDIR}. Use send_message to communicate. Complete your task."
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        for iteration in range(50):
            self._teammate_iterations[name] = iteration + 1
            self._print_teammate_iteration(name, iteration + 1)

            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            try:
                response = litellm.completion(
                    model=MODEL, messages=messages, tools=tools,
                    api_key=AZURE_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION,
                )
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
                except json.JSONDecodeError:
                    fn_args = {}
                output = self._exec(name, fn_name, fn_args)
                self._print_teammate_tool(name, fn_name, str(output))
                results.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)})
            messages.extend(results)

        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()
            self._print_teammate_idle(name)

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        return [
            {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write content to file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
            {"type": "function", "function": {"name": "send_message", "description": "Send message to a teammate.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}}},
            {"type": "function", "function": {"name": "read_inbox", "description": "Read and drain your inbox.", "parameters": {"type": "object", "properties": {}}}},
        ]

    def list_all(self) -> str:
        self._print_team_list()
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]

    def _print_teammate_spawned(self, name: str, role: str, prompt: str):
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "green"))
        print(logger._color(f"│  🤖 TEAMMATE SPAWNED" + " " * 56 + "│", "green"))
        print(logger._color(f"│  Name: {name}" + " " * (71 - len(name)) + "│", "green"))
        print(logger._color(f"│  Role: {role}" + " " * (71 - len(role)) + "│", "dim"))
        prompt_preview = prompt[:65] + "..." if len(prompt) > 65 else prompt
        print(logger._color(f"│  Prompt: {prompt_preview}" + " " * (69 - len(prompt_preview)) + "│", "dim"))
        print(logger._color(f"│  Status: 🔄 Working" + " " * 58 + "│", "yellow"))
        print(logger._color(f"└" + "─" * 78 + "┘", "green"))

    def _print_teammate_iteration(self, name: str, iteration: int):
        print(logger._color(f"\n  🔄 TEAMMATE [{name}] ITERATION #{iteration}", "magenta"))

    def _print_teammate_tool(self, name: str, tool_name: str, output: str):
        output_preview = output[:100] + "..." if len(output) > 100 else output
        print(logger._color(f"    ⚡ [{name}] {tool_name}: {output_preview}", "dim"))

    def _print_teammate_done(self, name: str, stop_reason: str):
        print(logger._color(f"\n  🏁 TEAMMATE [{name}] DONE: {stop_reason}", "green"))

    def _print_teammate_error(self, name: str, error: str):
        print(logger._color(f"\n  ❌ TEAMMATE [{name}] ERROR: {error}", "red"))

    def _print_teammate_idle(self, name: str):
        iterations = self._teammate_iterations.get(name, 0)
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "cyan"))
        print(logger._color(f"│  💤 TEAMMATE [{name}] NOW IDLE" + " " * (47 - len(name)) + "│", "cyan"))
        print(logger._color(f"│  Total iterations: {iterations}" + " " * (57 - len(str(iterations))) + "│", "dim"))
        print(logger._color(f"└" + "─" * 78 + "┘", "cyan"))

    def _print_team_list(self):
        if not self.config["members"]:
            print(logger._color(f"\n  📋 No teammates found.", "dim"))
            return
        working = sum(1 for m in self.config["members"] if m["status"] == "working")
        idle = sum(1 for m in self.config["members"] if m["status"] == "idle")
        total = len(self.config["members"])
        print(logger._color(f"\n{'╔' + '═' * 78 + '╗'}", "cyan"))
        print(logger._color(f"║  👥 TEAM: {self.config['team_name']}" + " " * (66 - len(self.config['team_name'])) + "║", "cyan"))
        print(logger._color(f"║  Total: {total} | 🔄 Working: {working} | 💤 Idle: {idle}" + " " * (78 - 45 - len(str([total, working, idle]))) + "║", "dim"))
        print(logger._color(f"╠" + "═" * 78 + "╣", "cyan"))
        status_icons = {"working": "🔄", "idle": "💤", "shutdown": "🛑"}
        for m in self.config["members"]:
            icon = status_icons.get(m["status"], "❓")
            line = f"  {icon} {m['name']} ({m['role']}): {m['status']}"
            print(logger._color(f"║{line}" + " " * (78 - len(line) - 1) + "║", "dim"))
        print(logger._color(f"╚" + "═" * 78 + "╝", "cyan"))

    def print_summary(self):
        print(logger._color(f"\n  📊 Team System Summary:", "cyan"))
        print(logger._color(f"      Team directory: {self.dir}", "dim"))
        print(logger._color(f"      Team name: {self.config['team_name']}", "dim"))
        print(logger._color(f"      Members: {len(self.config['members'])}", "dim"))
        print(logger._color(f"      Active threads: {len([t for t in self.threads.values() if t.is_alive()])}", "dim"))


TEAM = TeammateManager(TEAM_DIR)


# -- Base tool implementations --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def _run_read(path: str, limit: int = None) -> str:
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead tool dispatch (9 tools) --
TOOL_HANDLERS = {
    "bash":            lambda **kw: _run_bash(kw["command"]),
    "read_file":       lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":      lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":       lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "spawn_teammate", "description": "Spawn a persistent teammate.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}}},
    {"type": "function", "function": {"name": "list_teammates", "description": "List all teammates.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send a message to a teammate.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}}},
    {"type": "function", "function": {"name": "read_inbox", "description": "Read and drain the lead's inbox.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "broadcast", "description": "Send a message to all teammates.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}},
]


def agent_loop(messages: list):
    """Lead Agent 循环"""
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
    logger.header("LiteLLM Agent Teams - Azure GPT-5.2", "litellm-s09")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    TEAM.print_summary()
    BUS.print_summary()
    print(logger._color(f"\n  💡 Commands: /team (list teammates), /inbox (check messages)", "dim"))

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            query = input("\033[36mlitellm-s09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
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
