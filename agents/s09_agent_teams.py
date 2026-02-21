#!/usr/bin/env python3
"""
s09_agent_teams.py - Agent Teams

Persistent named agents with file-based JSONL inboxes. Each teammate runs
its own agent loop in a separate thread. Communication via append-only inboxes.

    Subagent (s04):  spawn -> execute -> return summary -> destroyed
    Teammate (s09):  spawn -> work -> idle -> work -> ... -> shutdown

    .team/config.json                   .team/inbox/
    +----------------------------+      +------------------+
    | {"team_name": "default",   |      | alice.jsonl      |
    |  "members": [              |      | bob.jsonl        |
    |    {"name":"alice",        |      | lead.jsonl       |
    |     "role":"coder",        |      +------------------+
    |     "status":"idle"}       |
    |  ]}                        |      send_message("alice", "fix bug"):
    +----------------------------+        open("alice.jsonl", "a").write(msg)

                                        read_inbox("alice"):
    spawn_teammate("alice","coder",...)   msgs = [json.loads(l) for l in ...]
         |                                open("alice.jsonl", "w").close()
         v                                return msgs  # drain
    Thread: alice             Thread: bob
    +------------------+      +------------------+
    | agent_loop       |      | agent_loop       |
    | status: working  |      | status: idle     |
    | ... runs tools   |      | ... waits ...    |
    | status -> idle   |      |                  |
    +------------------+      +------------------+

    5 message types (all declared, not all handled here):
    +-------------------------+-----------------------------------+
    | message                 | Normal text message               |
    | broadcast               | Sent to all teammates             |
    | shutdown_request        | Request graceful shutdown (s10)   |
    | shutdown_response       | Approve/reject shutdown (s10)     |
    | plan_approval_response  | Approve/reject plan (s10)         |
    +-------------------------+-----------------------------------+

Key insight: "Teammates that can talk to each other."
"""

import json
import os
import subprocess
import threading
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
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        # 日志：消息发送
        self._print_message_sent(sender, to, msg_type, content)

        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")

        # 日志：收件箱读取
        if messages:
            self._print_inbox_read(name, messages)

        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1

        # 日志：广播
        self._print_broadcast(sender, content, count)

        return f"Broadcast to {count} teammates"

    # -- 日志输出方法 --
    def _print_message_sent(self, sender: str, to: str, msg_type: str, content: str):
        """打印消息发送日志"""
        content_preview = content[:50] + "..." if len(content) > 50 else content
        print(logger._color(f"  📨 MESSAGE SENT", "green"))
        print(logger._color(f"      From: {sender} → To: {to}", "dim"))
        print(logger._color(f"      Type: {msg_type}", "dim"))
        print(logger._color(f"      Content: {content_preview}", "dim"))

    def _print_inbox_read(self, name: str, messages: list):
        """打印收件箱读取日志"""
        print(logger._color(f"  📬 INBOX READ: {name}", "yellow"))
        print(logger._color(f"      Messages: {len(messages)}", "dim"))
        for i, msg in enumerate(messages[:3]):
            msg_from = msg.get("from", "unknown")
            msg_type = msg.get("type", "message")
            content_preview = msg.get("content", "")[:40]
            print(logger._color(f"      [{i+1}] {msg_type} from {msg_from}: {content_preview}", "dim"))
        if len(messages) > 3:
            print(logger._color(f"      ... ({len(messages) - 3} more)", "dim"))

    def _print_broadcast(self, sender: str, content: str, count: int):
        """打印广播日志"""
        content_preview = content[:50] + "..." if len(content) > 50 else content
        print(logger._color(f"  📢 BROADCAST from {sender}", "cyan"))
        print(logger._color(f"      Recipients: {count}", "dim"))
        print(logger._color(f"      Content: {content_preview}", "dim"))

    def print_summary(self):
        """打印消息系统状态摘要"""
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
        self._teammate_iterations = {}  # 记录每个队友的迭代次数

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

        # 日志：队友启动
        self._print_teammate_spawned(name, role, prompt)

        self._teammate_iterations[name] = 0
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        for iteration in range(50):
            self._teammate_iterations[name] = iteration + 1

            # 日志：队友迭代开始
            self._print_teammate_iteration(name, iteration + 1)

            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            try:
                response = client.messages.create(
                    model=MODEL,
                    system=sys_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                )
            except Exception as e:
                self._print_teammate_error(name, str(e))
                break

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                self._print_teammate_done(name, response.stop_reason)
                break

            # 执行工具调用
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input)
                    self._print_teammate_tool(name, block.name, str(output))
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
            messages.append({"role": "user", "content": results})

        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()
            self._print_teammate_idle(name)

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # these base tools are unchanged from s02
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
        # these base tools are unchanged from s02
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
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

    # -- 日志输出方法 --
    def _print_teammate_spawned(self, name: str, role: str, prompt: str):
        """打印队友启动日志"""
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "green"))
        print(logger._color(f"│  🤖 TEAMMATE SPAWNED" + " " * 56 + "│", "green"))
        print(logger._color(f"│  Name: {name}" + " " * (71 - len(name)) + "│", "green"))
        print(logger._color(f"│  Role: {role}" + " " * (71 - len(role)) + "│", "dim"))
        prompt_preview = prompt[:65] + "..." if len(prompt) > 65 else prompt
        print(logger._color(f"│  Prompt: {prompt_preview}" + " " * (69 - len(prompt_preview)) + "│", "dim"))
        print(logger._color(f"│  Status: 🔄 Working" + " " * 58 + "│", "yellow"))
        print(logger._color(f"└" + "─" * 78 + "┘", "green"))

    def _print_teammate_iteration(self, name: str, iteration: int):
        """打印队友迭代日志"""
        indent = "  "
        print(logger._color(f"\n{indent}🔄 TEAMMATE [{name}] ITERATION #{iteration}", "magenta"))

    def _print_teammate_tool(self, name: str, tool_name: str, output: str):
        """打印队友工具调用日志"""
        indent = "  "
        output_preview = output[:100] + "..." if len(output) > 100 else output
        print(logger._color(f"{indent}  ⚡ [{name}] {tool_name}: {output_preview}", "dim"))

    def _print_teammate_done(self, name: str, stop_reason: str):
        """打印队友完成日志"""
        print(logger._color(f"\n  🏁 TEAMMATE [{name}] DONE: {stop_reason}", "green"))

    def _print_teammate_error(self, name: str, error: str):
        """打印队友错误日志"""
        print(logger._color(f"\n  ❌ TEAMMATE [{name}] ERROR: {error}", "red"))

    def _print_teammate_idle(self, name: str):
        """打印队友空闲日志"""
        iterations = self._teammate_iterations.get(name, 0)
        print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "cyan"))
        print(logger._color(f"│  💤 TEAMMATE [{name}] NOW IDLE" + " " * (47 - len(name)) + "│", "cyan"))
        print(logger._color(f"│  Total iterations: {iterations}" + " " * (57 - len(str(iterations))) + "│", "dim"))
        print(logger._color(f"└" + "─" * 78 + "┘", "cyan"))

    def _print_team_list(self):
        """打印团队列表"""
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
        """打印团队系统状态摘要"""
        print(logger._color(f"\n  📊 Team System Summary:", "cyan"))
        print(logger._color(f"      Team directory: {self.dir}", "dim"))
        print(logger._color(f"      Team name: {self.config['team_name']}", "dim"))
        print(logger._color(f"      Members: {len(self.config['members'])}", "dim"))
        print(logger._color(f"      Active threads: {len([t for t in self.threads.values() if t.is_alive()])}", "dim"))


TEAM = TeammateManager(TEAM_DIR)


# -- Base tool implementations (these base tools are unchanged from s02) --
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
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
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

# these base tools are unchanged from s02
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate that runs in its own thread.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates with name, role, status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate's inbox.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
]


def agent_loop(messages: list):
    """Lead Agent 循环"""
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)

        # 读取 lead 的收件箱
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages.",
            })

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
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
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
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                })

        messages.append({"role": "user", "content": results})
        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("s09 Agent Teams - Interactive Mode", "s09")

    # 显示当前日志配置
    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    # 显示系统状态
    TEAM.print_summary()
    BUS.print_summary()

    print(logger._color(f"\n  💡 Commands: /team (list teammates), /inbox (check messages)", "dim"))

    history = []
    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            inbox = BUS.read_inbox("lead")
            print(json.dumps(inbox, indent=2))
            continue

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
            if hasattr(block, "text"):
                print(block.text)
        print()
