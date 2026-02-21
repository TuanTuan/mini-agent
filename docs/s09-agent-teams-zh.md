# s09: Agent Teams (智能体团队)

> 持久化的队友通过 JSONL 收件箱将孤立的智能体转变为可通信的团队 -- spawn、message、broadcast 和 drain。

## 问题

子智能体 (s04) 是一次性的: 生成、工作、返回摘要、消亡。它们没有身份, 没有跨调用的记忆, 也无法接收后续指令。后台任务 (s08) 运行 shell 命令, 但不能做 LLM 引导的决策或交流发现。

真正的团队协作需要三样东西: (1) 存活时间超过单次 prompt 的持久化智能体, (2) 身份和生命周期管理, (3) 智能体之间的通信通道。没有消息机制, 即使持久化的队友也是又聋又哑的 -- 它们可以并行工作但永远无法协调。

解决方案将 TeammateManager (用于生成持久化的命名智能体) 与使用 JSONL 收件箱文件的 MessageBus 结合。每个队友在独立线程中运行自己的 agent loop, 每次 LLM 调用前检查收件箱, 可以向任何其他队友或领导发送消息。

关于 s06 到 s07 的桥梁: s03 的 TodoManager 条目随压缩 (s06) 消亡。基于文件的任务 (s07) 因为存储在磁盘上而能存活压缩。团队建立在同样的原则上 -- config.json 和收件箱文件持久化在上下文窗口之外。

## 解决方案

```
Teammate lifecycle:
  spawn -> WORKING -> IDLE -> WORKING -> ... -> SHUTDOWN

Communication:
  .team/
    config.json           <- team roster + statuses
    inbox/
      alice.jsonl         <- append-only, drain-on-read
      bob.jsonl
      lead.jsonl

                +--------+    send("alice","bob","...")    +--------+
                | alice  | -----------------------------> |  bob   |
                | loop   |    bob.jsonl << {json_line}    |  loop  |
                +--------+                                +--------+
                     ^                                         |
                     |        BUS.read_inbox("alice")          |
                     +---- alice.jsonl -> read + drain ---------+

5 message types:
+-------------------------+------------------------------+
| message                 | Normal text between agents   |
| broadcast               | Sent to all teammates        |
| shutdown_request        | Request graceful shutdown     |
| shutdown_response       | Approve/reject shutdown      |
| plan_approval_response  | Approve/reject plan          |
+-------------------------+------------------------------+
```

## 工作原理

1. TeammateManager 通过 config.json 维护团队名册。每个成员有名称、角色和状态。

```python
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
```

2. `spawn()` 创建队友并在线程中启动其 agent loop。重新 spawn 一个 idle 状态的队友会将其重新激活。

```python
def spawn(self, name: str, role: str, prompt: str) -> str:
    member = self._find_member(name)
    if member:
        if member["status"] not in ("idle", "shutdown"):
            return f"Error: '{name}' is currently {member['status']}"
        member["status"] = "working"
    else:
        member = {"name": name, "role": role, "status": "working"}
        self.config["members"].append(member)
    self._save_config()
    thread = threading.Thread(
        target=self._teammate_loop,
        args=(name, role, prompt), daemon=True)
    self.threads[name] = thread
    thread.start()
    return f"Spawned teammate '{name}' (role: {role})"
```

3. MessageBus 处理 JSONL 收件箱文件。`send()` 追加一行 JSON; `read_inbox()` 读取所有行并清空文件。

```python
class MessageBus:
    def send(self, sender, to, content,
             msg_type="message", extra=None):
        msg = {"type": msg_type, "from": sender,
               "content": content,
               "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name):
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return "[]"
        msgs = [json.loads(l)
                for l in path.read_text().strip().splitlines()
                if l]
        path.write_text("")  # drain
        return json.dumps(msgs, indent=2)
```

4. 每个队友在每次 LLM 调用前检查收件箱, 将收到的消息注入对话上下文。

```python
def _teammate_loop(self, name, role, prompt):
    sys_prompt = f"You are '{name}', role: {role}, at {WORKDIR}."
    messages = [{"role": "user", "content": prompt}]
    for _ in range(50):
        inbox = BUS.read_inbox(name)
        if inbox != "[]":
            messages.append({"role": "user",
                "content": f"<inbox>{inbox}</inbox>"})
            messages.append({"role": "assistant",
                "content": "Noted inbox messages."})
        response = client.messages.create(
            model=MODEL, system=sys_prompt,
            messages=messages, tools=TOOLS)
        messages.append({"role": "assistant",
                         "content": response.content})
        if response.stop_reason != "tool_use":
            break
        # execute tools, append results...
    self._find_member(name)["status"] = "idle"
    self._save_config()
```

5. `broadcast()` 向除发送者外的所有队友发送相同消息。

```python
def broadcast(self, sender, content, teammates):
    count = 0
    for name in teammates:
        if name != sender:
            self.send(sender, name, content, "broadcast")
            count += 1
    return f"Broadcast to {count} teammates"
```

## 核心代码

TeammateManager + MessageBus 核心 (来自 `agents/s09_team_messaging.py`):

```python
class TeammateManager:
    def spawn(self, name, role, prompt):
        member = self._find_member(name) or {
            "name": name, "role": role, "status": "working"
        }
        member["status"] = "working"
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt), daemon=True)
        thread.start()
        return f"Spawned '{name}'"

class MessageBus:
    def send(self, sender, to, content,
             msg_type="message", extra=None):
        msg = {"type": msg_type, "from": sender,
               "content": content, "timestamp": time.time()}
        if extra: msg.update(extra)
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")

    def read_inbox(self, name):
        path = self.dir / f"{name}.jsonl"
        if not path.exists(): return "[]"
        msgs = [json.loads(l)
                for l in path.read_text().strip().splitlines()
                if l]
        path.write_text("")
        return json.dumps(msgs, indent=2)
```

## 相对 s08 的变更

| 组件           | 之前 (s08)       | 之后 (s09)                         |
|----------------|------------------|------------------------------------|
| Tools          | 6                | 9 (+spawn/send/read_inbox)         |
| 智能体数量     | 单一             | 领导 + N 个队友                    |
| 持久化         | 无               | config.json + JSONL 收件箱         |
| 线程           | 后台命令         | 每线程完整 agent loop              |
| 生命周期       | 一次性           | idle -> working -> idle            |
| 通信           | 无               | 5 种消息类型 + broadcast           |

教学简化说明: 此实现未使用文件锁来保护收件箱访问。在生产中, 多个写入者并发追加需要文件锁或原子重命名。这里使用的单写入者-per-收件箱模式在教学场景下是安全的。

## 生产环境参考

Claude Code 通过 TeammateManager 实现队友管理, 通过 SendMessage 工具和 JSONL 收件箱文件实现消息传递。生产系统支持同样的 5 种消息类型并使用同样的追加/排空模式。消息异步传递 -- 发送者不会阻塞等待接收者读取。每次 LLM 调用前检查收件箱确保消息在对话的自然间断点到达。生产版本增加了消息去重、大小限制和适当的文件锁。

## 试一试

```sh
cd learn-claude-code
python agents/s09_team_messaging.py
```

可以尝试的提示:

1. `Spawn alice (coder) and bob (tester). Have alice send bob a message.`
2. `Broadcast "status update: phase 1 complete" to all teammates`
3. `Check the lead inbox for any messages`
4. 输入 `/team` 查看带状态的团队名册
5. 输入 `/inbox` 手动检查领导的收件箱
