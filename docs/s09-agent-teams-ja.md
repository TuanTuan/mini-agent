# s09: Agent Teams

> JSONL形式のインボックスを持つ永続的なチームメイトが、孤立したエージェントをコミュニケーションするチームに変える -- spawn、message、broadcast、drain。

## 問題

サブエージェント(s04)は使い捨てだ: 生成し、作業し、要約を返し、消滅する。アイデンティティもなく、呼び出し間の記憶もなく、フォローアップの指示を受け取る方法もない。バックグラウンドタスク(s08)はシェルコマンドを実行するが、LLM誘導の意思決定やフィードバックの伝達はできない。

本物のチームワークには3つのものが必要だ: (1)単一のプロンプトを超えて存続する永続的なエージェント、(2)アイデンティティとライフサイクル管理、(3)エージェント間の通信チャネル。メッセージングがなければ、永続的なチームメイトでさえ聾唖だ -- 並列に作業できるが協調することはない。

解決策は、名前付きの永続的エージェントを生成するTeammateManagerと、JONSLインボックスファイルを使うMessageBusの組み合わせだ。各チームメイトは自身のagent loopをスレッドで実行し、各LLM呼び出しの前にインボックスを確認し、他のチームメイトやリーダーにメッセージを送れる。

s06からs07への橋渡しについての注記: s03のTodoManagerアイテムは圧縮(s06)と共に死ぬ。ファイルベースのタスク(s07)はディスク上に存在するため圧縮後も生き残る。チームも同じ原則の上に構築されている -- config.jsonとインボックスファイルはコンテキストウィンドウの外に永続化される。

## 解決策

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

## 仕組み

1. TeammateManagerがチームの名簿としてconfig.jsonを管理する。各メンバーは名前、役割、ステータスを持つ。

```python
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
```

2. `spawn()`がチームメイトを作成し、そのagent loopをスレッドで開始する。アイドル状態のチームメイトを再spawnすると再活性化される。

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

3. MessageBusがJSONLインボックスファイルを処理する。`send()`がJSON行を追記し、`read_inbox()`がすべての行を読み取ってファイルをドレインする。

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

4. 各チームメイトは各LLM呼び出しの前にインボックスを確認し、受信メッセージを会話コンテキストに注入する。

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

5. `broadcast()`が送信者以外の全チームメイトに同じメッセージを送信する。

```python
def broadcast(self, sender, content, teammates):
    count = 0
    for name in teammates:
        if name != sender:
            self.send(sender, name, content, "broadcast")
            count += 1
    return f"Broadcast to {count} teammates"
```

## 主要コード

TeammateManager + MessageBusのコア(`agents/s09_team_messaging.py`):

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

## s08からの変更点

| Component      | Before (s08)     | After (s09)                |
|----------------|------------------|----------------------------|
| Tools          | 6                | 9 (+spawn/send/read_inbox) |
| Agents         | Single           | Lead + N teammates         |
| Persistence    | None             | config.json + JSONL inboxes|
| Threads        | Background cmds  | Full agent loops per thread|
| Lifecycle      | Fire-and-forget  | idle -> working -> idle    |
| Communication  | None             | 5 message types + broadcast|

教育上の簡略化: この実装ではインボックスアクセスにロックファイルを使用していない。本番環境では、複数ライターからの並行追記にはファイルロッキングまたはアトミックリネームが必要になる。ここで使用している単一ライター/インボックスパターンは教育シナリオでは安全だ。

## 本番環境との対応

Claude CodeではTeammateManagerによるチームメイト管理とSendMessageツールによるJSONLインボックスファイルを通じたメッセージングを実装している。本番システムは同じ5つのメッセージタイプをサポートし、同じappend/drainパターンを使用する。メッセージは非同期で配信される -- 送信者は受信者が読むのを待ってブロックしない。各LLM呼び出し前のインボックスチェックにより、メッセージは思考の途中ではなく会話の自然な区切りで到着する。本番環境ではメッセージの重複排除、サイズ制限、適切なファイルロッキングが追加されている。

## 試してみる

```sh
cd learn-claude-code
python agents/s09_team_messaging.py
```

試せるプロンプト例:

1. `Spawn alice (coder) and bob (tester). Have alice send bob a message.`
2. `Broadcast "status update: phase 1 complete" to all teammates`
3. `Check the lead inbox for any messages`
4. `/team`と入力してステータス付きのチーム名簿を確認する
5. `/inbox`と入力してリーダーのインボックスを手動確認する
