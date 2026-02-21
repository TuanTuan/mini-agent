# s11: Autonomous Agents

> タスクボードポーリング付きのアイドルサイクルにより、チームメイトが自分で作業を見つけて確保できるようになり、コンテキスト圧縮後にはアイデンティティの再注入が行われる。

## 問題

s09-s10では、チームメイトは明示的に指示された時のみ作業する。リーダーは各チームメイトを特定のプロンプトでspawnしなければならない。タスクボードに未割り当てのタスクが10個あっても、リーダーが手動で各タスクを割り当てなければならない。これはスケールしない。

真の自律性とは、チームメイトが自分で作業を見つけることだ。チームメイトが現在のタスクを完了したら、タスクボードで未確保の作業をスキャンし、タスクを確保し、作業を開始すべきだ -- リーダーからの指示なしに。

しかし自律エージェントには微妙な問題がある: コンテキスト圧縮後に、エージェントが自分が誰かを忘れる可能性がある。メッセージが要約されると、元のシステムプロンプトのアイデンティティ(「あなたはalice、役割はcoder」)が薄れる。アイデンティティの再注入は、圧縮されたコンテキストの先頭にアイデンティティブロックを挿入することでこれを解決する。

教育上の簡略化: ここで使用するトークン推定は大まかなもの(文字数 / 4)だ。本番システムでは適切なトークナイザーライブラリを使用する。nagの閾値3ラウンド(s03から)は教育目的の可視化のために低く設定されている。本番のClaude Codeでは閾値は約10。

## 解決策

```
Teammate lifecycle with idle cycle:

+-------+
| spawn |
+---+---+
    |
    v
+-------+   tool_use     +-------+
| WORK  | <------------- |  LLM  |
+---+---+                +-------+
    |
    | stop_reason != tool_use
    | (or idle tool called)
    v
+--------+
|  IDLE  |  poll every 5s for up to 60s
+---+----+
    |
    +---> check inbox --> message? ----------> WORK
    |
    +---> scan .tasks/ --> unclaimed? -------> claim -> WORK
    |
    +---> 60s timeout ----------------------> SHUTDOWN

Identity re-injection after compression:
  if len(messages) <= 3:
    messages.insert(0, identity_block)
    "You are 'alice', role: coder, team: my-team"
```

## 仕組み

1. チームメイトのループにはWORKとIDLEの2つのフェーズがある。WORKは標準的なagent loopを実行する。LLMがツール呼び出しを停止した時(または`idle`ツールを呼び出した時)、チームメイトはIDLEフェーズに入る。

```python
def _loop(self, name, role, prompt):
    while True:
        # -- WORK PHASE --
        messages = [{"role": "user", "content": prompt}]
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    self._set_status(name, "shutdown")
                    return
                messages.append(...)
            response = client.messages.create(...)
            if response.stop_reason != "tool_use":
                break
            # execute tools...
            if idle_requested:
                break

        # -- IDLE PHASE --
        self._set_status(name, "idle")
        resume = self._idle_poll(name, messages)
        if not resume:
            self._set_status(name, "shutdown")
            return
        self._set_status(name, "working")
```

2. IDLEフェーズがインボックスとタスクボードをループでポーリングする。

```python
def _idle_poll(self, name, messages):
    polls = IDLE_TIMEOUT // POLL_INTERVAL  # 60s / 5s = 12
    for _ in range(polls):
        time.sleep(POLL_INTERVAL)
        # Check inbox for new messages
        inbox = BUS.read_inbox(name)
        if inbox:
            messages.append({"role": "user",
                "content": f"<inbox>{inbox}</inbox>"})
            return True
        # Scan task board for unclaimed tasks
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task = unclaimed[0]
            claim_task(task["id"], name)
            messages.append({"role": "user",
                "content": f"<auto-claimed>Task #{task['id']}: "
                           f"{task['subject']}</auto-claimed>"})
            return True
    return False  # timeout -> shutdown
```

3. タスクボードスキャンがpendingかつ未割り当てかつブロックされていないタスクを探す。

```python
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed

def claim_task(task_id: int, owner: str):
    path = TASKS_DIR / f"task_{task_id}.json"
    task = json.loads(path.read_text())
    task["status"] = "in_progress"
    task["owner"] = owner
    path.write_text(json.dumps(task, indent=2))
```

4. アイデンティティの再注入は、コンテキストが短すぎる場合(圧縮が発生したことを示す)にアイデンティティブロックを挿入する。

```python
def make_identity_block(name, role, team_name):
    return {"role": "user",
            "content": f"<identity>You are '{name}', "
                       f"role: {role}, team: {team_name}. "
                       f"Continue your work.</identity>"}

# Before resuming work after idle:
if len(messages) <= 3:
    messages.insert(0, make_identity_block(
        name, role, team_name))
    messages.insert(1, {"role": "assistant",
        "content": f"I am {name}. Continuing."})
```

5. `idle`ツールにより、チームメイトはもう作業がないことを明示的にシグナルし、早期にアイドルポーリングフェーズに入る。

```python
{"name": "idle",
 "description": "Signal that you have no more work. "
                "Enters idle polling phase.",
 "input_schema": {"type": "object", "properties": {}}},
```

## 主要コード

自律ループ(`agents/s11_autonomous.py`):

```python
def _loop(self, name, role, prompt):
    while True:
        # WORK PHASE
        for _ in range(50):
            response = client.messages.create(...)
            if response.stop_reason != "tool_use":
                break
            for block in response.content:
                if block.name == "idle":
                    idle_requested = True
            if idle_requested:
                break

        # IDLE PHASE
        self._set_status(name, "idle")
        for _ in range(IDLE_TIMEOUT // POLL_INTERVAL):
            time.sleep(POLL_INTERVAL)
            inbox = BUS.read_inbox(name)
            if inbox: resume = True; break
            unclaimed = scan_unclaimed_tasks()
            if unclaimed:
                claim_task(unclaimed[0]["id"], name)
                resume = True; break
        if not resume:
            self._set_status(name, "shutdown")
            return
        self._set_status(name, "working")
```

## s10からの変更点

| Component      | Before (s10)     | After (s11)                |
|----------------|------------------|----------------------------|
| Tools          | 12               | 14 (+idle, +claim_task)    |
| Autonomy       | Lead-directed    | Self-organizing            |
| Idle phase     | None             | Poll inbox + task board    |
| Task claiming  | Manual only      | Auto-claim unclaimed tasks |
| Identity       | System prompt    | + re-injection after compress|
| Timeout        | None             | 60s idle -> auto shutdown  |

## 本番環境との対応

Claude Codeは同じidle/poll/claimサイクルで自律的なチームメイトを実装している。本番バージョンではステータス変更にTaskUpdateツールを使用し、アイドル通知システムを備えている。チームメイトはタスクボードから自動的に作業を発見し、リーダーの介入なしに確保する。アイデンティティの再注入により、圧縮後もチームメイトは自身の役割とコンテキストを維持する。本番のアイドルタイムアウトはチームメイトごとに設定可能だ。重要な洞察: リーダーがタスクボードを作成し、チームメイトが自己組織化してそれを完了する。

## 試してみる

```sh
cd learn-claude-code
python agents/s11_autonomous.py
```

試せるプロンプト例:

1. `Create 3 tasks on the board, then spawn alice and bob. Watch them auto-claim.`
2. `Spawn a coder teammate and let it find work from the task board itself`
3. `Create tasks with dependencies. Watch teammates respect the blocked order.`
4. `/tasks`と入力してオーナー付きのタスクボードを確認する
5. `/team`と入力して誰が作業中でアイドルかを監視する
