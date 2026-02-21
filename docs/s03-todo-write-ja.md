# s03: TodoWrite

> TodoManagerによりエージェントが自身の進捗を追跡でき、nagリマインダーの注入により更新を忘れた場合に強制的に更新させる。

## 問題

エージェントがマルチステップのタスクに取り組むとき、何を完了し何が残っているかを見失うことが多い。明示的な計画がなければ、モデルは作業を繰り返したり、ステップを飛ばしたり、脱線したりする可能性がある。ユーザーにはエージェントの内部計画が見えない。

これは見た目以上に深刻だ。長い会話ではモデルが「ドリフト」する -- コンテキストウィンドウがツール結果で埋まるにつれ、システムプロンプトの影響力が薄れていく。10ステップのリファクタリングタスクでステップ1-3を完了した後、モデルはステップ4-10の存在を忘れて即興で行動し始めるかもしれない。

解決策は構造化された状態管理だ: モデルが明示的に書き込むTodoManager。モデルは計画を作成し、作業中のアイテムをin_progressとしてマークし、完了時にcompletedとマークする。nagリマインダーは、モデルが3ラウンド以上todoを更新しなかった場合にナッジを注入する。

教育上の簡略化: nagの閾値3ラウンドは教育目的の可視化のために低く設定されている。本番のClaude Codeでは過剰なプロンプトを避けるため閾値は約10に設定されている。

## 解決策

```
+----------+      +-------+      +---------+
|   User   | ---> |  LLM  | ---> | Tools   |
|  prompt  |      |       |      | + todo  |
+----------+      +---+---+      +----+----+
                      ^               |
                      |   tool_result |
                      +---------------+
                            |
                +-----------+-----------+
                | TodoManager state     |
                | [ ] task A            |
                | [>] task B  <- doing  |
                | [x] task C            |
                +-----------------------+
                            |
                if rounds_since_todo >= 3:
                  inject <reminder> into tool_result
```

## 仕組み

1. TodoManagerはアイテムのリストをバリデーションして保持する。`in_progress`にできるのは一度に1つだけ。

```python
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated = []
        in_progress_count = 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({
                "id": item["id"],
                "text": item["text"],
                "status": status,
            })
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = validated
        return self.render()
```

2. `todo`ツールは他のツールと同様にディスパッチマップに追加される。

```python
TOOL_HANDLERS = {
    "bash":  lambda **kw: run_bash(kw["command"]),
    # ...other tools...
    "todo":  lambda **kw: TODO.update(kw["items"]),
}
```

3. nagリマインダーは、モデルが3ラウンド以上`todo`を呼び出さなかった場合にtool_resultメッセージに`<reminder>`タグを注入する。

```python
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        if rounds_since_todo >= 3 and messages:
            last = messages[-1]
            if (last["role"] == "user"
                    and isinstance(last.get("content"), list)):
                last["content"].insert(0, {
                    "type": "text",
                    "text": "<reminder>Update your todos.</reminder>",
                })
        # ... rest of loop ...
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
```

4. システムプロンプトがモデルにtodoによる計画を指示する。

```python
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks.
Mark in_progress before starting, completed when done.
Prefer tools over prose."""
```

## 主要コード

TodoManagerとnag注入(`agents/s03_structured_planning.py` 51-85行目および158-187行目):

```python
class TodoManager:
    def update(self, items: list) -> str:
        validated = []
        in_progress_count = 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({
                "id": item["id"],
                "text": item["text"],
                "status": status,
            })
        if in_progress_count > 1:
            raise ValueError("Only one in_progress")
        self.items = validated
        return self.render()

# In agent_loop:
if rounds_since_todo >= 3:
    last["content"].insert(0, {
        "type": "text",
        "text": "<reminder>Update your todos.</reminder>",
    })
```

## s02からの変更点

| Component      | Before (s02)     | After (s03)              |
|----------------|------------------|--------------------------|
| Tools          | 4                | 5 (+todo)                |
| Planning       | None             | TodoManager with statuses|
| Nag injection  | None             | `<reminder>` after 3 rounds|
| Agent loop     | Simple dispatch  | + rounds_since_todo counter|

## 本番環境との対応

Claude Codeも同じパターンをTodoWriteツールとTodoReadツールで実装している。モデルはステータス(pending, in_progress, completed)付きの構造化されたtodoリストを作成する。nagリマインダーの概念は「system reminders」として現れる -- モデルがドリフトしたときに軌道修正するために注入されるメッセージだ。本番のTodoManagerは優先度やセッション間の永続化をサポートしている。「一度にin_progressは1つだけ」という制約は逐次的な集中を強制し、モデルが頻繁にコンテキストスイッチするのを防ぐ。

## 試してみる

```sh
cd learn-claude-code
python agents/s03_structured_planning.py
```

試せるプロンプト例:

1. `Refactor the file hello.py: add type hints, docstrings, and a main guard`
2. `Create a Python package with __init__.py, utils.py, and tests/test_utils.py`
3. `Review all Python files and fix any style issues`
