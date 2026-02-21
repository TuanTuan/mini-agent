# s04: Subagents

> サブエージェントは新しいメッセージリストで実行され、親とファイルシステムを共有し、要約のみを返す -- 親のコンテキストをクリーンに保つ。

## 問題

エージェントが作業するにつれ、メッセージ配列は膨張する。すべてのツール呼び出し、ファイル読み取り、bash出力が蓄積されていく。20-30回のツール呼び出しの後、コンテキストウィンドウは無関係な履歴で溢れる。ちょっとした質問に答えるために500行のファイルを読むと、永久に500行がコンテキストに追加される。

これは探索的タスクで特に深刻だ。「このプロジェクトはどのテストフレームワークを使っているか」という質問には5つのファイルを読む必要があるかもしれないが、親エージェントには5つのファイルの内容すべては不要だ -- 「pytest with conftest.py configuration」という回答だけが必要なのだ。

解決策はプロセスの分離だ: `messages=[]`で子エージェントを生成する。子は探索し、ファイルを読み、コマンドを実行する。終了時には最終的なテキストレスポンスだけが親に返される。子のメッセージ履歴全体は破棄される。

## 解決策

```
Parent agent                     Subagent
+------------------+             +------------------+
| messages=[...]   |             | messages=[]      | <-- fresh
|                  |  dispatch   |                  |
| tool: task       | ---------->| while tool_use:  |
|   prompt="..."   |            |   call tools     |
|                  |  summary   |   append results |
|   result = "..." | <--------- | return last text |
+------------------+             +------------------+
          |
Parent context stays clean.
Subagent context is discarded.
```

## 仕組み

1. 親エージェントにサブエージェント生成をトリガーする`task`ツールが追加される。子は`task`を除くすべての基本ツールを取得する(再帰的な生成は不可)。

```python
PARENT_TOOLS = CHILD_TOOLS + [
    {"name": "task",
     "description": "Spawn a subagent with fresh context.",
     "input_schema": {
         "type": "object",
         "properties": {
             "prompt": {"type": "string"},
             "description": {"type": "string"},
         },
         "required": ["prompt"],
     }},
]
```

2. サブエージェントは委譲されたプロンプトのみを含む新しいメッセージリストで開始する。ファイルシステムは共有される。

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({
            "role": "assistant", "content": response.content
        })
        if response.stop_reason != "tool_use":
            break
        # execute tools, append results...
```

3. 最終テキストのみが親に返される。子の30回以上のツール呼び出し履歴は破棄される。

```python
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"
```

4. 親はこの要約を通常のtool_resultとして受け取る。

```python
if block.name == "task":
    output = run_subagent(block.input["prompt"])
results.append({
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": str(output),
})
```

## 主要コード

サブエージェント関数(`agents/s04_context_isolation.py` 110-128行目):

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({"role": "assistant",
                             "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input)
                results.append({"type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"
```

## s03からの変更点

| Component      | Before (s03)     | After (s04)               |
|----------------|------------------|---------------------------|
| Tools          | 5                | 5 (base) + task (parent)  |
| Context        | Single shared    | Parent + child isolation  |
| Subagent       | None             | `run_subagent()` function |
| Return value   | N/A              | Summary text only         |
| Todo system    | TodoManager      | Removed (not needed here) |

## 本番環境との対応

Claude Codeではこれを`spawn()`付きのTaskツールとして実装している。モデルがTaskを呼び出すと、`messages=[]`で子プロセスが作成される。子はファイルシステムとツール定義を引き継ぐが、独自のコンテキストウィンドウを持つ。Claude Codeではサブエージェントの説明文がサポートされており、親のUIに表示されるため、親コンテキストを汚すことなく子が何をしているか確認できる。本番バージョンではサブエージェントの深さ制限(TaskからTaskの再帰は不可)と最大反復回数の制限も行っている。

## 試してみる

```sh
cd learn-claude-code
python agents/s04_context_isolation.py
```

試せるプロンプト例:

1. `Use a subtask to find what testing framework this project uses`
2. `Delegate: read all .py files and summarize what each one does`
3. `Use a task to create a new module, then verify it from here`
