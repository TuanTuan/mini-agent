# s01: The Agent Loop

> AIコーディングエージェントの秘密はすべて、モデルが「終了」と判断するまでツール結果をモデルにフィードバックし続けるwhileループにある。

## 問題

なぜ言語モデルは単体でコーディングの質問に答えられないのか。それはコーディングが「現実世界とのインタラクション」を必要とするからだ。モデルはファイルを読み、テストを実行し、エラーを確認し、反復する必要がある。一回のプロンプト-レスポンスのやり取りではこれは実現できない。

agent loopがなければ、ユーザーが自分でモデルの出力をコピーペーストして戻す必要がある。つまりユーザー自身がループの役割を果たすことになる。agent loopはこれを自動化する: モデルを呼び出し、モデルが要求したツールを実行し、結果をフィードバックし、モデルが「完了」と言うまで繰り返す。

単純なタスクを考えてみよう: 「helloと出力するPythonファイルを作成せよ」。モデルは(1)ファイルを書くことを決定し、(2)書き、(3)動作を検証する必要がある。最低でも3回のツール呼び出しが必要だ。ループがなければ、そのたびに手動の介入が必要になる。

## 解決策

```
+----------+      +-------+      +---------+
|   User   | ---> |  LLM  | ---> |  Tool   |
|  prompt  |      |       |      | execute |
+----------+      +---+---+      +----+----+
                      ^               |
                      |   tool_result |
                      +---------------+
                      (loop continues)

The loop terminates when stop_reason != "tool_use".
That single condition is the entire control flow.
```

## 仕組み

1. ユーザーがプロンプトを入力する。これが最初のメッセージになる。

```python
history.append({"role": "user", "content": query})
```

2. メッセージ配列がツール定義と共にLLMに送信される。

```python
response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS, max_tokens=8000,
)
```

3. アシスタントのレスポンスがメッセージに追加される。

```python
messages.append({"role": "assistant", "content": response.content})
```

4. stop reasonを確認する。モデルがツールを呼び出さなかった場合、ループは終了する。これが唯一の終了条件だ。

```python
if response.stop_reason != "tool_use":
    return
```

5. レスポンス中の各tool_useブロックについて、ツール(このセッションではbash)を実行し、結果を収集する。

```python
for block in response.content:
    if block.type == "tool_use":
        output = run_bash(block.input["command"])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

6. 結果がuserメッセージとして追加され、ループが続行する。

```python
messages.append({"role": "user", "content": results})
```

## 主要コード

最小限のエージェント -- パターン全体が30行未満
(`agents/s01_agent_loop.py` 66-86行目):

```python
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

## 変更点

これはセッション1 -- 出発点である。前のセッションは存在しない。

| Component     | Before     | After                          |
|---------------|------------|--------------------------------|
| Agent loop    | (none)     | `while True` + stop_reason     |
| Tools         | (none)     | `bash` (one tool)              |
| Messages      | (none)     | Accumulating list              |
| Control flow  | (none)     | `stop_reason != "tool_use"`    |

## 本番環境との対応

実際のClaude Codeでも、同じループがメインREPLの中で動いている。コアとなるループロジックは`agent.ts`にあり、同一のパターンに従う: メッセージをAPIに送り、`stop_reason`を確認し、ツールを実行し、結果を追加する。本番バージョンではエラーハンドリング、トークンカウント、ストリーミング、リトライロジックが追加されているが、根本的な構造は変わらない。すべてのClaude Codeセッションはこのwhileループである。

## 試してみる

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

試せるプロンプト例:

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`
