# s06: Compact

> 3層の圧縮パイプラインにより、古いツール結果の戦略的な忘却、トークンが閾値を超えた時の自動要約、オンデマンドの手動圧縮を組み合わせて、エージェントを無期限に動作可能にする。

## 問題

コンテキストウィンドウは有限だ。十分なツール呼び出しの後、メッセージ配列がモデルのコンテキスト上限を超え、API呼び出しが失敗する。ハード制限に達する前でも、パフォーマンスは劣化する: モデルは遅くなり、精度が落ち、以前のメッセージを無視し始める。

200,000トークンのコンテキストウィンドウは大きく聞こえるが、1000行のソースファイルに対する一回の`read_file`で約4000トークンを消費する。30ファイルを読み20回のbashコマンドを実行すると、100,000トークン以上になる。何らかの圧縮がなければ、エージェントは大規模なコードベースで作業できない。

3層のパイプラインは積極性を段階的に上げて対処する:
第1層(micro-compact)は毎ターン静かに古いツール結果を置換する。
第2層(auto-compact)はトークンが閾値を超えた時に完全な要約を発動する。
第3層(manual compact)はモデル自身が圧縮をトリガーできる。

教育上の簡略化: ここでのトークン推定は大まかな「文字数/4」ヒューリスティックを使用している。本番システムでは正確なカウントのために適切なトークナイザーライブラリを使用する。

## 解決策

```
Every turn:
+------------------+
| Tool call result |
+------------------+
        |
        v
[Layer 1: micro_compact]        (silent, every turn)
  Replace tool_result > 3 turns old
  with "[Previous: used {tool_name}]"
        |
        v
[Check: tokens > 50000?]
   |               |
   no              yes
   |               |
   v               v
continue    [Layer 2: auto_compact]
              Save transcript to .transcripts/
              LLM summarizes conversation.
              Replace all messages with [summary].
                    |
                    v
            [Layer 3: compact tool]
              Model calls compact explicitly.
              Same summarization as auto_compact.
```

## 仕組み

1. **第1層 -- micro_compact**: 各LLM呼び出しの前に、直近3件以前のすべてのtool_resultエントリを見つけて内容を置換する。

```python
def micro_compact(messages: list) -> list:
    tool_results = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for j, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((i, j, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, part in to_clear:
        if len(part.get("content", "")) > 100:
            tool_id = part.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            part["content"] = f"[Previous: used {tool_name}]"
    return messages
```

2. **第2層 -- auto_compact**: 推定トークン数が50,000を超えた時、完全なトランスクリプトを保存し、LLMに要約を依頼する。

```python
def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity..."
            + json.dumps(messages, default=str)[:80000]}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    return [
        {"role": "user", "content": f"[Compressed]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. Continuing."},
    ]
```

3. **第3層 -- manual compact**: `compact`ツールが同じ要約処理をオンデマンドでトリガーする。

```python
if manual_compact:
    messages[:] = auto_compact(messages)
```

4. agent loopが3つの層すべてを統合する。

```python
def agent_loop(messages: list):
    while True:
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)
        response = client.messages.create(...)
        # ... tool execution ...
        if manual_compact:
            messages[:] = auto_compact(messages)
```

## 主要コード

3層パイプライン(`agents/s06_compression.py` 67-93行目および189-223行目):

```python
THRESHOLD = 50000
KEEP_RECENT = 3

def micro_compact(messages):
    # Replace old tool results with placeholders
    ...

def auto_compact(messages):
    # Save transcript, LLM summarize, replace messages
    ...

def agent_loop(messages):
    while True:
        micro_compact(messages)          # Layer 1
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)  # Layer 2
        response = client.messages.create(...)
        # ...
        if manual_compact:
            messages[:] = auto_compact(messages)  # Layer 3
```

## s05からの変更点

| Component      | Before (s05)     | After (s06)                |
|----------------|------------------|----------------------------|
| Tools          | 5                | 5 (base + compact)         |
| Context mgmt   | None             | Three-layer compression    |
| Micro-compact  | None             | Old results -> placeholders|
| Auto-compact   | None             | Token threshold trigger    |
| Manual compact | None             | `compact` tool             |
| Transcripts    | None             | Saved to .transcripts/     |
| Skills         | load_skill       | Removed (different focus)  |

## 本番環境との対応

Claude Codeは3つの層すべてを実装している。micro-compactionは自動的に行われ、古いツール結果は切り詰められるか要約で置換される。auto-compactionはコンテキストがモデルの上限に近づくとトリガーされ、会話を要約してリセットする。ユーザーは`/compact`で手動圧縮をトリガーすることもできる。トランスクリプトは保存されるため、完全な履歴は決して真に失われず、アクティブなコンテキストの外に移動されるだけだ。本番バージョンでは圧縮されるべきでない「ピン留め」メッセージ(CLAUDE.mdの指示など)も保持される。

## 試してみる

```sh
cd learn-claude-code
python agents/s06_compression.py
```

試せるプロンプト例:

1. `Read every Python file in the agents/ directory one by one`
   (micro-compactが古い結果を置換するのを観察する)
2. `Keep reading files until compression triggers automatically`
3. `Use the compact tool to manually compress the conversation`
