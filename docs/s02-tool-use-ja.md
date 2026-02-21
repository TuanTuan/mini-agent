# s02: Tools

> ディスパッチマップがツール呼び出しをハンドラ関数にルーティングする -- ループ自体はまったく変更しない。

## 問題

`bash`だけでは、エージェントはすべてをシェル経由で行う: ファイルの読み取り、書き込み、編集。これは動くが脆弱だ。`cat`の出力は予期しないタイミングで切り詰められる。`sed`による置換は特殊文字で失敗する。直接的な関数呼び出しの方がシンプルなのに、モデルはシェルパイプラインの構築にトークンを浪費する。

さらに重要なのは、bashがセキュリティ上の攻撃面であること。bashの呼び出しはシェルでできることなら何でもできてしまう。`read_file`や`write_file`のような専用ツールがあれば、モデルが危険な操作を避けることを期待するのではなく、ツールレベルでパスのサンドボックス化や危険なパターンのブロックを強制できる。

重要な洞察は、ツールを追加してもループを変更する必要がないということだ。s01のループはそのまま同一で維持される。ツール配列にエントリを追加し、ハンドラ関数を追加し、ディスパッチマップで接続するだけだ。

## 解決策

```
+----------+      +-------+      +------------------+
|   User   | ---> |  LLM  | ---> | Tool Dispatch    |
|  prompt  |      |       |      | {                |
+----------+      +---+---+      |   bash: run_bash |
                      ^          |   read: run_read |
                      |          |   write: run_wr  |
                      +----------+   edit: run_edit |
                      tool_result| }                |
                                 +------------------+

The dispatch map is a dict: {tool_name: handler_function}
One lookup replaces any if/elif chain.
```

## 仕組み

1. 各ツールのハンドラ関数を定義する。各関数はツールのinput_schemaに対応するキーワード引数を受け取り、文字列の結果を返す。

```python
def run_read(path: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]
```

2. ツール名とハンドラを結びつけるディスパッチマップを作成する。

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
}
```

3. agent loop内で、ハードコードの代わりに名前でハンドラをルックアップする。

```python
for block in response.content:
    if block.type == "tool_use":
        handler = TOOL_HANDLERS.get(block.name)
        output = handler(**block.input)
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

4. パスのサンドボックス化により、モデルがワークスペースの外に出ることを防ぐ。

```python
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
```

## 主要コード

ディスパッチパターン(`agents/s02_multi_tool.py` 93-129行目):

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
}

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
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler \
                    else f"Unknown tool: {block.name}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

## s01からの変更点

| Component      | Before (s01)       | After (s02)                |
|----------------|--------------------|----------------------------|
| Tools          | 1 (bash only)      | 4 (bash, read, write, edit)|
| Dispatch       | Hardcoded bash call | `TOOL_HANDLERS` dict       |
| Path safety    | None               | `safe_path()` sandbox      |
| Agent loop     | Unchanged          | Unchanged                  |

## 本番環境との対応

Claude Codeも同じディスパッチマップパターンを使用している。ツールはツールレジストリに登録され、メインループは名前でハンドラをルックアップする。本番のツールセットはより大きく(Bash, Read, Write, Edit, Glob, Grep, WebFetchなど)、しかし各ツールは同じ契約に従う: スキーマに合致する入力を受け取り、文字列の結果を返す。Claude Codeに新しいツールを追加するには、ハンドラ関数とスキーマエントリを追加するだけでよい -- ループは変更しない。

## 試してみる

```sh
cd learn-claude-code
python agents/s02_multi_tool.py
```

試せるプロンプト例:

1. `Read the file requirements.txt`
2. `Create a file called greet.py with a greet(name) function`
3. `Edit greet.py to add a docstring to the function`
4. `Read greet.py to verify the edit worked`
5. `Run the greet function with bash: python -c "from greet import greet; greet('World')"`
