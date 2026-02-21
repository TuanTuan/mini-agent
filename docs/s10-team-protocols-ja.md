# s10: Team Protocols

> 同じrequest_idハンドシェイクパターンがシャットダウンとプラン承認の両方を支える -- 1つのFSM、2つの適用。

## 問題

s09ではチームメイトが作業しコミュニケーションするが、構造化された協調はない。2つの問題が生じる:

**シャットダウン**: チームメイトをどうやってクリーンに停止するか。スレッドを強制終了するとファイルが中途半端に書かれ、config.jsonが不正な状態になる。グレースフルシャットダウンにはハンドシェイクが必要だ: リーダーが要求し、チームメイトが承認(終了処理を行い退出)するか拒否(作業を継続)するかを判断する。

**プラン承認**: 実行をどうやってゲーティングするか。リーダーが「認証モジュールをリファクタリングして」と言うと、チームメイトは即座に開始する。リスクの高い変更では、実行開始前にリーダーが計画をレビューすべきだ。ジュニアが提案し、シニアが承認する。

両方の問題は同じ構造を共有している: 一方がユニークなIDを持つリクエストを送り、もう一方がそのIDを参照してレスポンスする。有限状態機械が各リクエストをpending -> approved | rejectedの遷移で追跡する。

## 解決策

```
Shutdown Protocol            Plan Approval Protocol
==================           ======================

Lead             Teammate    Teammate           Lead
  |                 |           |                 |
  |--shutdown_req-->|           |--plan_req------>|
  | {req_id:"abc"}  |           | {req_id:"xyz"}  |
  |                 |           |                 |
  |<--shutdown_resp-|           |<--plan_resp-----|
  | {req_id:"abc",  |           | {req_id:"xyz",  |
  |  approve:true}  |           |  approve:true}  |
  |                 |           |                 |
  v                 v           v                 v
tracker["abc"]     exits     proceeds          tracker["xyz"]
 = approved                                     = approved

Shared FSM (identical for both protocols):
  [pending] --approve--> [approved]
  [pending] --reject---> [rejected]

Trackers:
  shutdown_requests = {req_id: {target, status}}
  plan_requests     = {req_id: {from, plan, status}}
```

## 仕組み

1. リーダーがrequest_idを生成し、インボックス経由でshutdown_requestを送信してシャットダウンを開始する。

```python
shutdown_requests = {}

def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    shutdown_requests[req_id] = {
        "target": teammate, "status": "pending",
    }
    BUS.send("lead", teammate, "Please shut down gracefully.",
             "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent (status: pending)"
```

2. チームメイトはインボックスでリクエストを受信し、`shutdown_response`ツールを呼び出して承認または拒否する。

```python
if tool_name == "shutdown_response":
    req_id = args["request_id"]
    approve = args["approve"]
    if req_id in shutdown_requests:
        shutdown_requests[req_id]["status"] = \
            "approved" if approve else "rejected"
    BUS.send(sender, "lead", args.get("reason", ""),
             "shutdown_response",
             {"request_id": req_id, "approve": approve})
    return f"Shutdown {'approved' if approve else 'rejected'}"
```

3. チームメイトのループが承認済みシャットダウンを確認して終了する。

```python
if (block.name == "shutdown_response"
        and block.input.get("approve")):
    should_exit = True
# ...
member["status"] = "shutdown" if should_exit else "idle"
```

4. プラン承認も同一のパターンに従う。チームメイトがプランを提出し、request_idを生成する。

```python
plan_requests = {}

if tool_name == "plan_approval":
    plan_text = args.get("plan", "")
    req_id = str(uuid.uuid4())[:8]
    plan_requests[req_id] = {
        "from": sender, "plan": plan_text,
        "status": "pending",
    }
    BUS.send(sender, "lead", plan_text,
             "plan_approval_request",
             {"request_id": req_id, "plan": plan_text})
    return f"Plan submitted (request_id={req_id})"
```

5. リーダーがレビューし、同じrequest_idでレスポンスする。

```python
def handle_plan_review(request_id, approve, feedback=""):
    req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown request_id '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback,
             "plan_approval_response",
             {"request_id": request_id,
              "approve": approve,
              "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"
```

6. 両プロトコルとも同じ`plan_approval`ツール名を2つのモードで使用する: チームメイトが提出(request_idなし)、リーダーがレビュー(request_idあり)。

```python
# Lead tool dispatch:
"plan_approval": lambda **kw: handle_plan_review(
    kw["request_id"], kw["approve"],
    kw.get("feedback", "")),
# Teammate: submit mode (generate request_id)
```

## 主要コード

2つのプロトコルハンドラ(`agents/s10_team_protocols.py`):

```python
shutdown_requests = {}
plan_requests = {}

# -- Shutdown --
def handle_shutdown_request(teammate):
    req_id = str(uuid.uuid4())[:8]
    shutdown_requests[req_id] = {
        "target": teammate, "status": "pending"
    }
    BUS.send("lead", teammate,
             "Please shut down gracefully.",
             "shutdown_request",
             {"request_id": req_id})

# -- Plan Approval --
def handle_plan_review(request_id, approve, feedback=""):
    req = plan_requests[request_id]
    req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback,
             "plan_approval_response",
             {"request_id": request_id,
              "approve": approve})

# Both use the same FSM:
# pending -> approved | rejected
# Both correlate by request_id across async inboxes
```

## s09からの変更点

| Component      | Before (s09)     | After (s10)                  |
|----------------|------------------|------------------------------|
| Tools          | 9                | 12 (+shutdown_req/resp +plan)|
| Shutdown       | Natural exit only| Request-response handshake   |
| Plan gating    | None             | Submit/review with approval  |
| Request tracking| None            | Two tracker dicts            |
| Correlation    | None             | request_id per request       |
| FSM            | None             | pending -> approved/rejected |

## 本番環境との対応

Claude CodeではSendMessageツールの`type: "shutdown_request"`によるシャットダウンとExitPlanModeによるプラン承認を実装している。チームメイトが`plan_mode_required`を設定されている場合、実行前にプランを提出しなければならない。本番バージョンではシャットダウンのタイムアウト処理(設定可能なウィンドウ後の強制終了)とプラン修正(拒否されたプランは修正して再提出可能)が追加されている。request_idパターンは任意の非同期承認ワークフローに一般化できる -- 何が承認されるかに関わらず同じFSMが適用される。

## 試してみる

```sh
cd learn-claude-code
python agents/s10_team_protocols.py
```

試せるプロンプト例:

1. `Spawn alice as a coder. Then request her shutdown.`
2. `List teammates to see alice's status after shutdown approval`
3. `Spawn bob with a risky refactoring task. Review and reject his plan.`
4. `Spawn charlie, have him submit a plan, then approve it.`
5. `/team`と入力してステータスを監視する
