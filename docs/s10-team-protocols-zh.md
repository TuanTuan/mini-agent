# s10: Team Protocols (团队协议)

> 同一个 request_id 握手模式驱动了关机和计划审批两种协议 -- 一个 FSM, 两种应用。

## 问题

在 s09 中, 队友可以工作和通信, 但没有结构化的协调。出现了两个问题:

**关机**: 如何干净地停止一个队友? 直接杀线程会留下写了一半的文件和错误状态的 config.json。优雅关机需要握手: 领导发起请求, 队友决定是批准 (完成并退出) 还是拒绝 (继续工作)。

**计划审批**: 如何控制执行门槛? 当领导说 "重构认证模块", 队友会立即开始。对于高风险变更, 领导应该在执行开始前审查计划。初级提出方案, 高级批准。

两个问题共享相同的结构: 一方发送带唯一 ID 的请求, 另一方引用该 ID 作出响应。一个有限状态机 (FSM) 跟踪每个请求经历 pending -> approved | rejected 的状态变迁。

## 解决方案

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

## 工作原理

1. 领导通过生成 request_id 并通过收件箱发送 shutdown_request 来发起关机。

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

2. 队友在收件箱中收到请求, 调用 `shutdown_response` 工具来批准或拒绝。

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

3. 队友的循环检查是否批准了关机并退出。

```python
if (block.name == "shutdown_response"
        and block.input.get("approve")):
    should_exit = True
# ...
member["status"] = "shutdown" if should_exit else "idle"
```

4. 计划审批遵循完全相同的模式。队友提交计划时生成一个 request_id。

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

5. 领导审查后使用同一个 request_id 作出响应。

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

6. 两个协议使用同一个 `plan_approval` 工具名, 有两种模式: 队友提交 (无 request_id), 领导审查 (带 request_id)。

```python
# Lead tool dispatch:
"plan_approval": lambda **kw: handle_plan_review(
    kw["request_id"], kw["approve"],
    kw.get("feedback", "")),
# Teammate: submit mode (generate request_id)
```

## 核心代码

双协议处理器 (来自 `agents/s10_team_protocols.py`):

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

## 相对 s09 的变更

| 组件           | 之前 (s09)       | 之后 (s10)                           |
|----------------|------------------|--------------------------------------|
| Tools          | 9                | 12 (+shutdown_req/resp +plan)        |
| 关机           | 仅自然退出       | 请求-响应握手                        |
| 计划门控       | 无               | 提交/审查与审批                      |
| 请求追踪       | 无               | 两个 tracker 字典                    |
| 关联           | 无               | 每个请求一个 request_id              |
| FSM            | 无               | pending -> approved/rejected         |

## 生产环境参考

Claude Code 通过 SendMessage 工具的 `type: "shutdown_request"` 实现关机, 通过 ExitPlanMode 实现计划审批。当队友设置了 `plan_mode_required` 时, 必须先提交计划才能执行。生产版本增加了关机超时处理 (可配置时间窗口后强制终止) 和计划修订 (被拒绝的计划可以修改后重新提交)。request_id 模式可推广到任何异步审批工作流 -- 无论审批什么, 同一个 FSM 都适用。

## 试一试

```sh
cd learn-claude-code
python agents/s10_team_protocols.py
```

可以尝试的提示:

1. `Spawn alice as a coder. Then request her shutdown.`
2. `List teammates to see alice's status after shutdown approval`
3. `Spawn bob with a risky refactoring task. Review and reject his plan.`
4. `Spawn charlie, have him submit a plan, then approve it.`
5. 输入 `/team` 监控状态
