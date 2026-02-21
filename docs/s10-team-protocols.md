# s10: Team Protocols

> The same request_id handshake pattern powers both shutdown and plan approval -- one FSM, two applications.

## The Problem

In s09, teammates work and communicate but there is no structured
coordination. Two problems arise:

**Shutdown**: How do you stop a teammate cleanly? Killing the thread
leaves files partially written and config.json in a wrong state.
Graceful shutdown requires a handshake: the lead requests, the teammate
decides whether to approve (finish and exit) or reject (keep working).

**Plan approval**: How do you gate execution? When the lead says
"refactor the auth module," the teammate starts immediately. For
high-risk changes, the lead should review the plan before execution
begins. A junior proposes, a senior approves.

Both problems share the same structure: one side sends a request with a
unique ID, the other side responds referencing that ID. A finite state
machine tracks each request through pending -> approved | rejected.

## The Solution

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

## How It Works

1. The lead initiates shutdown by generating a request_id and sending
   a shutdown_request through the inbox.

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

2. The teammate receives the request in its inbox and calls the
   `shutdown_response` tool to approve or reject.

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

3. The teammate loop checks for approved shutdown and exits.

```python
if (block.name == "shutdown_response"
        and block.input.get("approve")):
    should_exit = True
# ...
member["status"] = "shutdown" if should_exit else "idle"
```

4. Plan approval follows the identical pattern. The teammate submits
   a plan, generating a request_id.

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

5. The lead reviews and responds with the same request_id.

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

6. Both protocols use the same `plan_approval` tool name with two
   modes: teammates submit (no request_id), the lead reviews (with
   request_id).

```python
# Lead tool dispatch:
"plan_approval": lambda **kw: handle_plan_review(
    kw["request_id"], kw["approve"],
    kw.get("feedback", "")),
# Teammate: submit mode (generate request_id)
```

## Key Code

The dual protocol handlers (from `agents/s10_team_protocols.py`):

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

## What Changed From s09

| Component      | Before (s09)     | After (s10)                  |
|----------------|------------------|------------------------------|
| Tools          | 9                | 12 (+shutdown_req/resp +plan)|
| Shutdown       | Natural exit only| Request-response handshake   |
| Plan gating    | None             | Submit/review with approval  |
| Request tracking| None            | Two tracker dicts            |
| Correlation    | None             | request_id per request       |
| FSM            | None             | pending -> approved/rejected |

## Production Reference

Claude Code implements shutdown through the SendMessage tool with
`type: "shutdown_request"` and plan approval through ExitPlanMode.
When a teammate has `plan_mode_required` set, it must submit a plan
before executing. The production version adds timeout handling for
shutdown (force-kill after configurable window) and plan revisions
(rejected plans can be modified and resubmitted). The request_id
pattern generalizes to any async approval workflow -- the same FSM
applies regardless of what is being approved.

## Try It

```sh
cd learn-claude-code
python agents/s10_team_protocols.py
```

Example prompts to try:

1. `Spawn alice as a coder. Then request her shutdown.`
2. `List teammates to see alice's status after shutdown approval`
3. `Spawn bob with a risky refactoring task. Review and reject his plan.`
4. `Spawn charlie, have him submit a plan, then approve it.`
5. Type `/team` to monitor statuses
