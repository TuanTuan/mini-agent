# Learn Claude Code -- Build an AI Agent From Scratch

```
                    THE AGENT PATTERN
                    =================

    User --> messages[] --> LLM --> response
                                      |
                            stop_reason == "tool_use"?
                           /                          \
                         yes                           no
                          |                             |
                    execute tools                    return text
                    append results
                    loop back -----------------> messages[]


    That's it. Every AI coding agent is this loop.
    Everything else is refinement.
```

**Learn how modern AI agents work by building one from scratch -- 11 progressive sessions, from a simple loop to full autonomous teams.**

> **Disclaimer**: This is an independent educational project. It is not affiliated with, endorsed by, or sponsored by Anthropic. "Claude Code" is a trademark of Anthropic.

---

## Architecture

```
learn-claude-code/
|
|-- agents/                        # Python reference implementations
|   |-- s01_agent_loop.py          # while loop + bash
|   |-- s02_tool_use.py            # + Read, Write, Edit
|   |-- s03_todo_write.py          # + TodoWrite
|   |-- s04_subagent.py            # + Task tool / spawn
|   |-- s05_skill_loading.py       # + SKILL.md injection
|   |-- s06_context_compact.py     # + /compact (3-layer)
|   |-- s07_task_system.py         # + Tasks CRUD + deps
|   |-- s08_background_tasks.py    # + background threads
|   |-- s09_agent_teams.py         # + teammates + mailboxes
|   |-- s10_team_protocols.py      # + shutdown + plan approval
|   |-- s11_autonomous_agents.py   # + idle cycle + auto-claim
|   +-- s_full.py                  # full combined reference
|
|-- docs/                          # Mental-model-first documentation
|   |-- s01-the-agent-loop.md
|   |-- s02-tool-use.md
|   |-- ...
|   +-- s11-autonomous-agents.md
|
|-- web/                           # Interactive learning platform
|   |-- src/components/
|   |   |-- simulator/             #   Step-through agent execution
|   |   |-- architecture/          #   Flow diagrams, arch diagrams
|   |   |-- code/                  #   Python source viewer
|   |   +-- docs/                  #   Documentation renderer
|   +-- src/app/[locale]/(learn)/
|       +-- [version]/             #   Per-session learning page
|
|-- skills/                        # Skill files for s05
+-- .github/workflows/ci.yml      # CI: typecheck + build
```

## Learning Path

```
Phase 1: THE LOOP                   Phase 2: PLANNING & KNOWLEDGE
=================                   ==============================
s01: The Agent Loop                 s03: TodoWrite
|  bash is all you need             |  plan before you act
|  "The entire agent is a loop"     |  "Visible plans improve completion"
|                                   |
+-> s02: Tools                      s04: Subagents
    |  Read, Write, Edit, Bash      |  fresh context via Task tool
    |  "The loop didn't change"     |  "Process isolation = context isolation"
                                    |
                                    s05: Skills
                                    |  SKILL.md + tool_result injection
                                    |  "Load on demand, not upfront"
                                    |
                                    s06: Compact
                                       three-layer context compression
                                       "Strategic forgetting"

Phase 3: PERSISTENCE                Phase 4: TEAMS
=================                   =====================
s07: Tasks                          s09: Agent Teams
|  persistent CRUD + dependencies   |  teammates + mailboxes
|  "State survives /compact"        |  "Append to send, drain to read"
|                                   |
s08: Background Tasks               s10: Team Protocols
   fire-and-forget threads + notify |  shutdown + plan approval
   "Fire and forget"                |  "Same request_id, two protocols"
                                    |
                                    s11: Autonomous Agents
                                       idle cycle + auto-claim
                                       "Poll, claim, work, repeat"
```

## Quick Start

### Run Python Agents Locally

```sh
git clone https://github.com/anthropics/anthropic-cookbook
cd anthropic-cookbook/learn-claude-code

pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Run any session
python agents/s01_agent_loop.py       # Start here
python agents/s11_autonomous_agents.py  # Full autonomous team
```

### Web Platform (Visualization)

The web platform provides interactive visualizations for each session:
- Step-through simulator shows each agent loop iteration
- Architecture diagrams and execution flow visualizations
- Python source code viewer with syntax highlighting
- Session documentation with ASCII diagrams

```sh
cd web
npm install
npm run dev
# Open http://localhost:3000
```

## Structured Logging System

All agents include a unified logging system (`agents/logger.py`) for debugging and analysis.

### Command Line Arguments

```bash
# Default: verbose terminal output + show raw API data
python agents/s01_agent_loop.py

# Quiet mode: suppress terminal output
python agents/s01_agent_loop.py -q

# Hide raw API request/response data
python agents/s01_agent_loop.py --no-show-raw

# Output to Markdown file
python agents/s01_agent_loop.py -o logs/session.md

# Quiet + file output (for background runs)
python agents/s01_agent_loop.py -q -o logs/session.md

# File without raw data (smaller files)
python agents/s01_agent_loop.py -o logs/session.md --no-file-show-raw

# Append to existing log file
python agents/s01_agent_loop.py -o logs/session.md -a

# Combined: quiet mode + file with collapsible raw data
python agents/s01_agent_loop.py -q -o logs/session.md --file-show-raw
```

### Logging Arguments Reference

| Argument | Short | Description |
|----------|-------|-------------|
| `--verbose` / `--no-verbose` | - | Terminal verbose output (default: True) |
| `-q`, `--quiet` | -q | Quiet mode (equivalent to `--no-verbose`) |
| `--show-raw` / `--no-show-raw` | - | Show raw API data in terminal (default: True) |
| `-o`, `--log-file` | -o | Markdown log file path |
| `--file-show-raw` / `--no-file-show-raw` | - | Include raw API data in file (default: True) |
| `-a`, `--append` | -a | Append to existing log file (default: overwrite) |

### Markdown Log Features

When using `-o` to output to a file, the log includes:

- **Collapsible sections** (`<details>`) for full API request/response JSON
- **Structured headings** with iteration counts
- **Code blocks** with syntax highlighting
- **Timestamps** for session start/end and duration
- **Tool call traces** with inputs and outputs

Example Markdown output:

```markdown
# Agent Session Log

**Started:** 2024-01-15 10:30:00

---

## 🔄 Loop Iteration #1

#### 📤 API Request

**Request Summary:**
```json
{"model": "claude-3-5-sonnet", "messages": [...]}
```

<details>
<summary>📄 Full Request JSON (click to expand)</summary>

```json
{
  "model": "claude-3-5-sonnet",
  "system": "...",
  "messages": [...],
  "tools": [...]
}
```

</details>

#### ⚡ Tool Call: `bash`

- **Input:**
```json
{"command": "ls -la"}
```

#### ✓ Success Tool Result

<details>
<summary>Full Content (click to expand)</summary>

```
total 48
drwxr-xr-x  12 user  staff   384 Jan 15 10:30 .
...
```

</details>
```

### Programmatic Usage

```python
from logger import AgentLogger, create_logger_from_args, get_logger_config_string

# Method 1: Direct initialization
logger = AgentLogger(
    verbose=True,
    show_raw=True,
    log_file="logs/session.md",
    file_show_raw=True
)

# Method 2: From command-line arguments
from logger import parse_logger_args
args = parse_logger_args()
logger = create_logger_from_args(args)

# Display current config
print(f"Config: {get_logger_config_string(args)}")

# End session with summary
logger.session_end("Task completed successfully")
```

## Session Comparison

```
Session  Claude Code Feature    Tools  Core Addition              Key Insight
-------  --------------------  -----  -------------------------  ----------------------------
s01      The Agent Loop           1    while + stop_reason        Bash is all you need
s02      Tools                    4    Read/Write/Edit/Bash       The loop didn't change
s03      TodoWrite                5    TodoManager + nag          Plan before you act
s04      Subagents                5    Task tool + spawn          Fresh context per subagent
s05      Skills                   5    SKILL.md injection         Load on demand, not upfront
s06      Compact                  5    3-layer compression        Strategic forgetting
s07      Tasks                    8    CRUD + dependency graph    State survives /compact
s08      Background Tasks         6    threads + notifications    Fire and forget
s09      Agent Teams              9    teammates + mailboxes      Persistent agents + async mail
s10      Team Protocols          12    shutdown + plan approval   Same request_id, two protocols
s11      Autonomous Agents       14    idle cycle + auto-claim    Poll, claim, work, repeat
```

## The Core Pattern

```python
# Every AI agent is this loop:
def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM,
            messages=messages, tools=TOOLS,
        )
        messages.append({"role": "assistant",
                         "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = TOOL_HANDLERS[block.name](**block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

Each session adds ONE mechanism on top of this loop.

## Key Mechanisms

| Claude Code Feature | Session | What It Does                                    |
|---------------------|---------|-------------------------------------------------|
| Agent loop          | s01     | `while (stop_reason == "tool_use")` loop        |
| Tools               | s02     | Map of tool name -> handler function            |
| TodoWrite           | s03     | Create plan before execution, track completion   |
| Subagents           | s04     | Fresh message list per subagent via Task tool    |
| Skills              | s05     | SKILL.md content injected via tool_result        |
| Compact (micro)     | s06     | Old tool results replaced with placeholders      |
| Compact (auto)      | s06     | LLM summarizes conversation when tokens > limit  |
| Tasks API           | s07     | File-based tasks with dependency graph           |
| Background tasks    | s08     | Threaded commands + notification queue           |
| Agent Teams         | s09     | Named persistent agents with config.json         |
| Mailbox             | s09     | Append-only file-based messages, per-teammate     |
| Shutdown protocol   | s10     | request_id based FSM for graceful shutdown       |
| Plan approval       | s10     | Submit/review with request_id correlation        |
| Idle cycle          | s11     | Poll board, auto-claim unclaimed tasks           |

## Documentation

Each doc follows a mental-model-first structure with ASCII diagrams:

- [s01: The Agent Loop](./docs/s01-the-agent-loop.md)
- [s02: Tools](./docs/s02-tool-use.md)
- [s03: TodoWrite](./docs/s03-todo-write.md)
- [s04: Subagents](./docs/s04-subagent.md)
- [s05: Skills](./docs/s05-skill-loading.md)
- [s06: Compact](./docs/s06-context-compact.md)
- [s07: Tasks](./docs/s07-task-system.md)
- [s08: Background Tasks](./docs/s08-background-tasks.md)
- [s09: Agent Teams](./docs/s09-agent-teams.md)
- [s10: Team Protocols](./docs/s10-team-protocols.md)
- [s11: Autonomous Agents](./docs/s11-autonomous-agents.md)

## License

MIT

---

**The model is the agent. Our job is to give it tools and stay out of the way.**
