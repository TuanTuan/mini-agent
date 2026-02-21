#!/usr/bin/env python3
"""
litellm_s01.py - Agent Loop with LiteLLM SDK (Azure GPT-5.2)

基于 s01_agent_loop.py，使用 LiteLLM SDK 和 OpenAI 消息格式。

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)
"""

import json
import os
import subprocess

import litellm
from dotenv import load_dotenv

from logger_openai import OpenAILogger

load_dotenv(override=True)

# ============================================================================
# 配置
# ============================================================================
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5.2")

MODEL = f"azure/{AZURE_DEPLOYMENT}"
SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

logger = OpenAILogger(verbose=True, show_raw=True)

# ============================================================================
# OpenAI 格式的工具定义
# ============================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"],
            },
        },
    }
]

# ============================================================================
# 工具执行
# ============================================================================
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# ============================================================================
# Agent Loop
# ============================================================================
def agent_loop(messages: list):
    """
    核心 Agent 循环

    OpenAI 响应结构:
    {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "...",
                "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
            },
            "finish_reason": "tool_calls" | "stop"
        }],
        "usage": {"prompt_tokens": N, "completion_tokens": N}
    }
    """
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)
        logger.messages_snapshot(messages, "BEFORE LLM CALL")

        # 显示原始请求
        logger.request_raw(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000
        )

        # 调用 LiteLLM
        response = litellm.completion(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )

        # 转换为字典
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)

        # 显示原始响应
        logger.response_raw(response_dict)

        # 提取响应信息
        choice = (response_dict.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = message.get("tool_calls") or []

        # 追加 assistant 消息
        assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        logger.messages_snapshot(messages, "AFTER APPEND ASSISTANT")

        # 检查是否结束
        if finish_reason != "tool_calls":
            logger.loop_end(f"finish_reason = '{finish_reason}'")
            return

        # 执行工具调用
        logger.section("Executing Tool Calls", "🔧")
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args_str = fn.get("arguments", "{}")

            try:
                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            except json.JSONDecodeError:
                fn_args = {}

            # 显示工具调用
            logger.tool_call(fn_name, fn_args, tc_id)

            # 执行工具
            output = ""
            if fn_name == "bash":
                output = run_bash(fn_args.get("command", ""))
                print(f"\033[33m$ {fn_args.get('command', '')}\033[0m")
                print(output[:200] if len(output) > 200 else output)

            # 显示工具结果
            logger.tool_result(tc_id, output, is_error=output.startswith("Error:"))

            # 追加 tool 结果消息
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": output,
            })

        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    logger.header("LiteLLM Agent Loop - Azure GPT-5.2", "litellm-s01")
    logger.config(
        model=MODEL,
        api_base=AZURE_API_BASE,
        api_version=AZURE_API_VERSION
    )

    # OpenAI 格式: system 是第一条消息
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            query = input("\033[36mlitellm >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        logger.user_input(query)
        history.append({"role": "user", "content": query})
        agent_loop(history)

        logger.separator("FINAL RESPONSE")
        # 获取最后的 assistant 消息
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    print(content)
                break
        print()
