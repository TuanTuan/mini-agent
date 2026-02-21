"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  role: string;
  detail: string;
  color: string;
}

const TURNS: Message[][] = [
  [{ role: "user", detail: "Fix the login bug", color: "bg-blue-500" }],
  [
    { role: "assistant", detail: "tool_use: read_file", color: "bg-zinc-600" },
    { role: "tool_result", detail: "auth.ts contents...", color: "bg-emerald-500" },
  ],
  [
    { role: "assistant", detail: "tool_use: edit_file", color: "bg-zinc-600" },
    { role: "tool_result", detail: "file updated", color: "bg-emerald-500" },
  ],
  [{ role: "assistant", detail: "end_turn: Done!", color: "bg-purple-500" }],
];

export default function AgentLoop() {
  const [turnIndex, setTurnIndex] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [running, setRunning] = useState(true);

  const stopReason = turnIndex >= TURNS.length - 1 ? "end_turn" : "tool_use";

  const reset = useCallback(() => {
    setTurnIndex(0);
    setMessages([]);
    setRunning(true);
  }, []);

  useEffect(() => {
    if (!running || turnIndex >= TURNS.length) return;
    const timer = setTimeout(() => {
      setMessages((prev) => [...prev, ...TURNS[turnIndex]]);
      setTurnIndex((prev) => prev + 1);
      if (turnIndex >= TURNS.length - 1) {
        setRunning(false);
      }
    }, 1200);
    return () => clearTimeout(timer);
  }, [turnIndex, running]);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        The Agent While-Loop
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-3 flex items-center justify-between">
          <code className="rounded bg-zinc-100 px-2 py-1 text-xs dark:bg-zinc-800">
            while (stop_reason === &quot;tool_use&quot;)
          </code>
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-zinc-500">
              Turn {Math.min(turnIndex, TURNS.length)} / {TURNS.length}
            </span>
            <button
              onClick={reset}
              className="rounded bg-zinc-200 px-2 py-0.5 text-xs hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
            >
              Replay
            </button>
          </div>
        </div>

        <div className="mb-3 flex items-center gap-2">
          <span className="text-xs text-zinc-500">stop_reason:</span>
          <motion.span
            key={stopReason}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            className={`rounded px-2 py-0.5 font-mono text-xs font-medium text-white ${
              stopReason === "end_turn" ? "bg-purple-500" : "bg-amber-500"
            }`}
          >
            {stopReason}
          </motion.span>
          {!running && (
            <motion.span
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xs font-medium text-emerald-600 dark:text-emerald-400"
            >
              Loop exited
            </motion.span>
          )}
        </div>

        <div className="space-y-1.5">
          <span className="font-mono text-xs text-zinc-400">messages[]</span>
          <div className="flex flex-wrap gap-1.5">
            <AnimatePresence>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.7, y: 8 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  className={`rounded-md px-2.5 py-1.5 ${msg.color}`}
                >
                  <div className="font-mono text-[10px] font-medium text-white">
                    {msg.role}
                  </div>
                  <div className="text-[9px] text-white/80">{msg.detail}</div>
                </motion.div>
              ))}
            </AnimatePresence>
            {messages.length === 0 && (
              <span className="py-1 text-xs text-zinc-400">
                waiting for first message...
              </span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
