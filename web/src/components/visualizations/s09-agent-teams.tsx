"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";

type AgentStatus = "idle" | "working";

interface Agent {
  id: string;
  name: string;
  status: AgentStatus;
  inbox: string;
  cx: number;
  cy: number;
}

interface Envelope {
  id: number;
  from: string;
  to: string;
  label: string;
}

const AGENTS: Agent[] = [
  { id: "lead", name: "lead", status: "idle", inbox: "lead.jsonl", cx: 200, cy: 50 },
  { id: "coder", name: "coder", status: "idle", inbox: "coder.jsonl", cx: 80, cy: 150 },
  { id: "reviewer", name: "reviewer", status: "idle", inbox: "reviewer.jsonl", cx: 320, cy: 150 },
];

const MESSAGES: { from: string; to: string; label: string }[] = [
  { from: "lead", to: "coder", label: "Implement login API" },
  { from: "coder", to: "reviewer", label: "PR ready for review" },
  { from: "reviewer", to: "lead", label: "Approved with notes" },
];

export default function AgentTeams() {
  const [agents, setAgents] = useState<Agent[]>(AGENTS);
  const [envelopes, setEnvelopes] = useState<Envelope[]>([]);
  const [msgIndex, setMsgIndex] = useState(0);

  const reset = useCallback(() => {
    setAgents(AGENTS);
    setEnvelopes([]);
    setMsgIndex(0);
  }, []);

  useEffect(() => {
    if (msgIndex >= MESSAGES.length) {
      const t = setTimeout(reset, 3000);
      return () => clearTimeout(t);
    }
    const timer = setTimeout(() => {
      const msg = MESSAGES[msgIndex];
      setAgents((prev) =>
        prev.map((a) =>
          a.id === msg.from ? { ...a, status: "working" } : a
        )
      );
      setEnvelopes((prev) => [
        ...prev,
        { id: msgIndex, from: msg.from, to: msg.to, label: msg.label },
      ]);

      setTimeout(() => {
        setAgents((prev) =>
          prev.map((a) =>
            a.id === msg.from ? { ...a, status: "idle" } : a
          )
        );
        setMsgIndex((i) => i + 1);
      }, 1200);
    }, 1500);

    return () => clearTimeout(timer);
  }, [msgIndex, reset]);

  const getAgentPos = (id: string) => {
    const a = AGENTS.find((ag) => ag.id === id);
    return a ? { cx: a.cx, cy: a.cy } : { cx: 0, cy: 0 };
  };

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Agent Team Mailboxes
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <svg viewBox="0 0 400 210" className="w-full">
          {agents.map((agent) => (
            <g key={agent.id}>
              <motion.circle
                cx={agent.cx}
                cy={agent.cy}
                r="28"
                animate={{
                  fill: agent.status === "working" ? "#3b82f6" : "#a1a1aa",
                }}
                className="stroke-zinc-300 dark:stroke-zinc-600"
                strokeWidth="2"
              />
              <text
                x={agent.cx}
                y={agent.cy + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-white text-[10px] font-semibold"
              >
                {agent.name}
              </text>
              <text
                x={agent.cx}
                y={agent.cy + 42}
                textAnchor="middle"
                className="fill-zinc-400 text-[8px] font-mono dark:fill-zinc-500"
              >
                {agent.inbox}
              </text>
              <circle
                cx={agent.cx + 24}
                cy={agent.cy - 24}
                r="5"
                className={
                  agent.status === "working"
                    ? "fill-blue-400"
                    : "fill-zinc-300 dark:fill-zinc-600"
                }
              />
            </g>
          ))}

          {envelopes.map((env) => {
            const from = getAgentPos(env.from);
            const to = getAgentPos(env.to);
            return (
              <motion.g
                key={env.id}
                initial={{ x: from.cx, y: from.cy, opacity: 0 }}
                animate={{ x: to.cx, y: to.cy, opacity: [0, 1, 1, 0] }}
                transition={{ duration: 1.2 }}
              >
                <rect
                  x={-18}
                  y={-8}
                  width="36"
                  height="16"
                  rx="3"
                  className="fill-amber-400"
                />
                <text
                  x={0}
                  y={2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="fill-white text-[6px] font-medium"
                >
                  msg
                </text>
              </motion.g>
            );
          })}
        </svg>

        <div className="mt-2 text-center">
          <span className="text-xs text-zinc-400">
            {msgIndex < MESSAGES.length
              ? MESSAGES[msgIndex].label
              : "Cycle complete"}
          </span>
        </div>
        <div className="mt-2 text-center">
          <button
            onClick={reset}
            className="rounded bg-zinc-200 px-2 py-0.5 text-xs hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
          >
            Replay
          </button>
        </div>
      </div>
    </section>
  );
}
