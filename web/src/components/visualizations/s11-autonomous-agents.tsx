"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";

type Phase = "idle" | "poll" | "claim" | "work";
interface Agent { id: string; name: string; phase: Phase; taskId: string | null; idleTimer: number; }
interface Task { id: string; name: string; claimed: boolean; }

const INIT_TASKS: Task[] = [
  { id: "t1", name: "Fix auth bug", claimed: false },
  { id: "t2", name: "Add rate limiter", claimed: false },
  { id: "t3", name: "Write e2e tests", claimed: false },
  { id: "t4", name: "Update API docs", claimed: false },
];
const INIT_AGENTS: Agent[] = [
  { id: "a1", name: "Agent A", phase: "idle", taskId: null, idleTimer: 0 },
  { id: "a2", name: "Agent B", phase: "idle", taskId: null, idleTimer: 0 },
  { id: "a3", name: "Agent C", phase: "idle", taskId: null, idleTimer: 0 },
];
const PHASE_DOT: Record<Phase, string> = {
  idle: "bg-zinc-300 dark:bg-zinc-600", poll: "bg-amber-400",
  claim: "bg-blue-500", work: "bg-emerald-500",
};
const PHASE_FILL: Record<Phase, string> = {
  idle: "#a1a1aa", poll: "#fbbf24", claim: "#3b82f6", work: "#10b981",
};
const IDLE_TIMEOUT = 3;
const CX = 120, CY = 100, R = 70;
const positions = INIT_AGENTS.map((_, i) => {
  const a = (i * 2 * Math.PI) / INIT_AGENTS.length - Math.PI / 2;
  return { x: CX + R * Math.cos(a), y: CY + R * Math.sin(a) };
});

export default function AutonomousAgents() {
  const [agents, setAgents] = useState(INIT_AGENTS);
  const [tasks, setTasks] = useState(INIT_TASKS);
  const [running, setRunning] = useState(false);

  const reset = useCallback(() => {
    setAgents(INIT_AGENTS); setTasks(INIT_TASKS); setRunning(false);
  }, []);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      setAgents((prev) => prev.map((ag) => {
        if (ag.phase === "idle") {
          const n = ag.idleTimer + 1;
          return n >= IDLE_TIMEOUT ? { ...ag, phase: "poll" as Phase, idleTimer: 0 } : { ...ag, idleTimer: n };
        }
        if (ag.phase === "poll") {
          const avail = tasks.find((t) => !t.claimed);
          if (avail) {
            setTasks((ts) => ts.map((t) => t.id === avail.id ? { ...t, claimed: true } : t));
            return { ...ag, phase: "claim" as Phase, taskId: avail.id };
          }
          return { ...ag, phase: "idle" as Phase, idleTimer: 0 };
        }
        if (ag.phase === "claim") return { ...ag, phase: "work" as Phase };
        return { ...ag, phase: "idle" as Phase, taskId: null, idleTimer: 0 };
      }));
    }, 1000);
    return () => clearInterval(timer);
  }, [running, tasks]);

  const taskName = (id: string | null) => id ? (tasks.find((t) => t.id === id)?.name ?? "") : "";

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Autonomous Agent Cycle
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="flex gap-4">
          <div className="flex-1">
            <svg viewBox="0 0 240 200" className="w-full">
              <circle cx={CX} cy={CY} r="20" strokeWidth="1.5" strokeDasharray="4 2"
                className="fill-zinc-100 stroke-zinc-300 dark:fill-zinc-800 dark:stroke-zinc-600" />
              <text x={CX} y={CY + 2} textAnchor="middle" dominantBaseline="middle"
                className="fill-zinc-400 text-[7px] font-medium">Tasks</text>
              {agents.map((ag, i) => {
                const p = positions[i];
                return (
                  <g key={ag.id}>
                    {ag.phase === "poll" && (
                      <line x1={p.x} y1={p.y} x2={CX} y2={CY}
                        className="stroke-amber-400" strokeWidth="1" strokeDasharray="3 3" />
                    )}
                    {ag.phase === "claim" && (
                      <line x1={CX} y1={CY} x2={p.x} y2={p.y}
                        className="stroke-blue-500" strokeWidth="1.5" />
                    )}
                    <motion.circle cx={p.x} cy={p.y} r="22" strokeWidth="1.5"
                      animate={{ fill: PHASE_FILL[ag.phase] }}
                      className="stroke-zinc-300 dark:stroke-zinc-600" />
                    <text x={p.x} y={p.y - 4} textAnchor="middle" dominantBaseline="middle"
                      className="fill-white text-[8px] font-semibold">{ag.name}</text>
                    <text x={p.x} y={p.y + 7} textAnchor="middle" dominantBaseline="middle"
                      className="fill-white/80 text-[6px] font-mono">{ag.phase}</text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="flex-1 space-y-2">
            <div className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Task Board</div>
            {tasks.map((t) => (
              <motion.div key={t.id} animate={{ opacity: t.claimed ? 0.4 : 1 }}
                className="flex items-center gap-2 rounded border border-zinc-200 p-2 dark:border-zinc-700">
                <span className={`h-2 w-2 rounded-full ${t.claimed ? "bg-emerald-400" : "bg-zinc-300 dark:bg-zinc-600"}`} />
                <span className="text-xs text-zinc-700 dark:text-zinc-300">{t.name}</span>
                {t.claimed && <span className="ml-auto text-[10px] text-emerald-500">claimed</span>}
              </motion.div>
            ))}
            <div className="mt-2 space-y-1">
              {agents.map((ag) => (
                <div key={ag.id} className="flex items-center gap-2">
                  <span className={`h-1.5 w-1.5 rounded-full ${PHASE_DOT[ag.phase]}`} />
                  <span className="font-mono text-[10px] text-zinc-500">
                    {ag.name}
                    {ag.phase === "idle" && ` idle: ${ag.idleTimer}/${IDLE_TIMEOUT}`}
                    {ag.phase === "work" && ` -> ${taskName(ag.taskId)}`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-3 text-center">
          <button onClick={running ? reset : () => setRunning(true)}
            className="rounded-md bg-zinc-200 px-4 py-1.5 text-xs font-medium hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600">
            {running ? "Reset" : "Start Simulation"}
          </button>
        </div>
      </div>
    </section>
  );
}
