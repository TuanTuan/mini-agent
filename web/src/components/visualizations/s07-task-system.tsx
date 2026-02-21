"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";

type TaskStatus = "pending" | "in_progress" | "completed";

interface Task {
  id: string;
  name: string;
  status: TaskStatus;
  blockedBy: string[];
  x: number;
  y: number;
}

const INITIAL_TASKS: Task[] = [
  { id: "t1", name: "Setup DB schema", status: "completed", blockedBy: [], x: 50, y: 30 },
  { id: "t2", name: "Create API routes", status: "in_progress", blockedBy: ["t1"], x: 200, y: 20 },
  { id: "t3", name: "Build auth module", status: "pending", blockedBy: ["t1"], x: 200, y: 80 },
  { id: "t4", name: "Write integration tests", status: "pending", blockedBy: ["t2", "t3"], x: 370, y: 50 },
  { id: "t5", name: "Deploy to staging", status: "pending", blockedBy: ["t4"], x: 520, y: 50 },
];

const STATUS_COLORS: Record<TaskStatus, { fill: string; text: string; badge: string }> = {
  pending: {
    fill: "fill-zinc-200 dark:fill-zinc-700",
    text: "text-zinc-500 dark:text-zinc-400",
    badge: "bg-zinc-200 text-zinc-600 dark:bg-zinc-700 dark:text-zinc-400",
  },
  in_progress: {
    fill: "fill-amber-100 dark:fill-amber-900/30",
    text: "text-amber-700 dark:text-amber-300",
    badge: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
  },
  completed: {
    fill: "fill-emerald-100 dark:fill-emerald-900/30",
    text: "text-emerald-700 dark:text-emerald-300",
    badge: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
  },
};

function isUnblocked(task: Task, tasks: Task[]): boolean {
  return task.blockedBy.every(
    (depId) => tasks.find((t) => t.id === depId)?.status === "completed"
  );
}

export default function TaskSystem() {
  const [tasks, setTasks] = useState<Task[]>(INITIAL_TASKS);

  const completeNext = useCallback(() => {
    setTasks((prev) => {
      const inProgress = prev.find((t) => t.status === "in_progress");
      if (inProgress) {
        const updated = prev.map((t) =>
          t.id === inProgress.id ? { ...t, status: "completed" as TaskStatus } : t
        );
        const nextPending = updated.find(
          (t) => t.status === "pending" && isUnblocked(t, updated)
        );
        if (nextPending) {
          return updated.map((t) =>
            t.id === nextPending.id ? { ...t, status: "in_progress" as TaskStatus } : t
          );
        }
        return updated;
      }
      const nextPending = prev.find(
        (t) => t.status === "pending" && isUnblocked(t, prev)
      );
      if (nextPending) {
        return prev.map((t) =>
          t.id === nextPending.id ? { ...t, status: "in_progress" as TaskStatus } : t
        );
      }
      return INITIAL_TASKS;
    });
  }, []);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Task Dependency Graph
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <svg viewBox="0 0 650 120" className="w-full">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" className="fill-zinc-400 dark:fill-zinc-500" />
            </marker>
          </defs>
          {tasks.map((task) =>
            task.blockedBy.map((depId) => {
              const dep = tasks.find((t) => t.id === depId);
              if (!dep) return null;
              return (
                <line
                  key={`${depId}-${task.id}`}
                  x1={dep.x + 70} y1={dep.y + 15}
                  x2={task.x} y2={task.y + 15}
                  className="stroke-zinc-300 dark:stroke-zinc-600"
                  strokeWidth="1.5"
                  markerEnd="url(#arrow)"
                />
              );
            })
          )}
          {tasks.map((task) => (
            <g key={task.id}>
              <motion.rect
                x={task.x} y={task.y}
                width="140" height="30" rx="6"
                animate={{ opacity: 1 }}
                strokeWidth="1"
                className={`${STATUS_COLORS[task.status].fill} stroke-zinc-300 dark:stroke-zinc-600`}
              />
              <text
                x={task.x + 70} y={task.y + 19}
                textAnchor="middle"
                className={`text-[9px] font-medium ${STATUS_COLORS[task.status].text}`}
                fill="currentColor"
              >
                {task.name}
              </text>
            </g>
          ))}
        </svg>

        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center gap-1">
            <div className="h-3 w-3 rounded bg-zinc-200 dark:bg-zinc-700" />
            <span className="text-[10px] text-zinc-400">pending</span>
            <div className="ml-2 h-3 w-3 rounded bg-amber-200 dark:bg-amber-900/40" />
            <span className="text-[10px] text-zinc-400">in_progress</span>
            <div className="ml-2 h-3 w-3 rounded bg-emerald-200 dark:bg-emerald-900/40" />
            <span className="text-[10px] text-zinc-400">completed</span>
          </div>
          <button
            onClick={completeNext}
            className="rounded bg-zinc-200 px-3 py-1 text-xs font-medium hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
          >
            Advance
          </button>
        </div>

        <div className="mt-2 flex items-center gap-1.5 rounded-md bg-zinc-50 p-2 dark:bg-zinc-800/50">
          <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 text-zinc-400" fill="currentColor">
            <path d="M3.75 0.5A3.25 3.25 0 0 0 .5 3.75v8.5A3.25 3.25 0 0 0 3.75 15.5h8.5a3.25 3.25 0 0 0 3.25-3.25v-8.5A3.25 3.25 0 0 0 12.25.5h-8.5ZM2 3.75C2 2.784 2.784 2 3.75 2h8.5c.966 0 1.75.784 1.75 1.75v8.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5Z"/>
          </svg>
          <span className="font-mono text-[10px] text-zinc-500">
            .tasks/tasks.json -- persisted to disk
          </span>
        </div>
      </div>
    </section>
  );
}
