"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

type LaneState = "idle" | "running" | "done";

interface BackgroundTask {
  id: number;
  name: string;
  state: LaneState;
  progress: number;
}

export default function BackgroundTasks() {
  const [mainProgress, setMainProgress] = useState(0);
  const [tasks, setTasks] = useState<BackgroundTask[]>([
    { id: 1, name: "Run test suite", state: "idle", progress: 0 },
    { id: 2, name: "Lint codebase", state: "idle", progress: 0 },
  ]);
  const [notifications, setNotifications] = useState<string[]>([]);
  const [running, setRunning] = useState(false);

  const reset = useCallback(() => {
    setMainProgress(0);
    setTasks([
      { id: 1, name: "Run test suite", state: "idle", progress: 0 },
      { id: 2, name: "Lint codebase", state: "idle", progress: 0 },
    ]);
    setNotifications([]);
    setRunning(false);
  }, []);

  const start = useCallback(() => {
    reset();
    setRunning(true);
  }, [reset]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      setMainProgress((p) => Math.min(p + 5, 100));
      setTasks((prev) =>
        prev.map((t) => {
          if (t.state === "done") return t;
          if (t.state === "idle") return { ...t, state: "running", progress: 10 };
          const next = t.progress + Math.floor(Math.random() * 15) + 5;
          if (next >= 100) {
            setNotifications((n) => [...n, `${t.name} completed`]);
            return { ...t, state: "done", progress: 100 };
          }
          return { ...t, progress: Math.min(next, 95) };
        })
      );
    }, 600);

    return () => clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (mainProgress >= 100 && tasks.every((t) => t.state === "done")) {
      setRunning(false);
    }
  }, [mainProgress, tasks]);

  const LANE_COLORS: Record<LaneState, string> = {
    idle: "bg-zinc-200 dark:bg-zinc-700",
    running: "bg-blue-500",
    done: "bg-emerald-500",
  };

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Background Task Lanes
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="space-y-3">
          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                Main Thread
              </span>
              <span className="font-mono text-[10px] text-zinc-400">
                {mainProgress}%
              </span>
            </div>
            <div className="h-4 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
              <motion.div
                animate={{ width: `${mainProgress}%` }}
                className="h-full rounded-full bg-purple-500"
              />
            </div>
          </div>

          {tasks.map((task) => (
            <div key={task.id}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-xs text-zinc-500">{task.name}</span>
                <span
                  className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium text-white ${LANE_COLORS[task.state]}`}
                >
                  {task.state}
                </span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
                <motion.div
                  animate={{ width: `${task.progress}%` }}
                  className={`h-full rounded-full ${LANE_COLORS[task.state]}`}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-3 min-h-[40px]">
          <span className="text-xs text-zinc-500">Notifications (drained before next LLM call):</span>
          <AnimatePresence>
            {notifications.map((note, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-1 rounded bg-emerald-50 px-2 py-1 text-xs text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-300"
              >
                {note}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        <div className="mt-3 text-center">
          <button
            onClick={running ? reset : start}
            className="rounded-md bg-zinc-200 px-4 py-1.5 text-xs font-medium hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
          >
            {running ? "Reset" : "Start"}
          </button>
        </div>
      </div>
    </section>
  );
}
