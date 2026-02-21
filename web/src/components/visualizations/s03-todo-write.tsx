"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

type TodoStatus = "pending" | "in_progress" | "completed";

interface TodoItem {
  id: number;
  text: string;
  status: TodoStatus;
}

const INITIAL_TODOS: TodoItem[] = [
  { id: 1, text: "Write unit tests for auth module", status: "pending" },
  { id: 2, text: "Fix CSS layout on mobile", status: "pending" },
  { id: 3, text: "Add error handling to API calls", status: "pending" },
  { id: 4, text: "Update README with setup steps", status: "pending" },
];

const STATUS_STYLES: Record<TodoStatus, string> = {
  pending: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
  in_progress: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
  completed: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
};

export default function TodoWrite() {
  const [todos, setTodos] = useState<TodoItem[]>(INITIAL_TODOS);
  const [nagCount, setNagCount] = useState(0);
  const [showNag, setShowNag] = useState(false);
  const [roundsWithout, setRoundsWithout] = useState(0);

  const NAG_THRESHOLD = 3;

  const reset = useCallback(() => {
    setTodos(INITIAL_TODOS);
    setNagCount(0);
    setShowNag(false);
    setRoundsWithout(0);
  }, []);

  useEffect(() => {
    const pending = todos.filter((t) => t.status === "pending");
    const inProgress = todos.filter((t) => t.status === "in_progress");
    const allDone = todos.every((t) => t.status === "completed");
    if (allDone) return;

    const timer = setTimeout(() => {
      if (inProgress.length > 0) {
        setTodos((prev) =>
          prev.map((t) =>
            t.status === "in_progress" ? { ...t, status: "completed" } : t
          )
        );
        setRoundsWithout(0);
      } else if (pending.length > 0) {
        const nextRounds = roundsWithout + 1;
        setRoundsWithout(nextRounds);
        if (nextRounds >= NAG_THRESHOLD) {
          setNagCount((c) => c + 1);
          setShowNag(true);
          setTimeout(() => setShowNag(false), 2000);
          setRoundsWithout(0);
          setTodos((prev) => {
            const first = prev.find((t) => t.status === "pending");
            if (!first) return prev;
            return prev.map((t) =>
              t.id === first.id ? { ...t, status: "in_progress" } : t
            );
          });
        }
      }
    }, 1200);
    return () => clearTimeout(timer);
  }, [todos, roundsWithout]);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        TodoWrite Nag System
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-3 flex items-center justify-between">
          <span className="font-mono text-xs text-zinc-500">
            Rounds without progress: {roundsWithout} / {NAG_THRESHOLD}
          </span>
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-zinc-500">
              Nags: {nagCount}
            </span>
            <button
              onClick={reset}
              className="rounded bg-zinc-200 px-2 py-0.5 text-xs hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
            >
              Reset
            </button>
          </div>
        </div>

        <div className="space-y-2">
          {todos.map((todo) => (
            <motion.div
              key={todo.id}
              layout
              className="flex items-center gap-3 rounded-md border border-zinc-100 p-2 dark:border-zinc-800"
            >
              <motion.span
                key={todo.status}
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${STATUS_STYLES[todo.status]}`}
              >
                {todo.status}
              </motion.span>
              <span className="text-sm text-zinc-700 dark:text-zinc-300">
                {todo.text}
              </span>
            </motion.div>
          ))}
        </div>

        <AnimatePresence>
          {showNag && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-3 rounded-md border border-amber-300 bg-amber-50 p-2 text-center text-xs font-medium text-amber-700 dark:border-amber-700 dark:bg-amber-900/20 dark:text-amber-300"
            >
              NAG: {NAG_THRESHOLD} rounds without working on todos -- picking one up now!
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
