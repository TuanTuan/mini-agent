"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";

interface FSMState {
  id: string;
  label: string;
  x: number;
  y: number;
}

interface FSMTransition {
  from: string;
  to: string;
  label: string;
}

const SHUTDOWN_STATES: FSMState[] = [
  { id: "pending", label: "pending", x: 30, y: 50 },
  { id: "requested", label: "requested", x: 130, y: 50 },
  { id: "approved", label: "approved", x: 230, y: 20 },
  { id: "rejected", label: "rejected", x: 230, y: 80 },
];

const SHUTDOWN_TRANSITIONS: FSMTransition[] = [
  { from: "pending", to: "requested", label: "shutdown_request" },
  { from: "requested", to: "approved", label: "approve: true" },
  { from: "requested", to: "rejected", label: "approve: false" },
];

const PLAN_STATES: FSMState[] = [
  { id: "planning", label: "planning", x: 30, y: 50 },
  { id: "submitted", label: "submitted", x: 130, y: 50 },
  { id: "approved", label: "approved", x: 230, y: 20 },
  { id: "revision", label: "revision", x: 230, y: 80 },
];

const PLAN_TRANSITIONS: FSMTransition[] = [
  { from: "planning", to: "submitted", label: "exit_plan_mode" },
  { from: "submitted", to: "approved", label: "approve: true" },
  { from: "submitted", to: "revision", label: "approve: false" },
];

const REQUEST_ID = "req_a1b2c3";

function FSMDiagram({
  title,
  states,
  transitions,
  activeState,
  onAdvance,
  color,
}: {
  title: string;
  states: FSMState[];
  transitions: FSMTransition[];
  activeState: string;
  onAdvance: () => void;
  color: string;
}) {
  return (
    <div className="flex-1">
      <div className="mb-2 text-xs font-semibold text-zinc-700 dark:text-zinc-300">
        {title}
      </div>
      <svg viewBox="0 0 290 110" className="w-full">
        {transitions.map((t) => {
          const from = states.find((s) => s.id === t.from)!;
          const to = states.find((s) => s.id === t.to)!;
          return (
            <g key={`${t.from}-${t.to}`}>
              <line
                x1={from.x + 40}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                className="stroke-zinc-300 dark:stroke-zinc-600"
                strokeWidth="1.5"
                markerEnd="url(#fsm-arrow)"
              />
              <text
                x={(from.x + 40 + to.x) / 2}
                y={(from.y + to.y) / 2 - 6}
                textAnchor="middle"
                className="fill-zinc-400 text-[7px] font-mono dark:fill-zinc-500"
              >
                {t.label}
              </text>
            </g>
          );
        })}
        {states.map((state) => {
          const isActive = state.id === activeState;
          return (
            <g key={state.id}>
              <motion.rect
                x={state.x}
                y={state.y - 14}
                width="80"
                height="28"
                rx="14"
                animate={{
                  fill: isActive ? color : "#d4d4d8",
                  opacity: isActive ? 1 : 0.6,
                }}
                className="stroke-zinc-400 dark:stroke-zinc-500"
                strokeWidth="1"
              />
              <text
                x={state.x + 40}
                y={state.y + 3}
                textAnchor="middle"
                dominantBaseline="middle"
                className={`text-[9px] font-medium ${
                  isActive ? "fill-white" : "fill-zinc-600 dark:fill-zinc-300"
                }`}
              >
                {state.label}
              </text>
            </g>
          );
        })}
        <defs>
          <marker
            id="fsm-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" className="fill-zinc-400 dark:fill-zinc-500" />
          </marker>
        </defs>
      </svg>
      <button
        onClick={onAdvance}
        className="mt-1 w-full rounded bg-zinc-200 py-1 text-xs hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
      >
        Advance
      </button>
    </div>
  );
}

export default function TeamProtocols() {
  const [shutdownIdx, setShutdownIdx] = useState(0);
  const [planIdx, setPlanIdx] = useState(0);

  const shutdownOrder = ["pending", "requested", "approved"];
  const planOrder = ["planning", "submitted", "approved"];

  const advanceShutdown = useCallback(() => {
    setShutdownIdx((i) => (i + 1) % shutdownOrder.length);
  }, [shutdownOrder.length]);

  const advancePlan = useCallback(() => {
    setPlanIdx((i) => (i + 1) % planOrder.length);
  }, [planOrder.length]);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        FSM Team Protocols
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-3 text-center">
          <span className="rounded bg-purple-100 px-2 py-0.5 font-mono text-[10px] text-purple-700 dark:bg-purple-900/30 dark:text-purple-300">
            request_id: {REQUEST_ID}
          </span>
        </div>
        <div className="flex gap-4">
          <FSMDiagram
            title="Shutdown Protocol"
            states={SHUTDOWN_STATES}
            transitions={SHUTDOWN_TRANSITIONS}
            activeState={shutdownOrder[shutdownIdx]}
            onAdvance={advanceShutdown}
            color="#3b82f6"
          />
          <FSMDiagram
            title="Plan Approval Protocol"
            states={PLAN_STATES}
            transitions={PLAN_TRANSITIONS}
            activeState={planOrder[planIdx]}
            onAdvance={advancePlan}
            color="#10b981"
          />
        </div>
      </div>
    </section>
  );
}
