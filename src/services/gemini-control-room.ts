import { createServerFn } from "@tanstack/react-start";
import { getRequestIP } from "@tanstack/start-server-core";
import { askGeminiGridAssistant, type PolarStationContext } from "./gemini-service";
import { getPolarStationState, optimizePolarDispatch, runPolarRiskSimulation, type PolarScenario } from "./polar-station";

const WINDOW_MS = 60_000;
const MAX_REQUESTS = 10;
const TIMEOUT_MS = 30_000;
const FALLBACK_KEY = "unknown-client";
const buckets = new Map<string, { count: number; resetAt: number }>();

interface Question { readonly question: string; }

export const askGeminiControlRoom = createServerFn({ method: "POST" })
  .validator((input: unknown): Question => {
    if (!input || typeof input !== "object") throw new Error("Question is required.");
    const question = (input as { question?: unknown }).question;
    if (typeof question !== "string" || !question.trim()) throw new Error("Question is required.");
    return { question: question.trim() };
  })
  .handler(async ({ data }) => {
    enforceRateLimit();
    return withTimeout(async () => {
      const scenario = detectPolarScenario(data.question);
      const station = getPolarStationState(scenario);
      const risk = runPolarRiskSimulation(station);
      const optimizedRisk = optimizePolarDispatch(station);
      const polarStation: PolarStationContext = {
        dataType: "synthetic-prototype",
        state: station,
        risk,
        optimizedRisk,
      };

      return askGeminiGridAssistant({ question: data.question, polarStation });
    }, TIMEOUT_MS);
  });

function detectPolarScenario(question: string): PolarScenario {
  const q = question.toLowerCase();
  if (/low.?light|polar night|dark|solar/.test(q)) return "low-light";
  if (/wind|derat|calm/.test(q)) return "wind-derating";
  if (/storm|blizzard|extreme|cold|weather/.test(q)) return "polar-storm";
  return "nominal";
}

function enforceRateLimit(): void {
  const now = Date.now();
  const key = readKey();
  const bucket = buckets.get(key);
  if (!bucket || bucket.resetAt <= now) {
    buckets.set(key, { count: 1, resetAt: now + WINDOW_MS });
    return;
  }
  if (bucket.count >= MAX_REQUESTS) throw new Error("Too many AI requests. Please wait before retrying.");
  bucket.count += 1;
}

function readKey(): string {
  try { return getRequestIP({ xForwardedFor: true }) ?? FALLBACK_KEY; }
  catch { return FALLBACK_KEY; }
}

async function withTimeout<T>(fn: () => Promise<T>, timeoutMs: number): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      fn(),
      new Promise<T>((_, reject) => { timeout = setTimeout(() => reject(new Error("AI request timed out. Please retry.")), timeoutMs); }),
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}
