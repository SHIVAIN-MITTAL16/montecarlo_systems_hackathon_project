import type { PolarStationState, PolarRiskResult } from "./polar-station";

declare const process: { env: Record<string, string | undefined> };

const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models";
const GEMINI_TIMEOUT_MS = 20_000;
const GEMINI_MAX_OUTPUT_TOKENS = 900;

const SYSTEM_PROMPT = `You are Grid Sentinel AI, an energy-management decision-support assistant for an isolated polar research station, aligned to SIH26061.
Primary objectives: load forecasting, renewable integration, battery reserve protection, critical-load continuity, and fuel optimization under extreme polar conditions.
Rules: never invent telemetry, forecasts, probabilities, measurements, fuel levels, or operational facts; use only supplied backend data; label synthetic prototype values; never claim live Antarctic telemetry or utility-grade SCADA/EMS; recommendations are advisory and must prioritize critical loads and reserve protection; historical India/Texas data is research context only; if information is unavailable, say so explicitly.`.trim();

export interface PolarStationContext {
  readonly dataType: "synthetic-prototype";
  readonly state: PolarStationState;
  readonly risk: PolarRiskResult;
  readonly optimizedRisk: PolarRiskResult;
}

interface Input { readonly question: string; readonly polarStation: PolarStationContext; }
interface Response { readonly candidates?: readonly { readonly content?: { readonly parts?: readonly { readonly text?: string }[] } }[]; readonly error?: { readonly message?: string }; }

export interface GeminiGridAssistantResponse { readonly content: string; readonly sources: readonly string[]; }

export class GeminiServiceError extends Error {
  constructor(message: string) { super(message); this.name = "GeminiServiceError"; }
}

export async function askGeminiGridAssistant(input: Input): Promise<GeminiGridAssistantResponse> {
  const response = await requestGemini({
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [{ role: "user", parts: [{ text: buildPrompt(input) }] }],
    generationConfig: { temperature: 0.2, topP: 0.9, maxOutputTokens: GEMINI_MAX_OUTPUT_TOKENS },
  });
  const content = response.candidates?.[0]?.content?.parts?.map((p) => p.text ?? "").join("").trim();
  if (!content) throw new GeminiServiceError("Gemini returned no response text.");
  return { content, sources: ["Gemini 2.5 Flash", "Polar Station Digital Twin", "Polar Risk & Dispatch Engine"] };
}

function buildPrompt(input: Input): string {
  return [
    "Answer the operator question using only the supplied Polar Station backend context.",
    "Do not calculate or invent values absent from the context.",
    "All station values are synthetic prototype data unless explicitly stated otherwise.",
    `Operator question: ${input.question}`,
    "Polar Station backend context:",
    JSON.stringify({ deployment: "SIH26061 Polar Research Station Energy Management", polarStation: input.polarStation }, null, 2),
  ].join("\n");
}

async function requestGemini(body: unknown): Promise<Response> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new GeminiServiceError("Gemini is not configured.");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), GEMINI_TIMEOUT_MS);
  try {
    const response = await fetch(`${GEMINI_API_BASE_URL}/${GEMINI_MODEL}:generateContent`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    let data: Response;
    try { data = JSON.parse(text) as Response; } catch { throw new GeminiServiceError("Gemini returned an unreadable response."); }
    if (!response.ok) throw new GeminiServiceError("Gemini request failed. Please retry later.");
    return data;
  } catch (error) {
    if (error instanceof GeminiServiceError) throw error;
    if (error instanceof Error && error.name === "AbortError") throw new GeminiServiceError("Gemini request timed out. Please retry.");
    throw new GeminiServiceError("Gemini is unavailable. Please retry later.");
  } finally { clearTimeout(timeout); }
}
