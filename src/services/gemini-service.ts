import type { getNationalGridSnapshot } from "./grid-snapshot";
import type { optimizeGridDispatch } from "./grid-optimizer";
import type { runMonteCarloSimulation } from "./monte-carlo";
import type { PolarStationState, PolarRiskResult } from "./polar-station";

declare const process: { env: Record<string, string | undefined> };

type NationalGridSnapshot = Awaited<ReturnType<typeof getNationalGridSnapshot>>;
type MonteCarloResult = ReturnType<typeof runMonteCarloSimulation>;
type GridOptimizerResult = ReturnType<typeof optimizeGridDispatch>;

const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models";
const GEMINI_TIMEOUT_MS = 20_000;
const GEMINI_MAX_OUTPUT_TOKENS = 900;

const SYSTEM_PROMPT = `
You are Grid Sentinel AI, an energy-management decision-support assistant for an isolated polar research station, aligned to SIH26061.

Primary objectives:
- Forecast station load under extreme polar weather.
- Integrate variable solar and wind generation.
- Protect battery reserve and critical loads.
- Optimize backup-generator/fuel use.
- Quantify uncertainty using scenario simulation.
- Explain recommendations clearly to a station operator.

Rules:
- Never invent telemetry, forecasts, probabilities, measurements, fuel levels, or operational facts.
- Use only supplied backend data.
- Clearly label synthetic/prototype values when the context says they are simulated.
- Never claim this prototype is utility-grade SCADA/EMS or connected to live Antarctic telemetry.
- Prefer critical-load continuity and reserve protection over aggressive fuel minimization.
- Do not issue unsafe physical-control commands; recommendations are advisory.
- If historical Texas/India replay context is supplied, treat it only as supporting research evidence, not as the current station state.
- If information is unavailable, explicitly say it is unavailable.
- Be concise but technically accurate.
`.trim();

interface GeminiGridAssistantInput {
  readonly question: string;
  readonly snapshot: NationalGridSnapshot;
  readonly monteCarlo: MonteCarloResult;
  readonly optimizer: GridOptimizerResult;
  readonly polarStation?: PolarStationContext;
  readonly texasReplay?: TexasReplayPromptContext;
}

export interface PolarStationContext {
  readonly dataType: "synthetic-prototype";
  readonly state: PolarStationState;
  readonly risk: PolarRiskResult;
  readonly optimizedRisk: PolarRiskResult;
}

export interface GeminiGridAssistantResponse {
  readonly content: string;
  readonly sources: readonly string[];
}

export interface TexasReplayPromptContext {
  readonly requested: boolean;
  readonly dataAvailable: boolean;
  readonly status: string;
  readonly unavailableFields: readonly string[];
  readonly replayStart?: string;
  readonly replayEnd?: string;
  readonly peakDemandMw?: number | string;
  readonly peakRenewableGenerationMw?: number | string;
  readonly minimumReserveMarginPercent?: number | string;
  readonly maximumBlackoutProbability?: number | string;
  readonly maximumBlackoutProbabilityHour?: unknown;
  readonly majorTimelineEvents?: readonly unknown[];
  readonly summaryStatistics?: unknown;
}

interface GeminiResponse {
  readonly candidates?: readonly {
    readonly content?: { readonly parts?: readonly { readonly text?: string }[] };
  }[];
  readonly error?: { readonly message?: string };
}

interface GeminiProviderDiagnostics {
  readonly httpStatus?: number;
  readonly responseBody?: string;
  readonly providerErrorMessage?: string;
  readonly timeoutMs?: number;
  readonly timedOut: boolean;
}

export class GeminiServiceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GeminiServiceError";
  }
}

export async function askGeminiGridAssistant(
  input: GeminiGridAssistantInput,
): Promise<GeminiGridAssistantResponse> {
  const response = await requestGemini(buildRequestBody(input));
  const content = extractText(response);

  return {
    content,
    sources: [
      "Gemini 2.5 Flash",
      ...(input.polarStation ? ["Polar Station Digital Twin", "Polar Risk & Dispatch Engine"] : []),
      "Monte Carlo Engine",
      "Grid Optimizer (research context)",
    ],
  };
}

function buildRequestBody(input: GeminiGridAssistantInput) {
  return {
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [{ role: "user", parts: [{ text: buildUserPrompt(input) }] }],
    generationConfig: { temperature: 0.2, topP: 0.9, maxOutputTokens: GEMINI_MAX_OUTPUT_TOKENS },
  };
}

function buildUserPrompt(input: GeminiGridAssistantInput): string {
  return [
    "Answer the operator question using only this backend-generated Grid Sentinel context.",
    "Do not calculate or invent values that are absent from the context.",
    "",
    `Operator question: ${input.question}`,
    "",
    "Backend context JSON:",
    JSON.stringify(buildGridContext(input), null, 2),
  ].join("\n");
}

function buildGridContext(input: GeminiGridAssistantInput) {
  return {
    deployment: "SIH26061 Polar Research Station Energy Management",
    nationalResearchContext: buildNationalSummary(input.snapshot),
    monteCarloResearchContext: input.monteCarlo,
    gridOptimizerResearchContext: buildOptimizerSummary(input.optimizer),
    ...(input.polarStation ? { polarStation: input.polarStation } : {}),
    ...(input.texasReplay ? { historicalResearchReplay: input.texasReplay } : {}),
  };
}

function buildNationalSummary(snapshot: NationalGridSnapshot) {
  return {
    timestamp: snapshot.timestamp,
    nationalDemandMw: snapshot.nationalDemandMw,
    nationalRenewableGenerationMw: snapshot.nationalRenewableGenerationMw,
    nationalReserveMarginPercent: snapshot.nationalReserveMarginPercent,
    nationalRenewablePenetrationPercent: snapshot.nationalRenewablePenetrationPercent,
    nationalGridStressIndex: snapshot.nationalGridStressIndex,
    averageDemandConfidence: snapshot.averageDemandConfidence,
    highestRiskState: snapshot.highestRiskState,
    lowestRiskState: snapshot.lowestRiskState,
    systemHealthScore: snapshot.systemHealthScore,
  };
}

function buildOptimizerSummary(optimizer: GridOptimizerResult) {
  return {
    generatedAt: optimizer.generatedAt,
    systemPriority: optimizer.systemPriority,
    projectedReserveMarginPercent: optimizer.projectedReserveMarginPercent,
    residualRiskScore: optimizer.residualRiskScore,
    totalBatteryDispatchMw: optimizer.totalBatteryDispatchMw,
    totalDemandResponseMw: optimizer.totalDemandResponseMw,
    totalReserveProcurementMw: optimizer.totalReserveProcurementMw,
    totalRenewableCurtailmentMw: optimizer.totalRenewableCurtailmentMw,
    recommendedActions: optimizer.recommendedActions,
  };
}

async function requestGemini(body: unknown): Promise<GeminiResponse> {
  const apiKey = readApiKey();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), GEMINI_TIMEOUT_MS);
  const requestBody = JSON.stringify(body);
  try {
    const response = await fetch(buildGeminiUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
      body: requestBody,
      signal: controller.signal,
    });
    const responseBody = await response.text();
    const data = parseGeminiResponse(responseBody);
    if (!response.ok) {
      logGeminiProviderError({ httpStatus: response.status, responseBody, providerErrorMessage: data.error?.message, timeoutMs: GEMINI_TIMEOUT_MS, timedOut: false });
      throw new GeminiServiceError("Gemini request failed. Please retry later.");
    }
    return data;
  } catch (error) {
    if (isAbortError(error)) {
      logGeminiProviderError({ providerErrorMessage: "Gemini request aborted by server timeout.", timeoutMs: GEMINI_TIMEOUT_MS, timedOut: true });
      throw new GeminiServiceError("Gemini request timed out. Please retry.");
    }
    if (error instanceof GeminiServiceError) throw error;
    logGeminiProviderError({ providerErrorMessage: error instanceof Error ? error.message : "Unknown Gemini request failure.", timeoutMs: GEMINI_TIMEOUT_MS, timedOut: false });
    throw new GeminiServiceError("Gemini is unavailable. Please retry later.");
  } finally {
    clearTimeout(timeout);
  }
}

function readApiKey(): string {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new GeminiServiceError("Gemini is not configured.");
  return apiKey;
}

function buildGeminiUrl(): string {
  return `${GEMINI_API_BASE_URL}/${GEMINI_MODEL}:generateContent`;
}

function parseGeminiResponse(responseBody: string): GeminiResponse {
  try { return JSON.parse(responseBody) as GeminiResponse; }
  catch { throw new GeminiServiceError("Gemini returned an unreadable response."); }
}

function extractText(response: GeminiResponse): string {
  const text = response.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("").trim();
  if (!text) throw new GeminiServiceError("Gemini returned no response text.");
  return text;
}

function logGeminiProviderError(diagnostics: GeminiProviderDiagnostics): void {
  console.error("Gemini provider error", diagnostics);
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
