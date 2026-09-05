import { createFileRoute } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useEffect, useRef, useState } from "react";
import { Send, Sparkles } from "lucide-react";
import { askGeminiControlRoom } from "@/services/gemini-control-room";

export const Route = createFileRoute("/control-room")({
  head: () => ({
    meta: [
      { title: "Polar AI Control Room · Grid Sentinel AI" },
      { name: "description", content: "AI decision support for polar research station energy operations." },
    ],
  }),
  component: ControlRoom,
});

type Turn = { role: "operator" | "sentinel"; content: string; sources?: readonly string[]; ts: string };

const SUGGESTIONS = [
  "Why is station shortage risk increasing?",
  "How should we preserve battery reserve during a polar storm?",
  "How much critical load can be supported if wind derates?",
  "How can we reduce diesel fuel consumption?",
  "What dispatch should we use during low-light conditions?",
  "What is the safest response if renewable output drops sharply?",
];

function ts() { return new Date().toTimeString().slice(0, 8); }
function errorMessage(error: unknown) { return error instanceof Error && error.message ? error.message : "Gemini is unavailable. Please retry later."; }

function ControlRoom() {
  const askGemini = useServerFn(askGeminiControlRoom);
  const [turns, setTurns] = useState<Turn[]>([{
    role: "sentinel",
    content: "Polar Sentinel AI online. Gemini 2.5 Flash is grounded in the SIH26061 Polar Station Digital Twin. Ask about shortage risk, renewable availability, battery reserve, critical loads, or fuel-aware dispatch.",
    sources: ["Gemini 2.5 Flash", "Polar Station Digital Twin"],
    ts: ts(),
  }]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [lastFailedQuestion, setLastFailedQuestion] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }); }, [turns, thinking, lastFailedQuestion]);

  async function ask(q: string) {
    const question = q.trim();
    if (!question || thinking) return;
    setTurns((t) => [...t, { role: "operator", content: question, ts: ts() }]);
    setInput(""); setLastFailedQuestion(null); setThinking(true);
    try {
      const answer = await askGemini({ data: { question } });
      setTurns((t) => [...t, { role: "sentinel", content: answer.content, sources: answer.sources, ts: ts() }]);
    } catch (error) {
      setLastFailedQuestion(question);
      setTurns((t) => [...t, { role: "sentinel", content: `Gemini is unavailable: ${errorMessage(error)}\n\nNo response was generated.`, sources: ["Gemini 2.5 Flash"], ts: ts() }]);
    } finally { setThinking(false); }
  }

  return (
    <div className="px-6 py-6 grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
      <div className="panel p-0 flex flex-col h-[calc(100vh-12rem)] min-h-[640px] overflow-hidden">
        <div className="px-6 py-4 border-b border-[oklch(0.72_0.18_245/0.15)] flex items-center justify-between">
          <div className="flex items-center gap-3"><div className="relative w-9 h-9 grid place-items-center"><div className="absolute inset-0 rounded-full border border-[oklch(0.85_0.21_145/0.4)] animate-pulse-ring" /><Sparkles className="w-5 h-5 text-[oklch(0.85_0.21_145)]" /></div><div><div className="hud-label">AI Control Room</div><div className="font-display">Polar Station Operator Interface</div></div></div>
          <div className="text-[10px] font-mono text-muted-foreground">GEMINI 2.5 FLASH · SERVER-SIDE · ADVISORY</div>
        </div>
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {turns.map((t, i) => <div key={i} className="animate-fade-up"><div className="flex items-center gap-2 mb-1.5"><span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ color: t.role === "sentinel" ? "oklch(0.85 0.21 145)" : "oklch(0.72 0.18 245)", background: t.role === "sentinel" ? "oklch(0.85 0.21 145 / 0.1)" : "oklch(0.72 0.18 245 / 0.1)", border: `1px solid ${t.role === "sentinel" ? "oklch(0.85 0.21 145 / 0.35)" : "oklch(0.72 0.18 245 / 0.35)"}` }}>{t.role === "sentinel" ? "SENTINEL" : "OPERATOR"}</span><span className="text-[10px] font-mono text-muted-foreground">{t.ts}</span></div><div className="whitespace-pre-line text-sm leading-relaxed">{t.content}</div>{t.sources && <div className="mt-2 flex flex-wrap gap-1.5">{t.sources.map((s) => <span key={s} className="text-[10px] font-mono px-2 py-0.5 rounded bg-[oklch(0.16_0.028_260/0.7)] border border-[oklch(0.72_0.18_245/0.15)] text-muted-foreground">&gt; {s}</span>)}</div>}</div>)}
          {thinking && <div className="animate-fade-up flex items-center gap-2 text-[oklch(0.85_0.21_145)]"><span className="w-1.5 h-1.5 rounded-full bg-[oklch(0.85_0.21_145)] animate-flicker" /><span className="font-mono text-[11px]">Sentinel grounding Gemini in Polar Station context...</span></div>}
          {lastFailedQuestion && !thinking && <button type="button" onClick={() => ask(lastFailedQuestion)} className="text-[10px] font-mono px-2 py-1 rounded border border-[oklch(0.72_0.18_245/0.25)] text-[oklch(0.72_0.18_245)]">Retry last request</button>}
        </div>
        <div className="border-t border-[oklch(0.72_0.18_245/0.15)] p-4"><form onSubmit={(e) => { e.preventDefault(); ask(input); }} className="flex items-center gap-2"><input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about station risk, battery reserve, renewables, critical load, or fuel..." className="flex-1 bg-[oklch(0.16_0.028_260/0.7)] border border-[oklch(0.72_0.18_245/0.25)] rounded-lg px-4 py-3 text-sm font-mono placeholder:text-muted-foreground focus:outline-none focus:border-[oklch(0.72_0.18_245)]" /><button type="submit" className="px-4 py-3 rounded-lg bg-[oklch(0.72_0.18_245)] text-[oklch(0.1_0.02_260)] font-medium text-sm flex items-center gap-2"><Send size={14} /> Ask Sentinel</button></form></div>
      </div>
      <aside className="space-y-4">
        <div className="panel p-4"><div className="hud-label mb-3">Suggested queries</div><div className="space-y-2">{SUGGESTIONS.map((s) => <button key={s} onClick={() => ask(s)} className="w-full text-left p-3 rounded-lg text-sm border border-[oklch(0.72_0.18_245/0.12)] hover:border-[oklch(0.72_0.18_245/0.5)] transition-colors">{s}</button>)}</div></div>
        <div className="panel p-4"><div className="hud-label mb-2">AI grounding</div><div className="space-y-2 text-xs font-mono"><Ctx k="Deployment" v="SIH26061" tone="accent" /><Ctx k="Station model" v="Synthetic prototype" tone="primary" /><Ctx k="Decision mode" v="Advisory" tone="warning" /><Ctx k="Telemetry" v="Not live" tone="destructive" /><Ctx k="Primary assets" v="PV · Wind · BESS · Generator" tone="primary" /></div></div>
      </aside>
    </div>
  );
}

function Ctx({ k, v, tone }: { k: string; v: string; tone: "primary" | "accent" | "warning" | "destructive" }) {
  const c = { primary: "oklch(0.72 0.18 245)", accent: "oklch(0.85 0.21 145)", warning: "oklch(0.82 0.17 75)", destructive: "oklch(0.68 0.24 25)" }[tone];
  return <div className="flex items-center justify-between p-2 rounded bg-[oklch(0.16_0.028_260/0.6)] border border-[oklch(0.72_0.18_245/0.08)]"><span className="text-muted-foreground">{k}</span><span style={{ color: c }}>{v}</span></div>;
}
