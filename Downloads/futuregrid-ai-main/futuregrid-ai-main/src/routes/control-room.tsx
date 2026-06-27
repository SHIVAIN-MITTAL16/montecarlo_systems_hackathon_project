import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Send, Sparkles } from "lucide-react";

export const Route = createFileRoute("/control-room")({
  head: () => ({
    meta: [
      { title: "AI Control Room · Grid Sentinel AI" },
      { name: "description", content: "Enterprise AI intelligence interface for grid operators — diagnose risk, prescribe action, prevent blackouts." },
      { property: "og:title", content: "AI Control Room · Grid Sentinel AI" },
      { property: "og:description", content: "Predict. Simulate. Optimize. Prevent." },
    ],
  }),
  component: ControlRoom,
});

type Turn = { role: "operator" | "sentinel"; content: string; sources?: string[]; ts: string };

const SUGGESTIONS = [
  "Why is risk increasing in Maharashtra?",
  "Which state is most vulnerable in the next 24 hours?",
  "What action should be taken to stabilize Delhi?",
  "How can we reduce blackout probability nationally?",
  "Where should the next BESS investment go?",
  "What if the Mundra coal unit trips at 19:00?",
];

const ANSWERS: Record<string, { content: string; sources: string[] }> = {
  "Why is risk increasing in Maharashtra?": {
    content:
      "Maharashtra's composite risk has climbed to 81 over the past 90 minutes, driven by three converging factors.\n\nFirst, AC load in the MMR cluster is tracking 2.8 GW above the morning forecast band — humidity-corrected demand is responding non-linearly to a 4.1°C wet-bulb excursion. Second, the Tarapur–Boisar 400 kV corridor is at 92% headroom, limiting import options from Gujarat. Third, scheduled BESS at Dhule is at 52% SoC after an unscheduled morning discharge.\n\nThe model expects risk to peak near 88 by 19:40 IST. Recommended action: shed 600 MW of pre-enrolled industrial load (DR Tier 2), pre-dispatch the Uran gas peakers at 17:30, and pause Dhule charging until 22:00.",
    sources: ["POSOCO RLDC-West telemetry", "IMD wet-bulb advisory", "Sentinel-TS forecast horizon 6h"],
  },
  "Which state is most vulnerable in the next 24 hours?": {
    content:
      "Top-three vulnerability ranking (next 24h):\n\n1. Maharashtra — 81 risk · 24% blackout probability. Demand surge + corridor saturation.\n2. Delhi — 78 risk · 21% blackout probability. Transformer T-44 thermal margin narrowing.\n3. Uttar Pradesh — 64 risk · 14% blackout probability. NR pool import dependency rising.\n\nThe western corridor is the dominant concern. A coordinated DR + BESS strategy across MH–GJ would reduce composite national risk by an estimated 9.2 points.",
    sources: ["Sentinel-TS national rollup", "Monte Carlo · 10k scenarios"],
  },
  "What action should be taken to stabilize Delhi?": {
    content:
      "Three-step stabilization plan for Delhi (confidence 94%):\n\n1. Immediate (0–10 min): Engage secondary cooling on T-44; redirect 180 MW from Dadri-II through Mandola.\n2. Near-term (10–60 min): Arm DR Tier 2 (commercial cluster, 220 MW enrolled).\n3. Evening peak (17:30–21:00): Pre-dispatch Bawana CCGT at 78% load factor; hold Pragati-III in spinning reserve.\n\nExpected outcome: frequency band returns to 49.95–50.02 Hz; blackout probability falls from 21% to 6%.",
    sources: ["NR RLDC SCADA", "BSES Rajdhani DR roster", "Sentinel-TS prescriptive engine"],
  },
  "How can we reduce blackout probability nationally?": {
    content:
      "National blackout probability is 11% over the next 24h. Three high-leverage actions:\n\n• Coordinate inter-regional pool transfers (NR ↔ WR) — unlocks 1.8 GW of headroom.\n• Pre-position 3.4 GW of BESS in charge state by 16:00 IST.\n• Activate DR Tier 1 in Delhi, Mumbai, Bengaluru — 940 MW enrolled, <8 min ramp.\n\nCombined impact: probability falls to 4.2%. Estimated cost: ₹38 Cr. Avoided economic loss: ₹2,140 Cr.",
    sources: ["NLDC scheduling", "Sentinel-TS prescriptive engine", "MoP DR registry"],
  },
};

function defaultAnswer(q: string) {
  return {
    content:
      `Analyzing "${q}".\n\nSentinel-TS has run 10,240 scenarios in 18 ms across the national digital twin. Composite risk sits at 58/100, with western and northern corridors carrying the dominant tail risk.\n\nThe model recommends a hybrid response: pre-dispatch 1.4 GW of gas peakers, charge 2.1 GW of BESS into the evening peak, and arm DR Tier 1 in Mumbai and Delhi. This reduces 24h blackout probability from 11% to 4.6% at an estimated cost of ₹26 Cr.`,
    sources: ["Sentinel-TS national rollup", "Monte Carlo · 10k scenarios", "POSOCO live telemetry"],
  };
}

function ts() { return new Date().toTimeString().slice(0, 8); }

function ControlRoom() {
  const [turns, setTurns] = useState<Turn[]>([
    {
      role: "sentinel",
      content: "Grid Sentinel AI online. National risk index 58/100. Three states under elevated stress — Maharashtra, Delhi, Uttar Pradesh. Ask me anything about the grid state, risk drivers, or prescriptive actions.",
      sources: ["System bootstrap · model v7.3"],
      ts: ts(),
    },
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, thinking]);

  function ask(q: string) {
    if (!q.trim()) return;
    setTurns((t) => [...t, { role: "operator", content: q, ts: ts() }]);
    setInput("");
    setThinking(true);
    const a = ANSWERS[q] ?? defaultAnswer(q);
    setTimeout(() => {
      setTurns((t) => [...t, { role: "sentinel", content: a.content, sources: a.sources, ts: ts() }]);
      setThinking(false);
    }, 900);
  }

  return (
    <div className="px-6 py-6 grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
      <div className="panel p-0 flex flex-col h-[calc(100vh-12rem)] min-h-[640px] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[oklch(0.72_0.18_245/0.15)] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative w-9 h-9 grid place-items-center">
              <div className="absolute inset-0 rounded-full border border-[oklch(0.85_0.21_145/0.4)] animate-pulse-ring" />
              <Sparkles className="w-5 h-5 text-[oklch(0.85_0.21_145)]" />
            </div>
            <div>
              <div className="hud-label">AI Control Room</div>
              <div className="font-display">Sentinel · Operator Interface</div>
            </div>
          </div>
          <div className="text-[10px] font-mono text-muted-foreground">
            MODEL TS-49 · CONTEXT 412M · LATENCY 18 ms
          </div>
        </div>

        {/* Conversation */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {turns.map((t, i) => (
            <div key={i} className="animate-fade-up">
              <div className="flex items-center gap-2 mb-1.5">
                <span
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                  style={{
                    color: t.role === "sentinel" ? "oklch(0.85 0.21 145)" : "oklch(0.72 0.18 245)",
                    background: t.role === "sentinel" ? "oklch(0.85 0.21 145 / 0.1)" : "oklch(0.72 0.18 245 / 0.1)",
                    border: `1px solid ${t.role === "sentinel" ? "oklch(0.85 0.21 145 / 0.35)" : "oklch(0.72 0.18 245 / 0.35)"}`,
                  }}
                >
                  {t.role === "sentinel" ? "SENTINEL" : "OPERATOR"}
                </span>
                <span className="text-[10px] font-mono text-muted-foreground">{t.ts}</span>
              </div>
              <div className={`whitespace-pre-line text-sm leading-relaxed ${t.role === "sentinel" ? "text-foreground" : "text-foreground/90"}`}>
                {t.content}
              </div>
              {t.sources && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {t.sources.map((s) => (
                    <span key={s} className="text-[10px] font-mono px-2 py-0.5 rounded bg-[oklch(0.16_0.028_260/0.7)] border border-[oklch(0.72_0.18_245/0.15)] text-muted-foreground">
                      ▸ {s}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {thinking && (
            <div className="animate-fade-up flex items-center gap-2 text-[oklch(0.85_0.21_145)] text-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-[oklch(0.85_0.21_145)] animate-flicker" />
              <span className="font-mono text-[11px]">Sentinel solving · simulating 10,240 scenarios…</span>
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-[oklch(0.72_0.18_245/0.15)] p-4">
          <form
            onSubmit={(e) => { e.preventDefault(); ask(input); }}
            className="flex items-center gap-2"
          >
            <div className="flex-1 relative">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask Sentinel — risk drivers, prescriptive actions, contingency plans…"
                className="w-full bg-[oklch(0.16_0.028_260/0.7)] border border-[oklch(0.72_0.18_245/0.25)] rounded-lg px-4 py-3 pr-12 text-sm font-mono placeholder:text-muted-foreground focus:outline-none focus:border-[oklch(0.72_0.18_245)] focus:shadow-[0_0_24px_-6px_oklch(0.72_0.18_245/0.6)]"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-mono text-muted-foreground">⌘ + ⏎</span>
            </div>
            <button
              type="submit"
              className="px-4 py-3 rounded-lg bg-[oklch(0.72_0.18_245)] text-[oklch(0.1_0.02_260)] font-medium text-sm hover:shadow-[0_0_24px_-4px_oklch(0.72_0.18_245)] flex items-center gap-2"
            >
              <Send size={14} /> Dispatch
            </button>
          </form>
        </div>
      </div>

      {/* Right: suggestions + state */}
      <aside className="space-y-4">
        <div className="panel p-4">
          <div className="hud-label mb-3">Suggested queries</div>
          <div className="space-y-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => ask(s)}
                className="w-full text-left p-3 rounded-lg text-sm border border-[oklch(0.72_0.18_245/0.12)] hover:border-[oklch(0.72_0.18_245/0.5)] hover:bg-[oklch(0.72_0.18_245/0.06)] transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
        <div className="panel p-4">
          <div className="hud-label mb-2">Live context</div>
          <div className="space-y-1.5 text-xs font-mono">
            <Ctx k="National Risk" v="58 / 100" tone="warning" />
            <Ctx k="Frequency" v="49.98 Hz" tone="accent" />
            <Ctx k="RE share" v="42 %" tone="accent" />
            <Ctx k="Blackout 24h" v="11 %" tone="destructive" />
            <Ctx k="DR enrolled" v="6.2 GW" tone="primary" />
            <Ctx k="BESS SoC mean" v="58 %" tone="primary" />
          </div>
        </div>
      </aside>
    </div>
  );
}

function Ctx({ k, v, tone }: { k: string; v: string; tone: "primary" | "accent" | "warning" | "destructive" }) {
  const c = {
    primary: "oklch(0.72 0.18 245)",
    accent: "oklch(0.85 0.21 145)",
    warning: "oklch(0.82 0.17 75)",
    destructive: "oklch(0.68 0.24 25)",
  }[tone];
  return (
    <div className="flex items-center justify-between p-2 rounded bg-[oklch(0.16_0.028_260/0.6)] border border-[oklch(0.72_0.18_245/0.08)]">
      <span className="text-muted-foreground">{k}</span>
      <span style={{ color: c }}>{v}</span>
    </div>
  );
}
