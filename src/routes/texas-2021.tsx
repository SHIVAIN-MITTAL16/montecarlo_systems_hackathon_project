import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { Snowflake, AlertTriangle, Zap, Activity, Brain, Lightbulb } from "lucide-react";

export const Route = createFileRoute("/texas-2021")({
  head: () => ({
    meta: [
      { title: "Texas Winter Storm Uri 2021 · Grid Sentinel AI" },
      { name: "description", content: "An interactive replay of the February 2021 Texas grid collapse — demand vs supply, generation mix, blackout progression, and the prescriptive actions Grid Sentinel AI would have recommended." },
      { property: "og:title", content: "Texas Winter Storm Uri · Replay" },
      { property: "og:description", content: "Inside the 2021 ERCOT collapse — modeled, animated, and explained." },
    ],
  }),
  component: TexasReplay,
});

/* ---------- Timeline (placeholder narrative; numbers illustrative) ---------- */
type Phase = {
  t: string;             // local Texas time stamp
  day: string;           // human label
  title: string;
  detail: string;
  demandGW: number;      // illustrative
  supplyGW: number;
  outageGW: number;      // unserved load
  freqHz: number;
  state: "watch" | "warning" | "critical" | "recovery";
};

const TIMELINE: Phase[] = [
  { t: "Feb 10 · 18:00", day: "T−4d", title: "Cold front advisory issued",
    detail: "NWS warns of historic arctic mass entering Texas. ERCOT issues operating condition notice.",
    demandGW: 55, supplyGW: 70, outageGW: 0, freqHz: 60.00, state: "watch" },
  { t: "Feb 14 · 22:00", day: "T−1d", title: "Demand climbing, gas pressure dropping",
    detail: "Natural gas wellhead freeze-offs begin in Permian Basin. Wind generation iced over.",
    demandGW: 69, supplyGW: 67, outageGW: 0, freqHz: 59.96, state: "warning" },
  { t: "Feb 15 · 01:25", day: "Day 1", title: "EEA Level 3 declared",
    detail: "ERCOT orders rolling blackouts. Frequency drift toward 59.4 Hz threatens cascading trip.",
    demandGW: 76, supplyGW: 62, outageGW: 14, freqHz: 59.40, state: "critical" },
  { t: "Feb 15 · 02:30", day: "Day 1", title: "Forced load shed initiated",
    detail: "Approx. 4.5M customers offline. Generation losses peak at 52 GW (gas, coal, nuclear, wind).",
    demandGW: 78, supplyGW: 55, outageGW: 23, freqHz: 59.30, state: "critical" },
  { t: "Feb 16 · 12:00", day: "Day 2", title: "Sustained shed continues",
    detail: "Grid held off cascading collapse by minutes. Water systems and hospitals losing backup.",
    demandGW: 74, supplyGW: 58, outageGW: 16, freqHz: 59.65, state: "critical" },
  { t: "Feb 18 · 09:00", day: "Day 4", title: "Recovery phase",
    detail: "Generation gradually restored as thaw begins. ERCOT exits EEA-3 by evening.",
    demandGW: 64, supplyGW: 65, outageGW: 3, freqHz: 59.92, state: "recovery" },
  { t: "Feb 19 · 21:00", day: "Day 5", title: "Conservation appeal lifted",
    detail: "246 confirmed deaths, USD 195B in damages — the costliest weather disaster in Texas history.",
    demandGW: 58, supplyGW: 68, outageGW: 0, freqHz: 60.00, state: "recovery" },
];

const SENTINEL_TIMELINE: { offset: string; action: string; rationale: string }[] = [
  { offset: "T−96h", action: "Pre-stage 8.4 GW of dispatchable reserves",
    rationale: "Ensemble weather models converge on 4σ cold anomaly · 92% confidence" },
  { offset: "T−72h", action: "Mandate wellhead winterization on Permian gas supply",
    rationale: "Historical correlation: −10°C ambient → 18 GW gas freeze-off probability 0.71" },
  { offset: "T−48h", action: "Import 6 GW via SPP & MISO DC-tie pre-purchase",
    rationale: "Probabilistic supply gap of 14–22 GW exceeds intra-ERCOT reserve margin" },
  { offset: "T−24h", action: "Issue voluntary conservation order + activate 3.2 GW DR",
    rationale: "Shaves projected peak by 4.1 GW · avoids EEA-3 escalation in 78% of scenarios" },
  { offset: "T−6h",  action: "Begin controlled, rotating load curtailment of 2 GW",
    rationale: "Distributed shed prevents concentrated 23 GW emergency drop · saves est. 180 lives" },
  { offset: "T+0",   action: "Maintain BESS dispatch + cold-load pickup sequencing",
    rationale: "Restores 60.00 Hz within 4 h vs observed 96 h. Critical loads never lose power." },
];

function TexasReplay() {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const ref = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) return;
    ref.current = window.setInterval(() => {
      setIdx((i) => {
        if (i >= TIMELINE.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, 2200);
    return () => { if (ref.current) clearInterval(ref.current); };
  }, [playing]);

  const phase = TIMELINE[idx];
  const tone = stateTone(phase.state);

  return (
    <div className="px-6 py-8 max-w-[1500px] mx-auto space-y-10">
      {/* ===== Hero ===== */}
      <header className="relative panel p-8 overflow-hidden">
        <div className="absolute inset-0 pointer-events-none opacity-40"
             style={{ background: "radial-gradient(ellipse 60% 50% at 100% 0%, oklch(0.72 0.18 245 / 0.35), transparent 70%)" }} />
        <Snowflake className="absolute top-6 right-6 text-[oklch(0.82_0.14_200)]/40" size={120} strokeWidth={0.6} />
        <div className="relative">
          <div className="hud-label">Case Study · February 2021</div>
          <h1 className="display-lg mt-2 font-display">
            Texas Winter Storm <span className="text-[oklch(0.82_0.14_200)]">Uri</span>
          </h1>
          <p className="text-muted-foreground max-w-2xl mt-4">
            A 4σ arctic intrusion collapsed 52 GW of generation in 72 hours. ERCOT came within
            <span className="text-[oklch(0.68_0.24_25)]"> 4 minutes and 37 seconds </span>
            of an uncontrolled cascading blackout that would have darkened Texas for months. This
            is the storm — replayed through the eyes of Grid Sentinel AI.
          </p>
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3 max-w-3xl">
            <HeroStat label="Peak unserved load" value="23 GW" tone="critical" />
            <HeroStat label="Customers offline"  value="4.5 M" tone="critical" />
            <HeroStat label="Confirmed fatalities" value="246" tone="warning" />
            <HeroStat label="Total damages" value="USD 195 B" tone="warning" />
          </div>
        </div>
      </header>

      {/* ===== Timeline Replay ===== */}
      <section className="panel p-6">
        <div className="flex items-baseline justify-between mb-5">
          <div>
            <div className="hud-label">Interactive replay</div>
            <h2 className="display-md font-display">Timeline · the 96 hours that broke ERCOT</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setPlaying((p) => !p); }}
              className="px-3 py-1.5 rounded border border-[oklch(0.72_0.18_245/0.35)] text-xs font-mono hover:bg-[oklch(0.72_0.18_245/0.1)] transition"
            >
              {playing ? "❚❚ Pause" : "▶ Play"}
            </button>
            <button
              onClick={() => { setIdx(0); setPlaying(false); }}
              className="px-3 py-1.5 rounded border border-[oklch(0.72_0.18_245/0.2)] text-xs font-mono text-muted-foreground hover:text-foreground transition"
            >
              ⤺ Restart
            </button>
          </div>
        </div>

        {/* phase tracker */}
        <div className="relative mb-6">
          <div className="absolute left-0 right-0 top-1/2 h-px bg-[oklch(0.72_0.18_245/0.18)]" />
          <div className="absolute left-0 top-1/2 h-px transition-all duration-700"
               style={{ width: `${(idx / (TIMELINE.length - 1)) * 100}%`, background: tone, boxShadow: `0 0 12px ${tone}` }} />
          <div className="relative flex justify-between">
            {TIMELINE.map((p, i) => {
              const active = i === idx;
              const past = i < idx;
              return (
                <button
                  key={i}
                  onClick={() => { setIdx(i); setPlaying(false); }}
                  className="group flex flex-col items-center"
                >
                  <span
                    className="block w-3.5 h-3.5 rounded-full border-2 transition"
                    style={{
                      background: past || active ? stateTone(p.state) : "#070B14",
                      borderColor: past || active ? stateTone(p.state) : "oklch(0.72 0.18 245 / 0.35)",
                      boxShadow: active ? `0 0 16px ${stateTone(p.state)}` : "none",
                    }}
                  />
                  <span className={`mt-2 text-[10px] font-mono ${active ? "text-foreground" : "text-muted-foreground"}`}>
                    {p.day}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* phase detail + live readouts */}
        <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="font-mono text-xs text-muted-foreground">{phase.t}</span>
              <span className="px-2 py-0.5 text-[10px] font-mono rounded"
                    style={{ background: tone + "22", color: tone, border: `1px solid ${tone}` }}>
                {phase.state.toUpperCase()}
              </span>
            </div>
            <h3 className="display-md font-display mb-2">{phase.title}</h3>
            <p className="text-sm text-muted-foreground leading-relaxed max-w-xl">{phase.detail}</p>

            <div className="mt-6 grid grid-cols-2 gap-3">
              <Readout label="Demand"  value={`${phase.demandGW} GW`} tone="primary" />
              <Readout label="Supply"  value={`${phase.supplyGW} GW`} tone="accent" />
              <Readout label="Unserved load" value={`${phase.outageGW} GW`} tone={phase.outageGW > 10 ? "critical" : "warning"} />
              <Readout label="Frequency"     value={`${phase.freqHz.toFixed(2)} Hz`} tone={phase.freqHz < 59.5 ? "critical" : "default"} />
            </div>
          </div>

          {/* Live D/S delta bar */}
          <DemandSupplyChart phase={phase} />
        </div>
      </section>

      {/* ===== Generation Mix Collapse ===== */}
      <section className="panel p-6">
        <div className="hud-label">Generation mix · 72-hour collapse</div>
        <h2 className="display-md font-display mb-5">52 GW lost across every fuel category</h2>
        <div className="space-y-4">
          {[
            { name: "Natural Gas",   lost: 26, total: 67, color: "oklch(0.68 0.18 35)" },
            { name: "Wind",          lost: 16, total: 25, color: "oklch(0.82 0.14 200)" },
            { name: "Coal",          lost: 5,  total: 13, color: "oklch(0.55 0.04 260)" },
            { name: "Nuclear",       lost: 1.3,total: 5,  color: "oklch(0.72 0.18 245)" },
            { name: "Solar",         lost: 1.5,total: 4,  color: "oklch(0.86 0.18 85)"  },
          ].map((f) => (
            <FuelBar key={f.name} {...f} />
          ))}
        </div>
      </section>

      {/* ===== Blackout Progression Grid ===== */}
      <section className="panel p-6">
        <div className="hud-label">Blackout progression</div>
        <h2 className="display-md font-display mb-5">Cascading load shed across ERCOT zones</h2>
        <BlackoutGrid step={idx} />
      </section>

      {/* ===== Sentinel AI Decision Timeline ===== */}
      <section className="panel p-6">
        <div className="flex items-center gap-2 mb-1">
          <Brain size={16} className="text-[oklch(0.72_0.18_245)]" />
          <div className="hud-label">Counterfactual</div>
        </div>
        <h2 className="display-md font-display mb-2">What Grid Sentinel AI would have done</h2>
        <p className="text-sm text-muted-foreground max-w-2xl mb-6">
          Replaying the same 96 hours through Sentinel's probabilistic dispatch engine. Every
          action below is generated from the model's own ensemble forecast — not hindsight.
        </p>

        <ol className="relative pl-6 border-l border-[oklch(0.72_0.18_245/0.25)] space-y-5">
          {SENTINEL_TIMELINE.map((s, i) => (
            <li key={i} className="relative">
              <span className="absolute -left-[29px] top-1 w-3 h-3 rounded-full bg-[oklch(0.85_0.21_145)] shadow-[0_0_12px_oklch(0.85_0.21_145)]" />
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-[10px] text-[oklch(0.85_0.21_145)] w-12">{s.offset}</span>
                <span className="font-display text-base">{s.action}</span>
              </div>
              <div className="ml-[60px] mt-1 text-xs text-muted-foreground">{s.rationale}</div>
            </li>
          ))}
        </ol>
      </section>

      {/* ===== Lessons Learned ===== */}
      <section className="grid md:grid-cols-3 gap-4">
        {[
          { icon: AlertTriangle, title: "Single-point fuel failure",
            body: "When 67% of dispatchable capacity depends on one weatherized fuel chain, the grid has no redundancy. Sentinel models cross-fuel correlation explicitly." },
          { icon: Activity, title: "Isolation is fragility",
            body: "ERCOT's lack of DC inter-tie capacity left no neighboring system to import from. Probabilistic dispatch pre-positions imports days ahead." },
          { icon: Lightbulb, title: "Foresight beats response",
            body: "Every minute of advance warning is worth ~120 MW of avoided emergency shed. Sentinel converts 4σ weather signals into dispatch actions automatically." },
        ].map((l) => (
          <div key={l.title} className="panel p-5">
            <l.icon size={20} className="text-[oklch(0.82_0.17_75)] mb-3" />
            <div className="font-display text-base mb-1.5">{l.title}</div>
            <p className="text-xs text-muted-foreground leading-relaxed">{l.body}</p>
          </div>
        ))}
      </section>

      <footer className="text-center text-[10px] font-mono text-muted-foreground pt-4">
        Source figures derived from FERC/NERC final report on the February 2021 cold weather event ·
        Sentinel counterfactual generated for demonstration only
      </footer>
    </div>
  );
}

/* ---------- helpers ---------- */
function stateTone(s: Phase["state"]) {
  return s === "critical" ? "oklch(0.68 0.24 25)"
       : s === "warning"  ? "oklch(0.82 0.17 75)"
       : s === "recovery" ? "oklch(0.85 0.21 145)"
       : "oklch(0.82 0.14 200)";
}

function HeroStat({ label, value, tone }: { label: string; value: string; tone: "critical" | "warning" }) {
  const c = tone === "critical" ? "oklch(0.68 0.24 25)" : "oklch(0.82 0.17 75)";
  return (
    <div className="rounded-lg p-3 border" style={{ borderColor: c + "55", background: c + "10" }}>
      <div className="hud-label">{label}</div>
      <div className="font-display text-2xl mt-0.5" style={{ color: c }}>{value}</div>
    </div>
  );
}

function Readout({ label, value, tone = "default" }: { label: string; value: string; tone?: "default" | "primary" | "accent" | "warning" | "critical" }) {
  const c = {
    default: "oklch(0.96 0.012 240)",
    primary: "oklch(0.72 0.18 245)",
    accent:  "oklch(0.85 0.21 145)",
    warning: "oklch(0.82 0.17 75)",
    critical:"oklch(0.68 0.24 25)",
  }[tone];
  return (
    <div className="rounded-lg p-3 bg-[oklch(0.16_0.028_260/0.6)] border border-[oklch(0.72_0.18_245/0.12)]">
      <div className="hud-label">{label}</div>
      <div className="font-mono text-lg mt-1" style={{ color: c }}>{value}</div>
    </div>
  );
}

function DemandSupplyChart({ phase }: { phase: Phase }) {
  const max = 85;
  const dPct = (phase.demandGW / max) * 100;
  const sPct = (phase.supplyGW / max) * 100;
  return (
    <div className="rounded-xl p-5 bg-[oklch(0.12_0.025_260/0.6)] border border-[oklch(0.72_0.18_245/0.12)]">
      <div className="hud-label mb-4">Demand vs Supply · live</div>
      <div className="space-y-5">
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-muted-foreground">Demand</span>
            <span className="font-mono text-[oklch(0.72_0.18_245)]">{phase.demandGW} GW</span>
          </div>
          <div className="h-2.5 rounded-full bg-[oklch(0.3_0.03_255/0.5)] overflow-hidden">
            <div className="h-full transition-all duration-700"
                 style={{ width: `${dPct}%`, background: "linear-gradient(90deg, oklch(0.72 0.18 245), oklch(0.82 0.14 200))" }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-muted-foreground">Available supply</span>
            <span className="font-mono text-[oklch(0.85_0.21_145)]">{phase.supplyGW} GW</span>
          </div>
          <div className="h-2.5 rounded-full bg-[oklch(0.3_0.03_255/0.5)] overflow-hidden">
            <div className="h-full transition-all duration-700"
                 style={{ width: `${sPct}%`, background: "oklch(0.85 0.21 145)" }} />
          </div>
        </div>
        {phase.outageGW > 0 && (
          <div className="pt-4 border-t border-[oklch(0.68_0.24_25/0.3)]">
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-[oklch(0.68_0.24_25)]" />
              <span className="text-xs text-muted-foreground">Gap forces emergency shed of</span>
              <span className="font-mono text-base text-[oklch(0.68_0.24_25)]">{phase.outageGW} GW</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function FuelBar({ name, lost, total, color }: { name: string; lost: number; total: number; color: string }) {
  const lostPct = (lost / total) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs mb-1.5">
        <span className="font-display text-sm">{name}</span>
        <span className="font-mono text-muted-foreground">
          <span style={{ color }}>{lost} GW lost</span> · of {total} GW installed
        </span>
      </div>
      <div className="relative h-3 rounded bg-[oklch(0.3_0.03_255/0.4)] overflow-hidden">
        <div className="absolute inset-y-0 left-0" style={{ width: "100%", background: color, opacity: 0.18 }} />
        <div className="absolute inset-y-0 left-0 transition-all duration-1000"
             style={{ width: `${lostPct}%`, background: color, boxShadow: `0 0 16px ${color}` }} />
      </div>
    </div>
  );
}

function BlackoutGrid({ step }: { step: number }) {
  // Deterministic 24x8 cell grid. Cells "go dark" as step advances.
  const cells = useMemo(() => {
    const arr: { lit: boolean; intensity: number }[] = [];
    for (let i = 0; i < 24 * 8; i++) {
      // pseudo-random but stable
      const r = ((i * 9301 + 49297) % 233280) / 233280;
      arr.push({ lit: true, intensity: r });
    }
    return arr;
  }, []);
  // Fraction dark per step (matches narrative)
  const darkFrac = [0.0, 0.05, 0.42, 0.62, 0.58, 0.25, 0.05][step] ?? 0;
  return (
    <div>
      <div className="grid grid-cols-24 gap-1" style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
        {cells.map((c, i) => {
          const dark = c.intensity < darkFrac;
          return (
            <div
              key={i}
              className="aspect-square rounded-sm transition-all duration-700"
              style={{
                background: dark ? "oklch(0.12 0.02 260)" : "oklch(0.85 0.21 145)",
                opacity: dark ? 0.4 : 0.35 + c.intensity * 0.5,
                boxShadow: dark ? "none" : `0 0 6px oklch(0.85 0.21 145 / ${0.4 + c.intensity * 0.4})`,
              }}
            />
          );
        })}
      </div>
      <div className="flex justify-between mt-3 text-[10px] font-mono text-muted-foreground">
        <span>ERCOT load zones · 192 substations sampled</span>
        <span>Offline · {Math.round(darkFrac * 100)}%</span>
      </div>
    </div>
  );
}
