import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState, type ReactNode } from "react";
import { BatteryCharging, BrainCircuit, Fuel, ShieldCheck, Snowflake, Sun, Wind, Zap } from "lucide-react";

export const Route = createFileRoute("/digital-twin")({
  head: () => ({ meta: [
    { title: "Polar Station Digital Twin · Grid Sentinel AI" },
    { name: "description", content: "Weather-aware digital twin for AI-driven energy management of isolated polar research stations." },
  ]}),
  component: PolarStationTwin,
});

type Scenario = { id: string; label: string; weather: number; demand: number; renewable: number; note: string };

const SCENARIOS: Scenario[] = [
  { id: "nominal", label: "Nominal Weather", weather: 0.05, demand: 1.00, renewable: 1.00, note: "Normal polar operating conditions" },
  { id: "storm", label: "Polar Storm", weather: 0.55, demand: 1.18, renewable: 0.48, note: "Low renewables + heating demand surge" },
  { id: "dark", label: "Low-Light Event", weather: 0.38, demand: 1.10, renewable: 0.30, note: "Extended low solar availability" },
  { id: "wind", label: "Wind Derating", weather: 0.30, demand: 1.06, renewable: 0.58, note: "Wind resource temporarily constrained" },
];

function runMonteCarlo(s: Scenario, optimized: boolean) {
  const N = 5000;
  let deficitScenarios = 0;
  let totalDeficit = 0;
  let totalFuel = 0;
  let minSoc = 1;
  let seed = 918273;
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let i = 0; i < N; i++) {
    const weatherNoise = 0.75 + rand() * 0.50;
    const demandNoise = 0.90 + rand() * 0.25;
    const renewable = 78 * s.renewable * weatherNoise;
    const demand = 82 * s.demand * demandNoise;
    const initialSoc = 0.72 - rand() * 0.18;
    const usableBattery = initialSoc * 180;
    const reserve = optimized ? 0.22 * 180 : 0.10 * 180;
    const availableBattery = Math.max(0, usableBattery - reserve);
    const deficit = Math.max(0, demand - renewable);
    const batteryCover = Math.min(deficit, availableBattery);
    const remaining = deficit - batteryCover;
    const generatorLimit = optimized ? 34 : 28;
    const generator = Math.min(remaining, generatorLimit);
    const unmet = Math.max(0, remaining - generator);
    const socEnd = Math.max(0, (usableBattery - batteryCover) / 180);
    minSoc = Math.min(minSoc, socEnd);
    if (unmet > 0) { deficitScenarios++; totalDeficit += unmet; }
    totalFuel += generator * (optimized ? 0.028 : 0.035);
  }
  return { probability: (deficitScenarios / N) * 100, eue: totalDeficit / N, fuel: totalFuel / N, minSoc: minSoc * 100, scenarios: N };
}

function PolarStationTwin() {
  const [scenarioId, setScenarioId] = useState("storm");
  const [horizon, setHorizon] = useState("24h");
  const scenario = SCENARIOS.find(x => x.id === scenarioId)!;
  const baseline = useMemo(() => runMonteCarlo(scenario, false), [scenario]);
  const optimized = useMemo(() => runMonteCarlo(scenario, true), [scenario]);
  const improvement = baseline.eue > 0 ? ((baseline.eue - optimized.eue) / baseline.eue) * 100 : 0;

  return (
    <div className="px-6 py-6 space-y-5">
      <section className="flex flex-col xl:flex-row xl:items-end justify-between gap-4">
        <div>
          <div className="hud-label flex items-center gap-2"><Snowflake size={13} /> SIH26061 · POLAR STATION MODE</div>
          <h1 className="text-3xl font-display font-semibold tracking-tight">Polar Research Station <span className="text-[oklch(0.72_0.18_245)]">Digital Twin</span></h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-3xl">A virtual copy of the station energy system. Stress the weather, simulate thousands of futures, quantify risk, and test an optimized dispatch strategy before the real station is exposed.</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          {SCENARIOS.map(s => <button key={s.id} onClick={() => setScenarioId(s.id)} className={`px-3 py-2 rounded border text-xs font-mono transition ${scenarioId === s.id ? "border-[oklch(0.72_0.18_245)] bg-[oklch(0.72_0.18_245/0.12)] text-foreground" : "border-border text-muted-foreground hover:text-foreground"}`}>{s.label}</button>)}
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-5 gap-3">
        <Asset icon={Sun} label="Solar PV" value={`${Math.round(42 * scenario.renewable)} kW`} sub="weather adjusted" />
        <Asset icon={Wind} label="Wind" value={`${Math.round(36 * scenario.renewable)} kW`} sub="weather adjusted" />
        <Asset icon={BatteryCharging} label="Battery" value="180 kWh" sub="72% initial SOC" />
        <Asset icon={Fuel} label="Backup" value="50 kW" sub="dispatchable" />
        <Asset icon={Zap} label="Station Load" value={`${Math.round(82 * scenario.demand)} kW`} sub="critical + flexible" />
      </section>

      <div className="grid grid-cols-1 xl:grid-cols-[1.15fr_0.85fr] gap-5">
        <section className="panel p-5">
          <div className="flex items-start justify-between mb-5">
            <div><div className="hud-label">ENERGY DIGITAL TWIN</div><h2 className="text-xl font-display font-semibold">Station Energy Flow</h2></div>
            <div className="text-right"><div className="text-[10px] font-mono text-muted-foreground">HORIZON</div><select value={horizon} onChange={e => setHorizon(e.target.value)} className="bg-transparent border border-border rounded px-2 py-1 text-xs"><option>6h</option><option>12h</option><option>24h</option><option>48h</option></select></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-5 items-center gap-3 text-center">
            <FlowNode icon={Sun} title="SOLAR" value={`${Math.round(42 * scenario.renewable)} kW`} />
            <Arrow />
            <FlowNode icon={BatteryCharging} title="BATTERY" value="180 kWh" />
            <Arrow />
            <FlowNode icon={Zap} title="CRITICAL LOAD" value={`${Math.round(82 * scenario.demand)} kW`} />
          </div>
          <div className="flex justify-center my-3 text-muted-foreground text-xs">↕ coordinated with</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <FlowNode icon={Wind} title="WIND" value={`${Math.round(36 * scenario.renewable)} kW`} />
            <FlowNode icon={Fuel} title="BACKUP GENERATOR" value="50 kW" />
            <FlowNode icon={Snowflake} title="WEATHER" value={`${Math.round(scenario.weather * 100)}% stress`} />
          </div>
          <div className="mt-5 rounded border border-[oklch(0.72_0.18_245/0.18)] bg-[oklch(0.72_0.18_245/0.05)] p-4 text-sm"><b>{scenario.label}:</b> {scenario.note}. The twin propagates this uncertainty into the forecast, simulation and dispatch decision.</div>
        </section>

        <section className="space-y-4">
          <div className="panel p-5">
            <div className="hud-label flex items-center gap-2"><BrainCircuit size={13} /> MONTE CARLO RISK ENGINE</div>
            <div className="flex items-end justify-between mt-2"><div><div className="text-4xl font-mono font-semibold">{baseline.scenarios.toLocaleString()}</div><div className="text-xs text-muted-foreground">stochastic futures simulated</div></div><div className="text-right"><div className="text-2xl font-mono">{baseline.probability.toFixed(1)}%</div><div className="text-[10px] text-muted-foreground">baseline shortage probability</div></div></div>
            <div className="grid grid-cols-3 gap-3 mt-5"><Metric label="Baseline EUE" value={`${baseline.eue.toFixed(2)} kWh`} /><Metric label="Optimized EUE" value={`${optimized.eue.toFixed(2)} kWh`} /><Metric label="EUE change" value={`${improvement.toFixed(1)}%`} /></div>
          </div>
          <div className="panel p-5">
            <div className="hud-label flex items-center gap-2"><ShieldCheck size={13} /> OPTIMIZER RECOMMENDATION</div>
            <h3 className="text-lg font-display font-semibold mt-1">Preserve reserve, then dispatch backup</h3>
            <p className="text-sm text-muted-foreground mt-2">Maintain a 22% battery reserve for critical loads, use available renewable generation first, and prepare the backup generator for the residual deficit.</p>
            <div className="grid grid-cols-2 gap-3 mt-4"><Metric label="Min SOC" value={`${optimized.minSoc.toFixed(1)}%`} /><Metric label="Avg fuel" value={`${optimized.fuel.toFixed(2)} L`} /></div>
            <div className="mt-4 text-[10px] font-mono text-muted-foreground">TRACE: WEATHER → FORECAST → 5,000 SCENARIOS → RISK → CONSTRAINED DISPATCH</div>
          </div>
        </section>
      </div>
    </div>
  );
}

function Asset({ icon: Icon, label, value, sub }: { icon: typeof Sun; label: string; value: string; sub: string }) { return <div className="panel p-3"><Icon size={16} className="text-[oklch(0.72_0.18_245)]"/><div className="hud-label mt-2">{label}</div><div className="font-mono text-lg">{value}</div><div className="text-[10px] text-muted-foreground">{sub}</div></div>; }
function FlowNode({ icon: Icon, title, value }: { icon: typeof Sun; title: string; value: string }) { return <div className="rounded-lg border border-[oklch(0.72_0.18_245/0.18)] bg-[oklch(0.14_0.025_260/0.55)] p-4"><Icon className="mx-auto text-[oklch(0.85_0.21_145)]" size={24}/><div className="hud-label mt-2">{title}</div><div className="font-mono text-sm mt-1">{value}</div></div>; }
function Arrow() { return <div className="hidden md:block text-center text-xl text-muted-foreground">→</div>; }
function Metric({ label, value }: { label: string; value: string }) { return <div><div className="text-[10px] text-muted-foreground font-mono">{label}</div><div className="font-mono text-sm mt-1">{value}</div></div>; }
