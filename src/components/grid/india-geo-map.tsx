import { useMemo, useState } from "react";
import { geoMercator, geoPath, type GeoPermissibleObjects } from "d3-geo";
import { Sun, Wind, Droplet, Flame, BatteryCharging } from "lucide-react";
import indiaGeo from "@/lib/india-geo.json";
import { STATES, type StateData } from "@/lib/grid-data";

/* ---------- Name normalization (geojson → state code) ---------- */
const NAME_TO_ID: Record<string, string> = {
  "Jammu and Kashmir": "JK",
  "Punjab": "PB",
  "Delhi": "DL",
  "Rajasthan": "RJ",
  "Uttar Pradesh": "UP",
  "Gujarat": "GJ",
  "Maharashtra": "MH",
  "Madhya Pradesh": "MP",
  "Chhattisgarh": "CG",
  "Orissa": "OD",
  "West Bengal": "WB",
  "Bihar": "BR",
  "Jharkhand": "JH",
  "Andhra Pradesh": "AP",
  "Karnataka": "KA",
  "Kerala": "KL",
  "Tamil Nadu": "TN",
  "Assam": "AS",
  "Haryana": "HR",
  // remaining states/UTs render but stay un-instrumented (placeholders)
};

const DISPLAY_NAME: Record<string, string> = {
  "Orissa": "Odisha",
  "Uttaranchal": "Uttarakhand",
};

/* ---------- Renewable / generation hubs ---------- */
type HubKind = "solar" | "wind" | "hydro" | "thermal" | "battery";
type Hub = { kind: HubKind; lon: number; lat: number; label: string };

const HUBS: Hub[] = [
  // Solar
  { kind: "solar",   lon: 77.6, lat: 14.1, label: "Pavagada" },
  { kind: "solar",   lon: 71.9, lat: 27.0, label: "Bhadla"   },
  { kind: "solar",   lon: 71.3, lat: 23.2, label: "Charanka" },
  { kind: "solar",   lon: 78.5, lat: 17.9, label: "Kurnool"  },
  // Wind
  { kind: "wind",    lon: 77.5, lat: 8.4,  label: "Muppandal" },
  { kind: "wind",    lon: 69.6, lat: 23.2, label: "Kutch"     },
  { kind: "wind",    lon: 74.7, lat: 15.3, label: "Chitradurga" },
  // Hydro
  { kind: "hydro",   lon: 78.5, lat: 31.1, label: "Bhakra"    },
  { kind: "hydro",   lon: 78.4, lat: 25.5, label: "Rihand"    },
  { kind: "hydro",   lon: 76.5, lat: 10.1, label: "Idukki"    },
  // Thermal
  { kind: "thermal", lon: 82.1, lat: 22.7, label: "Korba"     },
  { kind: "thermal", lon: 78.7, lat: 21.4, label: "Mouda"     },
  { kind: "thermal", lon: 85.3, lat: 25.6, label: "Barh"      },
  { kind: "thermal", lon: 87.0, lat: 23.7, label: "Mejia"     },
  // Battery (BESS)
  { kind: "battery", lon: 77.2, lat: 28.6, label: "Delhi BESS" },
  { kind: "battery", lon: 72.9, lat: 19.1, label: "Mumbai BESS" },
  { kind: "battery", lon: 80.3, lat: 13.1, label: "Chennai BESS" },
];

/* Metro demand centres (pulse) */
const METROS: { lon: number; lat: number; name: string }[] = [
  { lon: 77.2, lat: 28.6, name: "Delhi" },
  { lon: 72.9, lat: 19.1, name: "Mumbai" },
  { lon: 80.3, lat: 13.1, name: "Chennai" },
  { lon: 88.4, lat: 22.6, name: "Kolkata" },
  { lon: 77.6, lat: 12.9, name: "Bengaluru" },
  { lon: 78.5, lat: 17.4, name: "Hyderabad" },
  { lon: 72.6, lat: 23.0, name: "Ahmedabad" },
  { lon: 73.9, lat: 18.5, name: "Pune" },
];

/* Inter-regional transmission corridors (lon/lat pairs) */
const CORRIDORS: { a: [number, number]; b: [number, number]; load: number }[] = [
  { a: [71.9, 27.0], b: [77.2, 28.6], load: 0.85 }, // Bhadla → Delhi
  { a: [72.6, 23.0], b: [72.9, 19.1], load: 0.92 }, // Gujarat → Mumbai
  { a: [77.6, 12.9], b: [80.3, 13.1], load: 0.74 }, // Bengaluru → Chennai
  { a: [78.5, 22.5], b: [80.9, 26.8], load: 0.68 }, // MP → UP
  { a: [82.1, 22.7], b: [85.8, 20.3], load: 0.55 }, // Korba → Odisha
  { a: [88.4, 22.6], b: [85.8, 20.3], load: 0.48 }, // Kolkata ↔ Odisha
  { a: [77.2, 28.6], b: [75.8, 30.7], load: 0.62 }, // Delhi → Punjab
  { a: [78.5, 17.4], b: [80.3, 13.1], load: 0.70 }, // Hyderabad → Chennai
  { a: [78.4, 25.5], b: [82.9, 25.3], load: 0.58 }, // Rihand → Varanasi
  { a: [69.6, 23.2], b: [72.6, 23.0], load: 0.80 }, // Kutch → Ahmedabad
  { a: [76.5, 10.1], b: [77.5, 8.4],  load: 0.45 }, // Idukki → Muppandal
];

/* ---------- Visual helpers ---------- */
function riskTone(risk: number) {
  if (risk >= 70) return "oklch(0.68 0.24 25)";  // critical
  if (risk >= 45) return "oklch(0.82 0.17 75)";  // watch
  if (risk >= 30) return "oklch(0.82 0.14 200)"; // elevated
  return "oklch(0.85 0.21 145)";                 // nominal
}

const HUB_COLOR: Record<HubKind, string> = {
  solar:   "oklch(0.86 0.18 85)",
  wind:    "oklch(0.82 0.14 200)",
  hydro:   "oklch(0.72 0.18 245)",
  thermal: "oklch(0.68 0.18 35)",
  battery: "oklch(0.85 0.21 145)",
};
const HUB_ICON: Record<HubKind, typeof Sun> = {
  solar: Sun, wind: Wind, hydro: Droplet, thermal: Flame, battery: BatteryCharging,
};

export interface HoverPayload {
  id?: string;
  name: string;
  data?: StateData;
}

interface Props {
  width?: number;
  height?: number;
  onHover?: (p: HoverPayload | null) => void;
}

const VB_W = 900;
const VB_H = 980;

export function IndiaGeoMap({ width, height = 720, onHover }: Props) {
  const [hover, setHover] = useState<HoverPayload | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);

  const { pathGen, project, features } = useMemo(() => {
    const fc = indiaGeo as unknown as GeoJSON.FeatureCollection;
    const projection = geoMercator().fitExtent(
      [[24, 24], [VB_W - 24, VB_H - 24]],
      fc as unknown as GeoPermissibleObjects,
    );
    return {
      pathGen: geoPath(projection),
      project: (lon: number, lat: number) => projection([lon, lat]) ?? [0, 0],
      features: fc.features,
    };
  }, []);

  const stateById = useMemo(() => {
    const m = new Map<string, StateData>();
    for (const s of STATES) m.set(s.id, s);
    return m;
  }, []);

  function handleHover(name: string, e: React.MouseEvent<SVGPathElement>) {
    const id = NAME_TO_ID[name];
    const data = id ? stateById.get(id) : undefined;
    const display = DISPLAY_NAME[name] ?? name;
    const payload: HoverPayload = { id, name: display, data };
    setHover(payload);
    onHover?.(payload);
    const svg = e.currentTarget.ownerSVGElement;
    if (svg) {
      const rect = svg.getBoundingClientRect();
      setPointer({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }
  }
  function handleLeave() {
    setHover(null);
    setPointer(null);
    onHover?.(null);
  }

  return (
    <div className="relative w-full" style={{ height, width }}>
      {/* base lighting */}
      <div className="absolute inset-0 rounded-2xl grid-bg opacity-50" />
      <div
        className="absolute inset-0 rounded-2xl pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 70% 55% at 50% 55%, oklch(0.72 0.18 245 / 0.18), transparent 70%)",
        }}
      />

      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="relative w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <filter id="state-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3.5" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="hub-glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="2.2" />
          </filter>
          <linearGradient id="state-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="oklch(0.22 0.04 258 / 0.95)" />
            <stop offset="100%" stopColor="oklch(0.14 0.03 260 / 0.95)" />
          </linearGradient>
          <linearGradient id="corridor" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"   stopColor="oklch(0.72 0.18 245 / 0)" />
            <stop offset="50%"  stopColor="oklch(0.72 0.18 245 / 0.85)" />
            <stop offset="100%" stopColor="oklch(0.85 0.21 145 / 0)" />
          </linearGradient>
        </defs>

        {/* sweep scanner */}
        <g style={{ transformOrigin: `${VB_W / 2}px ${VB_H / 2}px` }} className="animate-sweep" opacity="0.18">
          <defs>
            <radialGradient id="sweep-grad">
              <stop offset="0%" stopColor="oklch(0.72 0.18 245 / 0.35)" />
              <stop offset="100%" stopColor="transparent" />
            </radialGradient>
          </defs>
          <circle cx={VB_W / 2} cy={VB_H / 2} r={Math.min(VB_W, VB_H) / 2.2} fill="url(#sweep-grad)" />
        </g>

        {/* ----- STATE GEOMETRY ----- */}
        <g>
          {features.map((f, i) => {
            const name = (f.properties as { name: string }).name;
            const id = NAME_TO_ID[name];
            const data = id ? stateById.get(id) : undefined;
            const tone = data ? riskTone(data.risk) : "oklch(0.72 0.18 245 / 0.5)";
            const isHover = hover?.name === (DISPLAY_NAME[name] ?? name);
            const d = pathGen(f as unknown as GeoPermissibleObjects) ?? "";
            return (
              <g key={i}>
                {/* base shape */}
                <path
                  d={d}
                  fill="url(#state-fill)"
                  stroke={tone}
                  strokeOpacity={isHover ? 0.95 : 0.45}
                  strokeWidth={isHover ? 1.4 : 0.6}
                  style={{
                    cursor: "pointer",
                    transition: "stroke-opacity 200ms, fill 200ms",
                    filter: isHover ? "url(#state-glow)" : undefined,
                  }}
                  onMouseMove={(e) => handleHover(name, e)}
                  onMouseLeave={handleLeave}
                />
                {/* risk wash */}
                {data && (
                  <path
                    d={d}
                    fill={tone}
                    opacity={isHover ? 0.22 : 0.08}
                    pointerEvents="none"
                    style={{ transition: "opacity 200ms" }}
                  />
                )}
              </g>
            );
          })}
        </g>

        {/* ----- TRANSMISSION CORRIDORS ----- */}
        <g>
          {CORRIDORS.map((c, i) => {
            const [x1, y1] = project(c.a[0], c.a[1]);
            const [x2, y2] = project(c.b[0], c.b[1]);
            const mx = (x1 + x2) / 2 + (y2 - y1) * 0.08;
            const my = (y1 + y2) / 2 - (x2 - x1) * 0.08;
            const path = `M${x1},${y1} Q${mx},${my} ${x2},${y2}`;
            const dur = 2.8 + (1 - c.load) * 3.5;
            return (
              <g key={i}>
                <path d={path} stroke="url(#corridor)" strokeWidth={0.8 + c.load * 1.2}
                      fill="none" opacity={0.35 + c.load * 0.4} />
                <path d={path} stroke="oklch(0.72 0.18 245)" strokeWidth="0.4"
                      strokeDasharray="2 5" fill="none" opacity="0.3" className="animate-dash" />
                <circle r={1.4 + c.load * 1.4} fill="oklch(0.85 0.21 145)"
                        style={{ filter: "drop-shadow(0 0 6px oklch(0.85 0.21 145))" }}>
                  <animateMotion dur={`${dur}s`} repeatCount="indefinite" path={path} />
                </circle>
                <circle r="0.9" fill="oklch(1 0 0)">
                  <animateMotion dur={`${dur}s`} begin={`${dur * 0.5}s`} repeatCount="indefinite" path={path} />
                </circle>
              </g>
            );
          })}
        </g>

        {/* ----- METRO PULSES ----- */}
        <g>
          {METROS.map((m, i) => {
            const [x, y] = project(m.lon, m.lat);
            return (
              <g key={i}>
                <circle cx={x} cy={y} r="4" fill="oklch(0.72 0.18 245)" opacity="0.35">
                  <animate attributeName="r" values="4;14;4" dur="3.2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.5;0;0.5" dur="3.2s" repeatCount="indefinite" />
                </circle>
                <circle cx={x} cy={y} r="2.4" fill="oklch(0.96 0.012 240)" />
              </g>
            );
          })}
        </g>

        {/* ----- HUBS (icons rendered as foreignObject for crisp lucide) ----- */}
        <g>
          {HUBS.map((h, i) => {
            const [x, y] = project(h.lon, h.lat);
            const c = HUB_COLOR[h.kind];
            const Icon = HUB_ICON[h.kind];
            return (
              <g key={i} transform={`translate(${x - 11}, ${y - 11})`}>
                <circle cx="11" cy="11" r="12" fill={c} opacity="0.18" filter="url(#hub-glow)" />
                <circle cx="11" cy="11" r="10" fill="oklch(0.12 0.025 260 / 0.92)" stroke={c} strokeOpacity="0.85" strokeWidth="0.7" />
                <foreignObject x="3" y="3" width="16" height="16" pointerEvents="none">
                  <div style={{ color: c, width: 16, height: 16 }} className="grid place-items-center">
                    <Icon size={11} strokeWidth={2} />
                  </div>
                </foreignObject>
              </g>
            );
          })}
        </g>
      </svg>

      {/* ----- HOVER GLASS PANEL ----- */}
      {hover && pointer && (
        <HoverCard hover={hover} x={pointer.x} y={pointer.y} />
      )}

      {/* ----- LEGEND ----- */}
      <div className="absolute bottom-3 left-3 panel px-3 py-2 text-[10px] font-mono flex gap-3">
        <LegendDot color="oklch(0.85 0.21 145)" label="NOMINAL" />
        <LegendDot color="oklch(0.82 0.14 200)" label="ELEVATED" />
        <LegendDot color="oklch(0.82 0.17 75)" label="WATCH" />
        <LegendDot color="oklch(0.68 0.24 25)" label="CRITICAL" />
      </div>
      <div className="absolute top-3 right-3 panel px-3 py-1.5 text-[10px] font-mono text-muted-foreground flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-[oklch(0.85_0.21_145)] animate-flicker" />
        LIVE TELEMETRY · NLDC · {STATES.length} INSTRUMENTED NODES
      </div>
      <div className="absolute top-3 left-3 panel px-3 py-1.5 text-[10px] font-mono text-muted-foreground flex items-center gap-3">
        <Legend kind="solar"   label="Solar" />
        <Legend kind="wind"    label="Wind" />
        <Legend kind="hydro"   label="Hydro" />
        <Legend kind="thermal" label="Thermal" />
        <Legend kind="battery" label="BESS" />
      </div>
    </div>
  );
}

/* ---------- Sub-components ---------- */

function HoverCard({ hover, x, y }: { hover: HoverPayload; x: number; y: number }) {
  const d = hover.data;
  const fmt = (v?: number, unit = "", digits = 1) =>
    d && v !== undefined ? `${v.toFixed(digits)}${unit}` : "—";

  const stability = d ? Math.max(0, 100 - d.risk) : undefined;

  return (
    <div
      className="absolute z-30 panel panel-glow p-4 w-[300px] pointer-events-none animate-fade-up"
      style={{
        left: `min(calc(${x}px + 18px), calc(100% - 19rem))`,
        top:  `min(calc(${y}px + 18px), calc(100% - 22rem))`,
      }}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <div className="hud-label">State / UT</div>
          <div className="text-base font-display font-semibold">{hover.name}</div>
        </div>
        {d ? (
          <span
            className="px-2 py-0.5 rounded text-[10px] font-mono"
            style={{
              background: riskTone(d.risk) + "22",
              color: riskTone(d.risk),
              border: `1px solid ${riskTone(d.risk)}`,
            }}
          >
            RISK {d.risk}
          </span>
        ) : (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono text-muted-foreground border border-[oklch(0.72_0.18_245/0.2)]">
            UNINSTRUMENTED
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-y-1.5 gap-x-3 text-xs font-mono">
        <Row k="Demand"           v={fmt(d?.demand, " GW")} />
        <Row k="Renewable Output" v={fmt(d?.renewable, " GW")} />
        <Row k="Grid Stability"   v={d ? `${stability} %` : "—"} />
        <Row k="Battery Reserve"  v={d ? `${d.battery} %`   : "—"} />
        <Row k="Risk Index"       v={d ? `${d.risk} / 100`   : "—"} />
        <Row k="Blackout P"       v={fmt(d?.blackout, " %", 0)} />
        <Row k="Grid Frequency"   v={d ? "49.98 Hz" : "—"} />
        <Row k="Confidence"       v={d ? "94.2 %"   : "—"} />
      </div>

      <div className="mt-3 pt-3 border-t border-[oklch(0.72_0.18_245/0.18)]">
        <div className="hud-label mb-1">AI Recommendation</div>
        <div className="text-xs text-foreground/90 leading-relaxed min-h-[1.5em]">
          {d?.recommendation ?? "Awaiting telemetry stream — connect dispatch feed."}
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between text-[10px] font-mono text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[oklch(0.85_0.21_145)] animate-flicker" />
          LIVE
        </span>
        <span>Updated · just now</span>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <>
      <div className="text-muted-foreground">{k}</div>
      <div className="text-right">{v}</div>
    </>
  );
}
function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="w-2 h-2 rounded-full" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
      {label}
    </span>
  );
}
function Legend({ kind, label }: { kind: HubKind; label: string }) {
  const Icon = HUB_ICON[kind];
  const c = HUB_COLOR[kind];
  return (
    <span className="flex items-center gap-1.5" style={{ color: c }}>
      <Icon size={10} />
      <span className="text-muted-foreground">{label}</span>
    </span>
  );
}
