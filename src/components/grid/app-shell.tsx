import { Link, useRouterState } from "@tanstack/react-router";
import { Activity, BrainCircuit, FlaskConical, Globe2, Snowflake } from "lucide-react";
import type { ReactNode } from "react";

const NAV=[
 {to:"/",label:"Command Center",icon:Activity,code:"01"},
 {to:"/digital-twin",label:"India Digital Twin",icon:Globe2,code:"02"},
 {to:"/polar-station",label:"Polar Digital Twin",icon:Snowflake,code:"03"},
 {to:"/simulation",label:"Crisis Lab",icon:FlaskConical,code:"04"},
 {to:"/texas-2021",label:"Research Replay",icon:Snowflake,code:"05"},
 {to:"/control-room",label:"AI Control Room",icon:BrainCircuit,code:"06"},
] as const;

export function AppShell({children}:{children:ReactNode}){
 const pathname=useRouterState({select:s=>s.location.pathname});
 return <div className="min-h-screen flex flex-col"><header className="sticky top-0 z-40 border-b border-[oklch(0.72_0.18_245/0.15)] bg-[oklch(0.04_0.015_260/0.92)] backdrop-blur-xl"><div className="max-w-[1600px] mx-auto w-full px-6 h-16 flex items-center justify-between gap-6"><Link to="/" className="flex items-center gap-3 min-w-max"><div className="w-8 h-8 rounded-lg border border-[oklch(0.72_0.18_245/0.5)] grid place-items-center"><Activity size={16}/></div><div><div className="font-display font-semibold tracking-wide text-sm">GRID SENTINEL <span className="text-[oklch(0.72_0.18_245)]">AI</span></div><div className="hud-label">POLAR ENERGY OPERATIONS</div></div></Link><nav className="hidden lg:flex items-center gap-1">{NAV.map(({to,label,icon:Icon,code})=><Link key={to} to={to} className={`px-3 py-2 rounded-md text-xs flex items-center gap-2 transition-colors ${pathname===to?"bg-[oklch(0.72_0.18_245/0.12)] text-foreground":"text-muted-foreground hover:text-foreground hover:bg-[oklch(0.72_0.18_245/0.06)]"}`}><Icon size={13}/><span className="font-mono text-[9px] opacity-50">{code}</span>{label}</Link>)}</nav><div className="hidden xl:flex items-center gap-2 text-[9px] font-mono"><span className="px-2 py-1 rounded border border-[oklch(0.85_0.21_145/0.25)] text-[oklch(0.85_0.21_145)]">MODE: HYBRID</span><span className="px-2 py-1 rounded border border-[oklch(0.72_0.18_245/0.25)] text-[oklch(0.72_0.18_245)]">DIGITAL TWINS: ONLINE</span><span className="px-2 py-1 rounded border border-[oklch(0.82_0.17_75/0.25)] text-[oklch(0.82_0.17_75)]">DISPATCH: ADVISORY</span></div></div></header><main className="flex-1">{children}</main><footer className="border-t border-[oklch(0.72_0.18_245/0.1)] px-6 py-3"><div className="max-w-[1600px] mx-auto flex justify-between text-[9px] font-mono text-muted-foreground"><span>GRID SENTINEL AI · SIH26061 · INDIA + POLAR DIGITAL TWINS</span><span>SIMULATION ENVIRONMENT · DECISION SUPPORT ONLY</span></div></div>;
}
