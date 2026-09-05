import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState, type ReactNode } from "react";
import { Activity, BatteryCharging, CloudSnow, Fuel, ShieldAlert, Wind } from "lucide-react";

export const Route = createFileRoute("/simulation")({ head: () => ({ meta: [{ title: "Polar Crisis Lab · Grid Sentinel AI" }, { name: "description", content: "Stress-test a polar research station microgrid under extreme weather, renewable derating and fuel constraints." }] }), component: Simulation });

type State = { storm:number; lowLight:number; load:number; wind:number; battery:number; generator:number; fuel:number; critical:number };
const DEFAULTS: State = { storm:25, lowLight:15, load:10, wind:20, battery:85, generator:90, fuel:70, critical:60 };
const CONTROLS = [["storm","Polar storm severity","%","Cold, cloud and operational stress"],["lowLight","Low-light / polar night","%","Solar availability penalty"],["load","Station load surge","%","Heating, life-support and lab demand"],["wind","Wind derating","%","Turbine availability loss"],["battery","Battery available","%","Usable state of charge"],["generator","Backup generator","%","Generator availability"],["fuel","Fuel reserve","%","Diesel reserve remaining"],["critical","Critical-load share","%","Non-deferrable station demand"]] as const;

function Simulation(){
 const [s,setS]=useState<State>(DEFAULTS);
 const r=useMemo(()=>{
  const solar=Math.max(0,34*(1-s.lowLight/100)*(1-s.storm*.003));
  const wind=Math.max(0,28*(1-s.wind/100)*(1-s.storm*.002));
  const load=52*(1+s.load/100+s.storm*.0025);
  const battery=Math.max(0,18*s.battery/100);
  const generator=Math.max(0,30*s.generator/100);
  const renewable=solar+wind;
  const baseGap=Math.max(0,load-renewable-generator);
  const fuelPenalty=Math.max(0,20-s.fuel)*.45;
  const shortage=Math.min(99,Math.max(0,4+baseGap*1.35-battery*.55+s.critical*.08+fuelPenalty));
  const optimizedFuel=Math.max(0,Math.min(100,22+baseGap*.85+s.critical*.12-s.battery*.08));
  const baselineFuel=Math.max(0,Math.min(100,34+baseGap*1.15));
  const eue=baseGap*(shortage/100)*4.2;
  const minSoc=Math.max(5,s.battery-baseGap*.8);
  const reserve=Math.max(0,((renewable+battery+generator-load)/load)*100);
  const priority=s.critical>=65?"Critical loads first":"Balanced load shedding";
  let action="Maintain reserve and monitor weather.";
  if(shortage>35) action="Start backup generation, preserve battery reserve, defer non-critical loads.";
  else if(shortage>15) action="Pre-dispatch battery and generator; shift deferrable lab loads.";
  else if(s.lowLight>60) action="Protect battery SOC for the low-light window and minimize diesel cycling.";
  return {solar,wind,load,renewable,shortage,optimizedFuel,baselineFuel,eue,minSoc,reserve,priority,action,generator};
 },[s]);
 return <div className="px-6 py-6 space-y-6">
  <section className="panel p-6"><div className="hud-label mb-2 flex items-center gap-2"><ShieldAlert size={13}/> POLAR CRISIS LAB · DECISION-SUPPORT SIMULATION</div><h1 className="text-3xl font-display font-semibold">Stress-test the <span className="text-[oklch(0.72_0.18_245)]">station microgrid</span>.</h1><p className="text-muted-foreground mt-2 max-w-3xl">Explore extreme-weather, low-light and fuel-constrained conditions. The prototype recomputes generation, reserve, shortage risk and a dispatch recommendation as controls move.</p></section>
  <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-6">
   <aside className="panel p-5 space-y-4"><div className="flex justify-between items-center"><div><div className="hud-label">Scenario controls</div><div className="text-sm font-display">Station stress matrix</div></div><button onClick={()=>setS(DEFAULTS)} className="text-[10px] font-mono px-2 py-1 rounded border border-[oklch(0.72_0.18_245/0.3)]">RESET</button></div>{CONTROLS.map(([id,label,unit,hint])=>{const val=s[id];return <label key={id} className="block"><div className="flex justify-between"><span className="text-sm">{label}</span><span className="font-mono text-xs text-[oklch(0.72_0.18_245)]">{val}{unit}</span></div><div className="text-[10px] text-muted-foreground mb-1">{hint}</div><input type="range" min="0" max="100" step="1" value={val} onChange={e=>setS(x=>({...x,[id]:Number(e.target.value)}))} className="w-full"/></label>})}</aside>
   <main className="space-y-6"><div className="grid grid-cols-2 lg:grid-cols-4 gap-3"><Metric label="Load" value={`${r.load.toFixed(1)} kW`} icon={<Activity size={16}/>}/><Metric label="Renewables" value={`${r.renewable.toFixed(1)} kW`} icon={<Wind size={16>}/>} /><Metric label="Battery" value={`${r.minSoc.toFixed(0)}% min SOC`} icon={<BatteryCharging size={16}/>}/><Metric label="Fuel use" value={`${r.optimizedFuel.toFixed(0)}% index`} icon={<Fuel size={16}/>}/></div><section className="panel p-5"><div className="hud-label mb-4">Risk engine · scenario response</div><div className="grid md:grid-cols-2 gap-5"><Metric label="Shortage probability" value={`${r.shortage.toFixed(1)}%`}/><Metric label="Expected unserved energy" value={`${r.eue.toFixed(2)} kWh`}/><Metric label="Reserve margin" value={`${r.reserve.toFixed(1)}%`}/><Metric label="Critical-load policy" value={r.priority}/></div><div className="mt-5 p-4 rounded-lg border border-[oklch(0.85_0.21_145/0.25)] bg-[oklch(0.85_0.21_145/0.05)]"><div className="hud-label mb-1">Sentinel recommendation</div><div className="font-display text-lg">{r.action}</div><div className="text-xs text-muted-foreground mt-2">Fuel index: baseline {r.baselineFuel.toFixed(0)} → reserve-aware dispatch {r.optimizedFuel.toFixed(0)} · prototype simulation, not operational telemetry.</div></div></section><section className="panel p-5"><div className="hud-label mb-3">Station energy flow</div><div className="grid grid-cols-3 gap-3 text-center"><Flow icon={<Wind size={18}/>} label="Wind" value={`${r.wind.toFixed(1)} kW`}/><Flow icon={<CloudSnow size={18}/>} label="Solar" value={`${r.solar.toFixed(1)} kW`}/><Flow icon={<Fuel size={18}/>} label="Backup" value={`${r.generator.toFixed(1)} kW`}/></div></section></main>
  </div></div>;
}
function Metric({label,value,icon}:{label:string;value:string;icon?:ReactNode}){return <div className="panel p-4"><div className="hud-label flex items-center gap-2">{icon}{label}</div><div className="font-mono text-lg mt-2">{value}</div></div>}
function Flow({icon,label,value}:{icon:ReactNode;label:string;value:string}){return <div className="p-4 rounded-lg border border-[oklch(0.72_0.18_245/0.12)]"><div className="flex justify-center mb-2 text-[oklch(0.85_0.21_145)]">{icon}</div><div className="text-xs text-muted-foreground">{label}</div><div className="font-mono mt-1">{value}</div></div>}
