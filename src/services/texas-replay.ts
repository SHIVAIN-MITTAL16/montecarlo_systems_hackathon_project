import { getNationalGridSnapshot } from "./grid-snapshot";
import { runMonteCarloSimulation } from "./monte-carlo";

type NationalGridSnapshot = Awaited<ReturnType<typeof getNationalGridSnapshot>>;
type StateSnapshot = NationalGridSnapshot["states"][number];
type MonteCarloResult = ReturnType<typeof runMonteCarloSimulation>;

const DEFAULT_REPLAY_SEED = 2_021;
const DEFAULT_BLACKOUT_PROBABILITY_THRESHOLD = 50;
const SCORE_DENOMINATOR = 100;
const ROUND_MW_SCALE = 10;
const ROUND_PERCENT_SCALE = 10;
const SOLAR_HEAT_DERATE_START_C = 25;
const SOLAR_HEAT_DERATE_END_C = 45;
const SOLAR_HEAT_DERATE_MAX = 0.18;
const WIND_CUT_IN_KMH = 10;
const WIND_RATED_KMH = 45;
const WIND_HIGH_DERATE_START_KMH = 70;
const WIND_HIGH_DERATE_END_KMH = 100;
const WIND_HIGH_DERATE_MAX = 0.7;
const HYDRO_PRECIPITATION_UPLIFT_MAX = 0.2;
const HYDRO_PRECIPITATION_FULL_MM = 20;

interface HistoricalWeatherRecord {
  readonly timestamp: string;
  readonly temperatureCelsius: number;
  readonly cloudCoverPercent: number;
  readonly windSpeedKmh: number;
  readonly precipitationMm: number;
  readonly renewableGenerationMw?: number;
}

interface HistoricalDemandRecord {
  readonly timestamp: string;
  readonly demandMw: number;
}

interface HistoricalGroundTruthRecord {
  readonly timestamp: string;
  readonly actualDemandMw?: number;
  readonly actualRenewableGenerationMw?: number;
  readonly blackoutEvent?: boolean;
}

interface TexasReplayInput {
  readonly weather: readonly HistoricalWeatherRecord[];
  readonly demand: readonly HistoricalDemandRecord[];
  readonly groundTruth?: readonly HistoricalGroundTruthRecord[];
  readonly simulations?: number;
  readonly seed?: number;
  readonly baseSnapshot?: NationalGridSnapshot;
  readonly blackoutProbabilityThreshold?: number;
}

interface ReplayTimelinePoint {
  readonly timestamp: string;
  readonly blackoutProbability: number;
  readonly reserveMargin: number;
  readonly renewableGenerationMw: number;
  readonly demandMw: number;
  readonly lossOfLoadProbability: number;
  readonly expectedUnservedEnergyMwh: number;
  readonly predictedBlackout: boolean;
}

interface ReplaySummaryStatistics {
  readonly meanPredictionError: number | null;
  readonly peakDemandError: number | null;
  readonly renewableError: number | null;
  readonly blackoutDetectionAccuracy: number | null;
}

interface ReplayResult {
  readonly timeline: readonly ReplayTimelinePoint[];
  readonly summary: ReplaySummaryStatistics;
}

interface TimestepInput {
  readonly weather: HistoricalWeatherRecord;
  readonly demand: HistoricalDemandRecord;
  readonly groundTruth?: HistoricalGroundTruthRecord;
}

/**
 * Replays the Texas Winter Storm Uri sequence against the accepted snapshot and Monte Carlo pipeline.
 */
export async function runTexas2021Replay(input: TexasReplayInput): Promise<ReplayResult> {
  const baseSnapshot = input.baseSnapshot ?? await getNationalGridSnapshot();
  const timesteps = buildChronologicalTimesteps(input);
  const timeline = timesteps.map((timestep, index) =>
    runReplayTimestep(baseSnapshot, timestep, input, index),
  );

  return {
    timeline,
    summary: calculateSummaryStatistics(timeline, timesteps),
  };
}

function buildChronologicalTimesteps(input: TexasReplayInput): readonly TimestepInput[] {
  const demandByTimestamp = toTimestampMap(input.demand);
  const truthByTimestamp = toTimestampMap(input.groundTruth ?? []);

  return [...input.weather]
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    .flatMap((weather) => {
      const demand = demandByTimestamp.get(weather.timestamp);
      if (!demand) return [];
      return [{ weather, demand, groundTruth: truthByTimestamp.get(weather.timestamp) }];
    });
}

function runReplayTimestep(
  baseSnapshot: NationalGridSnapshot,
  timestep: TimestepInput,
  input: TexasReplayInput,
  index: number,
): ReplayTimelinePoint {
  const replaySnapshot = buildReplaySnapshot(baseSnapshot, timestep);
  const monteCarlo = runMonteCarloSimulation(replaySnapshot, {
    simulations: input.simulations,
    seed: (input.seed ?? DEFAULT_REPLAY_SEED) + index,
  });

  return toTimelinePoint(timestep.weather.timestamp, replaySnapshot, monteCarlo, input);
}

function buildReplaySnapshot(
  baseSnapshot: NationalGridSnapshot,
  timestep: TimestepInput,
): NationalGridSnapshot {
  const states = baseSnapshot.states.map((state) => buildReplayState(state, baseSnapshot, timestep));

  return {
    ...baseSnapshot,
    timestamp: timestep.weather.timestamp,
    states,
    nationalDemandMw: roundMw(timestep.demand.demandMw),
    nationalRenewableGenerationMw: roundMw(sum(states, (state) => state.energy.netRenewableGenerationMw)),
    nationalReserveMarginPercent: calculateNationalReserveMargin(states, timestep.demand.demandMw),
    nationalRenewablePenetrationPercent: calculateRenewablePenetration(states, timestep.demand.demandMw),
    nationalGridStressIndex: calculateNationalStress(states, timestep.demand.demandMw),
  };
}

function buildReplayState(
  state: StateSnapshot,
  baseSnapshot: NationalGridSnapshot,
  timestep: TimestepInput,
): StateSnapshot {
  const demandMw = allocateByWeight(timestep.demand.demandMw, state.demand.estimatedLoadMw, baseSnapshot.nationalDemandMw);
  const energy = buildReplayEnergy(state, timestep.weather, demandMw);
  const demand = { ...state.demand, estimatedLoadMw: roundMw(demandMw), peakLoadMw: Math.max(state.demand.peakLoadMw, roundMw(demandMw)) };

  return { ...state, observedAt: timestep.weather.timestamp, energy, demand };
}

function buildReplayEnergy(
  state: StateSnapshot,
  weather: HistoricalWeatherRecord,
  demandMw: number,
): StateSnapshot["energy"] {
  const solarGenerationMw = state.energy.solarGenerationMw * calculateSolarFactor(weather);
  const windGenerationMw = state.energy.windGenerationMw * calculateWindFactor(weather.windSpeedKmh);
  const hydroEstimateMw = state.energy.hydroEstimateMw * calculateHydroFactor(weather.precipitationMm);
  const renewableGenerationMw = solarGenerationMw + windGenerationMw + hydroEstimateMw;
  const gapMw = calculateReplayGap(state, demandMw, renewableGenerationMw);

  return {
    ...state.energy,
    observedAt: weather.timestamp,
    solarGenerationMw: roundMw(solarGenerationMw),
    windGenerationMw: roundMw(windGenerationMw),
    hydroEstimateMw: roundMw(hydroEstimateMw),
    estimatedDemandMw: roundMw(demandMw),
    netRenewableGenerationMw: roundMw(renewableGenerationMw),
    supplyDemandGapMw: roundMw(gapMw),
    reserveMarginPercent: calculateReserveMargin(gapMw, demandMw),
    renewablePenetrationPercent: roundPercent((renewableGenerationMw / demandMw) * SCORE_DENOMINATOR),
  };
}

function calculateReplayGap(
  state: StateSnapshot,
  demandMw: number,
  renewableGenerationMw: number,
): number {
  const baseDispatchableMw =
    state.energy.estimatedDemandMw + state.energy.supplyDemandGapMw - state.energy.netRenewableGenerationMw;

  return Math.max(0, baseDispatchableMw) + renewableGenerationMw - demandMw;
}

function toTimelinePoint(
  timestamp: string,
  snapshot: NationalGridSnapshot,
  monteCarlo: MonteCarloResult,
  input: TexasReplayInput,
): ReplayTimelinePoint {
  const threshold = input.blackoutProbabilityThreshold ?? DEFAULT_BLACKOUT_PROBABILITY_THRESHOLD;

  return {
    timestamp,
    blackoutProbability: monteCarlo.blackoutProbability,
    reserveMargin: monteCarlo.meanReserveMargin,
    renewableGenerationMw: snapshot.nationalRenewableGenerationMw,
    demandMw: snapshot.nationalDemandMw,
    lossOfLoadProbability: monteCarlo.lossOfLoadProbability,
    expectedUnservedEnergyMwh: monteCarlo.expectedUnservedEnergyMwh,
    predictedBlackout: monteCarlo.blackoutProbability >= threshold,
  };
}

function calculateSummaryStatistics(
  timeline: readonly ReplayTimelinePoint[],
  timesteps: readonly TimestepInput[],
): ReplaySummaryStatistics {
  return {
    meanPredictionError: calculateMeanPredictionError(timeline, timesteps),
    peakDemandError: calculatePeakDemandError(timeline, timesteps),
    renewableError: calculateRenewableError(timeline, timesteps),
    blackoutDetectionAccuracy: calculateBlackoutAccuracy(timeline, timesteps),
  };
}

function calculateMeanPredictionError(
  timeline: readonly ReplayTimelinePoint[],
  timesteps: readonly TimestepInput[],
): number | null {
  const errors = timeline.flatMap((point, index) => {
    const actual = timesteps[index]?.groundTruth?.actualDemandMw;
    return actual === undefined ? [] : [Math.abs(point.demandMw - actual)];
  });

  return errors.length > 0 ? roundMw(mean(errors)) : null;
}

function calculatePeakDemandError(
  timeline: readonly ReplayTimelinePoint[],
  timesteps: readonly TimestepInput[],
): number | null {
  const actualValues = timesteps.flatMap((item) => item.groundTruth?.actualDemandMw ?? []);
  if (actualValues.length === 0 || timeline.length === 0) return null;

  return roundMw(Math.max(...timeline.map((point) => point.demandMw)) - Math.max(...actualValues));
}

function calculateRenewableError(
  timeline: readonly ReplayTimelinePoint[],
  timesteps: readonly TimestepInput[],
): number | null {
  const errors = timeline.flatMap((point, index) => {
    const actual = timesteps[index]?.groundTruth?.actualRenewableGenerationMw;
    return actual === undefined ? [] : [Math.abs(point.renewableGenerationMw - actual)];
  });

  return errors.length > 0 ? roundMw(mean(errors)) : null;
}

function calculateBlackoutAccuracy(
  timeline: readonly ReplayTimelinePoint[],
  timesteps: readonly TimestepInput[],
): number | null {
  const comparisons = timeline.flatMap((point, index) => {
    const actual = timesteps[index]?.groundTruth?.blackoutEvent;
    return actual === undefined ? [] : [point.predictedBlackout === actual ? 1 : 0];
  });

  return comparisons.length > 0 ? roundPercent(mean(comparisons) * SCORE_DENOMINATOR) : null;
}

function calculateSolarFactor(weather: HistoricalWeatherRecord): number {
  const cloudFactor = 1 - clamp(weather.cloudCoverPercent, 0, SCORE_DENOMINATOR) / SCORE_DENOMINATOR;
  const heatDerate = scaleToUnit(weather.temperatureCelsius, SOLAR_HEAT_DERATE_START_C, SOLAR_HEAT_DERATE_END_C) * SOLAR_HEAT_DERATE_MAX;

  return clamp(cloudFactor * (1 - heatDerate), 0, 1);
}

function calculateWindFactor(windSpeedKmh: number): number {
  const ramp = scaleToUnit(windSpeedKmh, WIND_CUT_IN_KMH, WIND_RATED_KMH);
  const derate = scaleToUnit(windSpeedKmh, WIND_HIGH_DERATE_START_KMH, WIND_HIGH_DERATE_END_KMH) * WIND_HIGH_DERATE_MAX;

  return clamp(ramp * (1 - derate), 0, 1);
}

function calculateHydroFactor(precipitationMm: number): number {
  const uplift = scaleToUnit(precipitationMm, 0, HYDRO_PRECIPITATION_FULL_MM) * HYDRO_PRECIPITATION_UPLIFT_MAX;

  return 1 + uplift;
}

function calculateNationalReserveMargin(
  states: readonly StateSnapshot[],
  demandMw: number,
): number {
  const supplyMw = sum(states, (state) => state.energy.estimatedDemandMw + state.energy.supplyDemandGapMw);

  return calculateReserveMargin(supplyMw - demandMw, demandMw);
}

function calculateRenewablePenetration(states: readonly StateSnapshot[], demandMw: number): number {
  const renewableMw = sum(states, (state) => state.energy.netRenewableGenerationMw);

  return demandMw > 0 ? roundPercent((renewableMw / demandMw) * SCORE_DENOMINATOR) : 0;
}

function calculateNationalStress(states: readonly StateSnapshot[], demandMw: number): number {
  if (demandMw <= 0) return 0;
  return toScore(sum(states, (state) => state.energy.gridStressIndex * state.demand.estimatedLoadMw) / demandMw);
}

function calculateReserveMargin(gapMw: number, demandMw: number): number {
  return demandMw > 0 ? roundPercent((gapMw / demandMw) * SCORE_DENOMINATOR) : 0;
}

function allocateByWeight(total: number, weight: number, totalWeight: number): number {
  return totalWeight > 0 ? total * (weight / totalWeight) : 0;
}

function toTimestampMap<T extends { readonly timestamp: string }>(items: readonly T[]): Map<string, T> {
  return new Map(items.map((item) => [item.timestamp, item]));
}

function sum<T>(items: readonly T[], selector: (item: T) => number): number {
  return items.reduce((total, item) => total + selector(item), 0);
}

function mean(values: readonly number[]): number {
  return values.length > 0 ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function scaleToUnit(value: number, min: number, max: number): number {
  if (max <= min) return 0;
  return clamp((value - min) / (max - min), 0, 1);
}

function roundMw(value: number): number {
  return Math.round(value * ROUND_MW_SCALE) / ROUND_MW_SCALE;
}

function roundPercent(value: number): number {
  return Math.round(value * ROUND_PERCENT_SCALE) / ROUND_PERCENT_SCALE;
}

function toScore(value: number): number {
  return Math.round(clamp(value, 0, SCORE_DENOMINATOR));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
