const SCORE_DENOMINATOR = 100;
const ROUND_MW_SCALE = 10;
const ROUND_PERCENT_SCALE = 10;
const BLACKOUT_CERTAINTY_LOAD_SHED_MW = 10_000;
const CRITICAL_SHORTAGE_RATIO = 0.18;
const HIGH_STRESS_OUTAGE_MW = 45_000;
const LOW_FREQUENCY_HZ = 59.7;
const SIMULATION_INTERVAL_HOURS = 1;

export interface TexasReplayMetadata {
  readonly eventName: string;
  readonly timezone: string;
  readonly resolution: string;
  readonly sourceNotes: readonly string[];
  readonly sources: readonly string[];
  readonly missingDatasets?: readonly string[];
}

export interface TexasReplayRecord {
  readonly timestamp: string;
  readonly temperatureCelsius?: number;
  readonly weatherStation?: string;
  readonly windSpeedKmh?: number;
  readonly precipitationMm?: number;
  readonly demandMw?: number;
  readonly generationMw?: number;
  readonly renewableGenerationMw?: number;
  readonly forcedOutageMw?: number;
  readonly frequencyHz?: number;
  readonly loadShedMw?: number;
  readonly majorEvent?: string;
  readonly recommendation?: string;
  readonly generationByFuel?: GenerationByFuel;
}

export interface GenerationByFuel {
  readonly gasMw?: number;
  readonly coalMw?: number;
  readonly nuclearMw?: number;
  readonly windMw?: number;
  readonly solarMw?: number;
  readonly hydroMw?: number;
  readonly otherMw?: number;
}

export interface TexasReplayInput {
  readonly metadata: TexasReplayMetadata;
  readonly records: readonly TexasReplayRecord[];
}

export interface ReplayTimelinePoint {
  readonly timestamp: string;
  readonly temperatureCelsius?: number;
  readonly weatherStation?: string;
  readonly windSpeedKmh?: number;
  readonly precipitationMm?: number;
  readonly demandMw?: number;
  readonly generationMw?: number;
  readonly renewableGenerationMw?: number;
  readonly forcedOutageMw?: number;
  readonly reserveMarginPercent: number | null;
  readonly lossOfLoadProbability: number | null;
  readonly expectedUnservedEnergyMwh: number | null;
  readonly blackoutProbability: number | null;
  readonly systemStressIndex: number | null;
  readonly frequencyHz?: number;
  readonly loadShedMw?: number;
  readonly predictedBlackout: boolean | null;
  readonly majorEvent?: string;
  readonly recommendation?: string;
  readonly generationByFuel?: GenerationByFuel;
}

export interface ReplaySummaryStatistics {
  readonly eventName: string;
  readonly replayStart: string | null;
  readonly replayEnd: string | null;
  readonly replayStartTime: string | null;
  readonly replayEndTime: string | null;
  readonly replayDurationHours: number | null;
  readonly timelineEventCount: number;
  readonly peakDemandMw: number | null;
  readonly minimumDemandMw: number | null;
  readonly averageDemandMw: number | null;
  readonly peakGenerationMw: number | null;
  readonly minimumGenerationMw: number | null;
  readonly averageGenerationMw: number | null;
  readonly peakRenewableGenerationMw: number | null;
  readonly minimumRenewableGenerationMw: number | null;
  readonly averageRenewableGenerationMw: number | null;
  readonly peakGenerationShortageMw: number | null;
  readonly peakForcedOutageMw: number | null;
  readonly peakLoadShedMw: number | null;
  readonly maximumForcedOutageMw: number | null;
  readonly maximumLoadShedMw: number | null;
  readonly minimumReserveMarginPercent: number | null;
  readonly maximumReserveMarginPercent: number | null;
  readonly maximumBlackoutProbability: number | null;
  readonly maximumExpectedUnservedEnergyMwh: number | null;
  readonly worstEventTimestamp: string | null;
  readonly totalExpectedUnservedEnergyMwh: number | null;
  readonly blackoutHours: number;
  readonly sourceNotes: readonly string[];
  readonly sources: readonly string[];
}

export interface ReplayResult {
  readonly metadata: TexasReplayMetadata;
  readonly timeline: readonly ReplayTimelinePoint[];
  readonly summary: ReplaySummaryStatistics;
}

/**
 * Replays Winter Storm Uri from Texas-only historical records.
 */
export async function runTexas2021Replay(input: TexasReplayInput): Promise<ReplayResult> {
  const timeline = [...input.records]
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    .map(toTimelinePoint);

  return {
    metadata: input.metadata,
    timeline,
    summary: calculateSummary(input.metadata, timeline),
  };
}

function toTimelinePoint(record: TexasReplayRecord): ReplayTimelinePoint {
  const shortageMw = calculateShortageMw(record);
  const reserveMarginPercent = calculateReserveMargin(record);
  const lossOfLoadProbability = calculateLossOfLoadProbability(record, shortageMw);
  const blackoutProbability = calculateBlackoutProbability(record, shortageMw);
  const expectedUnservedEnergyMwh = shortageMw === null && record.loadShedMw === undefined
    ? null
    : roundMw(Math.max(shortageMw ?? 0, record.loadShedMw ?? 0) * SIMULATION_INTERVAL_HOURS);

  return {
    ...record,
    reserveMarginPercent,
    lossOfLoadProbability,
    expectedUnservedEnergyMwh,
    blackoutProbability,
    systemStressIndex: calculateSystemStress(record, reserveMarginPercent, blackoutProbability),
    predictedBlackout: blackoutProbability === null ? null : blackoutProbability >= 50,
  };
}

function calculateSummary(
  metadata: TexasReplayMetadata,
  timeline: readonly ReplayTimelinePoint[],
): ReplaySummaryStatistics {
  const replayStart = timeline[0]?.timestamp ?? null;
  const replayEnd = timeline.at(-1)?.timestamp ?? null;

  return {
    eventName: metadata.eventName,
    replayStart,
    replayEnd,
    replayStartTime: replayStart,
    replayEndTime: replayEnd,
    replayDurationHours: calculateReplayDurationHours(replayStart, replayEnd),
    timelineEventCount: timeline.length,
    peakDemandMw: maxValue(timeline, (point) => point.demandMw),
    minimumDemandMw: minValue(timeline, (point) => point.demandMw),
    averageDemandMw: averageValue(timeline, (point) => point.demandMw),
    peakGenerationMw: maxValue(timeline, (point) => point.generationMw),
    minimumGenerationMw: minValue(timeline, (point) => point.generationMw),
    averageGenerationMw: averageValue(timeline, (point) => point.generationMw),
    peakRenewableGenerationMw: maxValue(timeline, (point) => point.renewableGenerationMw),
    minimumRenewableGenerationMw: minValue(timeline, (point) => point.renewableGenerationMw),
    averageRenewableGenerationMw: averageValue(timeline, (point) => point.renewableGenerationMw),
    peakGenerationShortageMw: maxValue(timeline, (point) => calculateShortageMw(point)),
    peakForcedOutageMw: maxValue(timeline, (point) => point.forcedOutageMw),
    peakLoadShedMw: maxValue(timeline, (point) => point.loadShedMw),
    maximumForcedOutageMw: maxValue(timeline, (point) => point.forcedOutageMw),
    maximumLoadShedMw: maxValue(timeline, (point) => point.loadShedMw),
    minimumReserveMarginPercent: minValue(timeline, (point) => point.reserveMarginPercent),
    maximumReserveMarginPercent: maxValue(timeline, (point) => point.reserveMarginPercent),
    maximumBlackoutProbability: maxValue(timeline, (point) => point.blackoutProbability),
    maximumExpectedUnservedEnergyMwh: maxValue(timeline, (point) => point.expectedUnservedEnergyMwh),
    worstEventTimestamp: calculateWorstEventTimestamp(timeline),
    totalExpectedUnservedEnergyMwh: sumNullable(timeline, (point) => point.expectedUnservedEnergyMwh),
    blackoutHours: timeline.filter((point) => point.predictedBlackout).length,
    sourceNotes: metadata.sourceNotes,
    sources: metadata.sources,
  };
}

function calculateShortageMw(record: Pick<TexasReplayRecord, "demandMw" | "generationMw">): number | null {
  if (record.demandMw === undefined || record.generationMw === undefined) return null;
  return Math.max(0, record.demandMw - record.generationMw);
}

function calculateReserveMargin(record: Pick<TexasReplayRecord, "demandMw" | "generationMw">): number | null {
  if (record.demandMw === undefined || record.generationMw === undefined || record.demandMw <= 0) return null;
  return roundPercent(((record.generationMw - record.demandMw) / record.demandMw) * SCORE_DENOMINATOR);
}

function calculateLossOfLoadProbability(
  record: TexasReplayRecord,
  shortageMw: number | null,
): number | null {
  if ((record.loadShedMw ?? 0) > 0) return SCORE_DENOMINATOR;
  if (shortageMw === null || record.demandMw === undefined || record.demandMw <= 0) return null;
  if (shortageMw <= 0) return 0;
  return toScore((shortageMw / record.demandMw) / CRITICAL_SHORTAGE_RATIO);
}

function calculateBlackoutProbability(record: TexasReplayRecord, shortageMw: number | null): number | null {
  const risks: number[] = [];
  if (record.loadShedMw !== undefined) {
    risks.push(scaleToScore(record.loadShedMw, 0, BLACKOUT_CERTAINTY_LOAD_SHED_MW));
  }
  if (shortageMw !== null && record.demandMw !== undefined && record.demandMw > 0) {
    risks.push(toScore((shortageMw / record.demandMw) / CRITICAL_SHORTAGE_RATIO));
  }
  if (record.forcedOutageMw !== undefined) {
    risks.push(scaleToScore(record.forcedOutageMw, 0, HIGH_STRESS_OUTAGE_MW));
  }
  if (record.frequencyHz !== undefined) {
    risks.push(scaleToScore(60 - record.frequencyHz, 0, 60 - LOW_FREQUENCY_HZ));
  }

  return risks.length > 0 ? clampScore(Math.max(...risks)) : null;
}

function calculateSystemStress(
  record: TexasReplayRecord,
  reserveMarginPercent: number | null,
  blackoutProbability: number | null,
): number | null {
  const stresses: number[] = [];
  if (reserveMarginPercent !== null && reserveMarginPercent < 0) {
    stresses.push(scaleToScore(Math.abs(reserveMarginPercent), 0, 35) * 0.35);
  }
  if (record.forcedOutageMw !== undefined) {
    stresses.push(scaleToScore(record.forcedOutageMw, 0, HIGH_STRESS_OUTAGE_MW) * 0.25);
  }
  if (blackoutProbability !== null) {
    stresses.push(blackoutProbability * 0.25);
  }
  if (record.frequencyHz !== undefined) {
    stresses.push(scaleToScore(60 - record.frequencyHz, 0, 60 - LOW_FREQUENCY_HZ) * 0.15);
  }

  return stresses.length > 0 ? clampScore(stresses.reduce((total, value) => total + value, 0)) : null;
}

function maxValue<T>(items: readonly T[], selector: (item: T) => number | null | undefined): number | null {
  const values = items.map(selector).filter((value): value is number => value !== null && value !== undefined);
  return values.length > 0 ? Math.max(...values) : null;
}

function minValue<T>(items: readonly T[], selector: (item: T) => number | null | undefined): number | null {
  const values = items.map(selector).filter((value): value is number => value !== null && value !== undefined);
  return values.length > 0 ? Math.min(...values) : null;
}

function sumNullable<T>(items: readonly T[], selector: (item: T) => number | null | undefined): number | null {
  const values = items.map(selector).filter((value): value is number => value !== null && value !== undefined);
  return values.length > 0 ? roundMw(values.reduce((total, value) => total + value, 0)) : null;
}

function averageValue<T>(items: readonly T[], selector: (item: T) => number | null | undefined): number | null {
  const values = items.map(selector).filter((value): value is number => value !== null && value !== undefined);
  if (values.length === 0) return null;
  return roundMw(values.reduce((total, value) => total + value, 0) / values.length);
}

function calculateReplayDurationHours(start: string | null, end: string | null): number | null {
  if (!start || !end) return null;
  const startMs = Date.parse(start);
  const endMs = Date.parse(end);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs < startMs) return null;
  return roundHours((endMs - startMs) / 3_600_000);
}

function calculateWorstEventTimestamp(timeline: readonly ReplayTimelinePoint[]): string | null {
  const worst = timeline.reduce<ReplayTimelinePoint | null>((currentWorst, point) => {
    if (!currentWorst) return point;
    return calculateWorstEventScore(point) > calculateWorstEventScore(currentWorst)
      ? point
      : currentWorst;
  }, null);

  return worst?.timestamp ?? null;
}

function calculateWorstEventScore(point: ReplayTimelinePoint): number {
  const blackout = point.blackoutProbability ?? 0;
  const unserved = point.expectedUnservedEnergyMwh ?? 0;
  const reserveStress = point.reserveMarginPercent === null ? 0 : Math.max(0, -point.reserveMarginPercent);
  return blackout * 1_000_000 + unserved * 100 + reserveStress;
}

function scaleToScore(value: number, min: number, max: number): number {
  if (max <= min) return 0;
  return toScore((value - min) / (max - min));
}

function toScore(value: number): number {
  return Math.round(clamp(value, 0, 1) * SCORE_DENOMINATOR);
}

function clampScore(value: number): number {
  return Math.round(clamp(value, 0, SCORE_DENOMINATOR));
}

function roundMw(value: number): number {
  return Math.round(value * ROUND_MW_SCALE) / ROUND_MW_SCALE;
}

function roundPercent(value: number): number {
  return Math.round(value * ROUND_PERCENT_SCALE) / ROUND_PERCENT_SCALE;
}

function roundHours(value: number): number {
  return Math.round(value * ROUND_PERCENT_SCALE) / ROUND_PERCENT_SCALE;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
