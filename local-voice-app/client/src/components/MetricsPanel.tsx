import { useState } from 'react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import type { PipecatMetricsData } from '@pipecat-ai/client-js';
import {
  usePipecatClientTransportState,
  useRTVIClientEvent,
} from '@pipecat-ai/client-react';

import { STRINGS } from '../strings';
import { isConnected } from '../transportState';
import { CenteredNotice } from './CenteredNotice';

/**
 * 指标事件载荷上的 `tokens` 由服务端下发，但 `PipecatMetricsData` 没有声明它，
 * 所以下面读取该字段时要断言。
 */
type MetricsPayload = PipecatMetricsData & {
  tokens?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  }[];
};

/** 每个 processor 的最新一次读数。 */
type Readings = Record<string, number>;

const mergeReadings = (
  readings: PipecatMetricsData['ttfb'],
  into: Readings
): Readings => {
  if (!readings?.length) return into;
  const next = { ...into };
  for (const { processor, value } of readings) {
    next[processor] = value;
  }
  return next;
};

const MetricSection = ({
  title,
  rows,
}: {
  title: string;
  rows: [string, string][];
}) => (
  <section>
    <h2 className="text-xl font-semibold mb-2">{title}</h2>
    <div className="flex flex-col gap-1 font-mono text-xs">
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-4">
          <span className="text-muted-foreground truncate">{label}</span>
          <span>{value}</span>
        </div>
      ))}
    </div>
  </section>
);

/** 线上单位是秒（pipecat 侧 `TTFBMetricsData` / `ProcessingMetricsData` 均如此）。 */
const toMillisecondRows = (readings: Readings): [string, string][] =>
  Object.entries(readings).map(([processor, value]) => [
    processor,
    `${(value * 1000).toFixed(0)} ms`,
  ]);

const toCountRows = (readings: Readings): [string, string][] =>
  Object.entries(readings).map(([processor, value]) => [
    processor,
    String(Math.round(value)),
  ]);

export const MetricsPanel = () => {
  const transportState = usePipecatClientTransportState();
  const [ttfb, setTtfb] = useState<Readings>({});
  const [processing, setProcessing] = useState<Readings>({});
  const [characters, setCharacters] = useState<Readings>({});
  const [tokens, setTokens] = useState({ prompt: 0, completion: 0, total: 0 });

  useRTVIClientEvent(RTVIEvent.Connected, () => {
    setTtfb({});
    setProcessing({});
    setCharacters({});
    setTokens({ prompt: 0, completion: 0, total: 0 });
  });

  useRTVIClientEvent(RTVIEvent.Metrics, (data) => {
    const payload = data as MetricsPayload;
    setTtfb((previous) => mergeReadings(payload.ttfb, previous));
    setProcessing((previous) => mergeReadings(payload.processing, previous));
    setCharacters((previous) => mergeReadings(payload.characters, previous));

    const usage = payload.tokens?.[0];
    if (usage) {
      setTokens((previous) => ({
        prompt: previous.prompt + (usage.prompt_tokens ?? 0),
        completion: previous.completion + (usage.completion_tokens ?? 0),
        total: previous.total + (usage.total_tokens ?? 0),
      }));
    }
  });

  const hasReadings =
    Object.keys(ttfb).length > 0 ||
    Object.keys(processing).length > 0 ||
    Object.keys(characters).length > 0;

  if (!isConnected(transportState)) {
    return (
      <CenteredNotice
        title={STRINGS.conversation.notConnected}
        hint={STRINGS.conversation.notConnectedHint}
      />
    );
  }

  if (!hasReadings) {
    return <CenteredNotice title={STRINGS.metrics.waiting} />;
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <MetricSection
        title={STRINGS.metrics.tokenUsage}
        rows={[
          [STRINGS.metrics.promptTokens, String(tokens.prompt)],
          [STRINGS.metrics.completionTokens, String(tokens.completion)],
          [STRINGS.metrics.totalTokens, String(tokens.total)],
        ]}
      />
      {Object.keys(ttfb).length > 0 && (
        <MetricSection
          title={STRINGS.metrics.ttfb}
          rows={toMillisecondRows(ttfb)}
        />
      )}
      {Object.keys(processing).length > 0 && (
        <MetricSection
          title={STRINGS.metrics.processing}
          rows={toMillisecondRows(processing)}
        />
      )}
      {Object.keys(characters).length > 0 && (
        <MetricSection
          title={STRINGS.metrics.characters}
          rows={toCountRows(characters)}
        />
      )}
    </div>
  );
};
