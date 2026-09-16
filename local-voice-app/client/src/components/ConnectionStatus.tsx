import { useState } from 'react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import type { TransportState } from '@pipecat-ai/client-js';
import {
  usePipecatClientTransportState,
  useRTVIClientEvent,
} from '@pipecat-ai/client-react';
import { DataList, TextDashBlankslate, cn } from '@pipecat-ai/voice-ui-kit';

import { STRINGS } from '../strings';

const BUSY_STATES = ['initializing', 'authenticating', 'authenticated', 'connecting'];

const StatusValue = ({ state }: { state: TransportState | null }) => (
  <span
    className={cn(
      'mono-upper text-muted-foreground font-medium flex items-center gap-1.5 leading-none justify-end',
      {
        'text-active': state === 'connected' || state === 'ready',
        'text-destructive': state === 'error',
        'text-subtle/50 dark:text-subtle/80': state === 'disconnected',
        'text-subtle': !state,
        'animate-pulse': BUSY_STATES.includes(state ?? ''),
      }
    )}
  >
    {state ? STRINGS.transportState[state] : <TextDashBlankslate />}
  </span>
);

export const ConnectionStatus = () => {
  const transportState = usePipecatClientTransportState();
  const [botStatus, setBotStatus] = useState<TransportState | null>(null);

  useRTVIClientEvent(RTVIEvent.TransportStateChanged, (state) => {
    if (state === 'connecting') {
      setBotStatus('connecting');
    }
  });
  useRTVIClientEvent(RTVIEvent.BotReady, () => setBotStatus('ready'));
  useRTVIClientEvent(RTVIEvent.BotConnected, () => setBotStatus('connected'));
  useRTVIClientEvent(RTVIEvent.Disconnected, () => setBotStatus('disconnected'));
  useRTVIClientEvent(RTVIEvent.BotDisconnected, () =>
    setBotStatus('disconnected')
  );

  return (
    <DataList
      data={{
        [STRINGS.connection.client]: <StatusValue state={transportState} />,
        [STRINGS.connection.agent]: <StatusValue state={botStatus} />,
      }}
    />
  );
};
