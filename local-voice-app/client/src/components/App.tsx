import { useEffect, useState } from 'react';
import { RTVIEvent } from '@pipecat-ai/client-js';

import type { PipecatBaseChildProps } from '@pipecat-ai/voice-ui-kit';
import {
  Button,
  ConnectButton,
  EventsPanel,
  cn,
} from '@pipecat-ai/voice-ui-kit';
import type { ConnectButtonStateContent } from '@pipecat-ai/voice-ui-kit';

import type { TransportType } from '../config';
import { STRINGS } from '../strings';
import { useMicrophone } from '../useMicrophone';
import { BotAudioPanel } from './BotAudioPanel';
import { ChatPanel } from './ChatPanel';
import { ConnectionStatus } from './ConnectionStatus';
import { MicrophoneControl } from './MicrophoneControl';
import { TransportSelect } from './TransportSelect';
import { UserAudioPanel } from './UserAudioPanel';

interface AppProps extends PipecatBaseChildProps {
  transportType: TransportType;
  onTransportChange: (type: TransportType) => void;
  availableTransports: TransportType[];
}

/**
 * 连接按钮的每个状态都要给文案，否则会落到库内置的英文默认值。
 */
const CONNECT_BUTTON_CONTENT: ConnectButtonStateContent = {
  disconnected: { children: STRINGS.connectButton.connect, variant: 'active' },
  initialized: { children: STRINGS.connectButton.connect, variant: 'active' },
  initializing: {
    children: STRINGS.connectButton.initializing,
    variant: 'secondary',
  },
  authenticating: {
    children: STRINGS.connectButton.connecting,
    variant: 'secondary',
  },
  authenticated: {
    children: STRINGS.connectButton.connecting,
    variant: 'secondary',
  },
  connecting: {
    children: STRINGS.connectButton.connecting,
    variant: 'secondary',
  },
  connected: {
    children: STRINGS.connectButton.connecting,
    variant: 'secondary',
  },
  ready: { children: STRINGS.connectButton.disconnect, variant: 'destructive' },
  disconnecting: {
    children: STRINGS.connectButton.disconnecting,
    variant: 'secondary',
  },
  error: { children: STRINGS.connectButton.error, variant: 'destructive' },
};

export const App = ({
  client,
  handleConnect,
  handleDisconnect,
  transportType,
  onTransportChange,
  availableTransports,
}: AppProps) => {
  const [errorCount, setErrorCount] = useState(0);
  const [showDebug, setShowDebug] = useState(false);
  const microphone = useMicrophone();
  useEffect(() => {
    client?.initDevices();
    if (!client) return;
    const onError = () => setErrorCount((count) => count + 1);
    client.on(RTVIEvent.Error, onError);
    return () => {
      client.off(RTVIEvent.Error, onError);
    };
  }, [client]);

  const showTransportSelector = availableTransports.length > 1;

  const connect = async () => {
    if (client?.state === 'disconnected') {
      await client.initDevices();
    }
    await handleConnect?.();
  };

  return (
    <div className="flex flex-col w-full h-full">
      <div className="flex items-center justify-between gap-4 p-4">
        <div className="flex items-center gap-4">
          {showTransportSelector && (
            <TransportSelect
              transportType={transportType}
              onTransportChange={onTransportChange}
              availableTransports={availableTransports}
            />
          )}
          <Button
            size="lg"
            variant={showDebug ? 'active' : 'outline'}
            aria-pressed={showDebug}
            aria-label={STRINGS.app.debugToggle}
            onClick={() => setShowDebug((shown) => !shown)}
          >
            Debug
          </Button>
        </div>
        <div className="flex items-center gap-4">
          <MicrophoneControl {...microphone} />
          <ConnectButton
            size="lg"
            stateContent={CONNECT_BUTTON_CONTENT}
            onConnect={connect}
            onDisconnect={handleDisconnect}
          />
        </div>
      </div>
      <div className="flex-1 overflow-hidden px-4">
        <div className="voice-workspace grid h-full gap-4 overflow-hidden">
          <ChatPanel />
          <aside className="flex flex-col gap-4 overflow-auto">
            <ConnectionStatus />
            <UserAudioPanel enabled={microphone.enabled} />
            <BotAudioPanel />
            <div className="rounded-lg border p-4 text-sm">
              {STRINGS.app.errorCount}
              {errorCount}
            </div>
          </aside>
        </div>
      </div>
      <div className={cn('h-96 overflow-hidden px-4 pb-4', !showDebug && 'hidden')}>
        <EventsPanel />
      </div>
    </div>
  );
};
