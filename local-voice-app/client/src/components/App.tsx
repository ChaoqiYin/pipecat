import { useEffect, useState } from 'react';
import { RTVIEvent } from '@pipecat-ai/client-js';

import type { PipecatBaseChildProps } from '@pipecat-ai/voice-ui-kit';
import {
  ConnectButton,
  ConversationPanel,
  EventsPanel,
  BotAudioPanel,
  ClientStatus,
} from '@pipecat-ai/voice-ui-kit';

import type { TransportType } from '../config';
import { useMicrophone } from '../useMicrophone';
import { MicrophoneControl } from './MicrophoneControl';
import { TransportSelect } from './TransportSelect';
import { UserAudioPanel } from './UserAudioPanel';

interface AppProps extends PipecatBaseChildProps {
  transportType: TransportType;
  onTransportChange: (type: TransportType) => void;
  availableTransports: TransportType[];
}

export const App = ({
  client,
  handleConnect,
  handleDisconnect,
  transportType,
  onTransportChange,
  availableTransports,
}: AppProps) => {
  const [errorCount, setErrorCount] = useState(0);
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
        {showTransportSelector ? (
          <TransportSelect
            transportType={transportType}
            onTransportChange={onTransportChange}
            availableTransports={availableTransports}
          />
        ) : (
          <div /> /* Spacer */
        )}
        <div className="flex items-center gap-4">
          <MicrophoneControl {...microphone} />
          <ConnectButton
            size="lg"
            onConnect={connect}
            onDisconnect={handleDisconnect}
          />
        </div>
      </div>
      <div className="flex-1 overflow-hidden px-4">
        <div className="voice-workspace grid h-full gap-4 overflow-hidden">
          <ConversationPanel />
          <aside className="flex flex-col gap-4 overflow-auto">
            <ClientStatus />
            <UserAudioPanel enabled={microphone.enabled} />
            <BotAudioPanel />
            <div className="rounded-lg border p-4 text-sm">错误数量：{errorCount}</div>
          </aside>
        </div>
      </div>
      <div className="h-96 overflow-hidden px-4 pb-4">
        <EventsPanel />
      </div>
    </div>
  );
};
