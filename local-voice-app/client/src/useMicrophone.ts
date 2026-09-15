import { useRef, useState } from 'react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import { usePipecatClient, useRTVIClientEvent } from '@pipecat-ai/client-react';

export const useMicrophone = () => {
  const client = usePipecatClient();
  const [enabled, setEnabled] = useState(client?.isMicEnabled ?? false);
  const [pending, setPending] = useState(false);
  const requested = useRef<boolean | null>(null);

  const confirm = (active: boolean) => {
    setEnabled(active);
    if (requested.current === active) {
      requested.current = null;
      setPending(false);
    }
  };

  useRTVIClientEvent(RTVIEvent.TrackStarted, (track, participant) => {
    if (track.kind === 'audio' && participant?.local) confirm(true);
  });
  useRTVIClientEvent(RTVIEvent.TrackStopped, (track, participant) => {
    if (track.kind === 'audio' && participant?.local) confirm(false);
  });
  useRTVIClientEvent(RTVIEvent.Disconnected, () => {
    requested.current = null;
    setPending(false);
    setEnabled(false);
  });
  useRTVIClientEvent(RTVIEvent.Error, () => {
    requested.current = null;
    setPending(false);
  });

  const toggle = () => {
    if (!client || requested.current !== null) return;
    requested.current = !enabled;
    setPending(true);
    try {
      client.enableMic(requested.current);
    } catch (error) {
      requested.current = null;
      setPending(false);
      throw error;
    }
  };

  return { enabled, pending, toggle };
};
