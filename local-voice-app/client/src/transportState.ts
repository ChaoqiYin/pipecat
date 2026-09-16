import type { TransportState } from '@pipecat-ai/client-js';

export const isConnected = (state: TransportState) =>
  state === 'connected' || state === 'ready';

export const isConnecting = (state: TransportState) =>
  state === 'authenticating' || state === 'connecting';
