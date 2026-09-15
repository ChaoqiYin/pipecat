import { useMediaState, usePipecatClientMediaDevices } from '@pipecat-ai/client-react';
import { UserAudioComponent } from '@pipecat-ai/voice-ui-kit';

interface MicrophoneControlProps {
  enabled: boolean;
  pending: boolean;
  toggle: () => void;
}

export const MicrophoneControl = ({ enabled, pending, toggle }: MicrophoneControlProps) => {
  const devices = usePipecatClientMediaDevices();
  const { mic } = useMediaState();

  return (
    <UserAudioComponent
      {...devices}
      size="lg"
      isMicEnabled={enabled}
      onClick={toggle}
      unavailableText={mic.state === 'error' ? '无法访问麦克风' : undefined}
      buttonProps={{
        'aria-label': '切换麦克风静音',
        'aria-pressed': !enabled,
        isLoading: pending || mic.state === 'initializing' || mic.state === 'uninitialized',
      }}
      dropdownButtonProps={{ disabled: pending }}
    />
  );
};
