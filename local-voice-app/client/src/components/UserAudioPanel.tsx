import {
  useMediaState,
  usePipecatClientMediaTrack,
} from '@pipecat-ai/client-react';
import {
  Panel,
  PanelContent,
  PanelHeader,
  PanelTitle,
  VoiceVisualizer,
} from '@pipecat-ai/voice-ui-kit';

export const UserAudioPanel = ({ enabled }: { enabled: boolean }) => {
  const track = usePipecatClientMediaTrack('audio', 'local');
  const { mic } = useMediaState();
  const status =
    mic.state === 'error'
      ? '无法访问麦克风'
      : !enabled
        ? '麦克风已静音'
        : !track
          ? '等待麦克风输入'
          : null;

  return (
    <Panel className="user-audio-panel" role="region" aria-label="用户麦克风波形">
      <PanelHeader>
        <PanelTitle>用户音频</PanelTitle>
      </PanelHeader>
      <PanelContent className="user-audio-visualization">
        {status ? (
          <span role="status">{status}</span>
        ) : (
          <VoiceVisualizer
            participantType="local"
            backgroundColor="transparent"
            barColor="--color-client"
            barCount={20}
            barWidth={5}
            barGap={5}
            barMaxHeight={80}
            barOrigin="center"
          />
        )}
      </PanelContent>
    </Panel>
  );
};
