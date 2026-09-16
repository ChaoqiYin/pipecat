import { useEffect, useRef, useState } from 'react';
import { usePipecatClientMediaTrack } from '@pipecat-ai/client-react';
import {
  BotAudioControl,
  Panel,
  PanelContent,
  PanelHeader,
  PanelTitle,
  VoiceVisualizer,
} from '@pipecat-ai/voice-ui-kit';

import { STRINGS } from '../strings';

const BAR_COUNT = 10;
const MAX_VISUALIZER_WIDTH = 240;

/**
 * 与库里 `BotAudioPanel` 同一套尺寸计算：柱宽随容器宽度收缩，柱高封顶在 16:9。
 */
export const BotAudioPanel = () => {
  const track = usePipecatClientMediaTrack('audio', 'bot');
  const [maxHeight, setMaxHeight] = useState(48);
  const [barWidth, setBarWidth] = useState(4);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        const maxBarWidth = MAX_VISUALIZER_WIDTH / (2 * BAR_COUNT - 1);
        setBarWidth(
          Math.max(Math.min(width / (BAR_COUNT * 2), maxBarWidth), 2)
        );
        setMaxHeight(Math.max(Math.min(height, MAX_VISUALIZER_WIDTH / (16 / 9)), 20));
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <Panel className="flex-1 mt-auto">
      <PanelHeader className="justify-between gap-2">
        <PanelTitle>{STRINGS.botAudio.title}</PanelTitle>
        <BotAudioControl
          size="sm"
          variant="ghost"
          buttonProps={{ 'aria-label': STRINGS.botAudio.volumeLabel }}
          volumeSliderProps={{ label: STRINGS.botAudio.volumeLabel }}
        />
      </PanelHeader>
      <PanelContent className="overflow-hidden flex-1">
        <div ref={containerRef} className="relative flex h-full overflow-hidden">
          {track ? (
            <div className="m-auto">
              <VoiceVisualizer
                participantType="bot"
                backgroundColor="transparent"
                barColor="--color-agent"
                barCount={BAR_COUNT}
                barGap={barWidth}
                barLineCap="square"
                barMaxHeight={maxHeight}
                barOrigin="bottom"
                barWidth={barWidth}
              />
            </div>
          ) : (
            <div className="text-subtle flex w-full gap-2 items-center justify-center">
              <span className="font-semibold text-sm">
                {STRINGS.botAudio.noAudio}
              </span>
            </div>
          )}
        </div>
      </PanelContent>
    </Panel>
  );
};
