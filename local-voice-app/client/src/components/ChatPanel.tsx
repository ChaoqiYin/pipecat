import { useState } from 'react';
import {
  useConversationContext,
  usePipecatClientTransportState,
  usePipecatConversation,
} from '@pipecat-ai/client-react';
import {
  Conversation,
  Panel,
  PanelContent,
  PanelHeader,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  TextInput,
} from '@pipecat-ai/voice-ui-kit';
import type { TextRenderMode } from '@pipecat-ai/voice-ui-kit';

import { STRINGS } from '../strings';
import { isConnected, isConnecting } from '../transportState';
import { CenteredNotice } from './CenteredNotice';
import { MetricsPanel } from './MetricsPanel';

/**
 * 库里 `Conversation` 的空态文案没有属性出口，所以消息为空时改由这里渲染。
 */
const ConversationPlaceholder = () => {
  const transportState = usePipecatClientTransportState();
  const { botOutputSupported } = useConversationContext();

  let title: string = STRINGS.conversation.waiting;
  let hint: string | undefined;
  let destructive = false;

  if (isConnecting(transportState)) {
    title = STRINGS.conversation.connecting;
  } else if (!isConnected(transportState)) {
    title = STRINGS.conversation.notConnected;
    hint = STRINGS.conversation.notConnectedHint;
  } else if (botOutputSupported === false) {
    title = STRINGS.conversation.unsupported;
    hint = STRINGS.conversation.unsupportedHint;
    destructive = true;
  }

  return <CenteredNotice title={title} hint={hint} destructive={destructive} />;
};

export const ChatPanel = () => {
  const [textRenderMode, setTextRenderMode] =
    useState<TextRenderMode>('karaoke');
  const { messages } = usePipecatConversation();

  return (
    <Tabs className="h-full" defaultValue="conversation">
      <Panel className="h-full max-sm:border-none">
        <PanelHeader variant="noPadding" className="p-1.5 relative">
          <TabsList>
            <TabsTrigger value="conversation">
              {STRINGS.tabs.conversation}
            </TabsTrigger>
            <TabsTrigger value="metrics">{STRINGS.tabs.metrics}</TabsTrigger>
          </TabsList>
          <Select
            value={textRenderMode}
            onValueChange={(value) =>
              setTextRenderMode(value as TextRenderMode)
            }
          >
            <SelectTrigger
              variant="ghost"
              size="sm"
              className="ml-auto w-auto gap-1"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="karaoke">
                {STRINGS.textRenderMode.karaoke}
              </SelectItem>
              <SelectItem value="captions">
                {STRINGS.textRenderMode.captions}
              </SelectItem>
              <SelectItem value="instant">
                {STRINGS.textRenderMode.instant}
              </SelectItem>
            </SelectContent>
          </Select>
        </PanelHeader>
        <PanelContent className="p-0! overflow-hidden h-full">
          <TabsContent value="conversation" className="overflow-hidden h-full">
            <div className="flex h-full flex-col">
              <div className="chat-conversation">
                {messages.length > 0 ? (
                  <Conversation
                    noTextInput
                    textRenderMode={textRenderMode}
                    assistantLabel={STRINGS.conversation.roles.assistant}
                    clientLabel={STRINGS.conversation.roles.client}
                    systemLabel={STRINGS.conversation.roles.system}
                    functionCallLabel={STRINGS.conversation.roles.functionCall}
                  />
                ) : (
                  <ConversationPlaceholder />
                )}
              </div>
              <div className="p-3 border-t">
                <TextInput
                  classNames={{ container: 'items-center' }}
                  placeholder={STRINGS.textInput.placeholder}
                  noConnectedPlaceholder={STRINGS.textInput.noConnected}
                />
              </div>
            </div>
          </TabsContent>
          <TabsContent value="metrics" className="h-full">
            <MetricsPanel />
          </TabsContent>
        </PanelContent>
      </Panel>
    </Tabs>
  );
};
