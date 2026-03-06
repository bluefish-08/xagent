import React from "react";
import { Bot, Sparkles } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
import { useI18n } from "@/contexts/i18n-context";

export interface PromptCard {
  icon?: any;
  title?: string;
  description?: string;
  prompt: string;
  color?: string;
  bg?: string;
}

interface ChatStartScreenProps {
  title: string;
  description?: string;
  icon?: React.ReactNode | string; // URL string or ReactNode
  prompts?: (PromptCard | string)[];
  onSend: (message: string, files: File[], config?: any) => void;
  isSending?: boolean;
  inputValue?: string;
  onInputChange?: (value: string) => void;
  files?: File[];
  onFilesChange?: (files: File[]) => void;
  showModeToggle?: boolean;
  readOnlyConfig?: boolean;
  taskConfig?: any;
}

export function ChatStartScreen({
  title,
  description,
  icon,
  prompts,
  onSend,
  isSending = false,
  inputValue,
  onInputChange,
  files = [],
  onFilesChange,
  showModeToggle = false,
  readOnlyConfig = false,
  taskConfig
}: ChatStartScreenProps) {
  const { t } = useI18n();

  const handlePromptClick = (prompt: string) => {
    if (onInputChange) {
      onInputChange(prompt);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] py-16 text-center">
      <div className="relative mb-6">
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-[hsl(var(--gradient-from))]/20 to-[hsl(var(--gradient-to))]/10 flex items-center justify-center animate-float">
          {typeof icon === 'string' ? (
            <img
              src={icon}
              alt={title}
              className="w-10 h-10 rounded-lg object-cover"
            />
          ) : (
            icon || <Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />
          )}
        </div>
        <div className="absolute -inset-4 rounded-3xl bg-gradient-to-br from-primary/5 via-accent/5 to-transparent blur-xl -z-10" />
      </div>
      <h2 className="text-2xl font-bold mb-2 gradient-text">
        {title}
      </h2>
      {description && (
        <p className="text-xs text-muted-foreground/70 mb-8 max-w-md">{description}</p>
      )}

      <div className="w-full max-w-4xl mx-auto space-y-8">
        <ChatInput
          onSend={(msg, config) => onSend(msg, files, config)}
          isLoading={isSending}
          files={files}
          onFilesChange={onFilesChange || (() => {})}
          showModeToggle={showModeToggle}
          inputValue={inputValue}
          onInputChange={onInputChange}
          readOnlyConfig={readOnlyConfig}
          taskConfig={taskConfig}
        />

        {prompts && prompts.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground/80 px-1">
              <Sparkles className="w-4 h-4" />
              <span>{t("chatPage.page.startWith")}</span>
            </div>
            <div className={`grid grid-cols-1 sm:grid-cols-2 ${prompts.length <= 3 ? 'lg:grid-cols-3' : 'lg:grid-cols-4'} gap-4`}>
              {prompts.map((item, index) => {
                const isString = typeof item === 'string';
                const promptText = isString ? item : item.prompt;

                if (isString) {
                   return (
                      <div
                        key={index}
                        onClick={() => handlePromptClick(promptText)}
                        className="group relative p-4 h-24 rounded-xl border border-border/40 bg-card/30 hover:bg-card hover:border-primary/50 cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-lg flex flex-col justify-center text-left"
                      >
                        <p className="text-sm text-foreground/90 line-clamp-2">{promptText}</p>
                      </div>
                   );
                }

                // Card style for Task Page
                return (
                    <div
                      key={index}
                      onClick={() => handlePromptClick(promptText)}
                      className="group relative p-4 h-32 rounded-xl border border-border/40 bg-card/30 hover:bg-card hover:border-primary/50 cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-lg flex flex-col justify-between items-start text-left"
                    >
                      <div className={`w-10 h-10 rounded-lg ${item.bg} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                        {item.icon && <item.icon className={`w-5 h-5 ${item.color}`} />}
                      </div>
                      <div>
                        <h3 className="font-medium text-sm text-foreground/90">{item.title}</h3>
                        <p className="text-xs text-muted-foreground/70 mt-1">{item.description}</p>
                      </div>
                    </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
