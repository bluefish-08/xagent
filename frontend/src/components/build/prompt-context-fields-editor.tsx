"use client"

import React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useI18n } from "@/contexts/i18n-context"
import { cn } from "@/lib/utils"
import {
  PROMPT_CONTEXT_MAX_FIELDS,
  PROMPT_CONTEXT_MAX_LENGTH,
  PROMPT_CONTEXT_MIN_LENGTH,
  newPromptContextFieldDraft,
  validatePromptContextFields,
  type PromptContextFieldDraft,
} from "@/lib/prompt-context-fields"

interface PromptContextFieldsEditorProps {
  value: PromptContextFieldDraft[]
  onChange: (value: PromptContextFieldDraft[]) => void
  // A freshly added row is blank by design; flag it only once a save was attempted.
  revealEmpty: boolean
}

export function PromptContextFieldsEditor({
  value,
  onChange,
  revealEmpty,
}: PromptContextFieldsEditorProps) {
  const { t } = useI18n()
  const validation = validatePromptContextFields(value)
  const atLimit = value.length >= PROMPT_CONTEXT_MAX_FIELDS

  const updateRow = (index: number, patch: Partial<PromptContextFieldDraft>) => {
    onChange(value.map((row, i) => (i === index ? { ...row, ...patch } : row)))
  }

  return (
    <div className="space-y-2" data-testid="prompt-context-fields">
      <Label>{t("builds.configForm.promptContextFields.label")}</Label>
      <div className="text-xs text-muted-foreground mb-2 space-y-1">
        <p>{t("builds.configForm.promptContextFields.description")}</p>
        <p>{t("builds.configForm.promptContextFields.untrusted")}</p>
      </div>
      <div className="space-y-3">
        {value.length > 0 && (
          <div className="flex gap-2 text-xs font-medium text-muted-foreground" aria-hidden="true">
            <span className="flex-1">{t("builds.configForm.promptContextFields.nameColumn")}</span>
            <span className="w-28">{t("builds.configForm.promptContextFields.maxLengthColumn")}</span>
            <span className="invisible px-3 text-sm">{t("builds.configForm.promptContextFields.delete")}</span>
          </div>
        )}
        {value.map((row, index) => {
          const errors = validation.rows[index] ?? {}
          const nameError =
            errors.name === "required" && !revealEmpty ? undefined : errors.name
          const nameErrorId = `prompt-context-field-${index}-name-error`
          const lengthErrorId = `prompt-context-field-${index}-length-error`
          return (
            <div key={index} className="space-y-1">
              <div className="flex gap-2 items-start">
                <Input
                  value={row.name}
                  onChange={(e) => updateRow(index, { name: e.target.value })}
                  placeholder={t("builds.configForm.promptContextFields.namePlaceholder")}
                  aria-label={t("builds.configForm.promptContextFields.nameLabel", { index: index + 1 })}
                  aria-invalid={nameError ? true : undefined}
                  aria-describedby={nameError ? nameErrorId : undefined}
                  className={cn("flex-1 font-mono", nameError && "border-destructive")}
                />
                <Input
                  type="number"
                  inputMode="numeric"
                  min={PROMPT_CONTEXT_MIN_LENGTH}
                  max={PROMPT_CONTEXT_MAX_LENGTH}
                  step={1}
                  value={row.maxLength}
                  onChange={(e) => updateRow(index, { maxLength: e.target.value })}
                  aria-label={t("builds.configForm.promptContextFields.maxLengthLabel", { index: index + 1 })}
                  aria-invalid={errors.maxLength ? true : undefined}
                  aria-describedby={errors.maxLength ? lengthErrorId : undefined}
                  className={cn("w-28", errors.maxLength && "border-destructive")}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => onChange(value.filter((_, i) => i !== index))}
                >
                  {t("builds.configForm.promptContextFields.delete")}
                </Button>
              </div>
              {nameError && (
                <p id={nameErrorId} className="text-xs text-destructive">
                  {t(`builds.configForm.promptContextFields.errors.name.${nameError}`)}
                </p>
              )}
              {errors.maxLength && (
                <p id={lengthErrorId} className="text-xs text-destructive">
                  {t("builds.configForm.promptContextFields.errors.maxLength", {
                    min: PROMPT_CONTEXT_MIN_LENGTH,
                    max: PROMPT_CONTEXT_MAX_LENGTH,
                  })}
                </p>
              )}
            </div>
          )
        })}
        {validation.tooMany && (
          <p className="text-xs text-destructive">
            {t("builds.configForm.promptContextFields.errors.tooMany", { max: PROMPT_CONTEXT_MAX_FIELDS })}
          </p>
        )}
        <div className="flex items-center gap-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={atLimit}
            onClick={() => onChange([...value, newPromptContextFieldDraft()])}
          >
            {t("builds.configForm.promptContextFields.add")}
          </Button>
          <span className="text-xs text-muted-foreground">
            {t("builds.configForm.promptContextFields.limit", {
              count: value.length,
              max: PROMPT_CONTEXT_MAX_FIELDS,
            })}
          </span>
        </div>
      </div>
    </div>
  )
}
