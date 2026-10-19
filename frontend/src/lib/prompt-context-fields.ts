// Mirrors the server-side PromptContextFieldDeclaration rules; the server stays
// authoritative and rejects anything that slips past these checks with a 422.
export const PROMPT_CONTEXT_FIELD_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_.-]{0,63}$/
export const PROMPT_CONTEXT_MAX_FIELDS = 32
export const PROMPT_CONTEXT_MIN_LENGTH = 1
export const PROMPT_CONTEXT_MAX_LENGTH = 1024
export const PROMPT_CONTEXT_DEFAULT_LENGTH = 256

export interface PromptContextFieldDeclaration {
  name: string
  max_length: number
}

export interface PromptContextFieldDraft {
  name: string
  maxLength: string
}

export type PromptContextFieldNameError = "required" | "invalid" | "duplicate"

export interface PromptContextFieldRowErrors {
  name?: PromptContextFieldNameError
  maxLength?: "invalid"
}

export interface PromptContextFieldsValidation {
  rows: PromptContextFieldRowErrors[]
  tooMany: boolean
  valid: boolean
}

export function newPromptContextFieldDraft(): PromptContextFieldDraft {
  return { name: "", maxLength: String(PROMPT_CONTEXT_DEFAULT_LENGTH) }
}

export function draftsFromDeclarations(value: unknown): PromptContextFieldDraft[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return []
    const { name, max_length } = item as Record<string, unknown>
    if (typeof name !== "string" || typeof max_length !== "number") return []
    return [{ name, maxLength: String(max_length) }]
  })
}

function parseMaxLength(raw: string): number | null {
  const trimmed = raw.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const value = Number(trimmed)
  if (value < PROMPT_CONTEXT_MIN_LENGTH || value > PROMPT_CONTEXT_MAX_LENGTH) return null
  return value
}

export function validatePromptContextFields(
  drafts: PromptContextFieldDraft[],
): PromptContextFieldsValidation {
  const counts = new Map<string, number>()
  for (const draft of drafts) {
    const name = draft.name.trim()
    if (name) counts.set(name, (counts.get(name) ?? 0) + 1)
  }

  const rows = drafts.map((draft) => {
    const errors: PromptContextFieldRowErrors = {}
    const name = draft.name.trim()
    if (!name) errors.name = "required"
    else if (!PROMPT_CONTEXT_FIELD_NAME_PATTERN.test(name)) errors.name = "invalid"
    else if ((counts.get(name) ?? 0) > 1) errors.name = "duplicate"
    if (parseMaxLength(draft.maxLength) === null) errors.maxLength = "invalid"
    return errors
  })

  const tooMany = drafts.length > PROMPT_CONTEXT_MAX_FIELDS
  const valid = !tooMany && rows.every((row) => !row.name && !row.maxLength)
  return { rows, tooMany, valid }
}

// Only meaningful for drafts that passed validatePromptContextFields.
export function toPromptContextFieldsPayload(
  drafts: PromptContextFieldDraft[],
): PromptContextFieldDeclaration[] {
  return drafts.map((draft) => ({
    name: draft.name.trim(),
    max_length: Number(draft.maxLength.trim()),
  }))
}

export function promptContextFieldsKey(drafts: PromptContextFieldDraft[]): string {
  return JSON.stringify(drafts.map((draft) => [draft.name.trim(), draft.maxLength.trim()]))
}
