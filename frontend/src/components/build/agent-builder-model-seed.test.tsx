import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Server-side creation paths persisted no model config, so opening such an
// agent in the builder rendered "--" and the required-model guard refused to
// save. The edit-mode seed fills the slot from the owner's own default, and
// must not count as a user edit (that would disable Publish on open).

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    getUploadApiUrl: () => "http://api.local",
    getWsUrl: () => "ws://api.local",
  }
})

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: {
      messages: [],
      traceEvents: [],
      currentTask: null,
      isProcessing: false,
      isHistoryLoading: false,
      taskId: null,
      filePreview: { isOpen: false },
      dagExecution: null,
      steps: [],
    },
    setTaskId: vi.fn(),
    sendMessage: vi.fn(),
    dispatch: vi.fn(),
    closeFilePreview: vi.fn(),
    pauseTask: vi.fn(),
    resumeTask: vi.fn(),
    openFilePreview: vi.fn(),
    requestStatus: vi.fn(),
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", user: { id: "1", is_admin: false } }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}))

vi.mock("@/components/layout/resizable-three-column-layout", () => ({
  ResizableThreeColumnLayout: ({ middlePanel }: { middlePanel: React.ReactNode }) => (
    <div>{middlePanel}</div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({ AgentBuilderChat: () => null }))
vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))
vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
}))
vi.mock("@/components/chat/FileMentionDropdown", () => ({ FileMentionDropdown: () => null }))
vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    isOpen: false,
    items: [],
    selectedIndex: 0,
    selectItem: vi.fn(),
    close: vi.fn(),
  }),
}))
vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: (props: any) => (
    <div data-testid="multi-select" data-placeholder={props.placeholder}>
      {(props.options || []).map((o: any) => o.value).join("|")}
    </div>
  ),
}))
vi.mock("@/components/ui/select", () => ({ Select: () => null }))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { toast } from "sonner"

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"
const DEFAULT_MODEL_ID = 42

function agentResponse(models: unknown, canEdit = true) {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Seed Test Agent",
    description: "",
    instructions: "You are a test agent.",
    execution_mode: "balanced",
    models,
    knowledge_bases: [],
    skills: [],
    tool_categories: [],
    suggested_prompts: [],
    logo_url: null,
    status: "draft",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    can_edit: canEdit,
  }
}

const userDefault = (modelId: number) => [
  { config_type: "general", model: { id: modelId, model_name: "seeded-llm" } },
]

type Gate = { release: () => void }

function installApi(opts: {
  models: unknown
  userDefaults?: unknown[]
  llms?: unknown[]
  canEdit?: boolean
  gateAgent?: Gate
  gateDefaults?: Gate
}) {
  const defer = (gate: Gate | undefined, value: Response) => {
    if (!gate) return Promise.resolve(value)
    return new Promise<Response>(resolve => {
      gate.release = () => resolve(value)
    })
  }
  apiRequestMock.mockImplementation(
    (url: string, o?: { method?: string; body?: string }) => {
      if (o?.method === "PUT")
        return Promise.resolve(
          new Response(JSON.stringify(agentResponse(opts.models)), { status: 200 })
        )
      if (url.endsWith("/api/kb/collections"))
        return Promise.resolve(
          new Response(JSON.stringify({ collections: [] }), { status: 200 })
        )
      if (url.endsWith("/api/skills/"))
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      if (url.endsWith("/api/tools/available"))
        return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
      if (url.endsWith("/api/models/?category=llm"))
        return Promise.resolve(
          new Response(JSON.stringify(opts.llms ?? []), { status: 200 })
        )
      if (url.endsWith("/api/models/user-default"))
        return defer(
          opts.gateDefaults,
          new Response(
            JSON.stringify(opts.userDefaults ?? userDefault(DEFAULT_MODEL_ID)),
            { status: 200 }
          )
        )
      if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      if (url.endsWith(`/api/agents/${AGENT_ID}`))
        return defer(
          opts.gateAgent,
          new Response(
            JSON.stringify(agentResponse(opts.models, opts.canEdit ?? true)),
            { status: 200 }
          )
        )
      if (url.includes("/api/mcp/servers"))
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
    }
  )
}

const updateButton = () => screen.getByText("builds.editor.header.update")
const publishButton = () => screen.getByText("builds.editor.header.publish")
const nameBox = () => screen.getByDisplayValue("Seed Test Agent")
const loaded = () =>
  waitFor(() => expect(screen.getByDisplayValue("Seed Test Agent")).toBeInTheDocument())

const savedModels = async () => {
  await waitFor(() => {
    const put = apiRequestMock.mock.calls.find(([, o]) => (o as any)?.method === "PUT")
    expect(put).toBeTruthy()
  })
  const put = apiRequestMock.mock.calls.find(([, o]) => (o as any)?.method === "PUT")
  return JSON.parse((put![1] as any).body).models
}

beforeEach(() => {
  apiRequestMock.mockReset()
  ;(globalThis as any).WebSocket = vi.fn()
})

afterEach(() => cleanup())

describe("AgentBuilder edit-mode general-model seed", () => {
  it("saves an agent whose stored config is null", async () => {
    installApi({ models: null })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    fireEvent.change(nameBox(), { target: { value: "Renamed" } })
    fireEvent.click(updateButton())

    expect((await savedModels()).general).toBe(DEFAULT_MODEL_ID)
  })

  it("treats a truthy-empty stored config as unset too", async () => {
    installApi({ models: {} })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    fireEvent.change(nameBox(), { target: { value: "Renamed" } })
    fireEvent.click(updateButton())

    expect((await savedModels()).general).toBe(DEFAULT_MODEL_ID)
  })

  it("leaves Publish enabled and Update reachable on open", async () => {
    // Publish posts no body, so Update is the only flow that persists the
    // seeded slot -- it has to stay reachable, and Publish must not be
    // blocked for the very agents the seed exists to unblock.
    installApi({ models: null })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    await waitFor(() => expect(updateButton()).not.toBeDisabled())
    expect(publishButton()).not.toBeDisabled()
  })

  it("persists the seeded slot when Update is clicked with nothing else changed", async () => {
    installApi({ models: null })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    await waitFor(() => expect(updateButton()).not.toBeDisabled())
    fireEvent.click(updateButton())

    expect((await savedModels()).general).toBe(DEFAULT_MODEL_ID)
  })

  it("still blocks Publish once the user edits something else", async () => {
    installApi({ models: null })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    fireEvent.change(nameBox(), { target: { value: "Renamed" } })

    await waitFor(() => expect(publishButton()).toBeDisabled())
  })

  it("does not overwrite a slot the owner already chose", async () => {
    installApi({ models: { general: 7 } })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    fireEvent.change(nameBox(), { target: { value: "Renamed" } })
    fireEvent.click(updateButton())

    expect((await savedModels()).general).toBe(7)
  })

  it("preserves the other slots it does not fill", async () => {
    installApi({ models: { compact: 3 } })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    fireEvent.change(nameBox(), { target: { value: "Renamed" } })
    fireEvent.click(updateButton())

    const models = await savedModels()
    expect(models.general).toBe(DEFAULT_MODEL_ID)
    expect(models.compact).toBe(3)
  })

  it("ignores a malformed entry without losing a valid one", async () => {
    installApi({
      models: null,
      userDefaults: [
        { config_type: "general", model: { id: DEFAULT_MODEL_ID } },
        { config_type: "general", model: null },
      ],
    })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    await waitFor(() => expect(updateButton()).not.toBeDisabled())
    fireEvent.click(updateButton())

    expect((await savedModels()).general).toBe(DEFAULT_MODEL_ID)
  })

  it("does not fall back to the first available LLM in edit mode", async () => {
    // Silently pinning "whatever is first in the model list" onto an agent
    // that already exists is a choice the owner never made; the required-model
    // guard keeps holding instead.
    installApi({
      models: null,
      userDefaults: [],
      llms: [{ id: 99, model_name: "some-llm" }],
    })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()

    fireEvent.change(nameBox(), { target: { value: "Renamed" } })
    fireEvent.click(updateButton())

    await waitFor(() => expect(toast.error).toHaveBeenCalled())
    expect(
      apiRequestMock.mock.calls.find(([, o]) => (o as any)?.method === "PUT")
    ).toBeUndefined()
  })

  it("shows the seeded model in the flow view when the viewer owns the agent", async () => {
    installApi({
      models: null,
      llms: [{ id: DEFAULT_MODEL_ID, model_name: "seeded-llm" }],
    })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()
    fireEvent.click(screen.getByText("builds.editor.viewTabs.flow"))

    await waitFor(() => expect(screen.getByText("seeded-llm")).toBeInTheDocument())
  })

  it("does not seed a read-only cross-user view", async () => {
    // userDefaultGeneralRef holds the VIEWER's default; seeding here would
    // render someone else's model as this agent's configuration.
    installApi({
      models: null,
      canEdit: false,
      llms: [{ id: DEFAULT_MODEL_ID, model_name: "seeded-llm" }],
    })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()
    fireEvent.click(screen.getByText("builds.editor.viewTabs.flow"))

    await waitFor(() => expect(screen.getByText("—")).toBeInTheDocument())
    expect(screen.queryByText("seeded-llm")).toBeNull()
  })

  it("seeds when the agent load resolves last", async () => {
    const gateAgent: Gate = { release: () => {} }
    installApi({ models: null, gateAgent })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalled())
    gateAgent.release()
    await loaded()

    fireEvent.change(nameBox(), { target: { value: "Renamed" } })
    fireEvent.click(updateButton())

    expect((await savedModels()).general).toBe(DEFAULT_MODEL_ID)
  })

  it("seeds when the user-default fetch resolves last", async () => {
    // The seed effect must wait for BOTH mount fetches: keyed on the agent
    // load alone it would run while the default is still in flight and never
    // re-run once it landed.
    const gateDefaults: Gate = { release: () => {} }
    installApi({ models: null, gateDefaults })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await loaded()
    gateDefaults.release()

    await waitFor(() => expect(publishButton()).not.toBeDisabled())
    fireEvent.change(nameBox(), { target: { value: "Renamed" } })
    fireEvent.click(updateButton())

    expect((await savedModels()).general).toBe(DEFAULT_MODEL_ID)
  })
})
