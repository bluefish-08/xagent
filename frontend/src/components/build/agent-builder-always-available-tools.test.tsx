import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

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
      vars?.tools !== undefined ? `${key}=${vars.tools}` : key,
  }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: toastErrorMock, success: vi.fn() } }))

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
  MultiSelect: (props: { placeholder?: string }) => (
    <div data-testid="multi-select" data-placeholder={props.placeholder} />
  ),
}))
vi.mock("@/components/ui/select", () => ({ Select: () => null }))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"
const BLOCK_KEY = "builds.configForm.tools.alwaysAvailable"

const TOOLS_WITH_FLAGS = {
  tools: [
    { name: "execute_python_code", description: "", category: "basic", enabled: true, always_available: false },
    { name: "clock_a", description: "", category: "other", enabled: true, always_available: true },
    { name: "clock_b", description: "", category: "other", enabled: true, always_available: true },
    { name: "get_current_time", description: "", category: "other", enabled: true, always_available: false },
  ],
  skill_loader_tool: "skill_loader_x",
}

const TOOLS_WITHOUT_FLAGS = {
  tools: [
    { name: "execute_python_code", description: "", category: "basic", enabled: true },
    { name: "get_current_time", description: "", category: "other", enabled: true },
  ],
}

function agentResponse(skills: string[]) {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Agent",
    description: "",
    instructions: "You are an agent.",
    execution_mode: "balanced",
    models: { general: "10" },
    knowledge_bases: [],
    skills,
    tool_categories: ["basic"],
    suggested_prompts: [],
    logo_url: null,
    status: "draft",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    can_edit: true,
  }
}

function json(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
}

function installApi(toolsBody: unknown, agentSkills: string[]) {
  apiRequestMock.mockImplementation((url: string) => {
    if (url.endsWith("/api/kb/collections")) return json({ collections: [] })
    if (url.endsWith("/api/skills/")) return json([{ name: "writer" }])
    if (url.endsWith("/api/tools/available")) return json(toolsBody)
    if (url.endsWith("/api/models/?category=llm")) return json([])
    if (url.endsWith("/api/models/user-default")) return json([])
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`)) return json([])
    if (url.endsWith(`/api/agents/${AGENT_ID}`)) return json(agentResponse(agentSkills))
    if (url.includes("/api/mcp/servers")) return json([])
    return json({})
  })
}

async function renderLoaded(toolsBody: unknown, agentSkills: string[]) {
  installApi(toolsBody, agentSkills)
  render(<AgentBuilder agentId={AGENT_ID} />)
  await waitFor(() =>
    expect(screen.getByPlaceholderText("builds.configForm.name.placeholder")).toHaveValue("Agent")
  )
  await waitFor(() =>
    expect(
      screen
        .getAllByTestId("multi-select")
        .some((el) => el.getAttribute("data-placeholder") === "builds.configForm.tools.placeholder")
    ).toBe(true)
  )
}

const block = () => screen.queryByText((text) => text.startsWith(`${BLOCK_KEY}=`))

beforeEach(() => {
  apiRequestMock.mockReset()
  toastErrorMock.mockReset()
  vi.stubGlobal("WebSocket", vi.fn())
})

afterEach(() => cleanup())

describe("AgentBuilder always-available tools", () => {
  it("lists only backend-flagged tools and adds the skill loader once a skill is selected", async () => {
    await renderLoaded(TOOLS_WITH_FLAGS, [])

    expect(block()?.textContent).toBe(`${BLOCK_KEY}=clock_a, clock_b`)

    fireEvent.click(document.getElementById("selectAllSkills")!)

    await waitFor(() =>
      expect(block()?.textContent).toBe(`${BLOCK_KEY}=clock_a, clock_b, skill_loader_x`)
    )
  })

  it("includes the skill loader for an agent saved with skills", async () => {
    await renderLoaded(TOOLS_WITH_FLAGS, ["writer"])

    expect(block()?.textContent).toBe(`${BLOCK_KEY}=clock_a, clock_b, skill_loader_x`)
  })

  it("renders nothing when the response carries no always-available fields", async () => {
    await renderLoaded(TOOLS_WITHOUT_FLAGS, ["writer"])

    expect(block()).toBeNull()
  })
})
