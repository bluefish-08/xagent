/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const getWorkforceMock = vi.hoisted(() => vi.fn())
const getWorkforceRunMock = vi.hoisted(() => vi.fn())
const getWorkforceAgentExecutionMock = vi.hoisted(() => vi.fn())
const listWorkforceRunsMock = vi.hoisted(() => vi.fn())
const runWorkforceMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const closeFilePreviewMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())
const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const paramsMock = vi.hoisted(() => ({ id: "42" as string | string[] | undefined }))
const searchParamsMock = vi.hoisted(() => new URLSearchParams())
const translateMock = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) => {
    if (!vars) return key
    return Object.entries(vars).reduce(
      (value, [name, replacement]) => value.replace(`{${name}}`, String(replacement)),
      key,
    )
  },
)
const appState = vi.hoisted(() => ({
  currentTask: null as null | { id: string; status: string },
  traceEvents: [] as Array<Record<string, unknown>>,
  steps: [] as Array<Record<string, unknown>>,
  dagExecution: null as unknown,
  isHistoryLoading: false,
  messages: [] as Array<Record<string, unknown>>,
  filePreview: { isOpen: false, fileId: "", fileName: "", viewMode: "preview" },
}))

vi.mock("next/navigation", () => ({
  useParams: () => paramsMock,
  useRouter: () => ({ push: routerPushMock, replace: routerReplaceMock }),
  useSearchParams: () => searchParamsMock,
}))

vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ locale: "en-US", t: translateMock, tDynamic: translateMock }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    sendMessage: vi.fn(),
    setTaskId: setTaskIdMock,
    closeFilePreview: closeFilePreviewMock,
    dispatch: dispatchMock,
    openFilePreview: vi.fn(),
    getFileDownloadUrl: (fileId: string) => `/api/files/${fileId}/download`,
    state: appState,
  }),
  isTerminalTaskStatus: (status: string | null | undefined) => status === "completed" || status === "failed",
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: { id: "1", is_admin: false } }),
}))

vi.mock("@/components/config-dialog", () => ({
  ConfigDialog: ({ trigger }: { trigger: unknown }) => trigger,
}))

vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    dropdownPosition: null,
    fileList: [],
    filteredFiles: [],
    handleKeyDown: vi.fn(() => false),
    insertFile: vi.fn(),
    isLoadingFiles: false,
    resetMention: vi.fn(),
    selectedFileIndex: 0,
    showFilePicker: false,
  }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    hasAsrModel: false,
    startRecording: vi.fn(),
    status: "idle",
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => <div data-testid="task-conversation-panel" />,
}))

vi.mock("@/components/file/file-preview-content", () => ({
  FilePreviewContent: () => <div data-testid="file-preview" />,
}))

vi.mock("@/components/file/file-preview-action-buttons", () => ({
  FilePreviewActionButtons: () => <div data-testid="file-actions" />,
}))

vi.mock("@/lib/workforces-api", () => ({
  getWorkforce: getWorkforceMock,
  getWorkforceRun: getWorkforceRunMock,
  getWorkforceAgentExecution: getWorkforceAgentExecutionMock,
  listWorkforceRuns: listWorkforceRunsMock,
  runWorkforce: runWorkforceMock,
}))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>("@/lib/api-wrapper")
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("sonner", () => ({ toast: { error: toastErrorMock, success: vi.fn() } }))

import WorkforceRunPage from "./page-client"
import type { WorkforceDetail } from "@/types/workforce"

const workforceDetail: WorkforceDetail = {
  id: 42,
  name: "Launch Workforce",
  description: null,
  status: "active",
  manager: { id: 7, name: "Manager Agent", description: null, logo_url: null, status: "published" },
  workers: [],
  canvas_layout: null,
  scope_type: "user",
  scope_id: "1",
  owner_user_id: 1,
  created_at: null,
  updated_at: null,
}

function emptyJsonResponse(): Response {
  return new Response(JSON.stringify([]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function uploadResponse(fileId: string): Response {
  return new Response(JSON.stringify({ success: true, file_id: fileId }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function routeUploadResponse(fileId: string) {
  apiRequestMock.mockImplementation((url: string) =>
    typeof url === "string" && url.includes("/api/files/upload")
      ? Promise.resolve(uploadResponse(fileId))
      : Promise.resolve(emptyJsonResponse()),
  )
}

function getComposer(container: HTMLElement) {
  const editor = screen.getByRole("textbox")
  const submit = container.querySelector('form button[type="submit"]') as HTMLButtonElement
  return { editor, submit }
}

function typeMessage(editor: HTMLElement, text: string) {
  editor.innerHTML = text
  fireEvent.input(editor)
}

describe("workforce run first-message attachments", () => {
  beforeEach(() => {
    getWorkforceMock.mockReset()
    getWorkforceRunMock.mockReset()
    getWorkforceAgentExecutionMock.mockReset()
    listWorkforceRunsMock.mockReset().mockResolvedValue({ items: [], total: 0, page: 1, size: 20, pages: 0 })
    runWorkforceMock.mockReset()
    routerPushMock.mockReset()
    routerReplaceMock.mockReset()
    setTaskIdMock.mockReset()
    closeFilePreviewMock.mockReset()
    dispatchMock.mockReset()
    apiRequestMock.mockReset().mockImplementation(() => Promise.resolve(emptyJsonResponse()))
    toastErrorMock.mockReset()
    paramsMock.id = "42"
    searchParamsMock.delete("run")
    appState.currentTask = null
    appState.filePreview = { isOpen: false, fileId: "", fileName: "", viewMode: "preview" }
  })

  afterEach(() => cleanup())

  it("exposes a file upload control before the run starts", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)

    const { container } = render(<WorkforceRunPage />)
    await screen.findByText("workforces.run.placeholder")

    expect(container.querySelector('input[type="file"]')).not.toBeNull()
  })

  it("sends the first message with an uploaded file id", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    runWorkforceMock.mockResolvedValueOnce({ workforce_run_id: 5, task_id: 99, status: "running", redirect_url: "/task/99" })
    routeUploadResponse("file-123")

    const { container } = render(<WorkforceRunPage />)
    await screen.findByText("workforces.run.placeholder")
    const { editor, submit } = getComposer(container)

    const file = new File(["hello"], "brief.txt", { type: "text/plain", lastModified: 1 })
    fireEvent.change(container.querySelector('input[type="file"]') as HTMLInputElement, {
      target: { files: [file] },
    })
    await screen.findByText("brief.txt")

    typeMessage(editor, "Plan launch")
    await waitFor(() => expect(submit).not.toBeDisabled())
    fireEvent.click(submit)

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        message: "Plan launch",
        files: ["file-123"],
        is_visible: false,
      })
    })
  })

  it("sends the first message without files as an empty list", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    runWorkforceMock.mockResolvedValueOnce({ workforce_run_id: 5, task_id: 99, status: "running", redirect_url: "/task/99" })

    const { container } = render(<WorkforceRunPage />)
    await screen.findByText("workforces.run.placeholder")
    const { editor, submit } = getComposer(container)

    typeMessage(editor, "Plan launch")
    await waitFor(() => expect(submit).not.toBeDisabled())
    fireEvent.click(submit)

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        message: "Plan launch",
        files: [],
        is_visible: false,
      })
    })
  })

  it("keeps the attached file and the draft when the first run fails", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    runWorkforceMock.mockRejectedValueOnce(new Error("run failed"))
    routeUploadResponse("file-123")

    const { container } = render(<WorkforceRunPage />)
    await screen.findByText("workforces.run.placeholder")
    const { editor, submit } = getComposer(container)

    const file = new File(["hello"], "brief.txt", { type: "text/plain", lastModified: 1 })
    fireEvent.change(container.querySelector('input[type="file"]') as HTMLInputElement, {
      target: { files: [file] },
    })
    await screen.findByText("brief.txt")

    typeMessage(editor, "Plan launch")
    await waitFor(() => expect(submit).not.toBeDisabled())
    fireEvent.click(submit)

    await waitFor(() => expect(runWorkforceMock).toHaveBeenCalled())
    expect(await screen.findByText("brief.txt")).toBeInTheDocument()
    expect(editor).toHaveTextContent("Plan launch")
    expect(screen.queryByTestId("task-conversation-panel")).not.toBeInTheDocument()
    expect(toastErrorMock).toHaveBeenCalledTimes(1)
  })

  it("asks for a message instead of running with attachments only", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    routeUploadResponse("file-123")

    const { container } = render(<WorkforceRunPage />)
    await screen.findByText("workforces.run.placeholder")
    const { submit } = getComposer(container)

    const file = new File(["hello"], "brief.txt", { type: "text/plain", lastModified: 1 })
    fireEvent.change(container.querySelector('input[type="file"]') as HTMLInputElement, {
      target: { files: [file] },
    })
    await screen.findByText("brief.txt")
    await waitFor(() => expect(submit).not.toBeDisabled())
    fireEvent.click(submit)

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("workforces.run.messageRequired"))
    expect(toastErrorMock).toHaveBeenCalledTimes(1)
    expect(runWorkforceMock).not.toHaveBeenCalled()
    expect(screen.getByText("brief.txt")).toBeInTheDocument()
  })
})
