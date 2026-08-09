import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const toastWarningMock = vi.hoisted(() => vi.fn())
const inTeamMock = vi.hoisted(() => ({ value: false }))
const authUserMock = vi.hoisted(() => ({ value: { id: "7" } as { id: string } | null }))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ inTeam: inTeamMock.value, user: authUserMock.value }),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
  parseApiResponse: async (response: { json: () => Promise<unknown> }) => ({
    data: await response.json(),
    text: null,
    isHtml: false,
  }),
  // Mirrors api-wrapper.ts: detail wins over message. Getting this backwards
  // silently drops the backend sentence and makes assertions about it vacuous.
  getUploadErrorMessage: (
    _response: unknown,
    parsed: { data?: { detail?: string; message?: string } | null },
    messages: { generic: string }
  ) => parsed?.data?.detail || parsed?.data?.message || messages.generic,
  getApiErrorMessage: (
    _response: unknown,
    parsed: { data?: { detail?: string; message?: string } | null },
    generic: string
  ) => parsed?.data?.detail || parsed?.data?.message || generic,
  isJsonRecord: (value: unknown) => typeof value === "object" && value !== null && !Array.isArray(value),
  UPLOAD_ERROR_MESSAGES: {},
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
  cn: (...classes: Array<string | false | null | undefined>) => classes.filter(Boolean).join(" "),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

// The component imports toast from this wrapper, not from `sonner` directly:
// mocking the raw package would leave the wrapper's injected options in the
// asserted arguments and force a meaningless matcher for them.
vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
    warning: toastWarningMock,
  },
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Upload: Icon,
    Globe: Icon,
    Settings: Icon,
    CheckCircle: Icon,
    Clock: Icon,
    XCircle: Icon,
    AlertCircle: Icon,
    FileText: Icon,
    Cloud: Icon,
    Database: Icon,
    User: Icon,
    Users: Icon,
    ChevronDown: Icon,
    ChevronUp: Icon,
    ArrowRight: Icon,
    ArrowLeft: Icon,
  }
})

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))

vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}))

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) => <label {...props}>{children}</label>,
}))

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}))

vi.mock("@/components/ui/card", () => ({
  Card: ({
    children,
    ...props
  }: React.HTMLAttributes<HTMLDivElement> & { children: React.ReactNode }) => <div {...props}>{children}</div>,
}))

vi.mock("@/components/ui/dialog", () => ({
  // Esc and the overlay close through onOpenChange, which the real Dialog owns.
  // The button stands in for both so the guard on that prop stays testable.
  Dialog: ({ children, onOpenChange }: { children: React.ReactNode; onOpenChange?: (open: boolean) => void }) => (
    <div>
      <button data-testid="dismiss-dialog" onClick={() => onOpenChange?.(false)}>dismiss</button>
      {children}
    </div>
  ),
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/textarea", () => ({
  Textarea: (props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) => <textarea {...props} />,
}))

vi.mock("@/components/ui/progress", () => ({
  Progress: ({ value }: { value: number }) => <div data-testid="progress">{value}</div>,
}))

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsList: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsTrigger: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))

vi.mock("@/components/ui/select", () => ({
  Select: () => <div />,
}))

// Keeps the padding prop visible: the step bodies are empty divs, so nothing
// else here would notice the stepper losing the gap above the step body.
vi.mock("@/components/ui/stepper", () => ({
  Stepper: ({ contentClassName }: { contentClassName?: string }) => (
    <div data-testid="stepper" data-content-class={contentClassName} />
  ),
}))

vi.mock("./cloud-connect-dialog", () => ({
  CloudConnectDialog: ({
    open,
    provider,
    onConfirm,
  }: {
    open: boolean
    provider: { id: string } | null
    onConfirm: (files: Array<{ id: string; name: string; size?: string }>) => void
  }) => (
    open && provider ? (
      <button
        data-testid="mock-cloud-confirm"
        onClick={() => onConfirm([{ id: `${provider.id}-file-1`, name: "alpha.pdf", size: "1 KB" }])}
      >
        mock cloud confirm
      </button>
    ) : null
  ),
}))

import { KnowledgeBaseCreationDialog } from "./knowledge-base-creation-dialog"

function createJsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  }
}

function createSucceededJob(result: Record<string, unknown>) {
  return {
    id: "job-1",
    user_id: 1,
    job_type: "kb.ingest.document",
    queue: "kb",
    status: "succeeded",
    progress: { message: "Completed", completed: 1, total: 1 },
    result,
    error_message: null,
    celery_task_id: "task-1",
    attempts: 1,
    max_attempts: 3,
  }
}

function installApiMocks() {
  apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
    if (url === "http://api.local/api/models/?category=embedding") {
      return Promise.resolve(createJsonResponse([]))
    }
    if (url === "http://api.local/api/models/user-default") {
      return Promise.resolve(createJsonResponse({}))
    }
    if (url === "http://api.local/api/jobs/capabilities") {
      return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
    }
    if (url.endsWith("/reserve-team") || url.endsWith("/release-team-claim")) {
      return Promise.resolve(createJsonResponse(null, 204))
    }
    if (url === "http://api.local/api/knowledge-bases/team-status") {
      return Promise.resolve(createJsonResponse([]))
    }
    if (url === "http://api.local/api/kb/ingest/jobs") {
      return Promise.resolve(
        createJsonResponse(
          createSucceededJob({
            status: "success",
            collection: (options?.body as FormData).get("collection"),
            document_count: 1,
            chunks_count: 1,
            message: "ok",
          })
        )
      )
    }

    throw new Error(`Unhandled apiRequest: ${url}`)
  })
}

const IMPORT_TABS = ["file", "web", "cloud"] as const
type ImportTab = (typeof IMPORT_TABS)[number]

/** Walk the wizard to step 3 (where the create button lives) for one import tab. */
async function goToStep3(container: HTMLElement, tab: ImportTab, fileCount = 1) {
  fireEvent.click(screen.getByText("common.next"))

  if (tab === "file") {
    fireEvent.change(container.querySelector("#file-upload") as HTMLInputElement, {
      target: {
        files: Array.from(
          { length: fileCount },
          (_, index) => new File(["a"], `file ${index}!.txt`, { type: "text/plain" })
        ),
      },
    })
  } else if (tab === "web") {
    fireEvent.click(screen.getByText("kb.dialog.tabs.web"))
    fireEvent.change(container.querySelector("#start_url") as HTMLInputElement, {
      target: { value: "https://example.com/docs" },
    })
  } else {
    fireEvent.click(screen.getByText("kb.dialog.tabs.cloud"))
    fireEvent.click(screen.getByText("kb.dialog.cloudConnect.googleDrive"))
    fireEvent.click(await screen.findByTestId("mock-cloud-confirm"))
    await waitFor(() => {
      expect(screen.getByText("alpha.pdf")).toBeInTheDocument()
    })
  }

  fireEvent.click(screen.getByText("common.next"))
}

/** The shared guard must warn, park the user on step 1, and ingest nothing. */
async function expectNameRejected(container: HTMLElement) {
  await waitFor(() => {
    expect(toastErrorMock).toHaveBeenCalledWith("kb.errors.nameRequired")
  })

  // Step 1 is the only step rendering the name field, so its presence proves
  // the user is where the problem can actually be fixed.
  const nameInput = container.querySelector("#collection_name")
  expect(nameInput).not.toBeNull()

  // A toast alone leaves a screen reader with nothing at the field itself.
  expect(nameInput?.getAttribute("aria-required")).toBe("true")
  expect(nameInput?.getAttribute("aria-invalid")).toBe("true")
  expect(screen.getByText("kb.errors.nameRequired")).toBeInTheDocument()
  expect(nameInput?.getAttribute("aria-describedby")).toBe("collection_name_error")
  expect(container.querySelector("label[for=collection_name]")?.textContent).toContain("*")

  expect(
    apiRequestMock.mock.calls.filter(([url]) => String(url).includes("/api/kb/ingest"))
  ).toHaveLength(0)
}

describe("KnowledgeBaseCreationDialog collection naming", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    toastWarningMock.mockReset()
    inTeamMock.value = false
    installApiMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it.each([
    ["an empty", ""],
    ["a whitespace-only", "   "],
  ])("refuses to leave step 1 with %s collection name", async (_label, value) => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    if (value) {
      fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
        target: { value },
      })
    }

    fireEvent.click(screen.getByText("common.next"))

    await expectNameRejected(container)
    // Step 2 owns the file picker: never rendering it proves we did not advance.
    expect(container.querySelector("#file-upload")).toBeNull()
  })

  it("clears the name error once the user starts typing", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.click(screen.getByText("common.next"))
    await expectNameRejected(container)

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "t" },
    })

    expect(container.querySelector("#collection_name")?.getAttribute("aria-invalid")).toBe("false")
    expect(screen.queryByText("kb.errors.nameRequired")).toBeNull()
  })

  it("clears the name error when the dialog is closed and reopened", async () => {
    const { container, rerender } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.click(screen.getByText("common.next"))
    await expectNameRejected(container)

    // Escape, the overlay and the close button all bypass the cancel handler,
    // so only the parent's `open` flag flips.
    rerender(<KnowledgeBaseCreationDialog open={false} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)
    rerender(<KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(container.querySelector("#collection_name")?.getAttribute("aria-invalid")).toBe("false")
    })
    expect(screen.queryByText("kb.errors.nameRequired")).toBeNull()
  })

  it("keeps the step body clear of the step indicator", async () => {
    // The stepper carries no bottom margin any more and every step body passed
    // to it is empty, so this padding is the whole gap.
    render(<KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)
    expect(screen.getByTestId("stepper").getAttribute("data-content-class")).toBe("pt-6")
  })

  it("previews the name the user typed", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })

    await goToStep3(container, "file")

    expect(screen.getByText("team-docs")).toBeInTheDocument()
    // The step-1 gate is what keeps the preview from ever needing a stand-in
    // name; the "KB <date>" fallback it used to render is gone.
    expect(container.textContent).not.toMatch(/KB \d/)
  })

  it("uses the same explicit collection name for each uploaded file", async () => {
    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })

    await goToStep3(container, "file", 2)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      const ingestCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest/jobs")
      expect(ingestCalls).toHaveLength(2)
      for (const [, options] of ingestCalls) {
        expect((options?.body as FormData).get("collection")).toBe("team-docs")
      }
    })

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["team-docs", "team-docs"])
    })
  })

  it("advises picking another name when the entered one is taken", async () => {
    // The reported #1139 path: this is the only screen where the user typed the
    // name, so it is the only one allowed to tell them to change it.
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url === "http://api.local/api/kb/ingest/jobs") {
        return Promise.resolve(
          createJsonResponse(
            {
              detail:
                "Knowledge base name unavailable: test.",
            },
            409
          )
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "test" },
    })

    fireEvent.click(screen.getByText("common.next"))
    fireEvent.change(container.querySelector("#file-upload") as HTMLInputElement, {
      target: { files: [new File(["a"], "alpha.txt", { type: "text/plain" })] },
    })
    fireEvent.click(screen.getByText("common.next"))
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.errors.nameUnavailable",
        expect.objectContaining({ description: "kb.errors.nameUnavailableHint" })
      )
    })

    // The backend sentence is replaced by the localized copy, not appended.
    expect(JSON.stringify(toastErrorMock.mock.calls)).not.toContain(
      "Knowledge base name unavailable"
    )
  })

  it("uses the sync ingest endpoint when background jobs are unavailable", async () => {
    const onSuccess = vi.fn()
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "sync" }))
      }
      if (url === "http://api.local/api/kb/ingest") {
        return Promise.resolve(
          createJsonResponse({
            status: "success",
            collection: (options?.body as FormData).get("collection"),
            document_count: 1,
            chunks_count: 1,
            message: "ok",
          })
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "alpha" },
    })

    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      const syncCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest")
      const jobCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest/jobs")
      expect(syncCalls).toHaveLength(1)
      expect(jobCalls).toHaveLength(0)
    })

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["alpha"])
    })
  })

  it("keeps the dialog open for cloud partial failures and surfaces the failure message", async () => {
    const onOpenChange = vi.fn()
    const onSuccess = vi.fn()
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url === "http://api.local/api/kb/ingest-cloud") {
        return Promise.resolve(
          createJsonResponse([
            {
              status: "partial",
              message: "Cloud import partially failed",
              doc_id: "doc-1",
              chunk_count: 2,
              embedding_count: 0,
              completed_steps: [{ name: "register_document" }],
              failed_step: "compute_embeddings",
            },
          ])
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    try {
      const { container } = render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={onOpenChange} onSuccess={onSuccess} />
      )

      fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
        target: { value: "cloud-docs" },
      })

      await goToStep3(container, "cloud")
      fireEvent.click(screen.getByText("kb.dialog.createButton"))

      await waitFor(() => {
        expect(toastErrorMock).toHaveBeenCalledWith(
          "kb.errors.cloudIngestFailed",
          expect.objectContaining({
            description: "Cloud import partially failed",
          })
        )
      })

      expect(toastSuccessMock).not.toHaveBeenCalled()
      expect(onOpenChange).not.toHaveBeenCalledWith(false)
      expect(onSuccess).not.toHaveBeenCalled()
      expect(await screen.findByText("Cloud import partially failed")).toBeInTheDocument()
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  it("keeps the dialog open for web partial failures and surfaces the failure message", async () => {
    const onOpenChange = vi.fn()
    const onSuccess = vi.fn()
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url === "http://api.local/api/kb/ingest-web/jobs") {
        return Promise.resolve(
          createJsonResponse(
            createSucceededJob({
              status: "partial",
              collection: "web_collection",
              total_urls_found: 1,
              pages_crawled: 1,
              pages_failed: 1,
              documents_created: 0,
              chunks_created: 0,
              embeddings_created: 0,
              crawled_urls: [],
              failed_urls: {
                "https://example.com/docs": "embedding missing",
              },
              message: "Web import partially failed",
              warnings: [],
              elapsed_time_ms: 0,
            })
          )
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    try {
      const { container } = render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={onOpenChange} onSuccess={onSuccess} />
      )

      fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
        target: { value: "web_collection" },
      })

      await goToStep3(container, "web")
      fireEvent.click(screen.getByText("kb.dialog.createButton"))

      await waitFor(() => {
        expect(toastErrorMock).toHaveBeenCalledWith(
          "kb.errors.webIngestFailed",
          expect.objectContaining({
            description: "Web import partially failed",
          })
        )
      })

      expect(toastSuccessMock).not.toHaveBeenCalled()
      expect(onOpenChange).not.toHaveBeenCalledWith(false)
      expect(onSuccess).not.toHaveBeenCalled()
      expect(await screen.findByText("kb.dialog.webImport.status.failed")).toBeInTheDocument()
      expect(await screen.findByText("Web import partially failed")).toBeInTheDocument()
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  it("refuses to close while an ingest is still running", async () => {
    // The consumers keep this dialog mounted and only toggle `open`, so a
    // request outliving a close still lands in onSuccess — which, in the agent
    // builder, attaches the knowledge base the user thought they abandoned.
    const onOpenChange = vi.fn()
    const onSuccess = vi.fn()
    mockRoute(
      (url) => url === "http://api.local/api/kb/ingest/jobs",
      () => new Promise(() => {})
    )

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={onOpenChange} onSuccess={onSuccess} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "pending-docs" },
    })
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(screen.getByText("kb.dialog.fileUpload.processing")).toBeInTheDocument()
    })

    expect(screen.getByText("common.cancel")).toBeDisabled()
    fireEvent.click(screen.getByTestId("dismiss-dialog"))
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
  })
})

const RESERVE_URL = "http://api.local/api/knowledge-bases/team-docs/reserve-team"
const RELEASE_URL = "http://api.local/api/knowledge-bases/team-docs/release-team-claim"

function callsTo(url: string) {
  return apiRequestMock.mock.calls.filter(([called]) => called === url)
}

/** Override one route, leaving every other route on the installed mock.

    Replacing the whole implementation means restating the embedding-model,
    user-default and jobs-capability routes in each test, which buries the one
    line that actually differs. */
function mockRoute(
  match: (url: string) => boolean,
  respond: (url: string, options?: RequestInit) => unknown
) {
  const base = apiRequestMock.getMockImplementation()!
  apiRequestMock.mockImplementation((url: string, options?: RequestInit) =>
    match(url) ? respond(url, options) : base(url, options)
  )
}

function firstCallIndex(predicate: (url: string) => boolean) {
  const index = apiRequestMock.mock.calls.findIndex(([url]) => predicate(String(url)))
  // Reject the miss here: a bare -1 makes `toBeLessThan` pass for a call that
  // never happened, which is the opposite of what the ordering assertions mean.
  expect(index).toBeGreaterThan(-1)
  return index
}

/** Name the knowledge base and pick Team, which only exists when `inTeam`. */
function nameAndChooseTeam(container: HTMLElement) {
  fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
    target: { value: "team-docs" },
  })
  fireEvent.click(container.querySelector("#kb-ownership-team") as HTMLInputElement)
}

describe("KnowledgeBaseCreationDialog ownership", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    toastWarningMock.mockReset()
    inTeamMock.value = true
    installApiMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it("hides the selector and never touches the team endpoints outside a team", async () => {
    // The open-source single-node build has no team and no /api/knowledge-bases
    // routes: the selector must not render and no request may be made.
    inTeamMock.value = false
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })
    expect(container.querySelector("#kb-ownership-team")).toBeNull()
    expect(container.querySelector("#kb-ownership-personal")).toBeNull()

    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(1)
    })
    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).includes("/api/knowledge-bases/"))
    ).toHaveLength(0)
  })

  it("acts on a key pressed while a child of the card holds focus", () => {
    // The handler sits on the group and reads event.target, which is the card
    // only while the card itself has focus. Anything nested — the icon, a label
    // — would otherwise resolve to no index and drop the key silently.
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    const team = container.querySelector("#kb-ownership-team") as HTMLElement
    const inner = team.querySelector("span") as HTMLElement
    expect(inner).toBeTruthy()

    fireEvent.keyDown(inner, { key: "Enter" })
    expect(team.getAttribute("aria-checked")).toBe("true")
  })

  it("moves the ownership choice with the keyboard, not just the mouse", () => {
    // These are Cards standing in for radios, so the keyboard handling that a
    // real radio would give for free has to be written out.
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    const team = container.querySelector("#kb-ownership-team") as HTMLElement
    const personal = container.querySelector("#kb-ownership-personal") as HTMLElement
    expect(team.getAttribute("role")).toBe("radio")
    expect(team.getAttribute("aria-checked")).toBe("false")

    fireEvent.keyDown(team, { key: "Enter" })
    expect(team.getAttribute("aria-checked")).toBe("true")
    expect(personal.getAttribute("aria-checked")).toBe("false")

    fireEvent.keyDown(personal, { key: " " })
    expect(personal.getAttribute("aria-checked")).toBe("true")
    expect(team.getAttribute("aria-checked")).toBe("false")
  })

  it("keeps one tab stop and moves the choice with the arrow keys", () => {
    // The APG radiogroup pattern: tab reaches the group once, arrows choose
    // inside it. Two tab stops would make the group a keyboard trap to walk.
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    const personal = container.querySelector("#kb-ownership-personal") as HTMLElement
    const team = container.querySelector("#kb-ownership-team") as HTMLElement
    expect(personal.getAttribute("tabindex")).toBe("0")
    expect(team.getAttribute("tabindex")).toBe("-1")

    fireEvent.keyDown(personal, { key: "ArrowRight" })
    expect(team.getAttribute("aria-checked")).toBe("true")
    // The tab stop follows the selection, and so does the focus, or the arrow
    // key would strand the user on a card that is no longer tabbable.
    expect(team.getAttribute("tabindex")).toBe("0")
    expect(personal.getAttribute("tabindex")).toBe("-1")
    expect(document.activeElement).toBe(team)

    // Wrapping keeps every option reachable from either end of the group.
    fireEvent.keyDown(team, { key: "ArrowRight" })
    expect(personal.getAttribute("aria-checked")).toBe("true")
    fireEvent.keyDown(personal, { key: "ArrowUp" })
    expect(team.getAttribute("aria-checked")).toBe("true")
  })

  it("jumps to the first and last option with Home and End", () => {
    // APG lists both alongside the arrow keys for a radiogroup.
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    const personal = container.querySelector("#kb-ownership-personal") as HTMLElement
    const team = container.querySelector("#kb-ownership-team") as HTMLElement

    fireEvent.keyDown(personal, { key: "End" })
    expect(team.getAttribute("aria-checked")).toBe("true")
    expect(document.activeElement).toBe(team)

    fireEvent.keyDown(team, { key: "Home" })
    expect(personal.getAttribute("aria-checked")).toBe("true")
    expect(document.activeElement).toBe(personal)
  })

  it("associates the group with its visible label instead of repeating it", () => {
    // A second copy of the text as `aria-label` would be read out twice and
    // still leave the label unassociated with the group.
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    const group = container.querySelector('[role="radiogroup"]') as HTMLElement
    expect(group.getAttribute("aria-labelledby")).toBe("kb-ownership-label")
    expect(group.getAttribute("aria-label")).toBeNull()
    expect(container.querySelector("#kb-ownership-label")?.textContent).toBe("kb.ownership.label")
  })

  it("defaults to personal inside a team and reserves nothing", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    expect(container.querySelector("#kb-ownership-personal")?.className).toContain("border-primary")

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(1)
    })
    expect(callsTo(RESERVE_URL)).toHaveLength(0)
  })

  it("still reserves when a token refresh transiently drops the team context", async () => {
    // A token refresh re-runs the team lookup, dropping `inTeam` for the length
    // of one request and taking the cards with it. Skipping the claim there
    // would hand the user a personal KB after they asked for a team one — the
    // exact complaint in #1140. The server holds the authoritative answer.
    const { container, rerender } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    nameAndChooseTeam(container)
    expect(container.querySelector("#kb-ownership-team")?.getAttribute("aria-checked")).toBe("true")

    inTeamMock.value = false
    rerender(<KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)
    // The cards stay put while Team is the choice, so the refresh cannot strand
    // the user on a selection they can no longer see or change.
    expect(container.querySelector("#kb-ownership-team")?.getAttribute("aria-checked")).toBe("true")

    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(1)
    })
    // The choice the user actually made still reaches the server, which is the
    // only party that can say whether they are still in a team.
    expect(callsTo(RESERVE_URL)).toHaveLength(1)
  })

  it.each([
    ["own bare claim", 7, true, "kb.ownership.nameHeldByYou"],
    ["a teammate's bare claim", 99, true, "kb.ownership.nameHeldByTeammate"],
    ["a built team knowledge base", 99, false, "kb.ownership.nameIsTeamKnowledgeBase"],
  ])(
    "warns on step 1 that the name is %s",
    async (_label, createdBy, isEmpty, expected) => {
      // A claim and a built KB are the same row server-side, so the warning has
      // to name which one it is -- one says wait, the other says demote.
      mockRoute(
        (url) => url === "http://api.local/api/knowledge-bases/team-status",
        () =>
          createJsonResponse([
            { name: "team-docs", created_by_user_id: createdBy, is_empty: isEmpty },
          ])
      )
      const { container } = render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
      )

      fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
        target: { value: "team-docs" },
      })
      expect(await screen.findByText(expected)).toBeInTheDocument()
    }
  )

  it("says nothing about a name the team does not hold", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "brand-new" },
    })
    await waitFor(() => {
      expect(callsTo("http://api.local/api/knowledge-bases/team-status")).toHaveLength(1)
    })
    expect(screen.queryByText("kb.ownership.nameHeldByTeammate")).toBeNull()
    expect(screen.queryByText("kb.ownership.nameHeldByYou")).toBeNull()
  })

  it("keeps a way back to personal after team membership is lost for good", async () => {
    // The control is normally gated on inTeam. Losing membership permanently
    // would take it away while every submit still reserved, with no way out of
    // the choice short of closing the dialog.
    const { container, rerender } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    nameAndChooseTeam(container)

    inTeamMock.value = false
    rerender(<KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)

    const personal = container.querySelector("#kb-ownership-personal") as HTMLElement
    expect(personal).not.toBeNull()
    fireEvent.click(personal)

    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))
    await waitFor(() => {
      expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(1)
    })
    expect(callsTo(RESERVE_URL)).toHaveLength(0)
    // Once back on personal the control is gated on inTeam again.
    expect(container.querySelector("#kb-ownership-personal")).toBeNull()
  })

  it("forgets the ownership choice when the dialog is closed and reopened", async () => {
    const { container, rerender } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.click(container.querySelector("#kb-ownership-team") as HTMLElement)
    expect(container.querySelector("#kb-ownership-team")?.getAttribute("aria-checked")).toBe("true")

    // Escape, the overlay and the close button all bypass `resetState`, so a
    // stale Team would come back preselected in a dialog that looks fresh.
    rerender(<KnowledgeBaseCreationDialog open={false} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)
    rerender(<KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(container.querySelector("#kb-ownership-personal")?.getAttribute("aria-checked")).toBe("true")
    })
    expect(container.querySelector("#kb-ownership-team")?.getAttribute("aria-checked")).toBe("false")
  })

  it("reserves the team name once, before the first of several file ingests", async () => {
    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file", 2)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["team-docs", "team-docs"])
    })
    // Once, not once per file: the loop shares one collection.
    expect(callsTo(RESERVE_URL)).toHaveLength(1)
    expect(callsTo(RESERVE_URL)[0][1]).toMatchObject({ method: "POST" })
    // Ownership is resolved before the first byte is written, so ordering is
    // the whole point of the call.
    expect(firstCallIndex((url) => url === RESERVE_URL)).toBeLessThan(
      firstCallIndex((url) => url.includes("/api/kb/ingest"))
    )
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
  })

  it.each(["web", "cloud"] as const)("reserves the team name on the %s path", async (tab) => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url.endsWith("/reserve-team")) {
        return Promise.resolve(createJsonResponse(null, 204))
      }
      if (url === "http://api.local/api/kb/ingest-web/jobs") {
        return Promise.resolve(
          createJsonResponse(
            createSucceededJob({
              status: "success",
              collection: "team-docs",
              total_urls_found: 1,
              pages_crawled: 1,
              pages_failed: 0,
              documents_created: 1,
              chunks_created: 1,
              embeddings_created: 1,
              crawled_urls: ["https://example.com/docs"],
              failed_urls: {},
              message: "ok",
              warnings: [],
              elapsed_time_ms: 0,
            })
          )
        )
      }
      if (url === "http://api.local/api/kb/ingest-cloud") {
        return Promise.resolve(createJsonResponse([{ status: "success", message: "ok", doc_id: "d1" }]))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, tab)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo(RESERVE_URL)).toHaveLength(1)
    })
    expect(firstCallIndex((url) => url === RESERVE_URL)).toBeLessThan(
      firstCallIndex((url) => url.includes("/api/kb/ingest"))
    )
    // A claim that succeeded and produced documents must be kept, not handed back.
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
  })

  it("does not ingest at all when the reservation is refused", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url.endsWith("/reserve-team")) {
        return Promise.resolve(createJsonResponse({ detail: "team storage is offline" }, 500))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      // The server's sentence, not the generic fallback: it is the only thing
      // that says what to do about it.
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.errors.uploadFailed",
        expect.objectContaining({ description: "team storage is offline" })
      )
    })
    // Ingesting anyway would write the files into personal storage under a name
    // the user asked to be a team knowledge base.
    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).includes("/api/kb/ingest"))
    ).toHaveLength(0)
    // Nothing was reserved, so nothing may be rolled back.
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
    expect(onSuccess).not.toHaveBeenCalled()
  })

  it("reports a missing reserve endpoint instead of quietly creating a personal KB", async () => {
    // `inTeam` and these endpoints come from the same overlay, so a 404 means a
    // deployment is mid-upgrade rather than a build without teams. The two
    // existing callers of promote-team surface that; a silent downgrade would
    // hand the user the ownership they did not pick and say nothing.
    mockRoute(
      (url) => url.endsWith("/reserve-team"),
      () => Promise.resolve(createJsonResponse({ detail: "Not Found" }, 404))
    )

    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalled()
    })
    expect(onSuccess).not.toHaveBeenCalled()
    // Nothing was ingested under an ownership the user did not ask for.
    expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(0)
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
  })

  it("does not fall back to personal when the server refuses the reservation", async () => {
    // 403 is a real answer ("you are in no team"), not a missing endpoint:
    // creating a personal knowledge base here would ignore what was asked for.
    mockRoute(
      (url) => url.endsWith("/reserve-team"),
      () => Promise.resolve(createJsonResponse({ detail: "You are not in a team" }, 403))
    )

    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.errors.uploadFailed",
        expect.objectContaining({ description: "You are not in a team" })
      )
    })
    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).includes("/api/kb/ingest"))
    ).toHaveLength(0)
    expect(onSuccess).not.toHaveBeenCalled()
  })

  it("releases the reserved name when the ingest fails, without hiding why", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url.endsWith("/reserve-team")) {
        return Promise.resolve(createJsonResponse(null, 204))
      }
      if (url.endsWith("/release-team-claim")) {
        return Promise.resolve(createJsonResponse(null, 204))
      }
      if (url === "http://api.local/api/kb/ingest/jobs") {
        return Promise.resolve(createJsonResponse({ message: "ingest blew up" }, 500))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo(RELEASE_URL)).toHaveLength(1)
    })
    expect(callsTo(RELEASE_URL)[0][1]).toMatchObject({ method: "POST" })
    expect(toastErrorMock).toHaveBeenCalledWith(
      "kb.errors.uploadFailed",
      expect.objectContaining({ description: "ingest blew up" })
    )
    expect(toastWarningMock).not.toHaveBeenCalled()
    // Releasing has no timeout, so it must not sit between the failure and the
    // toast explaining it.
    expect(toastErrorMock.mock.invocationCallOrder[0]).toBeLessThan(
      apiRequestMock.mock.invocationCallOrder[firstCallIndex((url) => url === RELEASE_URL)]
    )
  })

  it.each([
    {
      tab: "web" as const,
      ingestUrl: "http://api.local/api/kb/ingest-web/jobs",
      message: "web ingest blew up",
      title: "kb.errors.webIngestFailed",
    },
    {
      tab: "cloud" as const,
      ingestUrl: "http://api.local/api/kb/ingest-cloud",
      message: "cloud ingest blew up",
      title: "kb.errors.cloudIngestFailed",
    },
  ])("releases the reserved name when the $tab ingest fails", async ({ tab, ingestUrl, message, title }) => {
    // Each path decides for itself that it failed before reaching the shared
    // release helper, so the file path's coverage says nothing about these two.
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url.endsWith("/reserve-team") || url.endsWith("/release-team-claim")) {
        return Promise.resolve(createJsonResponse(null, 204))
      }
      if (url === ingestUrl) {
        return Promise.resolve(createJsonResponse({ message }, 500))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    try {
      const { container } = render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
      )

      nameAndChooseTeam(container)
      await goToStep3(container, tab)
      fireEvent.click(screen.getByText("kb.dialog.createButton"))

      await waitFor(() => {
        expect(callsTo(RELEASE_URL)).toHaveLength(1)
      })
      expect(callsTo(RELEASE_URL)[0][1]).toMatchObject({ method: "POST" })
      expect(toastErrorMock).toHaveBeenCalledWith(
        title,
        expect.objectContaining({ description: message })
      )
      // Releasing has no timeout, so it must not sit between the failure and
      // the toast explaining it.
      expect(toastErrorMock.mock.invocationCallOrder[0]).toBeLessThan(
        apiRequestMock.mock.invocationCallOrder[firstCallIndex((url) => url === RELEASE_URL)]
      )
      expect(toastWarningMock).not.toHaveBeenCalled()
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  // 403 (not the claim's creator), 404 and a thrown request all mean the same
  // thing: the name is still claimed and nobody would otherwise notice.
  it.each([
    ["a 403 response", () => Promise.resolve(createJsonResponse({ detail: "forbidden" }, 403))],
    ["a 404 response", () => Promise.resolve(createJsonResponse({ detail: "no claim" }, 404))],
    ["a rejected request", () => Promise.reject(new Error("rollback exploded"))],
  ])("warns, but does not mask the ingest error, on %s from the release", async (_label, release) => {
    const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url.endsWith("/reserve-team")) {
        return Promise.resolve(createJsonResponse(null, 204))
      }
      if (url.endsWith("/release-team-claim")) {
        return release()
      }
      if (url === "http://api.local/api/kb/ingest/jobs") {
        return Promise.resolve(createJsonResponse({ message: "ingest blew up" }, 500))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    try {
      const { container } = render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
      )

      nameAndChooseTeam(container)
      await goToStep3(container, "file")
      fireEvent.click(screen.getByText("kb.dialog.createButton"))

      await waitFor(() => {
        expect(toastWarningMock).toHaveBeenCalledWith(
          "kb.ownership.releaseFailed",
          // Longer than the 8s the wrapper forces on every error toast, or the
          // error stacked over this one outlives the only leak signal there is.
          expect.objectContaining({ duration: expect.any(Number) })
        )
      })
      expect(toastWarningMock.mock.calls[0][1].duration).toBeGreaterThan(8000)
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.errors.uploadFailed",
        expect.objectContaining({ description: "ingest blew up" })
      )
    } finally {
      consoleWarnSpy.mockRestore()
    }
  })

  it("treats a 409 from the release as the expected outcome, not a failure", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url.endsWith("/reserve-team")) {
        return Promise.resolve(createJsonResponse(null, 204))
      }
      if (url.endsWith("/release-team-claim")) {
        // The server saw a real collection behind the claim and refused.
        return Promise.resolve(createJsonResponse({}, 409))
      }
      if (url === "http://api.local/api/kb/ingest/jobs") {
        return Promise.resolve(createJsonResponse({ message: "ingest blew up" }, 500))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.errors.uploadFailed",
        expect.objectContaining({ description: "ingest blew up" })
      )
    })
    expect(toastWarningMock).not.toHaveBeenCalled()
  })

  it("releases the name when the reserve request throws after the server took it", async () => {
    // fetchWithRetry retries POST with no idempotent-method allowlist, so a
    // network fault can leave a committed reservation behind a thrown request.
    // A claim nobody knows about has no TTL and no UI to find it.
    mockRoute(
      (url) => url.endsWith("/reserve-team"),
      () => Promise.reject(new Error("network down"))
    )

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo(RELEASE_URL)).toHaveLength(1)
    })
    expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(0)
  })

  it("reports a taken name when the reservation conflicts", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url.endsWith("/reserve-team")) {
        return Promise.resolve(createJsonResponse({ detail: "Knowledge base already exists" }, 409))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      // "Failed to update ownership" would hide the one thing the user can act
      // on: the name is taken, pick another. A reserve-time 409 now carries its
      // status to the same classifier an ingest-time 409 reaches.
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.errors.nameUnavailable",
        expect.objectContaining({ description: "kb.errors.nameUnavailableHint" })
      )
    })
    // Same recovery as an empty name: step 1 is where the name can be changed,
    // not step 3 with a toast and two "Previous" clicks.
    const nameInput = container.querySelector("#collection_name")
    expect(nameInput).not.toBeNull()
    expect(nameInput?.getAttribute("aria-invalid")).toBe("true")
    expect(screen.getByText("kb.errors.nameTaken")).toBeInTheDocument()
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
  })

  it("still releases once when only some files landed, leaving the verdict to the server", async () => {
    let ingestCalls = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url.endsWith("/reserve-team")) {
        return Promise.resolve(createJsonResponse(null, 204))
      }
      if (url.endsWith("/release-team-claim")) {
        // A collection exists now, so the server is the one that says no.
        return Promise.resolve(createJsonResponse({}, 409))
      }
      if (url === "http://api.local/api/kb/ingest/jobs") {
        ingestCalls += 1
        if (ingestCalls > 1) {
          return Promise.resolve(createJsonResponse({ message: "second file blew up" }, 500))
        }
        return Promise.resolve(
          createJsonResponse(
            createSucceededJob({
              status: "success",
              collection: (options?.body as FormData).get("collection"),
              document_count: 1,
              chunks_count: 1,
              message: "ok",
            })
          )
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    nameAndChooseTeam(container)
    await goToStep3(container, "file", 2)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["team-docs"])
    })
    // The frontend no longer guesses whether the claim is empty: it always
    // asks, and the 409 keeps the live team knowledge base intact.
    expect(callsTo(RELEASE_URL)).toHaveLength(1)
    expect(toastWarningMock).not.toHaveBeenCalled()
  })
})
