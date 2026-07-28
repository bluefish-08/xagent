import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    tDynamic: (_key: string, fallback: string) => fallback,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn() }),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
}))

vi.mock("./TraceEventRenderer", () => ({
  TraceEventRenderer: () => <div data-testid="trace-renderer" />,
}))

vi.mock("@/components/ui/markdown-renderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}))

vi.mock("./clarification-form", () => ({
  ClarificationForm: () => <div data-testid="clarification-form" />,
}))

import { ChatMessage } from "./ChatMessage"

// A tool call is the worst case for the widget: its arguments and output are
// the raw payload the trace renderer would print.
const TRACE_EVENTS = [
  {
    event_id: "tool-1",
    event_type: "tool_call",
    timestamp: 1000,
    data: { tool_name: "web_search", args: { query: "secret" } },
  },
] as any

afterEach(() => {
  cleanup()
})

describe("ChatMessage process view", () => {
  it("renders the trace for internal pages", () => {
    render(
      <ChatMessage
        role="assistant"
        content="Here is the answer"
        traceEvents={TRACE_EVENTS}
        showProcessView={true}
      />
    )

    expect(screen.getByTestId("trace-renderer")).toBeTruthy()
    expect(screen.getByText("Here is the answer")).toBeTruthy()
  })

  it("renders the answer without the trace when the process view is off", () => {
    render(
      <ChatMessage
        role="assistant"
        content="Here is the answer"
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
      />
    )

    expect(screen.queryByTestId("trace-renderer")).toBeNull()
    expect(screen.getByText("Here is the answer")).toBeTruthy()
  })

  it("drops a trace-only turn instead of leaving an empty bubble", () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={false}
      />
    )

    expect(screen.queryByTestId("trace-renderer")).toBeNull()
    expect(container.textContent).toBe("")
    // An empty bubble is not empty markup: it still carries the assistant
    // avatar, so assert the avatar icon is gone too.
    expect(container.querySelector("svg")).toBeNull()
  })

  it("keeps the bubble for a trace-only turn that is awaiting input", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={false}
        interactions={[{ type: "select_one", field: "dataset", label: "Dataset" }]}
      />
    )

    expect(screen.getByTestId("clarification-form")).toBeTruthy()
  })

  it("keeps a generic status line while the answer is still streaming", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={true}
        taskStatus="running"
      />
    )

    // The internal step title ("calling web_search") is part of the trace, so
    // the hidden-trace status line must fall back to the neutral wording.
    expect(screen.getByText("common.thinking")).toBeTruthy()
    expect(screen.queryByText(/web_search/)).toBeNull()
  })

  it("names the running step when the process view is on", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={true}
        showEmptyStatus={true}
        taskStatus="running"
      />
    )

    expect(screen.queryByText("common.thinking")).toBeNull()
  })
})
