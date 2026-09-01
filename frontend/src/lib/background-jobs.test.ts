import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return { ...actual, apiRequest: apiRequestMock }
})

import {
  waitForBackgroundJob,
  type BackgroundJobResponse,
} from "@/lib/background-jobs"

function job(status: string): BackgroundJobResponse {
  return {
    id: "job-1",
    user_id: 1,
    job_type: "kb_ingest_web",
    queue: "default",
    status,
    attempts: 1,
    max_attempts: 3,
  }
}

function ok(status: string) {
  return { ok: true, json: async () => job(status) } as unknown as Response
}

const badGateway = { ok: false, status: 502, json: async () => ({}) } as unknown as Response

beforeEach(() => {
  apiRequestMock.mockReset()
  // Every poll sleeps 1s; collapse it to a macrotask so the loop advances without
  // wall time while still yielding, or a non-terminating regression hangs the worker
  // instead of failing the test.
  const realSetTimeout = window.setTimeout
  vi.spyOn(window, "setTimeout").mockImplementation(((fn: () => void) =>
    realSetTimeout(fn, 0)) as unknown as typeof window.setTimeout)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("waitForBackgroundJob", () => {
  it("survives transient poll failures and returns the terminal job", async () => {
    apiRequestMock
      .mockResolvedValueOnce(badGateway)
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValueOnce(ok("running"))
      .mockResolvedValueOnce(badGateway)
      .mockResolvedValueOnce(ok("succeeded"))

    await expect(waitForBackgroundJob("http://api.local", job("running"))).resolves.toMatchObject({
      status: "succeeded",
    })
  })

  it("gives up once failures are consecutive enough to be real", async () => {
    apiRequestMock.mockResolvedValue(badGateway)

    await expect(waitForBackgroundJob("http://api.local", job("running"))).rejects.toThrow(
      "Failed to fetch background job job-1",
    )
    expect(apiRequestMock).toHaveBeenCalledTimes(10)
  })

  it("counts failures consecutively, not cumulatively", async () => {
    for (let i = 0; i < 9; i++) apiRequestMock.mockResolvedValueOnce(badGateway)
    apiRequestMock.mockResolvedValueOnce(ok("running"))
    for (let i = 0; i < 9; i++) apiRequestMock.mockResolvedValueOnce(badGateway)
    apiRequestMock.mockResolvedValueOnce(ok("succeeded"))

    await expect(waitForBackgroundJob("http://api.local", job("running"))).resolves.toMatchObject({
      status: "succeeded",
    })
    expect(apiRequestMock).toHaveBeenCalledTimes(20)
  })

  it("keeps rejecting a malformed payload immediately", async () => {
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => ({ nope: true }) })

    await expect(waitForBackgroundJob("http://api.local", job("running"))).rejects.toThrow(
      "Invalid background job response",
    )
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })
})
