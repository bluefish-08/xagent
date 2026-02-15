"use client"

import React, { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, Plus, Bot, Trash2, Send, MessageSquare } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useRouter, useSearchParams } from "next/navigation"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

interface Agent {
  id: number
  name: string
  description: string
  logo_url: string | null
  status: string
  created_at: string
}

export default function BuildsPage() {
  const { t } = useI18n()
  const router = useRouter()
  const searchParams = useSearchParams()
  const [searchTerm, setSearchTerm] = useState("")
  const [currentPage, setCurrentPage] = useState(1)
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const itemsPerPage = 10

  // Check for template parameter and redirect to create page
  useEffect(() => {
    const templateId = searchParams.get("template")
    if (templateId) {
      // Redirect to create page with template parameter
      router.replace(`/build/new?template=${templateId}`)
    }
  }, [searchParams, router])

  // Fetch agents on mount
  const fetchAgents = async () => {
    try {
      setLoading(true)
      const response = await apiRequest(`${getApiUrl()}/api/agents`)
      if (response.ok) {
        const data = await response.json()
        setAgents(data)
      }
    } catch (error) {
      console.error("Failed to fetch agents:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAgents()
  }, [])

  const handlePublish = async (agentId: number) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/publish`, {
        method: "POST",
      })
      if (response.ok) {
        fetchAgents() // Refresh list
      }
    } catch (error) {
      console.error("Failed to publish agent:", error)
    }
  }

  const handleUnpublish = async (agentId: number) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/unpublish`, {
        method: "POST",
      })
      if (response.ok) {
        fetchAgents() // Refresh list
      }
    } catch (error) {
      console.error("Failed to unpublish agent:", error)
    }
  }

  const handleDelete = async (agentId: number) => {
    if (!confirm(t('builds.list.actions.deleteConfirm'))) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`, {
        method: "DELETE",
      })
      if (response.ok) {
        fetchAgents() // Refresh list
      }
    } catch (error) {
      console.error("Failed to delete agent:", error)
    }
  }

  // Filter agents based on search term
  const filteredAgents = agents.filter(agent =>
    agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (agent.description && agent.description.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  // Pagination logic
  const totalItems = filteredAgents.length
  const totalPages = Math.ceil(totalItems / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = Math.min(startIndex + itemsPerPage, totalItems)
  const currentAgents = filteredAgents.slice(startIndex, endIndex)

  const handleCreate = () => {
    router.push("/build/new")
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString()
  }

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="border-b flex justify-between items-center p-8">
        <div>
          <h1 className="text-3xl font-bold mb-1">{t("builds.list.header.title")}</h1>
          <p className="text-muted-foreground">{t("builds.list.header.description")}</p>
        </div>
        <Button onClick={handleCreate}>
          <Plus className="mr-2 h-4 w-4" />
          {t("builds.list.header.create")}
        </Button>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-6 space-y-6 overflow-auto">
        {/* Search */}
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={t("builds.list.search.placeholder")}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Loading State */}
        {loading ? (
          <div className="flex items-center justify-center h-[400px]">
            <div className="text-muted-foreground">Loading...</div>
          </div>
        ) : (
          <>
            {/* List */}
            {currentAgents.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {currentAgents.map((agent) => (
                  <div
                    key={agent.id}
                    className="group relative flex flex-col justify-between space-y-4 rounded-xl border bg-card p-6 shadow-sm transition-all hover:shadow-md hover:border-primary/50"
                  >
                    <div
                      className="flex-1 cursor-pointer"
                      onClick={() => router.push(`/build/${agent.id}`)}
                    >
                      <div className="space-y-4">
                        <div className="flex items-start justify-between">
                          <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary overflow-hidden">
                            {agent.logo_url ? (
                              <img src={`${getApiUrl()}${agent.logo_url}`} alt={agent.name} className="h-full w-full object-cover" />
                            ) : (
                              <Bot className="h-6 w-6" />
                            )}
                          </div>
                          <div className={`text-xs px-2 py-1 rounded-full capitalize ${
                            agent.status === 'published'
                              ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                              : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400'
                          }`}>
                            {agent.status === 'published' ? t('builds.list.status.published') : t('builds.list.status.draft')}
                          </div>
                        </div>
                        <div className="space-y-2">
                          <h3 className="font-semibold leading-none tracking-tight">{agent.name}</h3>
                          <p className="text-sm text-muted-foreground line-clamp-2">
                            {agent.description || "No description"}
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-3 pt-4 border-t">
                      <div className="text-xs text-muted-foreground">
                        {t('builds.card.createdAt')}: {formatDate(agent.created_at)}
                      </div>
                      <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                        {agent.status === 'published' ? (
                          <>
                            <Button
                              variant="default"
                              size="sm"
                              className="flex-1"
                              onClick={() => router.push(`/agent/${agent.id}`)}
                            >
                              <MessageSquare className="mr-1 h-3 w-3" />
                              {t('builds.list.actions.chat')}
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              className="px-3"
                              onClick={() => handleUnpublish(agent.id)}
                            >
                              {t('builds.list.actions.unpublish')}
                            </Button>
                          </>
                        ) : (
                          <Button
                            variant="outline"
                            size="sm"
                            className="flex-1"
                            onClick={() => router.push(`/build/${agent.id}`)}
                          >
                            {t('builds.list.actions.edit')}
                          </Button>
                        )}
                        <Button
                          variant="outline"
                          size="sm"
                          className="px-3"
                          onClick={() => handleDelete(agent.id)}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-center space-y-4 border rounded-lg bg-muted/10 border-dashed">
                <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                  <Bot className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-semibold text-lg">{t("builds.list.empty.title")}</h3>
                  <p className="text-muted-foreground max-w-sm mx-auto">
                    {t("builds.list.empty.description")}
                  </p>
                </div>
                <Button onClick={handleCreate} variant="outline">
                  <Plus className="mr-2 h-4 w-4" />
                  {t("builds.list.empty.create")}
                </Button>
              </div>
            )}

            {/* Pagination */}
            {totalItems > 0 && (
              <div className="flex items-center justify-between pt-4 border-t">
                <div className="text-sm text-muted-foreground">
                  {t("builds.list.pagination.summary", {
                    from: startIndex + 1,
                    to: Math.min(endIndex, totalItems),
                    total: totalItems
                  })}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                  >
                    {t("builds.list.pagination.prev")}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                    disabled={currentPage === totalPages}
                  >
                    {t("builds.list.pagination.next")}
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
