"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Settings, Cpu, Zap, Eye } from "lucide-react"
import { cn, getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import { useState, useEffect } from "react"
import { useI18n } from "@/contexts/i18n-context";

interface Model {
  id: number
  model_id: string
  model_provider: string
  model_name: string
  base_url?: string
  temperature?: number
  is_default: boolean
  is_small_fast: boolean
  is_visual: boolean
  is_compact: boolean
  description?: string
  created_at?: string
  updated_at?: string
  is_active: boolean
}

interface Task {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused"
  description: string
  createdAt: string | number
  updatedAt: string | number
  modelName?: string
  smallFastModelName?: string
  visualModelName?: string
}

interface ModelInfoDisplayProps {
  currentTask?: Task | null
  onConfigChange?: () => void
  className?: string
}

export function ModelInfoDisplay({ currentTask, onConfigChange, className }: ModelInfoDisplayProps) {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)
  const { t } = useI18n();

  // Fetch models for mapping
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true)
        const response = await apiRequest(`${getApiUrl()}/api/models/?category=llm`)
        if (response.ok) {
          const data = await response.json()
          setModels(data)
        }
      } catch (error) {
        console.error('Failed to fetch models:', error)
      } finally {
        setLoading(false)
      }
    }

    // Only fetch if we have a task with model names
    if (currentTask && (currentTask.modelName || currentTask.smallFastModelName || currentTask.visualModelName)) {
      fetchModels()
    }
  }, [currentTask])

  // Create mapping from model_name to model_id
  const modelNameToIdMap = models.reduce((acc, model) => {
    acc[model.model_name] = model.model_id
    return acc
  }, {} as Record<string, string>)

  // Get display names (prefer model_id, fallback to model_name)
  const getDisplayName = (modelName?: string) => {
    if (!modelName) return null
    return modelNameToIdMap[modelName] || modelName
  }

  if (!currentTask || (!currentTask.modelName && !currentTask.smallFastModelName && !currentTask.visualModelName)) {
    // 如果没有任务或没有模型信息，显示配置按钮
    // 但是如果 onConfigChange 为 undefined，则不显示按钮（由父组件处理）
    if (!onConfigChange) {
      return null
    }
    return (
      <Button
        variant="ghost"
        size="sm"
        onClick={onConfigChange}
        className={cn(
          "h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md",
          className
        )}
        title={t('agent.input.actions.config')}
        aria-label={t('agent.input.actions.config')}
      >
        <Settings className="h-3.5 w-3.5" />
      </Button>
    )
  }

  const mainModelDisplay = getDisplayName(currentTask.modelName)
  const smallFastModelDisplay = getDisplayName(currentTask.smallFastModelName)
  const visualModelDisplay = getDisplayName(currentTask.visualModelName)

  return (
    <div className="flex items-center gap-2">
      {/* 主模型显示 */}
      {mainModelDisplay && (
        <Badge
          variant="secondary"
          className="text-xs bg-blue-500/10 text-blue-600 border-blue-500/20 flex items-center gap-1"
          title={`${t('agent.configDialog.modelSelect.main.label')} (${currentTask.modelName})`}
        >
          <Cpu className="h-3 w-3" />
          {mainModelDisplay}
        </Badge>
      )}

      {/* 快速模型显示 */}
      {smallFastModelDisplay && (
        <Badge
          variant="secondary"
          className="text-xs bg-green-500/10 text-green-600 border-green-500/20 flex items-center gap-1"
          title={`${t('agent.configDialog.modelSelect.smallFast.label')} (${currentTask.smallFastModelName})`}
        >
          <Zap className="h-3 w-3" />
          {smallFastModelDisplay}
        </Badge>
      )}

      {/* 视觉模型显示 */}
      {visualModelDisplay && (
        <Badge
          variant="secondary"
          className="text-xs bg-purple-500/10 text-purple-600 border-purple-500/20 flex items-center gap-1"
          title={`${t('agent.configDialog.modelSelect.visual.label')} (${currentTask.visualModelName})`}
        >
          <Eye className="h-3 w-3" />
          {visualModelDisplay}
        </Badge>
      )}

      {/* 配置按钮 */}
      {onConfigChange && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onConfigChange}
          className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md"
          title={t('agent.config.title')}
          aria-label={t('agent.config.title')}
        >
          <Settings className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  )
}
