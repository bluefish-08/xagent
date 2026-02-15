"use client"

import { useCallback, useEffect, useState, useRef } from "react"
import {
  ReactFlow,
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  NodeTypes,
  EdgeTypes,
  Connection,
  OnConnect,
  BackgroundVariant,
  ReactFlowProvider,
  useReactFlow,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  Handle,
  Position,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { JsonRenderer } from "@/components/ui/markdown-renderer"
import { formatTime } from "@/lib/time-utils"
import { Loader2, Brain, Network, Sparkles, Timer, XCircle, AlertCircle, RefreshCw, RotateCcw, LayoutDashboard,LayoutPanelLeft, Wrench, GitBranch } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context";


interface DAGNode extends Node {
  data: {
    label: string
    status: "pending" | "running" | "completed" | "failed" | "skipped"
    description?: string
    tool_names?: string[]
    started_at?: string | number
    completed_at?: string | number
    result?: unknown
    conditional_branches?: Record<string, string>
    required_branch?: string | null
    is_conditional?: boolean
  }
}

interface DAGEdge extends Edge {
  data: {
    label?: string
  }
}

interface DAGExecution {
  phase: "planning" | "executing" | "completed" | "failed"
  current_plan: Record<string, unknown>
  created_at: string | number
  updated_at: string | number
}

interface CenterPanelProps {
  dagExecution: DAGExecution | null
  dagNodes: DAGNode[]
  dagEdges: DAGEdge[]
  onNodeClick?: (node: DAGNode) => void
  onRefresh?: () => void
  isPlanning?: boolean
  hasError?: boolean
  dagLayout?: 'TB' | 'LR'
  onLayoutChange?: (layout: 'TB' | 'LR') => void
  currentTaskStatus?: "pending" | "running" | "completed" | "failed" | "paused"
  onFileClick?: (filePath: string, fileName: string) => void
}


const TimeInformationRenderer = ({ data }: { data: any }) => {
  const { t, locale } = useI18n();
  if (!data.started_at && !data.completed_at) {
    return null
  }

  return (
    <div className="space-y-2 border-t border-border pt-3">
      <div className="text-sm font-medium text-foreground mb-2">{t("agent.layout.center.time.title")}</div>

      {data.started_at && (
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">{t("agent.layout.center.time.startedAt")}</span>
          <span className="font-mono text-foreground">
            {new Date(data.started_at).toLocaleString(locale || "zh-CN")}
          </span>
        </div>
      )}

      {data.completed_at && (
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">{t("agent.layout.center.time.completedAt")}</span>
          <span className="font-mono text-foreground">
            {new Date(data.completed_at).toLocaleString(locale || "zh-CN")}
          </span>
        </div>
      )}

      {data.started_at && data.completed_at && (
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">{t("agent.layout.center.time.duration")}</span>
          <span className="font-mono text-[#F59E0B]">
            {(() => {
              try {
                const start = new Date(data.started_at).getTime()
                const end = new Date(data.completed_at).getTime()
                const duration = end - start
                if (duration < 1000) return `${duration}ms`
                if (duration < 60000) return `${(duration / 1000).toFixed(1)}s`
                return `${(duration / 60000).toFixed(1)}min`
              } catch {
                return t("agent.layout.common.unknown")
              }
            })()}
          </span>
        </div>
      )}
    </div>
  )
}

// Error State Component
const ErrorState = () => {
  const { t } = useI18n();
  return (
    <div className="flex flex-col items-center justify-center h-full bg-background/50 backdrop-blur-sm">
      {/* Background decorations */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-10 left-10 w-32 h-32 bg-red-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-10 right-10 w-40 h-40 bg-red-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/4 w-24 h-24 bg-red-500/10 rounded-full blur-2xl animate-pulse delay-500"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 text-center space-y-6">
        {/* Error icon */}
        <div className="relative">
          <div className="w-20 h-20 mx-auto relative">
            <XCircle className="w-full h-full text-red-500 animate-pulse" />
            <div className="absolute inset-0 bg-red-500/20 rounded-full animate-ping"></div>
          </div>
        </div>

        {/* Main text */}
        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-foreground animate-in slide-in-from-bottom-4 duration-500">
            {t("agent.layout.center.errors.taskFailedTitle")}
          </h3>
          <p className="text-muted-foreground animate-in slide-in-from-bottom-4 duration-500 delay-150">
            {t("agent.layout.center.errors.taskFailedDesc")}
          </p>
        </div>

        {/* Suggested actions */}
        <div className="space-y-3 max-w-md mx-auto">
          <div className="flex items-center justify-center space-x-4 text-sm text-muted-foreground">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-4 w-4 text-red-400" />
              <span>{t("agent.layout.center.errors.checkLeft")}</span>
            </div>
            <div className="flex items-center space-x-2">
              <RefreshCw className="h-4 w-4 text-blue-400" />
              <span>{t("agent.layout.center.errors.retryTask")}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Planning Loading State Component
const PlanningLoadingState = () => {
  const { t } = useI18n();
  const [currentStep, setCurrentStep] = useState(0)
  const steps = [
    { icon: Brain, text: t("agent.layout.center.planning.steps.analyze"), color: "text-blue-400" },
    { icon: Network, text: t("agent.layout.center.planning.steps.buildGraph"), color: "text-purple-400" },
    { icon: Sparkles, text: t("agent.layout.center.planning.steps.optimizePath"), color: "text-yellow-400" },
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length)
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  // Don't show loading state if the task has failed
  useEffect(() => {
    // This component should only be shown when isPlanning = true
    // The parent component handles the logic
    console.log('PlanningLoadingState mounted, isPlanning prop:', true)
  }, [])

  return (
    <div className="flex flex-col items-center justify-center h-full bg-background/50 backdrop-blur-sm">
      {/* Background decorations */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-10 left-10 w-32 h-32 bg-primary/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-10 right-10 w-40 h-40 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/4 w-24 h-24 bg-blue-500/10 rounded-full blur-2xl animate-pulse delay-500"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 text-center space-y-8">
        {/* Animated brain icon */}
        <div className="relative">
          <div className="w-24 h-24 mx-auto relative">
            <Brain className="w-full h-full text-primary animate-pulse" />
            <div className="absolute inset-0 bg-primary/20 rounded-full animate-ping"></div>
            {/* Orbiting nodes */}
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="absolute w-2 h-2 bg-primary rounded-full"
                style={{
                  top: `${50 + 40 * Math.cos((i * 60 * Math.PI) / 180)}%`,
                  left: `${50 + 40 * Math.sin((i * 60 * Math.PI) / 180)}%`,
                  transform: 'translate(-50%, -50%)',
                  animation: `orbit ${2 + i * 0.5}s linear infinite`,
                }}
              />
            ))}
          </div>
        </div>

        {/* Main text */}
        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-foreground animate-in slide-in-from-bottom-4 duration-500">
            {t("agent.layout.center.planning.title")}
          </h3>
          <p className="text-muted-foreground animate-in slide-in-from-bottom-4 duration-500 delay-150">
            {t("agent.layout.center.planning.subtitle")}
          </p>
        </div>

        {/* Progress steps */}
        <div className="space-y-4 max-w-md mx-auto">
          <div className="flex justify-center space-x-8">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = index === currentStep
              const isCompleted = index < currentStep

              return (
                <div
                  key={index}
                  className={cn(
                    "flex flex-col items-center space-y-2 transition-all duration-500",
                    isActive ? "scale-110" : "scale-100",
                    isCompleted ? "opacity-100" : "opacity-60"
                  )}
                >
                  <div
                    className={cn(
                      "w-12 h-12 rounded-full border-2 flex items-center justify-center transition-all duration-500",
                      isActive
                        ? "border-primary bg-primary/10 shadow-lg shadow-primary/20"
                        : isCompleted
                        ? "border-green-500/50 bg-green-500/10"
                        : "border-muted-foreground/30 bg-muted/20",
                      step.color
                    )}
                  >
                    <Icon className={cn(
                      "w-5 h-5 transition-all duration-500",
                      isActive ? "animate-pulse" : "",
                      isCompleted ? "text-green-400" : step.color
                    )} />
                  </div>
                  <span
                    className={cn(
                      "text-xs font-medium transition-all duration-500",
                      isActive ? "text-foreground" : "text-muted-foreground"
                    )}
                  >
                    {step.text}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Progress bar */}
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-purple-500 rounded-full transition-all duration-1000 ease-out"
              style={{
                width: `${((currentStep + 1) / steps.length) * 100}%`,
              }}
            />
          </div>
        </div>

        {/* Loading spinner */}
        <div className="flex items-center justify-center space-x-2 text-muted-foreground">
          <Loader2 className="w-4 h-4 animate-spin" />
          <Timer className="w-4 h-4" />
          <span className="text-sm">{t("agent.layout.center.planning.spinner")}</span>
        </div>
      </div>
    </div>
  )
}

const nodeTypes: NodeTypes = {
  default: ({ data, isConnectable }) => {
    const { t } = useI18n();
    const statusStyles = {
      pending: "border-border text-muted-foreground",
      running: "border-[rgba(59,130,246,0.5)] text-primary animate-pulse",
      completed: "border-[rgba(34,197,94,0.5)] text-[#E6EDF3]",
      failed: "border-[rgba(239,68,68,0.5)] text-destructive",
      skipped: "border-dashed border-2 border-gray-500/50 text-gray-400 bg-gray-500/5",
    }

    const statusBadges = {
      pending: { variant: "secondary" as const, label: t("agent.layout.status.pending") },
      running: { variant: "default" as const, label: t("agent.layout.status.running") },
      completed: { variant: "default" as const, label: t("agent.layout.status.completed") },
      failed: { variant: "destructive" as const, label: t("agent.layout.status.failed") },
      skipped: { variant: "secondary" as const, label: t("agent.layout.status.skipped") },
    }

    const getDuration = () => {
      if (!data.started_at) return ""
      if (!data.completed_at) return t("agent.layout.center.time.inProgress")
      try {
        let start, end

        // Handle number type (seconds from backend)
        if (typeof data.started_at === 'number') {
          start = data.started_at * 1000
        } else {
          start = new Date(data.started_at).getTime()
        }

        if (typeof data.completed_at === 'number') {
          end = data.completed_at * 1000
        } else {
          end = new Date(data.completed_at).getTime()
        }

        const duration = end - start
        if (duration < 1000) return `${duration}ms`
        if (duration < 60000) return `${(duration / 1000).toFixed(1)}s`
        return `${(duration / 60000).toFixed(1)}min`
      } catch {
        return ""
      }
    }

    return (
      <>
        <div
          className={cn(
            "relative px-3 py-2 rounded-xl border-2 backdrop-blur-sm shadow-lg min-w-[180px] max-w-[220px] text-center transition-all duration-200 hover:shadow-xl bg-card",
            statusStyles[data.status as keyof typeof statusStyles]
          )}
        >
          <Handle
            type="target"
            position={Position.Top}
            style={{
              background: '#3b82f6',
              width: 8,
              height: 8,
              border: '2px solid #ffffff',
              top: -4,
              left: '50%',
              transform: 'translateX(-50%)',
              position: 'absolute',
            }}
            isConnectable={isConnectable}
          />
          {/* Step Name */}
          <div className="font-medium text-sm text-[#E6EDF3] mb-2 truncate px-1" title={data.label}>
            {data.label}
          </div>

          {/* Status Badge */}
          <div className="mb-2">
            <Badge
              variant={statusBadges[data.status as keyof typeof statusBadges].variant}
              className={`text-xs ${data.status === 'completed' ? 'bg-green-500/10 text-green-500' : ''}`}
            >
              {statusBadges[data.status as keyof typeof statusBadges].label}
            </Badge>
          </div>

          {/* Conditional Branch Indicator */}
          {data.is_conditional && data.conditional_branches && Object.keys(data.conditional_branches).length > 0 && (
            <div className="mb-2 px-2 py-1 bg-purple-500/10 border border-purple-500/20 rounded">
              <div className="flex items-center gap-1 text-xs text-purple-400">
                <GitBranch className="h-3 w-3" />
                <span className="font-medium">{t("agent.layout.center.labels.conditionalBranchNode")}</span>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {t("agent.layout.center.labels.branches")} {Object.keys(data.conditional_branches).join(", ")}
              </div>
            </div>
          )}

          {/* Required Branch Indicator */}
          {data.required_branch && (
            <div className="mb-2 px-2 py-1 bg-blue-500/10 border border-blue-500/20 rounded">
              <div className="flex items-center gap-1 text-xs text-blue-400">
                <GitBranch className="h-3 w-3" />
                <span className="font-medium">{t("agent.layout.center.labels.requiredBranch")}</span>
                <code className="bg-blue-500/20 px-1 py-0.5 rounded text-xs">
                  {data.required_branch}
                </code>
              </div>
            </div>
          )}

          {/* Tools */}
          {data.tool_names && data.tool_names.length > 0 && (
            <div className="text-xs text-muted-foreground font-mono mb-2 px-2 py-1 bg-muted rounded-md">
              <Wrench className="h-3 w-3 inline mr-1" />
              <span className="break-words whitespace-normal" title={data.tool_names.join(", ")}>
                {data.tool_names.join(", ")}
              </span>
            </div>
          )}

          {/* Time Information */}
          {(data.started_at || data.completed_at) && (
            <div className="text-xs text-muted-foreground space-y-1 border-t border-border pt-2 mt-2">
              {data.started_at && (
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">{t("agent.layout.center.time.startedAt")}</span>
                  <span className="font-mono">{formatTime(data.started_at)}</span>
                </div>
              )}
              {data.completed_at && (
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">{t("agent.layout.center.time.completedAt")}</span>
                  <span className="font-mono">{formatTime(data.completed_at)}</span>
                </div>
              )}
              {(data.started_at || data.completed_at) && (
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">{t("agent.layout.center.time.duration")}</span>
                  <span className="font-mono text-[#F59E0B]">{getDuration()}</span>
                </div>
              )}
            </div>
          )}

          {/* Description (if available and short) */}
          {data.description && data.description.length < 30 && (
            <div className="text-xs text-muted-foreground mt-2 px-1 italic truncate" title={data.description}>
              {data.description}
            </div>
          )}

          <Handle
            type="source"
            position={Position.Bottom}
            style={{
              background: '#3b82f6',
              width: 8,
              height: 8,
              border: '2px solid #ffffff',
              bottom: -4,
              left: '50%',
              transform: 'translateX(-50%)',
              position: 'absolute',
            }}
            isConnectable={isConnectable}
          />
        </div>
      </>
    )
  },
}

// Inner component that uses ReactFlow hooks
function CenterPanelInner({
  dagExecution,
  dagNodes,
  dagEdges,
  onNodeClick,
  onRefresh,
  isPlanning,
  hasError,
  dagLayout = 'TB',
  onLayoutChange,
  currentTaskStatus,
  onFileClick,
}: CenterPanelProps) {
  const { t } = useI18n();
  const [nodes, setNodes, onNodesChange] = useNodesState(dagNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(dagEdges)
  const [selectedNode, setSelectedNode] = useState<DAGNode | null>(null)
  const { fitView, zoomIn, zoomOut } = useReactFlow()
  const hasFittedView = useRef(false)

  // Format DAG execution time with better error handling
  const formatDagExecutionTime = (timestamp: string | number) => {
    try {
      let timestampNum: number

      // Convert to number if it's a string
      if (typeof timestamp === 'string') {
        timestampNum = parseInt(timestamp, 10)
        if (isNaN(timestampNum)) {
          return 'Invalid timestamp'
        }
      } else {
        timestampNum = timestamp
      }

      // Check if it's in seconds or milliseconds
      // If it's less than 10000000000 (year 2286), assume it's seconds
      const correctedTimestamp = timestampNum > 10000000000 ? timestampNum : timestampNum * 1000

      return new Date(correctedTimestamp).toLocaleString()
    } catch (error) {
      return 'Time error'
    }
  }

  // Get display phase based on task status and dagExecution
  const getDisplayPhase = () => {
    // If task is completed or failed, use that status regardless of dagExecution
    if (currentTaskStatus === "completed") return "completed"
    if (currentTaskStatus === "failed") return "failed"
    if (currentTaskStatus === "running") return "executing"
    if (currentTaskStatus === "pending") return "planning"
    if (currentTaskStatus === "paused") return "executing" // Treat paused as still executing

    // Otherwise use dagExecution phase
    return dagExecution?.phase || "planning"
  }

  const displayPhase = getDisplayPhase()


  useEffect(() => {
    setNodes(dagNodes)
    setEdges(dagEdges)
    hasFittedView.current = false
  }, [dagNodes, dagEdges, setNodes, setEdges])

  useEffect(() => {
    // Auto-fit view when nodes change, but only once per update
    if (nodes.length > 0 && !hasFittedView.current) {
      const timer = setTimeout(() => {
        fitView({ padding: 0.2, duration: 800 })
        hasFittedView.current = true
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [nodes, fitView])

  const onConnect: OnConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds))
    },
    [setEdges]
  )

  const onNodeClickHandler = useCallback((event: React.MouseEvent, node: Node) => {
    const dagNode = node as DAGNode
    setSelectedNode(dagNode)
    onNodeClick?.(dagNode)
  }, [onNodeClick])

  const getPhaseBadge = (phase: DAGExecution["phase"]) => {
    const variants = {
      planning: "secondary",
      executing: "default",
      completed: "default",
      failed: "destructive",
    } as const

    const labels = {
      planning: t("agent.layout.common.inProgress"),
      executing: t("agent.layout.common.inProgress"),
      completed: t("agent.status.completed"),
      failed: t("agent.status.failed"),
    }

    const customStyles = {
      planning: "bg-muted/50 text-muted-foreground border-border",
      executing: "bg-primary/10 text-primary border-primary/20",
      completed: "bg-green-500/10 text-green-500 border-green-500/20",
      failed: "bg-destructive/10 text-destructive border-destructive/20",
    }

    return (
      <Badge
        variant={variants[phase]}
        className={`text-xs border ${customStyles[phase]}`}
      >
        {labels[phase]}
      </Badge>
    )
  }


  return (
    <div className="flex flex-col h-full bg-background/80">
      {/* Header */}
      <div className="p-4 border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">{t("agent.layout.center.titles.dag")}</h2>

          {/* Layout Controls */}
          {dagNodes.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 bg-muted/50 rounded-md p-1">
                <Button
                  variant={dagLayout === 'TB' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => onLayoutChange?.('TB')}
                  className="h-8 px-3 text-xs"
                  title={t("agent.layout.center.labels.layoutVerticalTitle")}
                >
                  <LayoutDashboard className="h-3.5 w-3.5 mr-1" />
                  {t("agent.layout.center.labels.vertical")}
                </Button>
                <Button
                  variant={dagLayout === 'LR' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => onLayoutChange?.('LR')}
                  className="h-8 px-3 text-xs"
                  title={t("agent.layout.center.labels.layoutHorizontalTitle")}
                >
                  <LayoutPanelLeft className="h-3.5 w-3.5 mr-1" />
                  {t("agent.layout.center.labels.horizontal")}
                </Button>
              </div>
            </div>
          )}
        </div>
        {dagExecution && (
          <div key={`${displayPhase}-${dagExecution.updated_at}`} className="mt-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">{t("agent.layout.center.labels.phase")}</span>
              {getPhaseBadge(displayPhase)}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("agent.layout.center.labels.updatedAt")}: {formatDagExecutionTime(dagExecution.updated_at)}
            </div>
          </div>
        )}
        </div>

      {/* DAG Visualization */}
      <div className="flex-1 relative bg-background w-full h-full min-h-[500px]">
        {hasError ? (
          <ErrorState />
        ) : isPlanning ? (
          <PlanningLoadingState />
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClickHandler}
            fitView
            minZoom={0.05}
            maxZoom={2}
            defaultEdgeOptions={{
              style: {
                stroke: '#3b82f6',
                strokeWidth: 3,
              },
              animated: true,
            }}
            nodeTypes={nodeTypes}
            nodesDraggable={false}
            nodesConnectable={false}
            deleteKeyCode={null}
          >
          <Background
            color="hsl(var(--border))"
            gap={16}
            variant={BackgroundVariant.Dots}
          />
          <Controls
            className="bg-card border-border"
          />
          <MiniMap
            className="bg-card border-border"
            nodeColor={(node) => {
              const data = node.data as DAGNode['data']
              const colors = {
                pending: 'hsl(var(--muted-foreground))',
                running: 'hsl(var(--primary))',
                completed: 'hsl(142, 76%, 36%)',
                failed: 'hsl(var(--destructive))',
                skipped: 'hsl(220, 10%, 50%)',
              }
              return colors[data.status] || colors.pending
            }}
          />
        </ReactFlow>
        )}
      </div>

      {/* Node Details Panel */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 right-4 z-10">
          <Card className="bg-card/95 backdrop-blur-sm border-border shadow-2xl">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center justify-between text-foreground">
                {selectedNode.data.label}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedNode(null)}
                  className="text-muted-foreground hover:text-foreground"
                >
                  ×
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">{t("agent.layout.center.labels.status")}</span>
                <Badge variant="outline" className="text-xs border-border">
                  {selectedNode.data.status}
                </Badge>
              </div>

              {/* Conditional Branch Indicator */}
              {selectedNode.data.is_conditional && selectedNode.data.conditional_branches && Object.keys(selectedNode.data.conditional_branches).length > 0 && (
                <div className="flex items-start gap-2 p-2 bg-purple-500/10 border border-purple-500/20 rounded">
                  <GitBranch className="h-4 w-4 text-purple-400 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-purple-400 mb-1">{t("agent.layout.center.labels.conditionalBranchNode")}</div>
                    <div className="text-xs text-muted-foreground">
                      {t("agent.layout.center.labels.optionalBranches")}{": "}{Object.keys(selectedNode.data.conditional_branches).join(", ")}
                    </div>
                  </div>
                </div>
              )}

              {/* Required Branch Indicator */}
              {selectedNode.data.required_branch && (
                <div className="flex items-start gap-2 p-2 bg-blue-500/10 border border-blue-500/20 rounded">
                  <GitBranch className="h-4 w-4 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-blue-400 mb-1">{t("agent.layout.center.labels.branchCondition")}</div>
                    <div className="text-xs text-muted-foreground">
                      {t("agent.layout.center.labels.requiredBranch")} <code className="bg-blue-500/20 px-1 py-0.5 rounded">{selectedNode.data.required_branch}</code>
                    </div>
                  </div>
                </div>
              )}

              {selectedNode.data.tool_names && selectedNode.data.tool_names.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t("agent.layout.center.labels.tools")}</span>
                    <div className="flex flex-wrap gap-1">
                      {selectedNode.data.tool_names.map((tool, index) => (
                        <span key={index} className="font-mono text-foreground bg-muted px-2 py-1 rounded text-xs">
                          <Wrench className="h-3 w-3 inline mr-1" />
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {selectedNode.data.tool_names && selectedNode.data.tool_names.length === 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">{t("agent.layout.center.labels.tools")}</span>
                  <span className="font-mono text-foreground bg-muted px-2 py-1 rounded text-xs">
                    <Brain className="h-3 w-3 inline mr-1" />
                    {t("agent.layout.center.labels.pureAnalysis")}
                  </span>
                </div>
              )}

              {selectedNode.data.description && (
                <div>
                  <span className="text-muted-foreground">{t("agent.layout.center.labels.description")}</span>
                  <p className="mt-1 text-muted-foreground leading-relaxed bg-muted p-2 rounded">
                    {selectedNode.data.description}
                  </p>
                </div>
              )}

              {/* Result Data (if available) */}
              {selectedNode?.data?.result ? (
                <div className="space-y-2 border-t border-border pt-3">
                  <div className="text-sm font-medium text-foreground mb-2">{t("agent.layout.center.labels.result")}</div>
                  <div className="max-h-32 overflow-y-auto">
                    <JsonRenderer data={selectedNode.data.result} onFileClick={onFileClick} />
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

// Main component wrapped with ReactFlowProvider

export function CenterPanel(props: CenterPanelProps) {
  return (
    <ReactFlowProvider>
      <CenterPanelInner {...props} />
    </ReactFlowProvider>
  )
}
