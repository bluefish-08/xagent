"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
// import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  Database,
  History,
  Send,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  Table,
  Search,
  Trash2,
  Brain,
  BarChart3,
  Copy,
  Download,
  PieChart,
  TrendingUp,
  LineChart,
  Plus
} from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import { useWebSocket } from "@/hooks/use-websocket"
import { SimpleBarChart, SimplePieChart, SimpleLineChart } from './components'
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { ThinkingTimeline } from "@/components/thinking-timeline"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"

interface Task {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused"
  description: string
  createdAt: string | number
  updatedAt: string | number
  agentType?: string
}

interface QueryResult {
  columns: string[]
  rows: any[]
  summary: string
}

interface Message {
  id: string
  type: "user" | "助手" | "system"
  content: string
  timestamp: number
  data?: any
  thinkingSteps?: ThinkingStep[] // 添加思考过程
  showThinking?: boolean // 是否显示思考过程
}

interface ThinkingStep {
  id: string
  name: string
  description: string
  status: "pending" | "running" | "completed" | "failed"
  type: "planning" | "analysis" | "sql_generation" | "execution" | "result"
  started_at?: string | number
  completed_at?: string | number
  dependencies?: string[]
  details?: {
    content?: string
    sql_query?: string
    result_data?: any
    error_message?: string
  }
  tool_names?: string[]
}

interface HistoricalTask {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused"
  created_at: string | number
  updated_at: string | number
}

export default function Text2SQLPage() {
  const { user, token } = useAuth()
  const { t } = useI18n()
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [historicalTasks, setHistoricalTasks] = useState<HistoricalTask[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isCreatingTask, setIsCreatingTask] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([])
  const [showThinking, setShowThinking] = useState(false)
  const [dagComplete, setDagComplete] = useState(false)
  const [isThinkingCollapsed, setIsThinkingCollapsed] = useState(false)
  const [selectedChart, setSelectedChart] = useState<string | null>(null)
  const [structuredQueryResult, setStructuredQueryResult] = useState<any>(null)
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table')

  // 从 URL 参数获取数据库信息
  const [databaseInfo, setDatabaseInfo] = useState({
    url: '',
    name: '',
    type: '',
    readOnly: true
  })

  // 输入框引用
  const inputRef = useRef<HTMLInputElement>(null)


  // 从 database_url 推断数据库类型的函数
  const inferDatabaseType = (url: string): string => {
    // 处理SQLAlchemy连接字符串格式，如 mysql+pymysql://
    if (url.startsWith('mysql+pymysql://') || url.startsWith('mysql://') || url.startsWith('mysql2://')) {
      return 'MySQL'
    } else if (url.startsWith('postgresql+psycopg://') || url.startsWith('postgresql://') || url.startsWith('postgres://')) {
      return 'PostgreSQL'
    } else if (url.startsWith('sqlite://')) {
      return 'SQLite'
    } else if (url.startsWith('mssql+pyodbc://') || url.startsWith('sqlserver://') || url.startsWith('mssql://')) {
      return 'SQL Server'
    } else if (url.startsWith('oracle+cx_oracle://') || url.startsWith('oracle://')) {
      return 'Oracle'
    } else if (url.startsWith('mongodb+')) {
      return 'MongoDB'
    }
    return 'SQLite' // 默认值
  }

  // 图表处理函数
  const handleChartAction = (chartType: string, data: any) => {
    setSelectedChart(chartType)
    console.log(`选择了 ${chartType} 图表`, data)

    switch (chartType) {
      case 'copy':
        navigator.clipboard.writeText(JSON.stringify(data, null, 2))
        console.log('数据已复制到剪贴板')
        break
      case 'export':
        // 简单的 CSV 导出
        const headers = data.columns || []
        const rows = data.rows || []
        let csv = headers.join(',') + '\n'
        rows.forEach((row: any) => {
          csv += headers.map((header: string) => `"${row[header] || ''}"`).join(',') + '\n'
        })

        const blob = new Blob([csv], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `query_result_${Date.now()}.csv`
        a.click()
        URL.revokeObjectURL(url)
        break
      default:
        // 图表类型切换
        break
    }
  }

  // 生成更具体的步骤描述（根据状态动态更新）
  const generateStepDescription = (data: any, toolNames: string[], status: string): string => {
    if (data.description && data.description !== '执行步骤中...') {
      return data.description
    }

    if (data.message) {
      return data.message
    }

    if (data.content) {
      return data.content
    }

    // 根据工具名称和状态生成描述
    if (toolNames && toolNames.length > 0) {
      const toolName = toolNames[0]
      if (toolName.includes('schema') || toolName.includes('analyze')) {
        return status === 'completed' ? t('agentStore.text2sql.chat.thinking.schemaCompleted') : t('agentStore.text2sql.chat.thinking.schemaRunning')
      } else if (toolName.includes('sql') || toolName.includes('query')) {
        return status === 'completed' ? t('agentStore.text2sql.chat.thinking.sqlGenCompleted') : t('agentStore.text2sql.chat.thinking.sqlGenRunning')
      } else if (toolName.includes('execute') || toolName.includes('run')) {
        return status === 'completed' ? t('agentStore.text2sql.chat.thinking.executeCompleted') : t('agentStore.text2sql.chat.thinking.executeRunning')
      } else if (toolName.includes('format') || toolName.includes('result')) {
        return status === 'completed' ? t('agentStore.text2sql.chat.thinking.formatCompleted') : t('agentStore.text2sql.chat.thinking.formatRunning')
      }
    }

    // 根据步骤名称和状态生成描述
    const stepName = data.step_name || data.name || ''
    if (stepName.toLowerCase().includes('plan') || stepName.toLowerCase().includes('规划')) {
      return status === 'completed' ? t('agentStore.text2sql.chat.thinking.planCompleted') : t('agentStore.text2sql.chat.thinking.planRunning')
    } else if (stepName.toLowerCase().includes('analyze') || stepName.toLowerCase().includes('分析')) {
      return status === 'completed' ? t('agentStore.text2sql.chat.thinking.analysisCompleted') : t('agentStore.text2sql.chat.thinking.analysisRunning')
    } else if (stepName.toLowerCase().includes('sql') || stepName.toLowerCase().includes('生成')) {
      return status === 'completed' ? t('agentStore.text2sql.chat.thinking.sqlStatementCompleted') : t('agentStore.text2sql.chat.thinking.sqlStatementRunning')
    } else if (stepName.toLowerCase().includes('execute') || stepName.includes('执行')) {
      return status === 'completed' ? t('agentStore.text2sql.chat.thinking.queryCompleted') : t('agentStore.text2sql.chat.thinking.queryRunning')
    }

    return status === 'completed' ? t('agentStore.text2sql.chat.thinking.stepCompleted') : t('agentStore.text2sql.chat.thinking.stepRunning')
  }

  // 从 trace 数据中解析 SQL 查询结果
  const parseQueryResultFromTrace = (traceData: any) => {
    try {
      if (!traceData) return null

      console.log('🎯 解析trace数据中的查询结果:', traceData)

      // 检查是否是 Text2SQL 结果数据
      if (traceData.type === 'text2sql_result' && traceData.structured_data) {
        console.log('🎯 找到Text2SQL结构化数据:', {
          type: traceData.type,
          hasStructuredData: !!traceData.structured_data,
          structuredDataKeys: traceData.structured_data ? Object.keys(traceData.structured_data) : [],
          hasColumns: traceData.structured_data?.columns,
          hasRows: traceData.structured_data?.rows,
          rowsCount: traceData.structured_data?.rows?.length,
          hasAISummary: !!traceData.ai_summary,
          dataKeys: Object.keys(traceData)
        })

        return {
          success: true,
          data: traceData,
          type: 'text2sql',
          message: traceData.ai_summary || t('agentStore.text2sql.chat.thinking.queryCompleted')
        }
      }

      return null
    } catch (e) {
      console.log('❌ 解析trace数据失败:', e)
      return null
    }
  }

  // 解析 SQL 查询结果
  const parseQueryResult = (messageContent: string): QueryResult | null => {
    if (!messageContent) return null;

    try {
      console.log('[FRONTEND] 解析消息内容:', messageContent.substring(0, 200));

      // 尝试解析 JSON 格式 (format_query_result 返回的格式)
      const trimmedContent = messageContent.trim();
      if (trimmedContent.startsWith('{') && trimmedContent.endsWith('}')) {
        const jsonData = JSON.parse(trimmedContent);
        console.log('[FRONTEND] 解析到 JSON 数据:', jsonData);

        if (jsonData.type === 'text2sql_result' && jsonData.success) {
          console.log('[FRONTEND] ✅ Text2SQL JSON 解析成功:', {
            hasColumns: !!jsonData.data?.columns,
            hasRows: !!jsonData.data?.rows,
            columnCount: jsonData.data?.columns?.length || 0,
            rowCount: jsonData.data?.rows?.length || 0,
            summary: jsonData.message
          });

          return {
            columns: jsonData.data?.columns || [],
            rows: jsonData.data?.rows || [],
            summary: jsonData.message
          };
        }
      }

      // 尝试查找 Text2SQL Result 格式 (向后兼容)
      const text2sqlMatch = messageContent.match(/\*\*Text2SQL Result:\*\*\n([\s\S]+?)(?=\n\n|$)/m);
      if (text2sqlMatch) {
        const text2sqlData = JSON.parse(text2sqlMatch[1]);
        console.log('[FRONTEND] 解析到Text2SQL结果数据 (旧格式):', text2sqlData);
        return {
          columns: text2sqlData.columns || [],
          rows: text2sqlData.rows || [],
          summary: text2sqlData.ai_summary || t('agentStore.text2sql.chat.thinking.queryCompleted')
        };
      }

      // 如果不是 JSON 格式，使用表格解析逻辑
      const tableResult = parseTableFromMessage(messageContent);
      if (tableResult && tableResult.columns && tableResult.rows) {
        console.log('[FRONTEND] 解析表格结果成功:', {
          columnCount: tableResult.columns.length,
          rowCount: tableResult.rows.length
        });
        return tableResult;
      }

      console.log('[FRONTEND] 未能解析查询结果');
      return null;
    } catch (error) {
      console.error('[FRONTEND] 解析查询结果失败:', error);
      return null;
    }
  };

  // 从消息中解析Markdown表格
  const parseTableFromMessage = (messageContent: string) => {
    try {
      console.log('🎯 [表格解析] 尝试从消息中解析表格');

      // 检查是否包含Markdown表格
      if (messageContent.includes('|') && messageContent.includes('---')) {
        const lines = messageContent.split('\n');
        const tableLines = lines.filter(line => line.trim().startsWith('|'));

        if (tableLines.length >= 3) {
          const headers = tableLines[0].split('|').map(h => h.trim()).filter(h => h);
          const rows = tableLines.slice(2).map(line =>
            line.split('|').map(cell => cell.trim()).filter(cell => cell)
          );

          if (headers.length > 0 && rows.length > 0) {
            console.log('🎯 [表格解析] 找到Markdown表格:', { headers, rowCount: rows.length });

            return {
              columns: headers,
              rows: rows,
              summary: t('agentStore.text2sql.chat.thinking.queryCompleted')
            };
          }
        }
      }

      return null;
    } catch (e) {
      console.log('❌ [表格解析] 失败:', e);
      return null;
    }
  }

  // 数据表格组件
  const DataDisplay = ({ data, type }: { data: any, type?: string }) => {
    if (!data) return null

    console.log('📊 DataDisplay 组件渲染:', {
      type,
      hasData: !!data,
      dataKeys: data ? Object.keys(data) : [],
      hasStructuredData: !!data.structured_data,
      hasAISummary: !!data.ai_summary,
      metadata: data.metadata
    });

    // 处理新的 Text2SQL Result 格式
    if (type === 'text2sql' && data.structured_data) {
      const { structured_data, ai_summary, metadata } = data

      console.log('📋 渲染 Text2SQL 数据表格:', {
        columns: structured_data.columns,
        rowsCount: structured_data.rows?.length,
        showAISummary: !!ai_summary,
        recordCount: metadata?.record_count
      });

      return (
        <div className="mt-4 space-y-4">
          {/* AI Summary */}
          {ai_summary && (
            <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <h4 className="font-medium text-sm mb-1 text-blue-800 dark:text-blue-200">💡 {t('agentStore.text2sql.chat.aiSummary.title')}</h4>
              <p className="text-sm text-blue-700 dark:text-blue-300">{ai_summary}</p>
            </div>
          )}

          {/* 原始数据表格 */}
          {structured_data.columns && structured_data.rows && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <h4 className="font-medium text-sm">
                  {t('agentStore.text2sql.chat.table.titleWithCount', { count: metadata?.record_count || structured_data.rows.length })}
                </h4>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">📊 {t('agentStore.text2sql.chat.table.buttons.chart')}</Button>
                  <Button size="sm" variant="outline">📋 {t('agentStore.text2sql.chat.table.buttons.copyData')}</Button>
                  <Button size="sm" variant="outline">💾 {t('agentStore.text2sql.chat.table.buttons.exportCsv')}</Button>
                </div>
              </div>

              <div className="border rounded-lg overflow-hidden">
                <div className="overflow-x-auto max-h-96">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50 sticky top-0">
                      <tr>
                        {structured_data.columns.map((col: string, idx: number) => (
                          <th key={idx} className="px-4 py-2 text-left font-medium border-b">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {structured_data.rows.map((row: any, rowIdx: number) => (
                        <tr key={rowIdx} className="hover:bg-muted/25 border-b">
                          {structured_data.columns.map((col: string, colIdx: number) => (
                            <td key={colIdx} className="px-4 py-2 border-r">
                              {row[col]?.toString() || '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )
    }


    // 处理 SQL 查询结果的格式：{ columns, rows, count, sql_query, success }
    if (data.columns && data.rows && Array.isArray(data.rows)) {
      const { columns, rows, count } = data

      return (
        <div className="mt-4 space-y-2">
          <div className="flex justify-between items-center">
            <h4 className="font-medium text-sm">{t('agentStore.text2sql.chat.table.titleWithCount', { count: count || rows.length })}</h4>
            <div className="flex gap-2">
              <Button size="sm" variant="outline">📊 {t('agentStore.text2sql.chat.table.buttons.chart')}</Button>
              <Button size="sm" variant="outline">📋 {t('agentStore.text2sql.chat.table.buttons.copyData')}</Button>
            </div>
          </div>

          <div className="border rounded-lg overflow-hidden">
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 sticky top-0">
                  <tr>
                    {columns.map((col: string, idx: number) => (
                      <th key={idx} className="px-4 py-2 text-left font-medium border-b">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row: any, rowIdx: number) => (
                    <tr key={rowIdx} className="hover:bg-muted/25 border-b">
                      {columns.map((col: string, colIdx: number) => (
                        <td key={colIdx} className="px-4 py-2 border-r">
                          {row[col]?.toString() || '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )
    }

    // 处理格式化查询结果的格式：{ type, data, total_count }
    if (data.type === 'table' && Array.isArray(data.data) && data.data.length > 0) {
      const columns = Object.keys(data.data[0])

      return (
        <div className="mt-4 space-y-2">
          <div className="flex justify-between items-center">
            <h4 className="font-medium text-sm">{t('agentStore.text2sql.chat.table.titleWithCount', { count: data.total_count || data.data.length })}</h4>
            <div className="flex gap-2">
              <Button size="sm" variant="outline">📊 {t('agentStore.text2sql.chat.table.buttons.chart')}</Button>
              <Button size="sm" variant="outline">📋 {t('agentStore.text2sql.chat.table.buttons.copyData')}</Button>
            </div>
          </div>

          <div className="border rounded-lg overflow-hidden">
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 sticky top-0">
                  <tr>
                    {columns.map((col, idx) => (
                      <th key={idx} className="px-4 py-2 text-left font-medium border-b">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.data.map((row: any, rowIdx: number) => (
                    <tr key={rowIdx} className="hover:bg-muted/25 border-b">
                      {columns.map((col, colIdx) => (
                        <td key={colIdx} className="px-4 py-2 border-r">
                          {row[col]?.toString() || '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )
    }

    // 非表格数据的显示
    return (
      <div className="mt-4 p-4 bg-muted/20 rounded-lg border">
        <div className="flex justify-between items-center mb-2">
          <h4 className="font-medium text-sm">{t('agentStore.text2sql.chat.table.title')}</h4>
          <Button size="sm" variant="outline">📋 {t('agentStore.text2sql.chat.table.buttons.copy')}</Button>
        </div>
        <pre className="text-sm whitespace-pre-wrap font-mono bg-background p-3 rounded border max-h-64 overflow-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    )
  }

  // 生成更友好的步骤名称
  const generateStepName = (data: any, toolNames: string[]): string => {
    if (data.step_name && data.step_name !== '未知步骤' && !data.step_name.startsWith('step')) {
      return data.step_name
    }

    if (data.name && data.name !== '未知步骤' && !data.name.startsWith('step')) {
      return data.name
    }

    // 根据工具名称生成友好的步骤名称
    if (toolNames && toolNames.length > 0) {
      const toolName = toolNames[0]
      if (toolName.includes('analyze_database_schema')) {
        return t('agentStore.text2sql.chat.steps.analyzeSchema')
      } else if (toolName.includes('execute_sql_query')) {
        return t('agentStore.text2sql.chat.steps.executeSql')
      } else if (toolName.includes('format_query_result')) {
        return t('agentStore.text2sql.chat.steps.formatResult')
      }
    }

    return t('agentStore.text2sql.chat.steps.defaultStep')
  }

  // 映射步骤类型到思考步骤类型
  const mapStepType = (stepType?: string, stepName?: string, toolNames?: string[]): ThinkingStep['type'] => {

    // 根据工具名称判断
    if (toolNames && toolNames.length > 0) {
      if (toolNames.some(tool => tool.includes('sql') || tool.includes('database'))) {
        return 'sql_generation'
      } else if (toolNames.some(tool => tool.includes('web_search'))) {
        return 'planning' // web搜索归类为规划（不正确的搜索行为）
      } else if (toolNames.some(tool => tool.includes('search') || tool.includes('analyze'))) {
        return 'analysis'
      } else if (toolNames.some(tool => tool.includes('execute') || tool.includes('query'))) {
        return 'execution'
      } else {
        return 'analysis' // 默认归类为分析
      }
    }

    // 根据步骤名称判断 - 改进的匹配逻辑
    if (stepName) {
      const name = stepName.toLowerCase()
      if (name.includes('plan') || name.includes('planning') || name.includes('思考') || name.includes('分析')) {
        return 'planning'
      } else if (name.includes('sql') || name.includes('生成') || name.includes('构建') || name.includes('query')) {
        return 'sql_generation'
      } else if (name.includes('execute') || name.includes('查询') || name.includes('执行') || name.includes('run')) {
        return 'execution'
      } else if (name.includes('result') || name.includes('结果') || name.includes('整理') || name.includes('output')) {
        return 'result'
      } else if (name.includes('analyze') || name.includes('探索') || name.includes('schema') || name.includes('check')) {
        return 'analysis'
      }
    }

    // 根据步骤类型判断
    if (stepType) {
      const type = stepType.toLowerCase()
      if (type.includes('agent') || type.includes('plan')) {
        return 'planning'
      } else if (type.includes('tool') || type.includes('sql')) {
        return 'sql_generation'
      } else if (type.includes('execute')) {
        return 'execution'
      }
    }

    return 'analysis' // 默认值
  }

  // 自动折叠思考过程
  const handleAutoCollapse = () => {
    setShowThinking(false)
    // 保留最近的思考步骤，但不展开显示
    // 用户可以手动重新展开查看
  }

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const databaseUrl = urlParams.get('database_url') || ''
    const databaseType = urlParams.get('database_type') || inferDatabaseType(databaseUrl)

    console.log('URL params:', {
      databaseType: urlParams.get('database_type'),
      databaseUrl,
      inferredType: inferDatabaseType(databaseUrl)
    })

    setDatabaseInfo({
      url: databaseUrl,
      name: urlParams.get('database_name') || '',
      type: databaseType,
      readOnly: urlParams.get('read_only') !== 'false'
    })
  }, [])

  // WebSocket 连接 - 只有在有任务时才连接
  const [wsEnabled, setWsEnabled] = useState(false)
  const { sendMessage, isConnected, lastMessage } = useWebSocket({
    taskId: wsEnabled && currentTask ? parseInt(currentTask.id) : undefined,
    onMessage: (message) => {
      // 只记录重要消息类型
      if (message.type === 'trace_event' && message.event_type && ['ai_message', 'user_message', 'dag_execution', 'dag_step_info', 'dag_step_start', 'dag_step_end'].includes(message.event_type)) {
        console.log('🎯 重要消息:', message.type, message.event_type, message.data)

        // 详细分析不同事件类型的数据结构
        if (message.event_type === 'ai_message') {
          const data = message.data as any
          console.log('AI消息详细数据:', {
            message: data?.message,
            content: data?.content,
            timestamp: data?.timestamp,
            data_keys: Object.keys(data || {})
          })
        } else if (message.event_type === 'user_message') {
          const data = message.data as any
          console.log('用户消息详细数据:', {
            message: data?.message,
            content: data?.content,
            timestamp: data?.timestamp,
            data_keys: Object.keys(data || {})
          })
        }
      } else if (message.type === 'chat') {
        console.log('聊天消息:', message.data)
      }

      if (message.type === 'chat' && message.data) {
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          type: "助手",
          content: (message.data as any).message || "处理完成",
          timestamp: Date.now(),
          data: (message.data as any).data
        }
        setMessages(prev => [...prev, assistantMessage])
        setIsLoading(false)
      } else if (message.type === 'trace_event') {
        // 主要处理 trace_event 类型，根据 event_type 分发
        const eventType = message.event_type || (message.data as any)?.event_type
        const eventData = message.data as any
        console.log('收到trace_event:', eventType, eventData)

        if (eventType === 'dag_execution') {
          console.log('DAG执行状态:', eventData)


          if (eventData.phase === 'planning') {
            console.log('DAG规划阶段开始')
            setShowThinking(true)
            setDagComplete(false)

            // 确保显示助手消息的思考过程
            setMessages(prev => {
              const updated = [...prev]
              const text2sqlMessageIndex = updated.findIndex(msg => msg.type === "助手" && msg.id.includes('text2sql-response'))

              if (text2sqlMessageIndex >= 0) {
                // 更新现有消息，添加规划步骤
                const planningStep = {
                  id: 'planning-step',
                  name: '查询策略规划',
                  description: '正在分析您的查询需求并制定最佳的执行策略...',
                  status: 'running' as const,
                  type: 'planning' as const
                }

                updated[text2sqlMessageIndex] = {
                  ...updated[text2sqlMessageIndex],
                  thinkingSteps: [planningStep],
                  showThinking: true,
                  content: updated[text2sqlMessageIndex].content || '正在处理您的查询...'
                }
              } else {
                // 创建新的助手消息来显示思考过程
                const assistantMessage: Message = {
                  id: `text2sql-response-${Date.now()}`,
                  type: "助手",
                  content: "正在处理您的查询...",
                  timestamp: Date.now(),
                  thinkingSteps: [{
                    id: 'planning-step',
                    name: '查询策略规划',
                    description: '正在分析您的查询需求并制定最佳的执行策略...',
                    status: 'running' as const,
                    type: 'planning'
                  }],
                  showThinking: true
                }
                console.log('🆕 创建新的助手消息 (规划阶段):', assistantMessage.id)
                updated.push(assistantMessage)
              }

              return updated
            })

            // 同时更新全局状态用于兼容
            setThinkingSteps([{
              id: 'planning-step',
              name: '查询策略规划',
              description: '正在分析您的查询需求并制定最佳的执行策略...',
              status: 'running',
              type: 'planning'
            }])
          } else if (eventData.phase === 'executing') {
            console.log('DAG执行阶段开始')
            setDagComplete(false)
          } else if (eventData.phase === 'completed' || eventData.phase === 'failed') {
            console.log('DAG执行完成')
            setDagComplete(true)
            setIsLoading(false)
          }
        } else if (eventType === 'dag_step_info' || eventType === 'dag_step_start' || eventType === 'dag_step_end') {
          console.log('🚀 DAG步骤信息事件:', eventType, eventData)
          console.log('🔍 检查是否包含 format_query_result 的结果:', {
            hasOutput: !!eventData.output,
            outputStartsWith: eventData.output ? eventData.output.substring(0, 100) : 'null',
            isJsonFormat: eventData.output && eventData.output.trim().startsWith('{'),
            stepName: eventData.step_name
          })

          // 确保有助手消息存在
          setMessages(prev => {
            const updated = [...prev]
            const text2sqlMessageIndex = updated.findIndex(msg => msg.type === "助手" && msg.id.includes('text2sql-response'))

            if (text2sqlMessageIndex < 0) {
              const newMessage: Message = {
                id: `text2sql-response-${Date.now()}`,
                type: "助手",
                content: '正在处理查询...',
                timestamp: Date.now(),
                thinkingSteps: [],
                showThinking: true
              }
              updated.push(newMessage)
              console.log('📝 创建助手消息用于接收步骤信息')
            }

            return updated
          })
          console.log('🔍 检查 trace 数据结构:', {
            hasAgentData: !!eventData.agent_data,
            agentDataKeys: eventData.agent_data ? Object.keys(eventData.agent_data) : [],
            hasResultData: !!eventData.result_data,
            resultDataKeys: eventData.result_data ? Object.keys(eventData.result_data) : [],
            stepId: eventData.step_id,
            stepName: eventData.step_name
          })

          const stepStatus = eventData.status || (eventType === 'dag_step_end' ? 'completed' : 'running')
          const toolNames = eventData.tool_name ? [eventData.tool_name] : (eventData.tool_names || [])

          const step: ThinkingStep = {
            id: eventData.step_id || eventData.id || `step-${Date.now()}`,
            name: generateStepName(eventData, toolNames),
            description: generateStepDescription(eventData, toolNames, stepStatus),
            status: stepStatus,
            type: mapStepType(eventData.type, eventData.step_name, eventData.tool_names),
            started_at: eventData.started_at || eventData.start_time,
            completed_at: eventData.completed_at || eventData.end_time,
            dependencies: eventData.dependencies,
            details: {
              content: eventData.description,
              sql_query: eventData.result_data?.sql_query || eventData.step_data?.sql_query,
              result_data: eventData.result_data || eventData.step_data?.result,
              error_message: eventData.result_data?.error || eventData.step_data?.error,
            },
            tool_names: toolNames
          }

          console.log('创建思考步骤:', step)


          // 向后兼容：检查旧的步骤结果数据格式
          if (eventData.result_data && eventData.result_data.columns && eventData.result_data.rows) {
            console.log('🎯 在步骤中发现Text2SQL结果数据 (旧格式):', eventData.result_data)
            const legacyData = {
              type: 'text2sql_result',
              structured_data: eventData.result_data,
              metadata: {
                record_count: eventData.result_data.rows.length,
                sql_query: eventData.result_data.sql_query,
                source: 'legacy_step_data'
              }
            }
            console.log('✅ Legacy data processed')
          }

          // 直接更新消息中的思考步骤
          setMessages(prev => {
            const updated = [...prev]
            const text2sqlMessageIndex = updated.findIndex(msg => msg.type === "助手" && msg.id.includes('text2sql-response'))

            if (text2sqlMessageIndex >= 0) {
              const currentSteps = updated[text2sqlMessageIndex].thinkingSteps || []
              const existingIndex = currentSteps.findIndex(s => s.id === step.id)

              let newSteps: ThinkingStep[]
              if (existingIndex >= 0) {
                // 更新现有步骤
                newSteps = [...currentSteps]
                newSteps[existingIndex] = { ...newSteps[existingIndex], ...step }
                console.log('✅ 更新现有步骤:', step.id, '->', step.status)
              } else {
                // 添加新步骤
                newSteps = [...currentSteps, step]
                console.log('➕ 添加新步骤:', step.id, '总数:', newSteps.length)
              }

              updated[text2sqlMessageIndex] = {
                ...updated[text2sqlMessageIndex],
                thinkingSteps: newSteps,
                showThinking: true // 确保显示思考过程
              }

              console.log('📝 消息更新后的思考步骤数:', updated[text2sqlMessageIndex]?.thinkingSteps?.length || 0)
            }

            return updated
          })

          // 同时更新全局状态用于兼容
          setThinkingSteps(prev => {
            const existingIndex = prev.findIndex(s => s.id === step.id)
            if (existingIndex >= 0) {
              const updated = [...prev]
              updated[existingIndex] = { ...updated[existingIndex], ...step }
              return updated
            } else {
              const newSteps = [...prev, step]
              console.log('添加思考步骤，总数:', newSteps.length)
              return newSteps
            }
          })
        } else if (eventType === 'task_completion') {
          console.log('任务完成')
          console.log('🔍 检查任务完成数据结构:', {
            hasAgentData: !!eventData.agent_data,
            agentDataKeys: eventData.agent_data ? Object.keys(eventData.agent_data) : [],
            hasResult: !!eventData.result,
            resultPreview: eventData.result ? eventData.result.substring(0, 100) + '...' : 'null'
          })
          setDagComplete(true)
          setIsLoading(false)

          // 任务完成后刷新任务列表
          loadHistoricalTasks()

          // 首先检查是否有 agent_data (结构化数据)
          let structuredQueryResult = null
          if (eventData.agent_data && eventData.agent_data.type === 'text2sql_result') {
            console.log('🎯 在任务完成中发现Text2SQL trace数据 (新格式):', eventData.agent_data)

            // 直接解析 trace 数据
            if (eventData.agent_data.structured_data) {
              const data = eventData.agent_data.structured_data
              console.log('🎯 找到结构化数据:', {
                hasColumns: !!data.columns,
                hasRows: !!data.rows,
                columnCount: data.columns?.length || 0,
                rowCount: data.rows?.length || 0
              })

              structuredQueryResult = {
                success: true,
                data: {
                  columns: data.columns,
                  rows: data.rows
                },
                type: 'text2sql',
                message: "Query completed successfully"
              }

              console.log('✅ 解析任务完成中的新格式结构化数据成功')
            }
          }

          // 向后兼容：检查旧的结果数据格式
          if (!structuredQueryResult && eventData.result) {
            try {
              const resultData = JSON.parse(eventData.result)
              if (resultData.structured_data && resultData.structured_data.columns) {
                console.log('🎯 在任务完成中发现Text2SQL结果数据 (旧格式):', resultData.structured_data)
                structuredQueryResult = {
                  success: true,
                  data: {
                    type: 'text2sql_result',
                    structured_data: resultData.structured_data,
                    metadata: {
                      record_count: resultData.structured_data.rows?.length || 0,
                      sql_query: resultData.structured_data.sql_query,
                      source: 'legacy_task_result'
                    }
                  },
                  type: 'text2sql',
                  message: t('agentStore.text2sql.chat.thinking.queryCompleted')
                }
                console.log('✅ 解析任务完成中的旧格式结构化数据成功')
              }
            } catch (e) {
              console.log('解析旧格式结果数据失败:', e)
            }
          }

          // 解析并显示最终结果，与思考过程集成在同一个消息中
          if (eventData.result) {
            try {
              const resultData = JSON.parse(eventData.result)
              console.log('解析任务完成结果:', resultData)

              if (resultData.success && resultData.output) {
                // 更新最后一条助手消息，添加最终结果
                setMessages(prev => {
                  const updated = [...prev]
                  // 查找最后一条助手消息
                  const lastAssistantIndex = updated.findIndex(msg => msg.type === "助手" && msg.id.includes('text2sql-response'))

                  if (lastAssistantIndex >= 0) {
                    // 更新现有消息的内容，保留已有的思考步骤和结构化数据
                    const existingSteps = updated[lastAssistantIndex].thinkingSteps || []
                    const existingData = updated[lastAssistantIndex].data || {}
                    updated[lastAssistantIndex] = {
                      ...updated[lastAssistantIndex],
                      content: resultData.output,
                      thinkingSteps: existingSteps, // 使用消息中已有的思考步骤
                      showThinking: true,
                      // 添加结构化数据到消息中，但保留现有的 structuredQueryResult
                      data: {
                        ...resultData,
                        structuredQueryResult: structuredQueryResult || existingData.structuredQueryResult // 优先使用新解析的，如果没有则保留现有的
                      }
                    }
                  } else {
                    // 如果没有找到，创建新的集成消息，使用全局累积的思考步骤
                    const integratedMessage: Message = {
                      id: `text2sql-response-${Date.now()}`,
                      type: "助手",
                      content: resultData.output,
                      timestamp: Date.now(),
                      data: {
                        ...resultData,
                        structuredQueryResult: structuredQueryResult
                      },
                      thinkingSteps: thinkingSteps, // 使用全局累积的思考步骤
                      showThinking: true
                    }
                    updated.push(integratedMessage)
                  }

                  return updated
                })
                console.log('已集成最终结果到思考过程消息中')
              } else if (resultData.error) {
                // 处理错误情况，也要集成思考过程
                setMessages(prev => {
                  const updated = [...prev]
                  const lastAssistantIndex = updated.findIndex(msg => msg.type === "助手" && msg.id.includes('text2sql-response'))

                  if (lastAssistantIndex >= 0) {
                    const existingSteps = updated[lastAssistantIndex].thinkingSteps || []
                    updated[lastAssistantIndex] = {
                      ...updated[lastAssistantIndex],
                      content: `查询执行出错: ${resultData.error}`,
                      thinkingSteps: existingSteps, // 使用消息中已有的思考步骤
                      showThinking: true
                    }
                  } else {
                    const errorMessage: Message = {
                      id: `text2sql-response-${Date.now()}`,
                      type: "助手",
                      content: `查询执行出错: ${resultData.error}`,
                      timestamp: Date.now(),
                      data: resultData,
                      thinkingSteps: thinkingSteps,
                      showThinking: true
                    }
                    updated.push(errorMessage)
                  }

                  return updated
                })
                console.log('已集成错误结果到思考过程消息中')
              }
            } catch (parseError) {
              console.error('解析任务完成结果失败:', parseError)
              // 如果解析失败，也要集成思考过程
              setMessages(prev => {
                const updated = [...prev]
                const lastAssistantIndex = updated.findIndex(msg => msg.type === "助手" && msg.id.includes('text2sql-response'))

                if (lastAssistantIndex >= 0) {
                  const existingSteps = updated[lastAssistantIndex].thinkingSteps || []
                  updated[lastAssistantIndex] = {
                    ...updated[lastAssistantIndex],
                    content: `任务完成，但结果解析失败。原始数据: ${eventData.result}`,
                    thinkingSteps: existingSteps,
                    showThinking: true
                  }
                } else {
                  const fallbackMessage: Message = {
                    id: `text2sql-response-${Date.now()}`,
                    type: "助手",
                    content: `任务完成，但结果解析失败。原始数据: ${eventData.result}`,
                    timestamp: Date.now(),
                    data: eventData,
                    thinkingSteps: thinkingSteps, // fallback时使用全局状态
                    showThinking: true
                  }
                  updated.push(fallbackMessage)
                }

                return updated
              })
            }
          }
        } else if (eventType === 'tool_execution_end') {
          console.log('🔧 检查所有 tool_execution_end 事件:', {
            toolName: eventData.tool_name,
            hasResult: !!eventData.result,
            resultType: typeof eventData.result
          })

          if (eventData.tool_name === 'execute_sql_query') {
            console.log('🔧 处理 execute_sql_query 工具执行结果:', eventData)

            // 解析工具结果中的结构化数据
            // 现在结果应该是字典格式，直接使用
            let parsedResult = eventData.result;

            if (parsedResult && parsedResult.structuredQueryResult) {
              const queryResult = parsedResult.structuredQueryResult
              console.log('🔧 ✅ 检测到 execute_sql_query 结构化结果:', {
                hasData: !!queryResult.data,
                hasColumns: !!queryResult.data?.columns,
                hasRows: !!queryResult.data?.rows,
                columnCount: queryResult.data?.columns?.length || 0,
                rowCount: queryResult.data?.rows?.length || 0
              })

              // 设置全局状态用于结果显示
              setStructuredQueryResult(queryResult.data)
              setViewMode('table')
              console.log('✅ 设置全局结构化查询结果')

              // 更新最后的助手消息
              setMessages(prev => {
                const updated = [...prev]
                const lastAssistantIndex = updated.findLastIndex(msg => msg.type === "助手")

                if (lastAssistantIndex >= 0) {
                  const existingData = updated[lastAssistantIndex].data || {}
                  updated[lastAssistantIndex] = {
                    ...updated[lastAssistantIndex],
                    content: queryResult.message || t('agentStore.text2sql.chat.thinking.queryCompleted'),
                    data: {
                      ...existingData,
                      success: true,
                      structuredQueryResult: queryResult
                    }
                  }
                  console.log('🔧 ✅ 更新助手消息，添加结构化数据')
                }

                return updated
              })
            }
          }

        } else if (eventType === 'user_message') {
          // 用户消息处理 - 使用原始时间戳，检查是否已存在
          const messageId = `user-${eventData.timestamp || Date.now()}`
          const messageContent = eventData.message || eventData.content || "用户消息"
          const messageTimestamp = eventData.timestamp ? new Date(eventData.timestamp).getTime() : Date.now()

          setMessages(prev => {
            // 检查是否已存在相同内容的用户消息
            const exists = prev.some(msg =>
              msg.type === "user" &&
              msg.content === messageContent &&
              Math.abs(msg.timestamp - messageTimestamp) < 1000 // 1秒内的相同消息认为是重复
            )

            if (!exists) {
              const userMessage: Message = {
                id: messageId,
                type: "user",
                content: messageContent,
                timestamp: messageTimestamp,
                data: eventData.data
              }
              return [...prev, userMessage]
            }
            return prev
          })
        } else if (eventType === 'ai_message') {
          // AI消息处理 - 不再创建独立消息，因为这些会通过task_completion统一处理
          // 这里可以选择性处理一些即时性的AI消息，但主要的查询结果应该通过task_completion
          console.log('收到AI消息，但主要结果将通过task_completion处理:', eventData)
          setIsLoading(false)
        }
      }
    }
  })

  // 当任务创建完成后启用 WebSocket
  useEffect(() => {
    if (currentTask && !wsEnabled) {
      // 使用 setTimeout 确保状态更新在渲染完成后进行
      setTimeout(() => setWsEnabled(true), 200)
    } else if (!currentTask && wsEnabled) {
      // 没有任务时禁用 WebSocket
      setWsEnabled(false)
    }
  }, [currentTask, wsEnabled])

  // 加载历史任务
  const loadHistoricalTasks = async () => {
    setIsLoadingHistory(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/tasks?agent_type=text2sql&per_page=20`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        const tasksArray = data.tasks || (Array.isArray(data) ? data : [])
        console.log('Loaded tasks:', tasksArray)
        // 确保每个任务都有正确的 id
        const processedTasks = tasksArray.map((task: any) => ({
          id: task.id?.toString() || task.task_id?.toString() || 'unknown',
          title: task.title || '未命名任务',
          status: task.status || 'pending',
          created_at: task.created_at || task.createdAt || Date.now(),
          updated_at: task.updated_at || task.updatedAt || Date.now()
        }))
        setHistoricalTasks(processedTasks)
      } else {
        console.error('Failed to load historical tasks:', response.status, response.statusText)
      }
    } catch (error) {
      console.error('Failed to load historical tasks:', error)
    } finally {
      setIsLoadingHistory(false)
    }
  }

  // 选择历史任务
  const selectHistoricalTask = async (taskId: string) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      })

      if (response.ok) {
        const taskData = await response.json()
        console.log('Selected task data:', taskData)
        const selectedTask: Task = {
          id: (taskData.task_id || taskData.id).toString(),
          title: taskData.title,
          status: taskData.status,
          description: taskData.description,
          agentType: "text2sql",
          createdAt: taskData.created_at || taskData.createdAt,
          updatedAt: taskData.updated_at || taskData.updatedAt
        }

        console.log('Setting current task:', selectedTask)

        // 先禁用 WebSocket 连接
        console.log('选择历史任务，禁用WebSocket')
        setWsEnabled(false)
        setCurrentTask(selectedTask)

        // 加载任务的消息历史
        await loadTaskMessages(taskId)

        // 延迟启用新的 WebSocket 连接，给足够时间建立连接
        console.log('重新启用WebSocket连接历史任务:', taskId)
        setTimeout(() => {
          console.log('WebSocket启用状态设置:', true)
          setWsEnabled(true)
        }, 300)
      }
    } catch (error) {
      console.error('Failed to load task details:', error)
    }
  }

  // 加载任务消息历史
  const loadTaskMessages = async (taskId: string) => {
    try {
      // WebSocket会自动推送历史消息，不需要手动加载
      // 清空当前消息，等待WebSocket推送历史消息
      setMessages([])
      console.log('等待WebSocket推送历史任务消息...')
    } catch (error) {
      console.error('Failed to load task messages:', error)
      setMessages([])
    }
  }

  // 删除任务
  const deleteTask = async (taskId: string, event: React.MouseEvent) => {
    event.stopPropagation() // 防止触发任务选择

    console.log('Deleting task with ID:', taskId)

    if (!confirm(t('agentStore.text2sql.chat.history.deleteConfirm'))) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`
        }
      })

      if (response.ok) {
        // 如果删除的是当前任务，清空当前任务和消息
        if (currentTask?.id === taskId) {
          setCurrentTask(null)
          setMessages([])
        }

        // 重新加载历史任务列表
        await loadHistoricalTasks()
      } else {
        alert(t('agentStore.text2sql.chat.history.deleteFailed'))
      }
    } catch (error) {
      console.error('Failed to delete task:', error)
      alert(t('agentStore.text2sql.chat.history.deleteFailed'))
    }
  }


  const createNewTask = async (): Promise<Task | null> => {
    if (!user || !token) return null

    setIsCreatingTask(true)
    try {
      // 从 URL 参数获取数据库配置
      const urlParams = new URLSearchParams(window.location.search)
      const databaseUrl = urlParams.get('database_url')
      const databaseName = urlParams.get('database_name')
      const readOnly = urlParams.get('read_only') !== 'false'

      // 验证必须的数据库配置
      if (!databaseUrl) {
        throw new Error(t('agentStore.text2sql.chat.login.description'))
      }

      const taskData = {
        title: `${t('agentStore.text2sql.title')} - ${databaseName || t('agentStore.text2sql.chat.database.unknown')}`,
        description: t('agentStore.text2sql.descriptionShort'),
        agent_type: "text2sql",
        agent_config: {
          database_url: databaseUrl,
          schema_info: null, // 将由 agent 自动发现
          read_only: readOnly,
          max_iterations: 3,
          database_name: databaseName || t('agentStore.text2sql.chat.database.unknown')
        }
        // 让后端从数据库读取用户的默认 LLM 配置
      }

      const response = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(taskData)
      })

      if (response.ok) {
        const data = await response.json()
        const newTask: Task = {
          id: data.task_id.toString(),
          title: taskData.title,
          status: "pending",
          description: taskData.description,
          agentType: "text2sql",
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }

        // 先设置任务，然后让 useEffect 处理 WebSocket 启用
        setCurrentTask(newTask)

        // 添加欢迎消息
        setMessages([{
          id: "welcome",
          type: "助手",
          content: `${t('agentStore.text2sql.chat.welcome.title')}！${t('agentStore.text2sql.chat.welcome.description')}\n\n${t('agentStore.text2sql.chat.welcome.featuresTitle')}\n\n• ${t('agentStore.text2sql.chat.welcome.features.naturalToSql')}\n• ${t('agentStore.text2sql.chat.welcome.features.analyzeSchema')}\n• ${t('agentStore.text2sql.chat.welcome.features.executeAndShow')}\n• ${t('agentStore.text2sql.chat.welcome.features.optimizeQuery')}\n\n${t('agentStore.text2sql.chat.input.title')}\n${t('agentStore.text2sql.chat.input.placeholder')}`,
          timestamp: Date.now()
        }])

        return newTask
      } else {
        console.error("Failed to create task:", response.statusText)
        return null
      }
    } catch (error) {
      console.error("Error creating task:", error)
      return null
    } finally {
      setIsCreatingTask(false)
    }
  }

  // 发送消息
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return

    // 如果没有当前任务，先创建一个
    let taskToUse = currentTask
    let isNewTask = false

    if (!taskToUse) {
      taskToUse = await createNewTask()
      isNewTask = true
      // 创建新任务后刷新任务列表
      await loadHistoricalTasks()
    }

    // 如果是新任务，清空欢迎消息
    if (isNewTask) {
      setMessages([])
    }

    // 简单等待连接建立，不依赖状态变量
    console.log('等待 WebSocket 连接建立...', { taskId: taskToUse?.id })
    await new Promise(resolve => setTimeout(resolve, 1000))

    // 直接发送消息，不检查连接状态（WebSocket hook 会处理）
    console.log('发送消息到任务:', taskToUse?.id)

    // 重置思考状态
    setThinkingSteps([])
    setShowThinking(false)
    setDagComplete(false)

    // 通过 WebSocket 发送消息，用户消息将由服务端返回后显示
    sendMessage({
      type: "chat",
      message: inputMessage
    })

    setInputMessage("")
    setIsLoading(true)
  }

  // 处理 WebSocket 消息
  useEffect(() => {
    if (!currentTask) return

    // 这里可以监听 WebSocket 事件并更新消息列表
    // 实际的消息处理逻辑在 useWebSocket hook 中
  }, [currentTask])

  // 页面加载时验证数据库配置并加载历史任务
  useEffect(() => {
    if (user) {
      // 验证是否有数据库配置
      const urlParams = new URLSearchParams(window.location.search)
      const databaseUrl = urlParams.get('database_url')
      const databaseName = urlParams.get('database_name')
      const readOnly = urlParams.get('read_only') !== 'false'

      if (!databaseUrl) {
        // 没有数据库配置，跳转回配置页面
        window.location.href = '/agent-store/text2sql'
        return
      }

      // 设置数据库信息状态
      setDatabaseInfo({
        url: databaseUrl,
        name: databaseName || '未知数据库',
        type: inferDatabaseType(databaseUrl),
        readOnly: readOnly
      })

      loadHistoricalTasks()
      // 不再自动创建任务，需要用户手动选择或创建
    }
  }, [user])


  // 获取状态图标
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "running":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  if (!user) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card className="w-96">
          <CardHeader className="text-center">
            <Database className="h-12 w-12 mx-auto mb-4 text-blue-500" />
            <CardTitle>{t('agentStore.text2sql.chat.login.title')}</CardTitle>
            <CardDescription>{t('agentStore.text2sql.chat.login.description')}</CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background">
      {/* 左侧历史记录面板 */}
      <div className="w-80 border-r border-border bg-card flex flex-col h-full">
        <div className="p-4 border-b border-border flex-shrink-0">
          <div className="flex items-center gap-2 mb-2">
            <Database className="h-5 w-5 text-blue-500" />
            <h3 className="font-semibold">{t('agentStore.text2sql.title')}</h3>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {currentTask && (
              <>
                {getStatusIcon(currentTask.status)}
                <span>{currentTask.status}</span>
              </>
            )}
            {isConnected && (
              <Badge variant="outline" className="text-green-600 border-green-600">
                {t('agentStore.text2sql.chat.connection.connected')}
              </Badge>
            )}
          </div>
        </div>

        <div className="overflow-y-auto flex-1 p-4" style={{ height: 'calc(100vh - 200px)' }}>
          <div className="space-y-4">
            <div className="flex items-center justify-between sticky top-0 bg-background pb-2">
              <h4 className="text-sm font-semibold text-foreground">
                {t('agentStore.text2sql.chat.history.title')}
              </h4>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  // 清空当前任务和消息
                  setCurrentTask(null)
                  setMessages([])
                  setThinkingSteps([])
                  setShowThinking(false)
                  setDagComplete(false)
                  setStructuredQueryResult(null)
                  setViewMode('table')
                  setSelectedChart(null)

                  // 聚焦输入框
                  setTimeout(() => {
                    inputRef.current?.focus()
                  }, 100)
                }}
                disabled={false}
                title={t('agentStore.text2sql.chat.history.newConversation')}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {isLoadingHistory ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            ) : historicalTasks.length === 0 ? (
              <div className="text-center py-8">
                <Database className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                <p className="text-sm font-medium text-foreground mb-1">{t('agentStore.text2sql.chat.history.emptyTitle')}</p>
                <p className="text-xs text-muted-foreground">{t('agentStore.text2sql.chat.history.emptyDescription')}</p>
              </div>
            ) : (
              <div className="space-y-2">
                {historicalTasks.map((task) => (
                  <div
                    key={task.id}
                    onClick={() => selectHistoricalTask(task.id)}
                    className={`
                      group p-3 rounded-lg border cursor-pointer transition-all duration-200
                      ${currentTask?.id === task.id
                        ? 'border-border bg-accent shadow-sm'
                        : 'border-border bg-card hover:bg-muted/30'
                      }
                    `}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        {getStatusIcon(task.status)}
                        <span className="text-sm font-medium text-foreground truncate">
                          {task.title}
                        </span>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 p-1 h-6 w-6"
                        onClick={(e) => deleteTask(task.id, e)}
                        title={t('agentStore.text2sql.chat.history.delete')}
                      >
                        <Trash2 className="h-3 w-3 text-red-500" />
                      </Button>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(task.created_at).toLocaleString('zh-CN', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 右侧主要内容区域 */}
      <div className="flex-1 flex flex-col">
        {/* 顶部信息栏 */}
        <div className="border-b border-border p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">{t('agentStore.text2sql.title')}</h1>
              <p className="text-muted-foreground">{t('agentStore.text2sql.descriptionShort')}</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-blue-600 border-blue-600">
                <Database className="h-3 w-3 mr-1" />
                {databaseInfo.type || 'SQLite'}
              </Badge>
              <Badge variant="outline">
                {databaseInfo.readOnly ? t('agentStore.text2sql.chat.mode.readOnly') : t('agentStore.text2sql.chat.mode.readWrite')}
              </Badge>
            </div>
          </div>
        </div>

        {/* 内容展示区域 */}
        {!currentTask && isCreatingTask ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-4">
              <Loader2 className="h-8 w-8 animate-spin mx-auto" />
              <p>{t('agentStore.text2sql.chat.initializing')}</p>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden">

            {/* 聊天消息区域 */}
            <div className="overflow-y-auto flex-1 p-4">
              {!currentTask ? (
                <div className="h-full flex items-center justify-center">
                  <Card className="w-full max-w-2xl">
                    <CardHeader className="text-center">
                      <Database className="h-12 w-12 mx-auto mb-4 text-blue-500" />
                      <CardTitle>{t('agentStore.text2sql.chat.welcome.title')}</CardTitle>
                      <CardDescription>
                        {t('agentStore.text2sql.chat.welcome.description')}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="text-center">
                        <p className="text-sm text-muted-foreground mb-4">
                          {t('agentStore.text2sql.chat.welcome.featuresTitle')}
                        </p>
                        <div className="grid grid-cols-1 gap-2 text-sm text-left">
                          <div className="p-3 bg-muted rounded-lg">• {t('agentStore.text2sql.chat.welcome.features.naturalToSql')}</div>
                          <div className="p-3 bg-muted rounded-lg">• {t('agentStore.text2sql.chat.welcome.features.analyzeSchema')}</div>
                          <div className="p-3 bg-muted rounded-lg">• {t('agentStore.text2sql.chat.welcome.features.executeAndShow')}</div>
                          <div className="p-3 bg-muted rounded-lg">• {t('agentStore.text2sql.chat.welcome.features.optimizeQuery')}</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              ) : messages.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <Card className="w-full max-w-2xl">
                    <CardHeader className="text-center">
                      <Search className="h-12 w-12 mx-auto mb-4 text-blue-500" />
                      <CardTitle>{t('agentStore.text2sql.chat.input.title')}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                        <Table className="h-4 w-4 text-blue-600" />
                        <p className="text-sm text-muted-foreground">
                          {t('agentStore.text2sql.chat.input.hint')}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              ) : (
                <div className="space-y-4 pb-4">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`
                        flex gap-3
                        ${message.type === "user" ? "justify-end" : "justify-start"}
                      `}
                    >
                      {message.type === "助手" && (
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
                          <Database className="h-4 w-4 text-white" />
                        </div>
                      )}
                      <div className={`
                        ${message.type === "user" ? "max-w-[80%]" : "min-w-[50%] max-w-[90%]"}
                        ${message.type === "user"
                          ? "bg-blue-500 text-white ml-auto p-3 rounded-lg"
                          : "bg-muted text-foreground"
                        }
                      `}>
                        {/* 助手消息包含思考过程 */}
                        {message.type === "助手" ? (
                          <div className="rounded-lg">
                            {/* 思考过程 - 优先使用消息中的思考步骤，回退到全局状态 */}
                            {(() => {
                              const stepsToUse = message.thinkingSteps || thinkingSteps;
                              console.log('💭 渲染思考过程:', {
                                messageId: message.id,
                                messageThinkingStepsCount: message.thinkingSteps?.length || 0,
                                globalThinkingStepsCount: thinkingSteps.length,
                                stepsToUseCount: stepsToUse.length,
                                showThinking: message.showThinking
                              });

                              return stepsToUse.length > 0 ? (
                                <div className="border-b border-border p-3">
                                  <ThinkingTimeline
                                    steps={stepsToUse}
                                    isComplete={stepsToUse.every(step => step.status === 'completed' || step.status === 'failed')}
                                  />
                                </div>
                              ) : null;
                            })()}

                            {/* 消息内容 */}
                            <div className="p-3">
                              <div className="text-sm">
                                {/* 显示结构化查询结果 - 在消息内容之前显示 */}
                                {(() => {
                                  console.log('🔍 开始解析消息的数据展现:', {
                                    messageId: message.id,
                                    messageType: message.type,
                                    hasData: !!message.data,
                                    dataKeys: message.data ? Object.keys(message.data) : [],
                                    hasStructuredQueryResult: !!message.data?.structuredQueryResult,
                                    contentLength: message.content.length,
                                    contentPreview: message.content.substring(0, 100) + '...',
                                    fullMessageData: message.data
                                  });

                                  // 详细检查 message.data 的所有可能字段
                                  if (message.data) {
                                    console.log('📋 详细检查 message.data:', {
                                      hasSuccess: !!message.data.success,
                                      hasOutput: !!message.data.output,
                                      hasIterations: !!message.data.iterations,
                                      iterationsCount: message.data.iterations?.length,
                                      hasHistory: !!message.data.history,
                                      historyCount: message.data.history?.length,
                                      outputPreview: message.data.output ? message.data.output.substring(0, 200) + '...' : 'null'
                                    });

                                    // 检查 iterations 中是否有结构化数据
                                    if (message.data.iterations && Array.isArray(message.data.iterations)) {
                                      message.data.iterations.forEach((iter: any, idx: number) => {
                                        if (iter.tool_results) {
                                          console.log(`🔧 检查 iteration ${idx} 的 tool_results:`, Object.keys(iter.tool_results));
                                          Object.entries(iter.tool_results).forEach(([toolName, result]: [string, any]) => {
                                            if (typeof result === 'object' && result.columns && result.rows) {
                                              console.log(`✅ 在 iteration ${idx} 的 ${toolName} 中找到结构化数据:`, {
                                                columns: result.columns,
                                                rowsCount: result.rows.length,
                                                sqlQuery: result.sql_query
                                              });
                                            }
                                          });
                                        }
                                      });
                                    }
                                  }

                                  // 查找包含 structuredQueryResult 的数据（从当前消息或所有消息中）
                                  let queryResult = null

                                  console.log('🔍 检查所有消息数据:', {
                                    totalMessages: messages.length,
                                    allMessageIds: messages.map(m => ({id: m.id, type: m.type, hasData: !!m.data, hasStructuredQueryResult: !!m.data?.structuredQueryResult})),
                                    currentMessageId: message.id
                                  })

                                  // 首先检查当前消息
                                  if (message.data?.structuredQueryResult) {
                                    queryResult = message.data.structuredQueryResult
                                    console.log('✅ 在当前消息中找到 structuredQueryResult')
                                  } else {
                                    // 如果当前消息没有，从所有消息中查找包含数据的消息
                                    const allMessages = messages // 使用全局 messages 状态
                                    console.log('🔍 检查所有消息，寻找 structuredQueryResult:', allMessages.map(m => ({id: m.id, hasData: !!m.data, dataKeys: m.data ? Object.keys(m.data) : [], hasStructuredQueryResult: !!m.data?.structuredQueryResult})))

                                  // 详细打印助手消息的 data 内容
                                  const assistantMessage = allMessages.find(m => m.type === "助手")
                                  if (assistantMessage && assistantMessage.data) {
                                    console.log('🔍 助手消息完整 data 内容:', {
                                      data: assistantMessage.data,
                                      structuredQueryResult: assistantMessage.data.structuredQueryResult,
                                      structuredQueryResultType: typeof assistantMessage.data.structuredQueryResult
                                    })
                                  }
                                    const dataMessage = allMessages.find(msg => msg.data?.structuredQueryResult)
                                    if (dataMessage) {
                                      queryResult = dataMessage.data.structuredQueryResult
                                      console.log('✅ 在其他消息中找到 structuredQueryResult:', dataMessage.id)
                                    }
                                  }

                                  if (!queryResult) {
                                    console.log('❌ 没有找到任何 structuredQueryResult，跳过渲染')
                                    return null
                                  }

                                  // queryResult 已经包含了数据，直接使用
                                  console.log('✅ 使用找到的查询结果数据:', {
                                    hasData: !!queryResult.data,
                                    hasColumns: !!queryResult.data?.columns,
                                    hasRows: !!queryResult.data?.rows,
                                    columnCount: queryResult.data?.columns?.length || 0,
                                    rowCount: queryResult.data?.rows?.length || 0,
                                    message: queryResult.message
                                  })
                                  // 修正数据结构，使其符合渲染期望的格式
                                  const finalQueryResult = {
                                    columns: queryResult.data?.columns || [],
                                    rows: queryResult.data?.rows || [],
                                    summary: queryResult.message || t('agentStore.text2sql.chat.thinking.queryCompleted')
                                  }

                                  console.log('✅ 最终查询结果数据:', {
                                    columns: finalQueryResult.columns.length,
                                    rows: finalQueryResult.rows.length,
                                    hasSummary: !!finalQueryResult.summary
                                  })
                                  console.log('🔍 检查消息数据结构:', {
                                    hasData: !!message.data,
                                    hasStructuredQueryResult: !!message.data?.structuredQueryResult,
                                    structuredQueryResultKeys: message.data?.structuredQueryResult ? Object.keys(message.data.structuredQueryResult) : [],
                                    hasSuccess: !!message.data?.structuredQueryResult?.success
                                  })

                                  if (message.data?.structuredQueryResult?.success) {
                                    const structuredData = message.data.structuredQueryResult.data
                                    queryResult = {
                                      columns: structuredData?.columns || [],
                                      rows: structuredData?.rows || [],
                                      summary: message.data.structuredQueryResult.message
                                    }
                                    console.log('🎯 从消息 data 中获取结构化数据:', queryResult)
                                  } else {
                                    // 从 content 中解析数据 (支持新的 JSON 格式和向后兼容)
                                    queryResult = parseQueryResult(message.content)
                                    if (queryResult) {
                                      console.log('🎯 从 content 解析获取结构化数据:', {
                                        hasColumns: !!queryResult.columns,
                                        hasRows: !!queryResult.rows,
                                        columnCount: queryResult.columns?.length || 0,
                                        rowCount: queryResult.rows?.length || 0
                                      })
                                    } else {
                                      console.log('❌ 未能解析到结构化数据')
                                    }
                                  }

                                  if (finalQueryResult && finalQueryResult.columns && finalQueryResult.rows) {
                                    console.log('✅ 数据展现解析成功:', {
                                      columnCount: finalQueryResult.columns.length,
                                      rowCount: finalQueryResult.rows.length,
                                      hasSummary: !!finalQueryResult.summary
                                    });
                                  } else {
                                    console.log('❌ 数据解析失败或缺少必要字段:', {
                                      hasQueryResult: !!finalQueryResult,
                                      hasColumns: !!finalQueryResult?.columns,
                                      hasRows: !!finalQueryResult?.rows,
                                      queryResultKeys: finalQueryResult ? Object.keys(finalQueryResult) : null
                                    });
                                  }

                                  console.log('🎯 即将渲染表格，条件检查:', {
                                    shouldRender: !!(finalQueryResult && finalQueryResult.columns && finalQueryResult.rows),
                                    hasColumns: !!finalQueryResult?.columns,
                                    columnsLength: finalQueryResult?.columns?.length || 0,
                                    hasRows: !!finalQueryResult?.rows,
                                    rowsLength: finalQueryResult?.rows?.length || 0,
                                    finalQueryResult: finalQueryResult
                                  });

                                  console.log('🚀 开始渲染表格组件');

                                  return (finalQueryResult && finalQueryResult.columns && finalQueryResult.rows) ? (
                                    <div className="mt-4 p-4 bg-muted/20 rounded-lg border">
                                      <div className="flex justify-between items-center mb-2">
                                        <h4 className="font-medium text-sm">
                                          {t('agentStore.text2sql.chat.table.titleWithCount', { count: finalQueryResult.rows.length })}
                                        </h4>
                                        <div className="flex gap-2 flex-wrap">
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            className="flex items-center gap-2"
                                            onClick={() => handleChartAction('bar', finalQueryResult)}
                                          >
                                            <BarChart3 className="h-4 w-4" />
                                            {t('agentStore.text2sql.chat.table.buttons.bar')}
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            className="flex items-center gap-2"
                                            onClick={() => handleChartAction('pie', finalQueryResult)}
                                          >
                                            <PieChart className="h-4 w-4" />
                                            {t('agentStore.text2sql.chat.table.buttons.pie')}
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            className="flex items-center gap-2"
                                            onClick={() => handleChartAction('line', finalQueryResult)}
                                          >
                                            <LineChart className="h-4 w-4" />
                                            {t('agentStore.text2sql.chat.table.buttons.line')}
                                          </Button>
                                            <Button
                                            size="sm"
                                            variant="outline"
                                            className="flex items-center gap-2"
                                            onClick={() => handleChartAction('copy', finalQueryResult)}
                                          >
                                            <Copy className="h-4 w-4" />
                                            {t('agentStore.text2sql.chat.table.buttons.copyData')}
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            className="flex items-center gap-2"
                                            onClick={() => handleChartAction('export', finalQueryResult)}
                                          >
                                            <Download className="h-4 w-4" />
                                            {t('agentStore.text2sql.chat.table.buttons.exportCsv')}
                                          </Button>
                                        </div>
                                      </div>

                                      {/* 数据显示区域：表格或图表 */}
                                      <div className="border rounded-lg overflow-hidden mb-4">
                                        {!selectedChart || selectedChart === 'copy' || selectedChart === 'export' ? (
                                          // 显示表格
                                          <div className="overflow-x-auto">
                                            <table className="w-full text-sm">
                                              <thead className="bg-muted/50">
                                                <tr>
                                                  {finalQueryResult.columns.map((col: string, idx: number) => (
                                                    <th key={idx} className="px-4 py-2 text-left font-medium border-b">
                                                      {col}
                                                    </th>
                                                  ))}
                                                </tr>
                                              </thead>
                                              <tbody>
                                                {finalQueryResult.rows.map((row: any, rowIdx: number) => (
                                                  <tr key={rowIdx} className="hover:bg-muted/25 border-b">
                                                    {finalQueryResult.columns.map((col: string, colIdx: number) => (
                                                      <td key={colIdx} className="px-4 py-2 border-r">
                                                        {row[col]?.toString() || '-'}
                                                      </td>
                                                    ))}
                                                  </tr>
                                                ))}
                                              </tbody>
                                            </table>
                                          </div>
                                        ) : (
                                          // 显示图表
                                          <div className="p-4">
                                            <div className="flex justify-between items-center mb-4">
                                              <h5 className="font-medium text-sm">
                                                {t('agentStore.text2sql.chat.table.buttons.chart')}
                                              </h5>
                                              <Button
                                                size="sm"
                                                variant="outline"
                                                onClick={() => setSelectedChart(null)}
                                              >
                                                <Table className="h-4 w-4 mr-1" />
                                                {t('agentStore.text2sql.charts.switchToTable')}
                                              </Button>
                                            </div>

                                            {selectedChart === 'bar' && <SimpleBarChart data={finalQueryResult} />}
                                            {selectedChart === 'pie' && <SimplePieChart data={finalQueryResult} />}
                                            {selectedChart === 'line' && <SimpleLineChart data={finalQueryResult} />}
                                                                                      </div>
                                        )}
                                      </div>
                                      </div>
                                  ) : null;
                                })()}

                                {/* 显示AI总结内容 */}
                                <MarkdownRenderer content={message.content} />
                              </div>
                            </div>
                          </div>
                        ) : (
                          /* 用户消息 */
                          <div className="p-3 rounded-lg">
                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                          </div>
                        )}
                      </div>
                      {message.type === "user" && (
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-500 flex items-center justify-center">
                          <span className="text-white text-sm">U</span>
                        </div>
                      )}
                    </div>
                  ))}


                  </div>
              )}
            </div>

            {/* 底部输入区域 */}
            <div className="border-t border-border p-4 bg-background">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {currentTask ? (wsEnabled ? (isConnected ? t('agentStore.text2sql.chat.status.ready') : t('agentStore.text2sql.chat.status.connecting')) : t('agentStore.text2sql.chat.status.initializing')) : t('agentStore.text2sql.chat.status.ready')}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {currentTask && wsEnabled && (
                    <Badge variant={isConnected ? "default" : "secondary"} className="text-xs">
                      {isConnected ? t('agentStore.text2sql.chat.connection.connected') : t('agentStore.text2sql.chat.connection.connecting')}
                    </Badge>
                  )}
                </div>
              </div>
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                  placeholder={t('agentStore.text2sql.chat.input.placeholder')}
                  className="flex-1 px-3 py-2 border border-border rounded-md bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isLoading || Boolean(currentTask && wsEnabled && !isConnected)}
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
              {currentTask && !wsEnabled && (
                <p className="text-xs text-yellow-600 mt-2">
                  {t('agentStore.text2sql.chat.connection.initializing')}
                </p>
              )}
              {currentTask && wsEnabled && !isConnected && (
                <p className="text-xs text-orange-500 mt-2">
                  {t('agentStore.text2sql.chat.connection.wsConnecting')}
                </p>
              )}
              {currentTask && wsEnabled && isConnected && (
                <p className="text-xs text-green-600 mt-2">
                  {t('agentStore.text2sql.chat.connection.ready')}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
