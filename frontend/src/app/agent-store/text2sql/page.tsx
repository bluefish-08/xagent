"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Database,
  Plus,
  Settings,
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  Edit,
  Trash2,
  RefreshCw
} from "lucide-react"
import { useAuth } from "@/contexts/auth-context"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import Link from "next/link"
import { useI18n } from "@/contexts/i18n-context"

interface DatabaseConfig {
  id: string
  name: string
  type: string
  url: string
  status: "connected" | "disconnected" | "error"
  created_at: string
  table_count?: number
  read_only: boolean
}

export default function Text2SQLConfigPage() {
  const { user, token } = useAuth()
  const { t } = useI18n()
  const [databases, setDatabases] = useState<DatabaseConfig[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedDbId, setSelectedDbId] = useState<string | null>(null)
  const [isAddingDb, setIsAddingDb] = useState(false)
  const [showAddForm, setShowAddForm] = useState(false)
  const [testingDbId, setTestingDbId] = useState<string | null>(null)

  useEffect(() => {
    if (user) {
      loadDatabases()
    }
  }, [user])

  const loadDatabases = async () => {
    setIsLoading(true)
    try {
      // 调用实际的 API 来获取用户的数据库配置
      const response = await apiRequest(`${getApiUrl()}/api/text2sql/databases`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      })

      console.log('API response:', response.status, response.ok)

      if (response.ok) {
        const data = await response.json()
        console.log('API data:', data)
        // 后端直接返回数组，不是包装在 databases 字段中
        setDatabases(Array.isArray(data) ? data : data.databases || [])
      } else {
        const errorData = await response.json()
        console.error('API error:', errorData)
        setDatabases([])
      }
    } catch (error) {
      console.error("Failed to load databases:", error)
      setDatabases([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddDatabase = async (formData: FormData) => {
    setIsAddingDb(true)
    try {
      const dbConfig = {
        name: formData.get('name') as string,
        type: formData.get('type') as string,
        url: formData.get('url') as string,
        read_only: formData.get('readonly') === 'on'
      }

      // 验证必填字段
      if (!dbConfig.name || !dbConfig.type || !dbConfig.url) {
        alert(t('agentStore.text2sql.config.alerts.fillAll'))
        return
      }

      // 调用 API 添加数据库
      const response = await apiRequest(`${getApiUrl()}/api/text2sql/databases`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(dbConfig)
      })

      if (response.ok) {
        console.log('Database added successfully')

        // 清空表单
        const form = document.getElementById('add-db-form') as HTMLFormElement
        if (form) form.reset()

        // 关闭表单并显示成功状态
        setShowAddForm(false)

        // 重新加载数据库列表
        console.log('Reloading databases...')
        await loadDatabases()
        console.log('Databases reloaded')
      } else {
        const errorData = await response.json()
        console.error('Add database error:', errorData)
        alert(t('agentStore.text2sql.config.alerts.addFailedPrefix') + (errorData.detail || t('agentStore.text2sql.config.alerts.unknownError')))
      }
    } catch (error) {
      console.error('Failed to add database:', error)
      alert(t('agentStore.text2sql.config.alerts.addFailed'))
    } finally {
      setIsAddingDb(false)
    }
  }

  const handleTestConnection = async (databaseId: string) => {
    setTestingDbId(databaseId)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/text2sql/databases/${databaseId}/test`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`
        }
      })

      if (response.ok) {
        const result = await response.json()
        alert(t('agentStore.text2sql.config.alerts.testSuccess', { message: result.message }))
        // 重新加载数据库列表以更新状态
        await loadDatabases()
      } else {
        const errorData = await response.json()
        alert(t('agentStore.text2sql.config.alerts.testFailedPrefix') + errorData.detail)
      }
    } catch (error) {
      console.error('Failed to test connection:', error)
      alert(t('agentStore.text2sql.config.alerts.testFailed'))
    } finally {
      setTestingDbId(null)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "connected":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "disconnected":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      connected: "default" as const,
      disconnected: "destructive" as const,
      error: "destructive" as const
    }

    const labels = {
      connected: t('agentStore.text2sql.config.statusBadge.connected'),
      disconnected: t('agentStore.text2sql.config.statusBadge.disconnected'),
      error: t('agentStore.text2sql.config.statusBadge.error')
    }

    return (
      <Badge variant={variants[status as keyof typeof variants] || "default"}>
        {labels[status as keyof typeof labels] || status}
      </Badge>
    )
  }

  if (!user) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card className="w-96">
          <CardHeader className="text-center">
            <Database className="h-12 w-12 mx-auto mb-4 text-blue-500" />
            <CardTitle>{t('agentStore.text2sql.config.loginRequiredTitle')}</CardTitle>
            <CardDescription>{t('agentStore.text2sql.config.loginRequiredDescription')}</CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      {/* 顶部标题 */}
      <div className="text-center space-y-2 mb-8">
        <h1 className="text-2xl font-bold">{t('agentStore.text2sql.title')}</h1>
        <p className="text-muted-foreground">{t('agentStore.text2sql.config.subtitle')}</p>
      </div>

      {/* 上半部分：已有数据库 */}
      <div className="space-y-4 mb-8">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">{t('agentStore.text2sql.config.connectionsTitle')}</h2>
          {selectedDbId && (
            <Link href={`/agent-store/text2sql/chat?${new URLSearchParams({
              db: selectedDbId,
              database_url: databases.find(db => db.id === selectedDbId)?.url || '',
              database_name: databases.find(db => db.id === selectedDbId)?.name || '',
              read_only: databases.find(db => db.id === selectedDbId)?.read_only ? 'true' : 'false'
            }).toString()}`}>
              <Button>
                <Play className="h-4 w-4 mr-2" />
                {t('agentStore.text2sql.config.startQuery')}
              </Button>
            </Link>
          )}
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        ) : databases.length === 0 ? (
          <Card className="p-8 text-center">
            <Database className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
            <p className="text-muted-foreground">{t('agentStore.text2sql.config.noConnections')}</p>
          </Card>
        ) : (
          <div className="grid gap-3">
            {databases.map((db) => (
              <Card
                key={db.id}
                className={`
                  cursor-pointer transition-all duration-200
                  ${selectedDbId === db.id
                    ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-500/20'
                    : 'hover:border-gray-300'
                  }
                `}
                onClick={() => setSelectedDbId(db.id)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Database className="h-5 w-5 text-blue-500" />
                      <div>
                        <div className="font-medium">{db.name}</div>
                        <div className="text-sm text-muted-foreground">{db.type}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {getStatusBadge(db.status)}
                      {db.read_only && <Badge variant="secondary">{t('agentStore.text2sql.config.readonlyBadge')}</Badge>}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleTestConnection(db.id)
                        }}
                        disabled={testingDbId === db.id}
                      >
                        {testingDbId === db.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <RefreshCw className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* 下半部分：添加新数据库 */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">{t('agentStore.text2sql.config.addNewDatabaseTitle')}</h2>
        <Card>
          <CardContent className="p-6">
            {!showAddForm ? (
              <div className="text-center">
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => setShowAddForm(true)}
                >
                  <Plus className="h-4 w-4 mr-2" />
                  {t('agentStore.text2sql.config.addDatabase')}
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">{t('agentStore.text2sql.config.addNewDatabaseTitle')}</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowAddForm(false)}
                  >
                    {t('common.cancel')}
                  </Button>
                </div>
                <form id="add-db-form" onSubmit={(e) => {
                  e.preventDefault()
                  handleAddDatabase(new FormData(e.currentTarget))
                }} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">{t('agentStore.text2sql.config.form.nameLabel')}</label>
                      <input
                        name="name"
                        type="text"
                        placeholder={t('agentStore.text2sql.config.form.namePlaceholder')}
                        className="w-full px-3 py-2 border rounded-md bg-background"
                        required
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">{t('agentStore.text2sql.config.form.typeLabel')}</label>
                      <select name="type" className="w-full px-3 py-2 border rounded-md bg-background">
                        <option value="sqlite">SQLite</option>
                        <option value="postgresql">PostgreSQL</option>
                        <option value="mysql">MySQL</option>
                        <option value="sqlserver">SQL Server</option>
                      </select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">{t('agentStore.text2sql.config.form.urlLabel')}</label>
                    <input
                      name="url"
                      type="text"
                      placeholder={t('agentStore.text2sql.config.form.urlPlaceholder')}
                      className="w-full px-3 py-2 border rounded-md bg-background"
                      required
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <input type="checkbox" id="readonly" name="readonly" defaultChecked />
                    <label htmlFor="readonly" className="text-sm">{t('agentStore.text2sql.config.form.readonlyLabel')}</label>
                  </div>
                  <Button type="submit" className="w-full" disabled={isAddingDb}>
                    {isAddingDb ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        {t('agentStore.text2sql.config.adding')}
                      </>
                    ) : (
                      <>
                        <Plus className="h-4 w-4 mr-2" />
                        {t('agentStore.text2sql.config.addDatabase')}
                      </>
                    )}
                  </Button>
                </form>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
