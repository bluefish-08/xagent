"use client"

import { useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { apiHooks } from "@/hooks/use-api"
import { useWebSocket } from "@/hooks/use-websocket"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"

export function ApiExample() {
  const { user, token, refreshToken } = useAuth()
  const { t } = useI18n()
  const [manualResult, setManualResult] = useState<any>(null)
  const [wsStatus, setWsStatus] = useState<string>(t('agent.header.connection.disconnected'))

  // 测试WebSocket连接（使用一个虚拟的task id）
  const { isConnected: wsConnected, connectionError: wsError } = useWebSocket({
    taskId: 999, // 虚拟任务ID用于测试
    onConnect: () => setWsStatus(t('agent.header.connection.connected')),
    onDisconnect: () => setWsStatus(t('agent.header.connection.disconnected')),
    onError: (error) => {
      setWsStatus(`${t('vibe.descriptions.process.examples.apiExample.labels.wsError')} ${error.message}`)
    },
  })

  // 使用自动token刷新的API调用
  const { data: models, loading, error, refetch } = apiHooks.useGet(
    "/api/models/?category=llm",
    {
      onError: (error) => {
        console.error("Failed to fetch models:", error)
      },
    }
  )

  // 手动API调用示例
  const handleManualCall = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/models/?category=llm`, {
        headers: {
          "X-Username": user?.username || "",
        },
      })

      if (response.ok) {
        const data = await response.json()
        setManualResult(data)
      } else {
        setManualResult({ error: t('vibe.descriptions.process.examples.apiExample.manualApi.failed') })
      }
    } catch (error) {
      setManualResult({ error: error instanceof Error ? error.message : t('common.errors.unknown') })
    }
  }

  if (!user) {
    return <div>{t('agentStore.loginRequiredTitle')}</div>
  }

  return (
    <div className="p-6 space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold mb-4">{t('vibe.descriptions.process.examples.apiExample.title')}</h2>

        <div className="space-y-4">
          <div className="p-4 bg-gray-50 rounded">
            <p><strong>{t('vibe.descriptions.process.examples.apiExample.labels.currentUser')}</strong> {user.username}</p>
            <p><strong>{t('vibe.descriptions.process.examples.apiExample.labels.accessTokenStatus')}</strong> {token ? t('vibe.descriptions.process.examples.apiExample.status.obtained') : t('vibe.descriptions.process.examples.apiExample.status.notObtained')}</p>
            <p><strong>{t('vibe.descriptions.process.examples.apiExample.labels.refreshTokenStatus')}</strong> {refreshToken ? t('vibe.descriptions.process.examples.apiExample.status.obtained') : t('vibe.descriptions.process.examples.apiExample.status.notObtained')}</p>
            <p><strong>{t('vibe.descriptions.process.examples.apiExample.labels.wsStatus')}</strong> {wsStatus}</p>
            {wsError && <p className="text-red-600"><strong>{t('vibe.descriptions.process.examples.apiExample.labels.wsError')}</strong> {wsError.message}</p>}
          </div>

          <div className="p-4 bg-blue-50 rounded">
            <h3 className="font-semibold mb-2">{t('vibe.descriptions.process.examples.apiExample.autoApi.title')}</h3>
            {loading && <p>{t('common.loading')}</p>}
            {error && <p className="text-red-600">{t('vibe.descriptions.process.examples.apiExample.common.errorPrefix')} {error.message}</p>}
            {models as any && Array.isArray(models) && (
              <div>
                <p className="text-green-600">{t('vibe.descriptions.process.examples.apiExample.models.success', { count: (models.length || 0) as number })}</p>
                <button
                  onClick={() => refetch()}
                  className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  {t('vibe.descriptions.process.examples.apiExample.models.refetch')}
                </button>
              </div>
            )}
          </div>

          <div className="p-4 bg-green-50 rounded">
            <h3 className="font-semibold mb-2">{t('vibe.descriptions.process.examples.apiExample.manualApi.title')}</h3>
            <button
              onClick={handleManualCall}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              {t('vibe.descriptions.process.examples.apiExample.manualApi.button')}
            </button>
            {manualResult && (
              <pre className="mt-2 p-2 bg-gray-100 rounded text-sm overflow-auto">
                {JSON.stringify(manualResult, null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>

      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
        <div className="flex">
          <div className="ml-3">
            <p className="text-sm text-yellow-700">
              <strong>{t('vibe.descriptions.process.examples.apiExample.guide.title')}</strong>
            </p>
            <ul className="mt-1 text-sm text-yellow-700 list-disc list-inside">
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.accessToken')}</li>
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.refreshToken')}</li>
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.autoRefresh')}</li>
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.offlineRecovery')}</li>
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.wsSupport')}</li>
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.tokenRotation')}</li>
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.seamless')}</li>
              <li>{t('vibe.descriptions.process.examples.apiExample.guide.smartCache')}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
