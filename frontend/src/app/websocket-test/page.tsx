"use client"

import { useEffect } from "react"
import { useWebSocket } from "@/hooks/use-websocket"

export default function WebSocketTestPage() {
  const { isConnected, connect, disconnect, connectionError } = useWebSocket({
    autoConnect: false,
  })

  useEffect(() => {
    console.log("WebSocket Test Page - Connection status:", isConnected)
    console.log("WebSocket Test Page - Error:", connectionError)
  }, [isConnected, connectionError])

  return (
    <div className="min-h-screen bg-[#0D1117] text-white p-8">
      <h1 className="text-2xl font-bold mb-4">WebSocket 连接测试</h1>

      <div className="space-y-4">
        <div className="p-4 bg-[#161B22] rounded-lg">
          <p className="text-lg">连接状态:
            <span className={`ml-2 ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              {isConnected ? '已连接' : '未连接'}
            </span>
          </p>
        </div>

        {connectionError && (
          <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg">
            <p className="text-red-400">连接错误: {connectionError.message}</p>
          </div>
        )}

        <div className="flex gap-4">
          <button
            onClick={() => connect()}
            disabled={isConnected}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg transition-colors"
          >
            连接 WebSocket
          </button>

          <button
            onClick={() => disconnect()}
            disabled={!isConnected}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded-lg transition-colors"
          >
            断开连接
          </button>
        </div>

        <div className="p-4 bg-[#161B22] rounded-lg">
          <h2 className="text-lg font-semibold mb-2">测试说明</h2>
          <ul className="text-sm text-gray-400 space-y-1">
            <li>• 点击"连接 WebSocket"测试连接</li>
            <li>• 点击"断开连接"测试断开</li>
            <li>• 检查浏览器控制台是否有 "The message port closed before a response was received" 错误</li>
            <li>• 刷新页面测试组件卸载时的清理逻辑</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
