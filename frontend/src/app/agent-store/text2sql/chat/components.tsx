// 专业的图表组件
import React, { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart as RechartsPieChart, Pie, Cell, LineChart, Line, ResponsiveContainer } from 'recharts'
import { TrendingUp, Loader2 } from 'lucide-react'
import { apiRequest } from '@/lib/api-wrapper'
import { getApiUrl } from '@/lib/utils'
import { useI18n } from '@/contexts/i18n-context'

// 预测功能的Hook
const usePrediction = () => {
  const { t } = useI18n()
  const [isPredicting, setIsPredicting] = useState(false)
  const [predictionData, setPredictionData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const generatePrediction = async (chartType: string, data: any, mapping?: any) => {
    setIsPredicting(true)
    setError(null)
    setSuccessMessage(null)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/text2sql/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chartType,
          data: {
            columns: data.columns,
            rows: data.rows
          },
          mapping,
          predictPeriods: 5
        })
      })

      if (response.ok) {
        const result = await response.json()

        if (result.success) {
          setPredictionData(result)
          setSuccessMessage(result.trendAnalysis || t('agentStore.text2sql.charts.predict.successDefault'))
          return result
        } else {
          setError(result.error || t('agentStore.text2sql.charts.predict.failedDefault'))
          return null
        }
      } else {
        setError(t('agentStore.text2sql.charts.predict.requestFailedHttp', { status: response.status }))
        return null
      }
    } catch (error) {
      console.error('Prediction error:', error)
      setError(t('agentStore.text2sql.charts.predict.requestFailedNetwork'))
      return null
    } finally {
      setIsPredicting(false)
    }
  }

  return {
    isPredicting,
    predictionData,
    error,
    successMessage,
    generatePrediction
  }
}

// 合并历史数据和预测数据的函数
const mergeWithPrediction = (historicalData: any[], predictionData: any[], labelKey: string = 'name', valueKey: string = 'value') => {
  const merged = [...historicalData]

  predictionData.forEach((prediction: any) => {
    merged.push({
      [labelKey]: prediction.period,
      [valueKey]: prediction.predictedValue,
      isPrediction: true,
      confidenceLower: prediction.confidenceLower,
      confidenceUpper: prediction.confidenceUpper
    })
  })

  return merged
}

// 专业的条形图组件
export const SimpleBarChart = ({ data }: { data: any }) => {
  const { t } = useI18n()
  const { isPredicting, generatePrediction, predictionData, error, successMessage } = usePrediction()
  const [showPrediction, setShowPrediction] = useState(false)

  if (!data.rows || data.rows.length === 0) return null

  // 找到数值列，排除 ID 类列
  const numericColumns = data.columns.filter((col: string) => {
    // 排除常见的 ID 列
    if (col.toLowerCase().includes('id') || col.toLowerCase().includes('key')) {
      return false
    }

    return data.rows.some((row: any) => {
      const value = row[col]
      // 只选择真正的数值列，而不是 ID
      return typeof value === 'number' && !col.toLowerCase().includes('id') ||
             (typeof value === 'string' && !isNaN(parseFloat(value)) && !col.toLowerCase().includes('id'))
    })
  })

  if (numericColumns.length === 0) {
    return <div className="text-center p-4 text-muted-foreground">{t('agentStore.text2sql.charts.noNumericData')}</div>
  }

  // 只取前10条数据
  const displayData = data.rows.slice(0, 10)
  // 优先选择文本列作为标签，排除 ID 列
    const labelColumn = data.columns.find((col: string) =>
      col !== numericColumns[0] &&
      !col.toLowerCase().includes('id') &&
      !col.toLowerCase().includes('key') &&
      data.rows.some((row: any) => typeof row[col] === 'string')
    ) || data.columns.find((col: string) => col !== numericColumns[0]) || data.columns[0]
  const valueColumn = numericColumns[0]

  // 为 Recharts 准备数据
  const historicalChartData = displayData.map((row: any) => ({
    name: row[labelColumn],
    value: Number(row[valueColumn]) || 0
  }))

  // 合并预测数据
  const chartData = showPrediction && predictionData?.predictedData
    ? mergeWithPrediction(historicalChartData, predictionData.predictedData)
    : historicalChartData

  const handlePrediction = async () => {
    const result = await generatePrediction('bar', data, {
      xAxis: labelColumn,
      valueAxis: valueColumn
    })

    if (result) {
      setShowPrediction(true)
    }
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h5 className="font-medium">{t('agentStore.text2sql.charts.bar.rankTop', { valueColumn, top: 10 })}</h5>
        <button
          onClick={handlePrediction}
          disabled={isPredicting}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '4px',
            fontSize: '14px',
            cursor: isPredicting ? 'not-allowed' : 'pointer',
            opacity: isPredicting ? 0.6 : 1
          }}
        >
          {isPredicting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <TrendingUp className="h-4 w-4" />
          )}
          {isPredicting ? t('agentStore.text2sql.charts.predict.buttonPredicting') : t('agentStore.text2sql.charts.predict.buttonPredict')}
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-800">
            <strong>✗ </strong>{error}
          </p>
        </div>
      )}

      {/* 趋势分析结果 */}
      {showPrediction && predictionData?.trendAnalysis && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
          <p className="text-sm text-blue-800">
            <strong>{t('agentStore.text2sql.charts.analysis.trendAnalysisLabel')}</strong>{predictionData.trendAnalysis}
            <span className="ml-2 text-xs bg-blue-100 px-2 py-1 rounded">
              {t('agentStore.text2sql.charts.analysis.confidence', { confidence: predictionData.confidence })}
            </span>
          </p>
        </div>
      )}

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="name"
            angle={-45}
            textAnchor="end"
            height={60}
            interval={0}
            tick={{ fontSize: 12 }}
          />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" fill="#3b82f6" name={t('agentStore.text2sql.charts.legend.historical')} />
          {showPrediction && predictionData?.predictedData && (
            <Bar
              dataKey="value"
              fill="#10b981"
              name={t('agentStore.text2sql.charts.legend.prediction')}
              fillOpacity={0.6}
              stroke="#10b981"
              strokeDasharray="5 5"
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// 专业的饼图组件
export const SimplePieChart = ({ data }: { data: any }) => {
  const { t } = useI18n()
  const { isPredicting, generatePrediction, predictionData, error, successMessage } = usePrediction()
  const [showPrediction, setShowPrediction] = useState(false)

  if (!data.rows || data.rows.length === 0) return null

  // 找到数值列，排除 ID 类列
  const numericColumns = data.columns.filter((col: string) => {
    // 排除常见的 ID 列
    if (col.toLowerCase().includes('id') || col.toLowerCase().includes('key')) {
      return false
    }

    return data.rows.some((row: any) => !isNaN(Number(row[col])))
  })

  if (numericColumns.length === 0) {
    return <div className="text-center p-4 text-muted-foreground">{t('agentStore.text2sql.charts.noNumericData')}</div>
  }

  // 取前8条数据
  const displayData = data.rows.slice(0, 8)
  // 优先选择文本列作为标签，排除 ID 列
    const labelColumn = data.columns.find((col: string) =>
      col !== numericColumns[0] &&
      !col.toLowerCase().includes('id') &&
      !col.toLowerCase().includes('key') &&
      data.rows.some((row: any) => typeof row[col] === 'string')
    ) || data.columns.find((col: string) => col !== numericColumns[0]) || data.columns[0]
  const valueColumn = numericColumns[0]

  // 为 Recharts 准备数据
  const historicalChartData = displayData.map((row: any) => ({
    name: row[labelColumn],
    value: Number(row[valueColumn]) || 0
  }))

  // 合并预测数据
  const chartData = showPrediction && predictionData?.predictedData
    ? mergeWithPrediction(historicalChartData, predictionData.predictedData)
    : historicalChartData

  const handlePrediction = async () => {
    const result = await generatePrediction('pie', data, {
      valueAxis: valueColumn,
      xAxis: labelColumn
    })

    if (result) {
      setShowPrediction(true)
    }
  }

  const COLORS = [
    '#3b82f6', '#10b981', '#eab308', '#ef4444',
    '#a855f7', '#ec4899', '#14b8a6', '#f97316'
  ]

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h5 className="font-medium">{t('agentStore.text2sql.charts.pie.distributionTop', { valueColumn, top: 8 })}</h5>
        <button
          onClick={handlePrediction}
          disabled={isPredicting}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '4px',
            fontSize: '14px',
            cursor: isPredicting ? 'not-allowed' : 'pointer',
            opacity: isPredicting ? 0.6 : 1
          }}
        >
          {isPredicting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <TrendingUp className="h-4 w-4" />
          )}
          {isPredicting ? t('agentStore.text2sql.charts.predict.buttonPredicting') : t('agentStore.text2sql.charts.predict.buttonPredict')}
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-800">
            <strong>✗ </strong>{error}
          </p>
        </div>
      )}

      {/* 趋势分析结果 */}
      {showPrediction && predictionData?.trendAnalysis && (
        <div className="mb-4 p-3 bg-purple-50 border border-purple-200 rounded-md">
          <p className="text-sm text-purple-800">
            <strong>{t('agentStore.text2sql.charts.analysis.trendAnalysisLabel')}</strong>{predictionData.trendAnalysis}
            <span className="ml-2 text-xs bg-purple-100 px-2 py-1 rounded">
              {t('agentStore.text2sql.charts.analysis.confidence', { confidence: predictionData.confidence })}
            </span>
          </p>
        </div>
      )}

      <ResponsiveContainer width="100%" height={300}>
        <RechartsPieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry: any, index: number) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.isPrediction ? '#10b981' : COLORS[index % COLORS.length]}
                stroke={entry.isPrediction ? '#10b981' : 'none'}
                strokeDasharray={entry.isPrediction ? '3 3' : 'none'}
              />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </RechartsPieChart>
      </ResponsiveContainer>
    </div>
  )
}

// 专业的折线图组件（最适合预测展示）
export const SimpleLineChart = ({ data }: { data: any }) => {
  const { t } = useI18n()
  const { isPredicting, generatePrediction, predictionData, error, successMessage } = usePrediction()
  const [showPrediction, setShowPrediction] = useState(false)

  if (!data.rows || data.rows.length === 0) return null

  // 找到数值列，排除 ID 类列
  const numericColumns = data.columns.filter((col: string) => {
    // 排除常见的 ID 列
    if (col.toLowerCase().includes('id') || col.toLowerCase().includes('key')) {
      return false
    }

    return data.rows.some((row: any) => {
      const value = row[col]
      // 只选择真正的数值列，而不是 ID
      return typeof value === 'number' && !col.toLowerCase().includes('id') ||
             (typeof value === 'string' && !isNaN(parseFloat(value)) && !col.toLowerCase().includes('id'))
    })
  })

  if (numericColumns.length === 0) {
    return <div className="text-center p-4 text-muted-foreground">{t('agentStore.text2sql.charts.noNumericData')}</div>
  }

  // 只取前20条数据（折线图适合显示更多数据点）
  const displayData = data.rows.slice(0, 20)

  // 优先选择文本列或时间列作为X轴标签，排除 ID 列
  const labelColumn = data.columns.find((col: string) =>
    col !== numericColumns[0] &&
    !col.toLowerCase().includes('id') &&
    !col.toLowerCase().includes('key') &&
    data.rows.some((row: any) => typeof row[col] === 'string')
  ) || data.columns.find((col: string) => col !== numericColumns[0]) || data.columns[0]

  const valueColumn = numericColumns[0]

  // 为 Recharts 准备数据
  const historicalChartData = displayData.map((row: any) => ({
    name: row[labelColumn],
    value: Number(row[valueColumn]) || 0
  }))

  // 合并预测数据
  const chartData = showPrediction && predictionData?.predictedData
    ? mergeWithPrediction(historicalChartData, predictionData.predictedData)
    : historicalChartData

  const handlePrediction = async () => {
    const result = await generatePrediction('line', data, {
      xAxis: labelColumn,
      valueAxis: valueColumn
    })

    if (result) {
      setShowPrediction(true)
    }
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h5 className="font-medium">{t('agentStore.text2sql.charts.line.trendTop', { valueColumn, top: 20 })}</h5>
        <button
          onClick={handlePrediction}
          disabled={isPredicting}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '4px',
            fontSize: '14px',
            cursor: isPredicting ? 'not-allowed' : 'pointer',
            opacity: isPredicting ? 0.6 : 1
          }}
        >
          {isPredicting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <TrendingUp className="h-4 w-4" />
          )}
          {isPredicting ? t('agentStore.text2sql.charts.predict.buttonPredicting') : t('agentStore.text2sql.charts.predict.buttonPredict')}
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-800">
            <strong>✗ </strong>{error}
          </p>
        </div>
      )}

      {/* 趋势分析结果 */}
      {showPrediction && predictionData?.trendAnalysis && (
        <div className="mb-4 p-3 bg-purple-50 border border-purple-200 rounded-md">
          <p className="text-sm text-purple-800">
            <strong>{t('agentStore.text2sql.charts.analysis.trendAnalysisLabel')}</strong>{predictionData.trendAnalysis}
            <span className="ml-2 text-xs bg-purple-100 px-2 py-1 rounded">
              {t('agentStore.text2sql.charts.analysis.confidence', { confidence: predictionData.confidence })}
            </span>
          </p>
        </div>
      )}

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="name"
            angle={-45}
            textAnchor="end"
            height={60}
            interval={0}
            tick={{ fontSize: 12 }}
          />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ fill: '#3b82f6', r: 4 }}
            activeDot={{ r: 6 }}
            name={t('agentStore.text2sql.charts.legend.historical')}
          />
          {showPrediction && predictionData?.predictedData && (
            <Line
              type="monotone"
              dataKey="value"
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#10b981', r: 3 }}
              name={t('agentStore.text2sql.charts.legend.prediction')}
              connectNulls={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
