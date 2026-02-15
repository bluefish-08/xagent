/**
 * 时间格式化工具函数
 * 统一处理时间戳和ISO格式的时间显示
 */

/**
 * 格式化时间为本地时间字符串
 * @param timestamp 时间戳（秒或毫秒）或ISO字符串
 * @param format 输出格式：'time' | 'date' | 'datetime'
 * @returns 格式化后的时间字符串
 */
export function formatTime(
  timestamp: string | number | null | undefined,
  format: 'time' | 'date' | 'datetime' = 'time'
): string {
  if (!timestamp) {
    return ''
  }

  try {
    let date: Date

    if (typeof timestamp === 'number') {
      // 自动判断是秒级还是毫秒级时间戳
      // 10000000000 大约是 2286 年，所以小于这个值认为是秒级
      date = new Date(timestamp * (timestamp < 10000000000 ? 1000 : 1))
    } else {
      date = new Date(timestamp)
    }

    if (isNaN(date.getTime())) {
      return String(timestamp)
    }

    switch (format) {
      case 'time':
        return date.toLocaleTimeString()
      case 'date':
        return date.toLocaleDateString()
      case 'datetime':
        return date.toLocaleString()
      default:
        return date.toLocaleTimeString()
    }
  } catch {
    return String(timestamp)
  }
}

/**
 * 计算时间间隔
 * @param start 开始时间（时间戳或ISO字符串）
 * @param end 结束时间（时间戳或ISO字符串）
 * @returns 时间间隔（毫秒）
 */
export function getTimeDuration(
  start: string | number | null | undefined,
  end: string | number | null | undefined
): number {
  if (!start || !end) {
    return 0
  }

  try {
    let startDate: Date
    let endDate: Date

    if (typeof start === 'number') {
      startDate = new Date(start * (start < 10000000000 ? 1000 : 1))
    } else {
      startDate = new Date(start)
    }

    if (typeof end === 'number') {
      endDate = new Date(end * (end < 10000000000 ? 1000 : 1))
    } else {
      endDate = new Date(end)
    }

    if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
      return 0
    }

    return endDate.getTime() - startDate.getTime()
  } catch {
    return 0
  }
}

/**
 * 格式化时间间隔
 * @param duration 时间间隔（毫秒）
 * @returns 格式化后的时间字符串（如：1h 30m 25s）
 */
export function formatDuration(duration: number): string {
  if (duration <= 0) {
    return '0s'
  }

  const seconds = Math.floor(duration / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)

  const parts: string[] = []

  if (days > 0) {
    parts.push(`${days}d`)
  }
  if (hours % 24 > 0) {
    parts.push(`${hours % 24}h`)
  }
  if (minutes % 60 > 0) {
    parts.push(`${minutes % 60}m`)
  }
  if (seconds % 60 > 0) {
    parts.push(`${seconds % 60}s`)
  }

  return parts.join(' ') || '0s'
}

/**
 * 获取当前时间戳（秒级）
 * @returns 当前时间戳
 */
export function getCurrentTimestamp(): number {
  return Math.floor(Date.now() / 1000)
}

/**
 * 检查时间是否过期
 * @param timestamp 时间戳（秒级）
 * @param seconds 过期时间（秒）
 * @returns 是否过期
 */
export function isExpired(timestamp: number, seconds: number): boolean {
  return getCurrentTimestamp() - timestamp > seconds
}
