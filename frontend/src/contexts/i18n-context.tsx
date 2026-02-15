"use client"

import React, { createContext, useContext, useEffect, useMemo, useState } from "react"
import { translations } from "@/i18n/translations"

export type Locale = "en" | "zh"

type TFunc = (key: string, vars?: Record<string, string | number>) => string

interface I18nContextValue {
  locale: Locale
  setLocale: (l: Locale) => void
  t: TFunc
}

const I18nContext = createContext<I18nContextValue | undefined>(undefined)

function interpolate(str: string, vars?: Record<string, string | number>) {
  if (!vars) return str
  return Object.entries(vars).reduce((s, [k, v]) => s.replace(new RegExp(`\\{${k}\\}`, "g"), String(v)), str)
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en")

  // 初始化：优先读取本地存储；否则跟随浏览器语言
  useEffect(() => {
    try {
      const stored = typeof window !== "undefined" ? localStorage.getItem("app_locale") : null
      if (stored === "en" || stored === "zh") {
        setLocaleState(stored as Locale)
      } else {
        const navLang = typeof navigator !== "undefined"
          ? (navigator.languages?.[0] || navigator.language || "en").toLowerCase()
          : "en"
        setLocaleState(navLang.includes("zh") ? "zh" : "en")
      }
    } catch {
      // ignore
    }
  }, [])

  const setLocale = (l: Locale) => {
    setLocaleState(l)
    try {
      localStorage.setItem("app_locale", l)
    } catch {
      // ignore
    }
  }

  // 同步 <html lang> 属性
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.lang = locale
    }
  }, [locale])

  const t: TFunc = useMemo(() => {
    return (key, vars) => {
      const dict: any = translations[locale] || {}
      const value = key.split(".").reduce((acc: any, part: string) => (acc && acc[part] !== undefined ? acc[part] : undefined), dict)
      const str = typeof value === "string" ? value : key
      return interpolate(str, vars)
    }
  }, [locale])

  const value = useMemo(() => ({ locale, setLocale, t }), [locale, t])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error("useI18n must be used within I18nProvider")
  return ctx
}
