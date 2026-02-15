"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Sparkles, Database, ArrowRight, Star, Users, Clock } from "lucide-react"
import Link from "next/link"
import { useI18n } from "@/contexts/i18n-context"

export default function AgentStorePage() {
  const { t } = useI18n()
  return (
    <div className="container mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
          {t("agentStore.title")}
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          {t("agentStore.subtitle")}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* 内置 Agents */}
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-purple-400" />
            <h2 className="text-2xl font-semibold">{t("agentStore.builtIn.title")}</h2>
            <Badge variant="outline" className="ml-2">{t("agentStore.builtIn.badgeOfficial")}</Badge>
          </div>

          <div className="grid gap-4">
            {/* Text2SQL Agent Card */}
            <Card className="group hover:shadow-lg transition-all duration-300 border-2 hover:border-blue-400/50">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                      <Database className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                    </div>
                    <div>
                      <CardTitle className="text-xl">{t("agentStore.text2sql.title")}</CardTitle>
                      <CardDescription className="text-sm">
                        {t("agentStore.text2sql.descriptionShort")}
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Star className="h-4 w-4 text-yellow-400 fill-current" />
                    <span className="text-sm text-muted-foreground">4.9</span>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  {t("agentStore.text2sql.featuresParagraph")}
                </p>

                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Users className="h-4 w-4" />
                    <span>{t("agentStore.stats.users", { count: 128 })}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    <span>{t("agentStore.stats.usesToday", { count: 42 })}</span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">SQLite</Badge>
                  <Badge variant="secondary">PostgreSQL</Badge>
                  <Badge variant="secondary">MySQL</Badge>
                  <Badge variant="secondary">{t("agentStore.badges.smartCorrection")}</Badge>
                </div>

                <Link href="/agent-store/text2sql">
                  <Button className="w-full group-hover:bg-blue-600 transition-colors">
                    {t("agentStore.text2sql.startUsing")}
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              </CardContent>
            </Card>

            {/* 占位卡片 - 未来的内置 Agents */}
            <Card className="opacity-60">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                    <Sparkles className="h-6 w-6 text-gray-600 dark:text-gray-400" />
                  </div>
                  <div>
                    <CardTitle className="text-xl">{t("agentStore.moreBuiltIn.title")}</CardTitle>
                    <CardDescription>{t("agentStore.moreBuiltIn.comingSoon")}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  {t("agentStore.moreBuiltIn.description")}
                </p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* VIBD 部署的 Agents */}
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-green-400" />
            <h2 className="text-2xl font-semibold">{t("agentStore.deployments.title")}</h2>
            <Badge variant="outline" className="ml-2">{t("agentStore.deployments.badgeCommunity")}</Badge>
          </div>

          <Card className="border-dashed border-2 border-muted-foreground/30">
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <Sparkles className="h-12 w-12 text-muted-foreground/50 mb-4" />
              <h3 className="text-lg font-semibold mb-2">{t("agentStore.deployments.noneTitle")}</h3>
              <p className="text-muted-foreground mb-4 max-w-md">
                {t("agentStore.deployments.noneDescription")}
              </p>
              <div className="space-y-2">
                <Link href="/agent/vibe">
                  <Button variant="outline">
                    {t("agentStore.actions.createAgent")}
                  </Button>
                </Link>
                <p className="text-xs text-muted-foreground">
                  {t("agentStore.deployments.helperText")}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* 示例部署的 Agents (当有实际部署时显示) */}
          <div className="space-y-4 hidden">
            {/* 这里将来会显示通过 VIBD 部署的 Agents */}
          </div>
        </div>
      </div>

      {/* 底部信息 */}
      <div className="text-center py-8 border-t border-border">
        <h3 className="text-lg font-semibold mb-2">{t("agentStore.footer.title")}</h3>
        <p className="text-muted-foreground mb-4">
          {t("agentStore.footer.description")}
        </p>
        <div className="flex justify-center gap-4">
          <Link href="/agent-store/text2sql">
            <Button>{t("agentStore.footer.tryText2SQL")}</Button>
          </Link>
          <Link href="/agent/vibe">
            <Button variant="outline">{t("agentStore.footer.createCustomAgent")}</Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
