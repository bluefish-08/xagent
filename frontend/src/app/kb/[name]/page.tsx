import KnowledgeBaseDetailPage from "@/components/pages/knowledge-base-detail"

export default function KnowledgeBaseDetail({ params }: { params: Promise<{ name: string }> }) {
  return <KnowledgeBaseDetailPage params={params} />
}
