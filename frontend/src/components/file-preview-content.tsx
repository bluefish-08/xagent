"use client"

import { useEffect } from "react"
import { useApp } from "@/contexts/app-context-chat"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { Loader2, XIcon } from "lucide-react"
import { DocxPreviewRenderer } from "@/components/docx-preview-renderer"

interface FilePreviewContentProps {
  open: boolean
}

export function FilePreviewContent({ open }: FilePreviewContentProps) {
  const { state, dispatch } = useApp()
  const { filePreview } = state
  const { t } = useI18n()

  // Load file content when the preview is open within container
  useEffect(() => {
    if (open && filePreview.fileId && !filePreview.content && !filePreview.error) {
      const loadFileContent = async () => {
        try {
          const apiUrl = getApiUrl()

          // PPTX files are converted to PDF by backend, treat as PDF
          const isPptx = filePreview.fileName.match(/\.pptx$/i)
          const isPdf = isPptx || filePreview.fileName.match(/\.pdf$/i)
          const isDocx = filePreview.fileName.match(/\.docx$/i)

          const url = `${apiUrl}/api/files/preview/${filePreview.fileId}`

          const response = await apiRequest(url, {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          })

          if (response.ok) {
            let fileContent

            // Get MIME type from response headers (more reliable than file extension)
            const contentType = response.headers.get('content-type') || ''
            const mimeType = contentType.split(';')[0].trim()

          // Determine file type based on MIME type instead of file extension
          const isImage = mimeType.startsWith('image/')
          const isPdf = mimeType.startsWith('application/pdf') || mimeType === 'application/pdf'
          const isDocx = mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

          console.log('File preview debug:', {
            fileName: filePreview.fileName,
            mimeType,
            isImage,
            isDocx,
            isPdf,
            contentType: response.headers.get('content-type')
          })

          if (isImage || isPdf || isDocx || filePreview.fileName.match(/\.(docx|pdf|jpg|jpeg|png|gif|webp|svg|pptx)$/i)) {
            const arrayBuffer = await response.arrayBuffer()

            // Use modern, efficient base64 conversion
            const bytes = new Uint8Array(arrayBuffer)
            const binaryString = Array.from(bytes, (byte) => String.fromCharCode(byte)).join('')
            fileContent = btoa(binaryString)

            console.log('Base64 conversion completed:', {
              mimeType,
              originalSize: arrayBuffer.byteLength,
              base64Size: fileContent.length
            })
            } else {
              fileContent = await response.text()
            }

            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: fileContent, mimeType, error: null }
            })
          } else {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: t('files.previewDialog.errors.loadFailed') }
            })
          }
        } catch (error) {
          if ((error as any)?.name === 'TypeError' && (error as any)?.message?.includes('Failed to fetch')) {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: t('files.previewDialog.errors.cors') }
            })
          } else {
            const msg = (error as any)?.message || t('common.errors.unknown')
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: t('files.previewDialog.errors.networkErrorWithMsg', { msg }) }
            })
          }
        }
      }

      loadFileContent()
    }
  }, [open, filePreview.fileId, filePreview.content, filePreview.error, dispatch, t, filePreview.fileName])

  const processHtmlContent = (htmlContent: string, fileId: string) => {
    if (!htmlContent || !fileId) return htmlContent

    const apiUrl = getApiUrl()

    return htmlContent.replace(
      /(src|href)=["']([^"']+)["']/g,
      (match, attr, path) => {
        if (path.match(/^(https?:\/|data:|\/\/|#)/)) return match

        return `${attr}="${apiUrl}/api/files/public/preview/${encodeURIComponent(fileId)}?relative_path=${encodeURIComponent(path)}"`
      }
    )
  }

  return (
    <div className="w-full h-full">
      <div className="flex-1 overflow-hidden flex flex-col min-h-0 h-full">
        {filePreview.isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground">{t('files.previewDialog.loading')}</span>
            </div>
          </div>
        ) : filePreview.error ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-2 text-center">
              <XIcon className="h-8 w-8 text-destructive" />
              <span className="text-sm text-muted-foreground">{filePreview.error}</span>
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-auto bg-muted/30 rounded border">
            {filePreview.mimeType?.startsWith('image/') ? (
              <div className="flex items-center justify-center h-full p-4">
                <img
                  src={`data:${filePreview.mimeType};base64,${filePreview.content || ''}`}
                  alt={filePreview.fileName}
                  className="max-w-full max-h-full object-contain"
                  onError={(e) => {
                    console.error('Image load error:', e)
                    e.currentTarget.style.display = 'none'
                    const fallback = e.currentTarget.nextElementSibling as HTMLElement
                    if (fallback) fallback.style.display = 'flex'
                  }}
                />
                <div className="hidden flex-col items-center justify-center h-full text-muted-foreground">
                  <span>{t('files.previewDialog.imageError.title')}</span>
                  <span className="text-sm">{t('files.previewDialog.imageError.hint')}</span>
                </div>
              </div>
            ) : filePreview.mimeType === 'application/pdf' || filePreview.fileName.toLowerCase().endsWith('.pdf') || filePreview.fileName.toLowerCase().endsWith('.pptx') ? (
              <div className="flex items-center justify-center h-full p-4">
                <iframe
                  src={`data:application/pdf;base64,${filePreview.content || ''}`}
                  className="w-full h-full border-0"
                  title={filePreview.fileName}
                />
              </div>
            ) : filePreview.mimeType?.includes('wordprocessingml') || filePreview.fileName.toLowerCase().endsWith('.docx') ? (
              <DocxPreviewRenderer base64Content={filePreview.content || ''} />
            ) : filePreview.fileName.endsWith('.html') || filePreview.fileName.endsWith('.htm') ? (
              <iframe
                srcDoc={processHtmlContent(filePreview.content, filePreview.fileId)}
                className="w-full h-full border-0"
                sandbox="allow-same-origin allow-scripts"
                title={filePreview.fileName}
              />
            ) : (
              <pre className="p-4 text-sm font-mono whitespace-pre-wrap break-words">
                {filePreview.content || t('files.previewDialog.emptyContent')}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
