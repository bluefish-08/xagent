import { MCPServer } from "@/app/tools/page"

// Shared shape for a connector catalog entry / connected MCP app. Kept in one
// place so the connector dialog and the settings dialog can't drift apart.
export interface AppIntegration {
  id: string
  name: string
  description: string
  icon: string
  is_connected?: boolean
  users?: string
  provider?: string
  category?: string
  is_local?: boolean
  server_id?: number
  transport?: string
  connected_account?: string
  is_custom?: boolean
  server?: MCPServer
  launch_config?: {
    command?: string
    args?: string[]
    required_env?: string[]
  }
  // Key-based apps: a shared key (platform-admin or app-injected, e.g. team)
  // already covers required_env, so the user can connect without their own.
  shared_env_available?: boolean
  // Key-based apps: this user has set their own per-user key.
  user_env_configured?: boolean
}
