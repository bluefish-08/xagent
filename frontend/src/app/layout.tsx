import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/contexts/auth-context";
import { ThemeProvider } from "@/contexts/theme-context";
import { AuthGuard } from "@/components/auth/auth-guard";
import { LayoutContent } from "@/components/layout/layout-content";
import { getBrandingFromEnv } from "@/lib/branding";
import { I18nProvider } from "@/contexts/i18n-context";

const branding = getBrandingFromEnv();

export const metadata: Metadata = {
  title: branding.appName,
  description: branding.description,
  icons: {
    icon: branding.logoPath,
    apple: branding.logoPath,
  },
};

// 获取主题配置
const getThemeConfig = () => {
  const themeName = process.env.NEXT_PUBLIC_THEME || 'dark';
  const themeMode = themeName === 'light' ? 'light' : 'dark';
  return { themeName, themeMode };
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const { themeMode } = getThemeConfig();

  return (
    <html lang="en" className={themeMode}>
      <body className="antialiased bg-background text-foreground" suppressHydrationWarning>
        <I18nProvider>
          <ThemeProvider>
            <AuthProvider>
              <AuthGuard>
                <LayoutContent>{children}</LayoutContent>
              </AuthGuard>
            </AuthProvider>
          </ThemeProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
