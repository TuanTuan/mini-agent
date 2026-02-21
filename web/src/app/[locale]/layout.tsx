import type { Metadata } from "next";
import { I18nProvider } from "@/lib/i18n";
import { Header } from "@/components/layout/header";
import "../globals.css";

export const metadata: Metadata = {
  title: "Learn Claude Code",
  description: "Build an AI coding agent from scratch, one concept at a time",
};

const locales = ["en", "zh", "ja"];

export function generateStaticParams() {
  return locales.map((locale) => ({ locale }));
}

export default async function RootLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;

  return (
    <html lang={locale} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            var theme = localStorage.getItem('theme');
            if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
              document.documentElement.classList.add('dark');
            }
          })();
        `}} />
      </head>
      <body className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)] antialiased">
        <I18nProvider locale={locale}>
          <Header />
          <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
            {children}
          </main>
        </I18nProvider>
      </body>
    </html>
  );
}
