import type React from "react"
import type { Metadata } from "next"
import { Trispace } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"

const trispace = Trispace({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-trispace",
})

export const metadata: Metadata = {
  title: "SwarmLabs - Drone Swarm Simulation",
  description: "Advanced drone swarm simulation platform",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={trispace.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
