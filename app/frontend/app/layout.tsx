import type { Metadata } from "next";
import { Manrope } from "next/font/google";
import "./globals.css";
import { PlayerProvider } from "@/components/PlayerContext";
import { Sidebar } from "@/components/Sidebar";
import { PlayerBar } from "@/components/PlayerBar";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope", display: "swap" });

export const metadata: Metadata = {
  title: "DJ-Lite — transition-aware mixes",
  description: "Build DJ-style playlists that respect key, tempo and an energy arc.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={manrope.variable}>
      <body>
        <PlayerProvider>
          <div className="flex min-h-screen flex-col md:flex-row">
            <Sidebar />
            <div className="flex min-w-0 flex-1 flex-col">
              <main className="flex-1 px-4 pb-8 pt-6 sm:px-6 lg:px-10">{children}</main>
              <footer className="border-t border-border px-4 py-5 pb-[112px] text-xs text-muted sm:px-6 lg:px-10">
                Artwork &amp; playback via Spotify. Audio features from the Spotify 1.2M+ Songs dataset (Kaggle).
              </footer>
            </div>
          </div>
          <PlayerBar />
        </PlayerProvider>
      </body>
    </html>
  );
}
