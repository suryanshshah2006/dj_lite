"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo, LogoMark } from "./Logo";
import { API_MODE } from "@/lib/api";

const NAV = [
  { href: "/", label: "Home", icon: HomeIcon },
  { href: "/mix", label: "Mix Builder", icon: MixIcon },
  { href: "/how", label: "How It Works", icon: InfoIcon },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <>
      {/* Desktop */}
      <aside className="hidden md:flex w-60 shrink-0 flex-col gap-6 border-r border-border bg-surface/60 px-4 py-6">
        <Link href="/" className="px-2" aria-label="DJ-Lite home"><Logo /></Link>
        <nav className="flex flex-col gap-1" aria-label="Main">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-semibold transition-colors ${
                  active ? "bg-surface-3 text-text" : "text-text-2 hover:bg-surface-2 hover:text-text"
                }`}
              >
                <Icon active={active} />
                {label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto rounded-xl border border-border bg-surface-2 p-3 text-xs text-text-2">
          <p className="font-semibold text-text">Transition-aware mixes</p>
          <p className="mt-1 leading-relaxed">Key, tempo and energy-aware playlists built from audio features.</p>
          {API_MODE === "mock" && (
            <p className="mt-2 inline-flex rounded-full bg-amber/15 px-2 py-0.5 font-semibold text-amber">Mock data</p>
          )}
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="md:hidden sticky top-0 z-30 flex items-center justify-between border-b border-border bg-bg/90 px-4 py-3 backdrop-blur">
        <Link href="/" aria-label="DJ-Lite home"><LogoMark size={28} /></Link>
        <nav className="flex gap-1" aria-label="Main">
          {NAV.map(({ href, label }) => {
            const active = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`rounded-full px-3 py-1.5 text-xs font-semibold ${active ? "bg-accent text-white" : "text-text-2"}`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </header>
    </>
  );
}

function HomeIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M3 10.5 12 3l9 7.5V21a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z" strokeLinejoin="round" />
    </svg>
  );
}
function MixIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M3 17c3 0 4-10 7-10s3 8 6 8 3-5 5-5" strokeLinecap="round" strokeLinejoin="round" />
      {active && <circle cx="10" cy="7" r="2.2" fill="currentColor" />}
    </svg>
  );
}
function InfoIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="9" fill={active ? "currentColor" : "none"} fillOpacity={active ? 0.2 : 0} />
      <path d="M12 11v6M12 7.5v.5" strokeLinecap="round" />
    </svg>
  );
}
