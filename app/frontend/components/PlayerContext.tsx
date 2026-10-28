"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import type { Track } from "@/lib/types";

interface PlayerState {
  current: Track | null;
  play: (t: Track) => void;
  stop: () => void;
}

const Ctx = createContext<PlayerState | null>(null);

export function PlayerProvider({ children }: { children: ReactNode }) {
  const [current, setCurrent] = useState<Track | null>(null);
  return (
    <Ctx.Provider value={{ current, play: setCurrent, stop: () => setCurrent(null) }}>
      {children}
    </Ctx.Provider>
  );
}

export function usePlayer(): PlayerState {
  const v = useContext(Ctx);
  if (!v) throw new Error("usePlayer must be used inside <PlayerProvider>");
  return v;
}
