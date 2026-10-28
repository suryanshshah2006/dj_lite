// DJ-Lite mark: a vinyl disc with a waveform crossing the label.
export function LogoMark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="djl-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--accent-hi)" />
          <stop offset="1" stopColor="var(--accent-lo)" />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="15" fill="url(#djl-g)" />
      <circle cx="16" cy="16" r="11.5" fill="none" stroke="#0b0b10" strokeOpacity="0.35" strokeWidth="0.8" />
      <circle cx="16" cy="16" r="8.5" fill="none" stroke="#0b0b10" strokeOpacity="0.35" strokeWidth="0.8" />
      <circle cx="16" cy="16" r="5.2" fill="#0b0b10" />
      <polyline
        points="5,16 8,16 9.5,12 11.5,20 13.5,9.5 15.5,22.5 17.5,11 19.5,19 21.5,14 23,16 27,16"
        fill="none" stroke="#f3f2f8" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark />
      <span className="text-lg font-extrabold tracking-tight text-text">
        DJ<span className="text-accent-hi">-</span>Lite
      </span>
    </span>
  );
}
