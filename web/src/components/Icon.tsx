/* Line icons in one weight (24 grid, 1.8 stroke), drawn for this console. */
const P: Record<string, string> = {
  box: 'M12 2.8l8 4.4v9.6l-8 4.4-8-4.4V7.2zM4 7.2l8 4.4 8-4.4M12 11.6v9.6',
  decision: 'M4 4h16v16H4zM8.5 12.2l2.4 2.4 4.8-5',
  pulse: 'M2.5 12h4l2.5-6 5 12 2.5-6h5',
  list: 'M8.5 6.5H20M8.5 12H20M8.5 17.5H20M4 6.5h.01M4 12h.01M4 17.5h.01',
  chart: 'M4 20V4M4 20h16M8.5 16v-4M12.5 16V8M16.5 16v-6',
  refresh: 'M20 11a8 8 0 00-14.3-4.6L4 8.5M4 4v4.5h4.5M4 13a8 8 0 0014.3 4.6L20 15.5M20 20v-4.5h-4.5',
  shield: 'M12 2.8l7.5 3v6.2c0 4.6-3.2 7.8-7.5 9.2-4.3-1.4-7.5-4.6-7.5-9.2V5.8z',
  shieldCheck: 'M12 2.8l7.5 3v6.2c0 4.6-3.2 7.8-7.5 9.2-4.3-1.4-7.5-4.6-7.5-9.2V5.8zM8.8 12l2.2 2.2 4.2-4.4',
  settings: 'M12 15.2a3.2 3.2 0 100-6.4 3.2 3.2 0 000 6.4zM19.4 13.5l1.6 1.2-2 3.4-1.9-.7a7.5 7.5 0 01-2.3 1.3L14.5 21h-4l-.3-2.3a7.5 7.5 0 01-2.3-1.3l-1.9.7-2-3.4 1.6-1.2a7.6 7.6 0 010-2.9L4 9.4l2-3.4 1.9.7a7.5 7.5 0 012.3-1.3L10.5 3h4l.3 2.3a7.5 7.5 0 012.3 1.3l1.9-.7 2 3.4-1.6 1.2a7.6 7.6 0 010 2.9z',
  search: 'M10.8 18a7.2 7.2 0 100-14.4 7.2 7.2 0 000 14.4zM16 16l4.5 4.5',
  chevronDown: 'M6 9l6 6 6-6',
  chevronRight: 'M9 6l6 6-6 6',
  updown: 'M8 9l4-4 4 4M8 15l4 4 4-4',
  panel: 'M4 4h16v16H4zM9.5 4v16',
  history: 'M3.5 12a8.5 8.5 0 102.5-6M3.5 4v4h4M12 7.5V12l3 2',
  help: 'M12 21a9 9 0 100-18 9 9 0 000 18zM9.5 9.3a2.6 2.6 0 015 .8c0 1.7-2.5 2.2-2.5 3.6M12 17h.01',
  cpu: 'M7 7h10v10H7zM10 2.5V7M14 2.5V7M10 17v4.5M14 17v4.5M2.5 10H7M2.5 14H7M17 10h4.5M17 14h4.5',
  plus: 'M12 5v14M5 12h14',
  upload: 'M12 15V4M7.5 8.5L12 4l4.5 4.5M4 15v5h16v-5',
  download: 'M12 4v11M7.5 10.5L12 15l4.5-4.5M4 15v5h16v-5',
  filter: 'M3.5 5h17l-6.5 8v6l-4 2v-8z',
  x: 'M6 6l12 12M18 6L6 18',
  check: 'M4.5 12.5l5 5L19.5 7',
  flag: 'M5 21V4M5 4h11l-2 4 2 4H5',
  link: 'M9.5 14.5l5-5M10.5 6.5l1.2-1.2a4 4 0 015.7 5.7l-1.2 1.2M13.5 17.5l-1.2 1.2a4 4 0 01-5.7-5.7l1.2-1.2',
  lock: 'M6 11h12v9.5H6zM8.5 11V8a3.5 3.5 0 017 0v3',
  info: 'M12 21a9 9 0 100-18 9 9 0 000 18zM12 11v5.5M12 7.5h.01',
  lines: 'M4 6h16M4 10.5h16M4 15h10',
  dollar: 'M12 21a9 9 0 100-18 9 9 0 000 18zM14.8 9.2c-.5-1-1.6-1.5-2.8-1.5-1.6 0-2.7.8-2.7 2 0 2.8 5.6 1.6 5.6 4.4 0 1.2-1.2 2.1-2.9 2.1-1.3 0-2.4-.6-2.9-1.6M12 6v1.7M12 16.2V18',
  user: 'M12 12a4 4 0 100-8 4 4 0 000 8zM4.5 21c.8-3.8 3.8-6 7.5-6s6.7 2.2 7.5 6',
  route: 'M6.5 19.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5zM17.5 9.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5zM9 17h6.5a3.5 3.5 0 000-7h-7a3.5 3.5 0 010-7H15',
  play: 'M7 4.5v15l12-7.5z',
  pause: 'M7 5h3.5v14H7zM13.5 5H17v14h-3.5z',
  calendar: 'M4 6h16v14H4zM4 10h16M8.5 3.5v4M15.5 3.5v4',
  copy: 'M9 9h11v11H9zM5 15H4V4h11v1',
  more: 'M12 6h.01M12 12h.01M12 18h.01',
  hub: 'M5 3.5h14v17H5zM8.5 7.5h7M8.5 11h7M8.5 14.5h4',
  bolt: 'M13 2.5L5 13.5h6.5L10.5 21.5 19 10.5h-6.5z',
  stack: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5',
  logout: 'M15 4h4.5v16H15M10 8l-4 4 4 4M6 12h10',
  eye: 'M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12zM12 15a3 3 0 100-6 3 3 0 000 6z',
  scale: 'M12 4v16M5 20h14M6 8h12M6 8l-3 6a3 3 0 006 0zM18 8l-3 6a3 3 0 006 0zM12 4h.01',
  tag: 'M3.5 12.5V4h8.5l8.5 8.5-8.5 8.5zM8 8h.01',
  store: 'M4 9.5L5.5 4h13L20 9.5M4 9.5h16M4 9.5a2.7 2.7 0 005.3 0 2.7 2.7 0 005.4 0 2.7 2.7 0 005.3 0M5.5 12v8h13v-8M10 20v-4.5h4V20',
  pin: 'M12 21s-6.5-6.2-6.5-11a6.5 6.5 0 0113 0c0 4.8-6.5 11-6.5 11zM12 12.3a2.3 2.3 0 100-4.6 2.3 2.3 0 000 4.6z',
  takeover: 'M10 12a4 4 0 100-8 4 4 0 000 8zM3 21c.7-3.6 3.5-5.8 7-5.8 1.2 0 2.3.3 3.3.7M16 15l5 5M21 15l-5 5',
  inbox: 'M3.5 13.5l2.6-8h11.8l2.6 8v6h-17zM3.5 13.5h5l1.5 2.5h4l1.5-2.5h5',
  sparkle: 'M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18',
}

export type IconName = keyof typeof P

export function Icon({ name, size = 18, stroke = 1.8, className, style }: { name: IconName; size?: number; stroke?: number; className?: string; style?: React.CSSProperties }) {
  const filled = name === 'play' || name === 'pause'
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
      style={{ flex: 'none', ...style }}
      fill={filled ? 'currentColor' : 'none'}
      stroke={filled ? 'none' : 'currentColor'}
      strokeWidth={stroke}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={P[name]} />
    </svg>
  )
}
