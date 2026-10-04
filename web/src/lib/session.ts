import { createContext, useContext } from 'react'

/*
  Console sign-in. This is a front-end gate for the demo: the API has no accounts,
  so the name only labels who is at the desk. It is not access control.
*/
const KEY = 'fs-analyst'
const THEME_KEY = 'fs-theme'

export function readAnalyst(): string | null {
  try {
    return localStorage.getItem(KEY)
  } catch {
    return null
  }
}

export function writeAnalyst(name: string | null): void {
  try {
    if (name) localStorage.setItem(KEY, name)
    else localStorage.removeItem(KEY)
  } catch {
    /* storage blocked: the session lasts until reload */
  }
}

export type Session = { analyst: string; signOut: () => void }
export const SessionContext = createContext<Session>({ analyst: '', signOut: () => {} })
export const useSession = () => useContext(SessionContext)

export function applySavedTheme(): void {
  try {
    const t = localStorage.getItem(THEME_KEY)
    if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t
  } catch {
    /* default: follow the system */
  }
}

export function toggleTheme(): void {
  const r = document.documentElement
  const dark = r.dataset.theme ? r.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  r.dataset.theme = dark ? 'light' : 'dark'
  try {
    localStorage.setItem(THEME_KEY, r.dataset.theme)
  } catch {
    /* ignore */
  }
}
