// Render docs/figures/*.html into one PNG per chart (white background, 2x) for slides.
// Usage: cd e2e && node render_charts.mjs   (after scripts/build_charts.py)
import { chromium } from '@playwright/test'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const pages = ['lightgbm_charts.html', 'slide_charts.html']
const b = await chromium.launch()
for (const name of pages) {
  const p = await b.newPage({ viewport: { width: 1100, height: 1400 }, colorScheme: 'light', deviceScaleFactor: 2 })
  const errs = []
  p.on('pageerror', (e) => errs.push(e.message))
  await p.goto(pathToFileURL(path.resolve('../docs/figures', name)).href)
  await p.waitForTimeout(500)
  if (process.env.OUT) await p.screenshot({ path: path.join(process.env.OUT, name.replace('.html', '.png')), fullPage: true })
  for (const f of await p.locator('figure').all()) {
    await f.screenshot({ path: path.resolve('../docs/figures', `${await f.getAttribute('data-png')}.png`) })
  }
  console.log(name, 'errors:', errs)
}
await b.close()
