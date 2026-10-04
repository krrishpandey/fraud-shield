import { expect, test } from '@playwright/test'

// This spec starts signed out (the config pre-signs every other spec in).
test.use({ storageState: { cookies: [], origins: [] } })

test('sign-in: an unsigned visitor sees the sign-in page, signs in to the console, and signing out returns to it', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))

  await page.goto('/#/')
  await expect(page.getByTestId('login-page')).toBeVisible()
  await expect(page.getByTestId('nav-score')).toHaveCount(0)

  // validation: a name and a code of at least 4 characters
  await page.getByTestId('login-submit').click()
  await expect(page.getByTestId('login-error')).toContainText('analyst name')
  await page.getByTestId('login-name').fill('Analyst E2E')
  await page.getByTestId('login-code').fill('12')
  await page.getByTestId('login-submit').click()
  await expect(page.getByTestId('login-error')).toContainText('at least 4')

  await page.getByTestId('login-code').fill('1234')
  await page.getByTestId('login-submit').click()
  await expect(page.getByTestId('login-page')).toHaveCount(0)
  await expect(page.getByTestId('nav-score')).toBeVisible()
  await expect(page.getByTestId('nav-redteam')).toBeVisible()

  // the session survives a reload, and signing out returns to the sign-in page
  await page.reload()
  await expect(page.getByTestId('nav-score')).toBeVisible()
  await page.locator('button.sb-user').click() // the account menu at the foot of the sidebar
  await page.getByTestId('sign-out').click()
  await expect(page.getByTestId('login-page')).toBeVisible()
  expect(errors).toEqual([])
})
