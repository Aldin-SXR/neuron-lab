import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  use: { baseURL: 'http://127.0.0.1:5173', browserName: 'chromium', viewport: { width: 1440, height: 1080 }, screenshot: 'only-on-failure' },
  webServer: { command: 'npm run dev -- --port 5173 --strictPort', url: 'http://127.0.0.1:5173', reuseExistingServer: !process.env.CI, timeout: 30_000 },
});
