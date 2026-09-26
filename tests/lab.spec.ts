import { test, expect, type Page } from '@playwright/test';

function trackErrors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  page.on('requestfailed', request => errors.push(`Failed request: ${request.url()}`));
  return errors;
}

test('learn the real calculations, undo an exact weight update, and train continuously', async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'See learning happen.' })).toBeVisible();
  await expect(page.getByTestId('step-number')).toHaveText('1');
  await expect(page.getByRole('button', { name: 'Step backward', exact: true })).toBeDisabled();
  await page.screenshot({ path: 'test-results/neuron-lab-desktop.png', fullPage: true });
  for (let i = 0; i < 18; i++) await page.getByRole('button', { name: 'Step forward', exact: true }).click();
  await expect(page.getByText('A small step toward better.')).toBeVisible();
  const weights = () => page.locator('.wire-hit').evaluateAll(nodes => nodes.map(node => node.getAttribute('aria-label')));
  const before = await weights();
  await page.getByRole('button', { name: 'Step forward', exact: true }).click();
  const after = await weights();
  expect(after).not.toEqual(before);
  await page.getByRole('button', { name: 'Step backward', exact: true }).click();
  expect(await weights()).toEqual(before);
  await page.getByRole('button', { name: 'Step forward', exact: true }).click();
  expect(await weights()).toEqual(after);
  await page.screenshot({ path: 'test-results/neuron-lab-weight-update.png', fullPage: true });

  await page.getByRole('button', { name: 'Train', exact: true }).click();
  await page.getByRole('button', { name: 'Step forward', exact: true }).click();
  await expect(page.getByTestId('epoch')).toHaveText('1.25');
  const epochWeights = await weights();
  await page.getByRole('button', { name: 'Step backward', exact: true }).click();
  expect(await weights()).toEqual(after);
  await page.getByRole('button', { name: 'Step forward', exact: true }).click();
  expect(await weights()).toEqual(epochWeights);
  await page.getByLabel('Playback speed').selectOption('4');
  await page.getByRole('button', { name: 'Start training', exact: true }).click();
  await expect.poll(async () => Number(await page.getByTestId('epoch').textContent()), { timeout: 30_000 }).toBeGreaterThan(180);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await expect.poll(async () => Number(await page.getByTestId('loss').textContent())).toBeLessThan(0.08);
  await page.getByRole('button', { name: 'Save lab', exact: true }).click();
  const saved = await weights();
  await page.reload();
  await page.getByRole('button', { name: 'Restore saved', exact: true }).click();
  expect(await weights()).toEqual(saved);
  await expect(page.getByTestId('epoch')).toHaveText('0');
  expect(errors).toEqual([]);
});

test('edit architecture and parameters, use softmax, and explore 2D points', async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Add neuron to hidden layer 1' }).click();
  await expect(page.getByRole('button', { name: 'Layer 1 neuron 5, not calculated yet', exact: true })).toBeVisible();
  await page.getByLabel('Hidden layer 1 activation').selectOption('relu');
  await page.getByRole('button', { name: 'Add hidden layer', exact: true }).click();
  await expect(page.getByLabel('Hidden layer 3 activation')).toBeVisible();
  await page.getByRole('button', { name: 'Remove hidden layer 3', exact: true }).click();
  await page.getByLabel('Output activation', { exact: true }).selectOption('softmax');
  await expect(page.getByLabel('Output neurons', { exact: true })).toHaveValue('2');
  await page.getByRole('button', { name: 'Edit this neuron\'s parameters', exact: true }).click();
  await page.getByLabel('Weight from input 1', { exact: true }).fill('0.75');
  await page.getByLabel('Bias', { exact: true }).fill('0.1');
  await page.getByRole('button', { name: 'Apply parameters', exact: true }).click();
  await expect(page.locator('.wire-hit').first()).toHaveAttribute('aria-label', /0.7500/);
  await page.getByLabel('02 PICK A PROBLEM').selectOption('circle');
  await expect(page.getByRole('heading', { name: 'Find the inner circle' })).toBeVisible();
  await expect(page.locator('.scatter circle')).toHaveCount(80);
  await page.getByRole('button', { name: /^Inspect point 1, class/ }).click();
  await expect(page.getByTestId('step-number')).toHaveText('1');
  await page.getByRole('button', { name: 'Use recommended network', exact: true }).click();
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  await page.getByLabel('Playback speed').selectOption('4');
  await page.getByRole('button', { name: 'Start training', exact: true }).click();
  await expect.poll(async () => Number(await page.getByTestId('epoch').textContent()), { timeout: 30_000 }).toBeGreaterThan(100);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.screenshot({ path: 'test-results/neuron-lab-circle.png', fullPage: true });
  await page.getByRole('button', { name: 'CNN, planned architecture' }).click();
  await expect(page.getByRole('heading', { name: 'One architecture at a time.' })).toBeVisible();
  await page.getByRole('button', { name: 'Close dialog' }).click();
  await page.getByRole('button', { name: 'Quick guide', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your first aha moment starts here.' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('dialog')).not.toBeVisible();
  expect(errors).toEqual([]);
});

test('mobile layout stays within the viewport and retains working controls', async ({ page }) => {
  const errors = trackErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Step forward', exact: true }).click();
  await expect(page.getByTestId('step-number')).toHaveText('2');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/neuron-lab-mobile.png', fullPage: true });
  expect(errors).toEqual([]);
});
