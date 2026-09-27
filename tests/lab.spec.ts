import { test, expect, type Page } from '@playwright/test';

function trackErrors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  page.on('requestfailed', request => errors.push(`Failed request: ${request.url()}`));
  return errors;
}
/** Opens the lab as a returning visitor (no welcome dialog), optionally on a given architecture. */
async function open(page: Page, arch = 'ann') {
  await page.addInitScript(a => { if (!sessionStorage.getItem('seeded')) { localStorage.setItem('neuron-lab-welcomed', '1'); localStorage.setItem('neuron-lab-arch', a); localStorage.setItem('neuron-lab-lang', 'en'); sessionStorage.setItem('seeded', '1'); } }, arch);
  await page.goto('/');
}
/** Labs stay mounted (hidden) after a visit, so scope lookups to the visible one. */
const lab = (page: Page) => page.locator('.lab:not([hidden])');
const stepForward = (page: Page) => page.getByRole('button', { name: 'Step forward', exact: true }).click();
const trainMode = (page: Page) => page.getByRole('group', { name: 'Learning mode' }).getByRole('button', { name: /^Train/ }).click();
async function trainUntil(page: Page, testId: 'epoch' | 'accuracy', minimum: number) {
  await lab(page).getByLabel('Playback speed').selectOption('4');
  await page.getByRole('button', { name: 'Start training', exact: true }).click();
  await expect.poll(async () => parseFloat((await lab(page).getByTestId(testId).textContent())!), { timeout: 40_000 }).toBeGreaterThanOrEqual(minimum);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
}

test('learn the real ANN calculations, undo an exact weight update, and train continuously', async ({ page }) => {
  const errors = trackErrors(page);
  await open(page);
  await expect(page.getByRole('heading', { name: 'Feedforward network (ANN)' })).toBeVisible();
  await expect(lab(page).getByTestId('step-number')).toHaveText('1');
  await expect(page.getByRole('button', { name: 'Step backward', exact: true })).toBeDisabled();
  await page.screenshot({ path: 'test-results/neuron-lab-desktop.png', fullPage: true });
  for (let i = 0; i < 17; i++) await stepForward(page);
  await page.keyboard.press('ArrowRight');
  await expect(lab(page).getByTestId('step-number')).toHaveText('19');
  await expect(page.getByText('Weight update')).toBeVisible();
  const weights = () => page.locator('.wire-hit').evaluateAll(nodes => nodes.map(node => node.getAttribute('aria-label')));
  const before = await weights();
  await stepForward(page);
  const after = await weights();
  expect(after).not.toEqual(before);
  await page.getByRole('button', { name: 'Step backward', exact: true }).click();
  expect(await weights()).toEqual(before);
  await stepForward(page);
  expect(await weights()).toEqual(after);
  await page.screenshot({ path: 'test-results/neuron-lab-weight-update.png', fullPage: true });

  await trainMode(page);
  await stepForward(page);
  await expect(lab(page).getByTestId('epoch')).toHaveText('1.25');
  const epochWeights = await weights();
  await page.getByRole('button', { name: 'Step backward', exact: true }).click();
  expect(await weights()).toEqual(after);
  await stepForward(page);
  expect(await weights()).toEqual(epochWeights);
  await trainUntil(page, 'epoch', 180);
  await expect.poll(async () => parseFloat((await lab(page).getByTestId('loss').textContent())!)).toBeLessThan(0.08);
  await page.getByRole('button', { name: 'Save lab', exact: true }).click();
  const saved = await weights();
  await page.reload();
  await page.getByRole('button', { name: 'Restore saved', exact: true }).click();
  expect(await weights()).toEqual(saved);
  await expect(lab(page).getByTestId('epoch')).toHaveText('0');
  expect(errors).toEqual([]);
});

test('edit the ANN architecture and parameters, use softmax, and explore 2D points', async ({ page }) => {
  const errors = trackErrors(page);
  await open(page);
  await page.getByRole('button', { name: 'Add neuron to hidden layer 1' }).click();
  await expect(page.getByRole('button', { name: 'Layer 1 neuron 5, not calculated yet', exact: true })).toBeVisible();
  await lab(page).getByLabel('Hidden layer 1 activation').selectOption('relu');
  await page.getByRole('button', { name: 'Add hidden layer', exact: true }).click();
  await expect(lab(page).getByLabel('Hidden layer 3 activation')).toBeVisible();
  await page.getByRole('button', { name: 'Remove hidden layer 3', exact: true }).click();
  await lab(page).getByLabel('Output activation', { exact: true }).selectOption('softmax');
  await expect(lab(page).getByLabel('Output neurons', { exact: true })).toHaveValue('2');
  await page.getByRole('button', { name: 'Edit this neuron’s parameters', exact: true }).click();
  await lab(page).getByLabel('Weight from input 1', { exact: true }).fill('0.75');
  await lab(page).getByLabel('Bias', { exact: true }).fill('0.1');
  await page.getByRole('button', { name: 'Apply parameters', exact: true }).click();
  await expect(page.locator('.wire-hit').first()).toHaveAttribute('aria-label', /0.7500/);
  await lab(page).getByLabel('Problem').selectOption('circle');
  await expect(page.getByRole('heading', { name: 'Circle', exact: true })).toBeVisible();
  await expect(lab(page).getByLabel('Learning rate')).toHaveValue('0.1');
  await expect(page.locator('.scatter circle')).toHaveCount(80);
  await page.getByRole('button', { name: /^Inspect point 1, class/ }).click();
  await expect(lab(page).getByTestId('step-number')).toHaveText('1');
  await page.getByRole('button', { name: 'Reset to recommended settings', exact: true }).click();
  await trainMode(page);
  await trainUntil(page, 'epoch', 150);
  await expect.poll(async () => parseFloat((await lab(page).getByTestId('accuracy').textContent())!)).toBeGreaterThanOrEqual(85);
  await page.screenshot({ path: 'test-results/neuron-lab-circle.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('CNN: convolve, pool, backpropagate into filters, undo, train, and draw an image', async ({ page }) => {
  const errors = trackErrors(page);
  await open(page);
  await page.getByRole('radio', { name: /CNN/ }).click();
  await expect(page.getByRole('heading', { name: 'Line direction' })).toBeVisible();
  // tanh keeps every filter's gradient nonzero (a ReLU filter can be inactive for a whole image).
  await lab(page).getByRole('combobox', { name: /Activation/ }).selectOption('tanh');
  await expect(page.getByRole('button', { name: 'Feature map 1, row 1, column 1, not calculated yet' })).toBeVisible();
  const filterWeights = () => page.locator('.kernel-grid .cell').evaluateAll(cells => cells.map(c => c.getAttribute('title')));
  const initial = await filterWeights();
  await stepForward(page);
  await expect(page.getByRole('button', { name: /^Feature map 1, row 1, column 1: / })).toBeVisible();
  await page.getByRole('button', { name: /^Feature map 1, row 2, column 3: / }).click();
  await expect(page.getByText('Feature map cell')).toBeVisible();
  await expect(page.locator('.pixel-grid .in-field')).toHaveCount(9);
  for (let i = 0; i < 15; i++) await stepForward(page);
  await expect(lab(page).getByTestId('step-number')).toHaveText('17');
  await expect(page.getByText('Update · filter 2')).toBeVisible();
  const updated = await filterWeights();
  expect(updated.slice(0, 9)).not.toEqual(initial.slice(0, 9));
  expect(updated.slice(9)).not.toEqual(initial.slice(9));
  await page.screenshot({ path: 'test-results/neuron-lab-cnn.png', fullPage: true });
  await page.getByRole('button', { name: 'Step backward', exact: true }).click();
  expect((await filterWeights()).slice(9)).toEqual(initial.slice(9));
  await trainMode(page);
  await trainUntil(page, 'epoch', 40);
  await expect.poll(async () => parseFloat((await lab(page).getByTestId('accuracy').textContent())!)).toBeGreaterThanOrEqual(90);
  await page.getByRole('tab', { name: 'Draw' }).click();
  for (const column of [1, 2, 3, 4, 5, 6]) await page.getByRole('button', { name: `Toggle pixel row 3, column ${column}` }).click();
  await expect(page.locator('.draw-result .is-top .class-name')).toHaveText('Horizontal');
  await lab(page).getByLabel('Problem').selectOption('shapes');
  await expect(page.getByRole('heading', { name: 'Shapes', exact: true })).toBeVisible();
  await expect(lab(page).getByTestId('epoch')).toHaveText('0');
  expect(errors).toEqual([]);
});

test('RNN: learn a custom word through time and write it back', async ({ page }) => {
  const errors = trackErrors(page);
  await open(page);
  await page.getByRole('radio', { name: /RNN/ }).click();
  await expect(page.getByRole('heading', { name: 'Next letter' })).toBeVisible();
  await lab(page).getByLabel('Word to learn').fill('ab');
  await expect(page.getByText('Use 3–12 letters')).toBeVisible();
  await lab(page).getByLabel('Word to learn').fill('banana');
  await page.getByRole('button', { name: 'Apply', exact: true }).click();
  await expect(page.getByText('“banana”', { exact: true })).toBeVisible();
  await expect(lab(page).getByTestId('step-number')).toHaveText('1');
  await expect(page.locator('.time-column')).toHaveCount(5);
  for (let i = 0; i < 3; i++) await stepForward(page);
  await page.getByRole('button', { name: 'Time step 2, input a' }).click();
  await expect(page.getByText('Hidden state update')).toBeVisible();
  for (let i = 0; i < 10; i++) await stepForward(page);
  await expect(page.getByText('Update · recurrent weights Wh')).toBeVisible();
  await trainMode(page);
  await trainUntil(page, 'accuracy', 100);
  await expect(lab(page).getByTestId('generated')).toHaveText('banana');
  await page.screenshot({ path: 'test-results/neuron-lab-rnn.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('LSTM in Bosnian: switch language, remember the first bit, and keep preferences after reload', async ({ page }) => {
  const errors = trackErrors(page);
  await open(page);
  await page.getByRole('group', { name: 'Language' }).getByRole('button', { name: 'BS' }).click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'bs');
  await expect(page.getByRole('heading', { name: 'Unaprijedna mreža (ANN)' })).toBeVisible();
  await page.getByRole('radio', { name: /LSTM/ }).click();
  await lab(page).getByLabel('Problem').selectOption('memory');
  await expect(page.getByRole('heading', { name: 'Prvi bit' })).toBeVisible();
  // Move focus off the controls (arrow keys there navigate the control) so the lesson shortcuts apply.
  await lab(page).locator('h1').click();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await expect(page.getByText('LSTM kapije')).toBeVisible();
  await page.keyboard.press('ArrowLeft');
  await expect(lab(page).getByTestId('step-number')).toHaveText('2');
  await page.getByRole('group', { name: 'Način učenja' }).getByRole('button', { name: /^Treniraj/ }).click();
  await lab(page).getByLabel('Brzina reprodukcije').selectOption('4');
  await page.getByRole('button', { name: 'Pokreni treniranje', exact: true }).click();
  await expect.poll(async () => parseFloat((await lab(page).getByTestId('accuracy').textContent())!), { timeout: 40_000 }).toBe(100);
  await page.getByRole('button', { name: 'Pauza', exact: true }).click();
  await page.screenshot({ path: 'test-results/neuron-lab-lstm-bs.png', fullPage: true });
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Sljedeće slovo' })).toBeVisible();
  await expect(page.getByRole('radio', { name: /LSTM/ })).toHaveAttribute('aria-checked', 'true');
  expect(errors).toEqual([]);
});

test('first visit: welcome dialog, guided tour, guide, and text size', async ({ page }) => {
  const errors = trackErrors(page);
  await page.addInitScript(() => { if (!sessionStorage.getItem('seeded')) { localStorage.clear(); sessionStorage.setItem('seeded', '1'); } });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Neural network visualizer' })).toBeVisible();
  await page.screenshot({ path: 'test-results/neuron-lab-welcome.png' });
  await page.getByRole('button', { name: 'Take the tour (1 min)' }).click();
  await expect(page.getByRole('heading', { name: 'Network type', exact: true })).toBeVisible();
  for (let i = 0; i < 8; i++) await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByText('9 of 9')).toBeVisible();
  await page.getByRole('button', { name: 'Done' }).click();
  await expect(page.locator('.tour-layer')).toHaveCount(0);
  await page.reload();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.getByRole('button', { name: 'Guide', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'How to use Neuron Lab' })).toBeVisible();
  await page.getByRole('dialog').getByRole('button', { name: /Convolutional network/ }).click();
  await expect(page.getByRole('heading', { name: 'Line direction' })).toBeVisible();
  await page.getByRole('button', { name: 'Larger text' }).click();
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).fontSize)).toBe('18px');
  await page.getByRole('button', { name: 'Take the tour' }).click();
  await page.keyboard.press('Escape');
  await expect(page.locator('.tour-layer')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('mobile layout stays within the viewport for every network', async ({ page }) => {
  const errors = trackErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  for (const arch of ['ANN', 'CNN', 'RNN', 'LSTM']) {
    await page.getByRole('radio', { name: new RegExp(arch) }).click();
    await stepForward(page);
    await expect(lab(page).getByTestId('step-number')).toHaveText('2');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), arch).toBe(true);
    await expect(page.getByRole('button', { name: 'Larger text' })).toBeVisible();
    await page.screenshot({ path: `test-results/neuron-lab-mobile-${arch.toLowerCase()}.png`, fullPage: true });
  }
  expect(errors).toEqual([]);
});

test('keyboard, labels, notifications, and localized numbers behave consistently across labs', async ({ page }) => {
  const errors = trackErrors(page);
  await open(page);
  // Arrow keys move through the network picker without advancing the lesson.
  await page.getByRole('radio', { name: /ANN/ }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('heading', { name: 'Line direction' })).toBeVisible();
  await expect(page.getByRole('radio', { name: /CNN/ })).toBeFocused();
  await expect(lab(page).getByTestId('step-number')).toHaveText('1');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('radio', { name: /LSTM/ })).toHaveAttribute('aria-checked', 'true');
  // Every mounted lab has unique element ids, so labels reach their own controls.
  expect(await page.evaluate(() => { const ids = [...document.querySelectorAll('[id]')].map(e => e.id); return ids.filter((id, i) => ids.indexOf(id) !== i); })).toEqual([]);
  await lab(page).locator('label', { hasText: 'Learning rate' }).click();
  await expect(lab(page).getByRole('slider', { name: 'Learning rate' })).toBeFocused();
  await page.getByRole('button', { name: 'Remove a hidden unit' }).click();
  await page.getByRole('button', { name: 'Remove a hidden unit' }).click();
  await page.getByRole('button', { name: 'Remove a hidden unit' }).click();
  await expect(lab(page).locator('.network-heading p')).toContainText('1 hidden unit·');
  // CNN tabs respond to arrow keys too.
  await page.getByRole('radio', { name: /CNN/ }).click();
  await page.getByRole('tab', { name: 'Gallery' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Draw' })).toHaveAttribute('aria-selected', 'true');
  // Repeating the same notification restarts its timer.
  await page.getByRole('tab', { name: 'Gallery' }).click();
  await page.getByRole('button', { name: /^Inspect image 2,/ }).click();
  await page.waitForTimeout(3000);
  await page.getByRole('button', { name: /^Inspect image 3,/ }).click();
  await page.waitForTimeout(2500);
  await expect(page.locator('.toast')).toBeVisible();
  // Bosnian uses a decimal comma in metrics and prose.
  await page.getByRole('group', { name: 'Language' }).getByRole('button', { name: 'BS' }).click();
  await expect(lab(page).getByTestId('loss')).toHaveText(/^\d+,\d{4}$/);
  // Tab stays inside the tour card.
  await page.getByRole('button', { name: 'Kreni u obilazak' }).click();
  for (let i = 0; i < 6; i++) await page.keyboard.press('Tab');
  expect(await page.evaluate(() => Boolean(document.activeElement?.closest('.tour-card')))).toBe(true);
  await page.keyboard.press('Escape');
  expect(errors).toEqual([]);
});
