#!/usr/bin/env node
// Optional animation authoring: Playwright 1.62.1 and a Chromium browser.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const target = path.join(root, 'docs/demo');
const cache = path.join(root, '.cache/readme-demo');
const frames = path.join(cache, 'frames');

async function main() {
  fs.mkdirSync(frames, {recursive: true});
  for (const filename of fs.readdirSync(frames)) {
    if (/^frame-\d{4}\.png$/.test(filename)) fs.unlinkSync(path.join(frames, filename));
  }
  const options = {headless: true, args: ['--disable-gpu']};
  if (process.env.DEMO_BROWSER) options.channel = process.env.DEMO_BROWSER;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({viewport: {width: 1200, height: 950}, deviceScaleFactor: 1});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const url = pathToFileURL(path.join(target, 'index.html')).href + '?capture=1';
    await page.goto(url);
    await page.waitForFunction(() => Boolean(window.demo));
    await page.evaluate(() => window.demo.render(window.demo.duration));
    assert.equal(await page.locator('#reward-after').textContent(), '74.63%');
    assert.equal(await page.locator('#cost-after').textContent(), '$1.55');
    assert.equal(await page.locator('#reward-change').textContent(), '+3.59 percentage points');
    assert.equal(await page.locator('#cost-change').textContent(), '25.3% lower cost');
    await page.locator('#demo').screenshot({path: path.join(target, 'routing-poster.png')});

    // Capture exact animation times, independent of wall-clock rendering speed.
    for (let index = 0; index < 128; index++) {
      await page.evaluate(time => window.demo.render(time), index / 10);
      await page.locator('#demo').screenshot({
        path: path.join(frames, `frame-${String(index).padStart(4, '0')}.png`)
      });
    }

    const layouts = [];
    for (const width of [320, 768, 1200]) {
      await page.setViewportSize({width, height: 1000});
      await page.evaluate(() => window.demo.render(window.demo.duration));
      const sizes = await page.evaluate(() => ({width: innerWidth, scroll: document.documentElement.scrollWidth}));
      assert.ok(sizes.scroll <= sizes.width, `Horizontal overflow at ${width}px`);
      const height = await page.evaluate(() => document.documentElement.scrollHeight);
      await page.setViewportSize({width, height});
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.screenshot({path: path.join(cache, `preview-${width}.png`)});
      layouts.push(sizes);
    }
    await page.locator('#play').click();
    await page.waitForFunction(() => Number(document.querySelector('#timeline').value) < 2);
    await page.locator('#play').click();
    assert.equal(await page.locator('#play').textContent(), 'Play');
    await page.locator('#prompt-choice').selectOption({label: 'Solve the equation 3x + 7 = 22.'});
    await page.evaluate(() => { window.demo.playback(false); window.demo.render(2); });
    assert.equal(await page.locator('#prompt-text').textContent(), 'Solve the equation 3x + 7 = 22.');
    await page.locator('#timeline').evaluate(element => {
      element.value = '10.8'; element.dispatchEvent(new Event('input', {bubbles: true}));
    });
    assert.equal(await page.locator('#step-number').textContent(), '4');
    assert.equal(await page.locator('#reward-after').textContent(), '74.63%');
    await page.locator('#replay').click();
    await page.evaluate(() => window.demo.playback(false));
    assert.ok(Number(await page.locator('#timeline').inputValue()) < 1);

    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.reload();
    await page.waitForFunction(() => Boolean(window.demo));
    // A normal (non-capture) page opens at the final, stationary state.
    await page.goto(url.replace('?capture=1', ''));
    await page.waitForFunction(() => Boolean(window.demo));
    assert.equal(await page.locator('#play').textContent(), 'Play');
    assert.equal(await page.locator('#timeline').inputValue(), '12.8');
    assert.deepEqual(errors, []);
    const report = {passed: true, captured_frames: 128, duration_ms: 12800,
      browser: await browser.version(), playwright: require((process.env.PLAYWRIGHT_MODULE || 'playwright') + '/package.json').version,
      layouts, checks: ['final metrics', 'no horizontal overflow', 'play/pause', 'replay', 'prompt selection', 'timeline', 'reduced motion', 'no page errors']};
    fs.writeFileSync(path.join(cache, 'browser-check.json'), JSON.stringify(report, null, 2) + '\n');
    fs.writeFileSync(path.join(target, 'browser-check.json'), JSON.stringify(report, null, 2) + '\n');
    process.stdout.write(JSON.stringify(report, null, 2) + '\n');
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
