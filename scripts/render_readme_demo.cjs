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
    const url = pathToFileURL(path.join(target, 'index.html')).href;
    await page.goto(url + '?capture=1');
    await page.waitForFunction(() => Boolean(window.demo));
    assert.equal(await page.locator('button, input, select, video[controls], audio[controls]').count(), 0);
    assert.ok(!(await page.locator('body').textContent()).toLowerCase().includes('in motion'));
    await page.evaluate(() => window.demo.render(window.demo.duration));
    assert.equal(await page.locator('#selection-check').getAttribute('opacity'), '1.0000');
    assert.ok((await page.locator('#embedding-summary').textContent()).includes('0.6615 to 0.7007'));
    assert.ok((await page.locator('#quality-summary').textContent()).includes('71.04% to 74.63%'));
    assert.ok((await page.locator('#quality-summary').textContent()).includes('$2.07 to $1.55'));
    assert.ok(await page.locator('.comparison img').evaluate(img => img.complete && img.naturalWidth > 0));
    await page.locator('#demo').screenshot({path: path.join(target, 'routing-poster.png')});

    const animation = await page.evaluate(() => window.demo.data.animation);
    const count = Math.round(animation.duration_seconds * animation.fps);
    for (let index = 0; index < count; index++) {
      await page.evaluate(time => window.demo.render(time), index / animation.fps);
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

    // Exercise actual automatic playback and a complete loop using the browser clock.
    await page.setViewportSize({width: 1200, height: 950});
    await page.clock.install({time: new Date('2026-01-01T00:00:00Z')});
    await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
    await page.goto(url);
    await page.clock.runFor(300);
    const start = await page.evaluate(() => window.demo.state);
    assert.ok(start.visible && start.running, 'Autoplay must start when the animation is visible');
    await page.clock.runFor(2500);
    const advanced = await page.evaluate(() => window.demo.state);
    assert.ok(advanced.time > start.time + 2, 'Animation must advance without input');
    await page.clock.runFor(6200);
    assert.equal(await page.locator('#selection-check').getAttribute('opacity'), '1.0000');
    await page.clock.runFor(3000);
    const repeated = await page.evaluate(() => window.demo.state);
    assert.ok(repeated.cycles >= 1 && repeated.time < 3, 'Completed sequence must repeat automatically');

    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.reload();
    assert.equal(await page.evaluate(() => window.demo.state.running), false);
    assert.equal(await page.evaluate(() => window.demo.state.time), animation.duration_seconds);
    assert.deepEqual(errors, []);
    const report = {passed: true, captured_frames: count, duration_ms: 1000 * animation.duration_seconds,
      browser: await browser.version(), playwright: require((process.env.PLAYWRIGHT_MODULE || 'playwright') + '/package.json').version,
      layouts, checks: ['no playback controls', 'no motion heading', 'numbers in sentences',
        'original model-selection sequence', 'plot loaded', 'no horizontal overflow',
        'autoplay on view', 'automatic advancement', 'automatic repeat after completion',
        'reduced motion', 'no page errors']};
    fs.writeFileSync(path.join(cache, 'browser-check.json'), JSON.stringify(report, null, 2) + '\n');
    fs.writeFileSync(path.join(target, 'browser-check.json'), JSON.stringify(report, null, 2) + '\n');
    process.stdout.write(JSON.stringify(report, null, 2) + '\n');
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
