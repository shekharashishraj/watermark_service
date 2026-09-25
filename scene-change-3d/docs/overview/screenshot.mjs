import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';
const file = 'file://' + process.cwd() + '/scene_change_3d.html';
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' }).catch(async () => chromium.launch());
for (const scheme of ['light', 'dark']) {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: scheme });
    const errors = [];
    page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
    page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
    await page.goto(file, { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(800);
    const el = await page.$('#robust');
    await el.screenshot({ path: `robust_${scheme}_${width}.png` });
    // hover a chart to test the tooltip
    const svg = await page.$('#robust-charts svg'); await svg.scrollIntoViewIfNeeded();
    const box = await svg.boundingBox();
    await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.4);
    await page.waitForTimeout(200);
    const tip = await page.$eval('#tip', (t) => ({ on: t.classList.contains('on'), html: t.innerHTML.slice(0, 300) }));
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    console.log(scheme, width, 'errors:', errors.filter((e) => !/three|OrbitControls|cdnjs|jsdelivr|fonts/i.test(e)).slice(0, 5), 'tip:', tip.on, 'hscroll:', overflow);
    if (scheme === 'light' && width === 1280) console.log(tip.html);
    await page.close();
  }
}
await browser.close();
