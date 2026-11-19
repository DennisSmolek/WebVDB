import type { Page } from "@playwright/test";

/**
 * Collect a page's console errors and uncaught exceptions for a "no errors" assertion.
 *
 * A 404 for a URL under `/fixtures/` is not collected: demo pages probe optional, git-ignored fixtures (gallery
 * assets, WDAS, files copied from a mjurczyk/openvdb clone), fall back without them and report what's missing in
 * their state. The browser still logs each probe as "Failed to load resource", which on CI (no optional fixtures)
 * failed every spec that asserted an empty error list. Any other failed load, including a 404 elsewhere, is kept.
 */
export function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const text = msg.text();
    if (/Failed to load resource: .*\b404\b/.test(text) && isFixtureUrl(msg.location().url)) return;
    errors.push(text);
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

function isFixtureUrl(url: string): boolean {
  try {
    return new URL(url).pathname.includes("/fixtures/");
  } catch {
    return false;
  }
}
