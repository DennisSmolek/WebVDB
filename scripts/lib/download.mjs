// Shared download helper for the fixture-fetch scripts (Node 20+, no deps).
import { createWriteStream } from "node:fs";
import { mkdir, rename, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export async function exists(file) {
  try {
    await stat(file);
    return true;
  } catch {
    return false;
  }
}

/**
 * Download `url` to `dest`, atomically (via a .part file) and idempotently
 * (skips if `dest` already exists). Prints coarse progress to stderr.
 */
export async function download(url, dest, { headers = {} } = {}) {
  if (await exists(dest)) {
    console.error(`  ✓ already present, skipping: ${dest}`);
    return dest;
  }
  await mkdir(path.dirname(dest), { recursive: true });

  console.error(`  ↓ ${url}`);
  const res = await fetch(url, {
    redirect: "follow",
    headers: { "user-agent": "webvdb-fixture-fetch/0.1", ...headers },
  });
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
  }

  const total = Number(res.headers.get("content-length")) || 0;
  let seen = 0;
  let lastPct = -10;
  const progress = new TransformStream({
    transform(chunk, controller) {
      seen += chunk.byteLength;
      if (total) {
        const pct = Math.floor((seen / total) * 100);
        if (pct >= lastPct + 10) {
          lastPct = pct;
          console.error(`    … ${pct}% of ${(total / 1e6).toFixed(0)} MB`);
        }
      }
      controller.enqueue(chunk);
    },
  });

  const part = `${dest}.part`;
  await pipeline(Readable.fromWeb(res.body.pipeThrough(progress)), createWriteStream(part));
  await rename(part, dest);
  console.error(`  ✓ saved ${dest} (${(seen / 1e6).toFixed(1)} MB)`);
  return dest;
}

/** Fetch a page and return all absolute href/src URLs matching `pattern`. */
export async function scrapeLinks(pageUrl, pattern) {
  const res = await fetch(pageUrl, {
    redirect: "follow",
    headers: { "user-agent": "webvdb-fixture-fetch/0.1" },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${pageUrl}`);
  const html = await res.text();
  const urls = new Set();
  for (const m of html.matchAll(/(?:href|src)=["']([^"']+)["']/g)) {
    const raw = m[1];
    let abs;
    try {
      abs = new URL(raw, pageUrl).toString();
    } catch {
      continue;
    }
    if (pattern.test(abs)) urls.add(abs);
  }
  return [...urls];
}
