// Shared download helper for the fixture-fetch scripts (Node 20+, no deps).
import { createWriteStream } from "node:fs";
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, stat } from "node:fs/promises";
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

export async function verifyFile(file, { expectedBytes, expectedSha256 } = {}) {
  const info = await stat(file);
  const actualBytes = info.size;
  const actualSha256 = expectedSha256
    ? createHash("sha256").update(await readFile(file)).digest("hex")
    : undefined;
  return {
    ok:
      (expectedBytes === undefined || actualBytes === expectedBytes) &&
      (expectedSha256 === undefined || actualSha256 === expectedSha256),
    actualBytes,
    ...(actualSha256 ? { actualSha256 } : {}),
  };
}

/**
 * Download `url` to `dest`, atomically (via a .part file) and idempotently
 * (skips if `dest` already exists). Prints coarse progress to stderr.
 */
export async function download(
  url,
  dest,
  { headers = {}, expectedBytes, expectedSha256 } = {},
) {
  if (await exists(dest)) {
    const integrity = await verifyFile(dest, { expectedBytes, expectedSha256 });
    if (integrity.ok) {
      console.error(`  ✓ already present, verified: ${dest}`);
      return dest;
    }
    console.error(`  ! replacing invalid existing file: ${dest}`);
    await rm(dest, { force: true });
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
  await rm(part, { force: true });
  try {
    await pipeline(Readable.fromWeb(res.body.pipeThrough(progress)), createWriteStream(part));
    const integrity = await verifyFile(part, { expectedBytes, expectedSha256 });
    if (!integrity.ok)
      throw new Error(
        `integrity mismatch for ${url}: got ${integrity.actualBytes} bytes` +
          `${integrity.actualSha256 ? ` / sha256 ${integrity.actualSha256}` : ""}`,
      );
    await rename(part, dest);
  } catch (error) {
    await rm(part, { force: true });
    throw error;
  }
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
