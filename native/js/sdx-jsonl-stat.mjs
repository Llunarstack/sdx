#!/usr/bin/env node
/**
 * Stream JSONL manifest stats (Node 18+).
 * Usage: node sdx-jsonl-stat.mjs path/to/manifest.jsonl
 */
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";

function keys(obj) {
  const image =
    obj.image_path ?? obj.path ?? obj.image ?? "";
  const cap = (obj.caption ?? obj.text ?? "").trim();
  return { image: String(image).trim(), cap };
}

const file = process.argv[2];
if (!file) {
  console.error("usage: node sdx-jsonl-stat.mjs <manifest.jsonl>");
  process.exit(2);
}

let total = 0;
let emptySkip = 0;
let parseErr = 0;
let missing = 0;
let ok = 0;
const capLens = [];

const rl = createInterface({
  input: createReadStream(file, { encoding: "utf8" }),
  crlfDelay: Infinity,
});

for await (const line of rl) {
  total += 1;
  const t = line.trim();
  if (!t) {
    emptySkip += 1;
    continue;
  }
  let obj;
  try {
    obj = JSON.parse(t);
  } catch {
    parseErr += 1;
    continue;
  }
  const { image, cap } = keys(obj);
  if (image && cap) {
    ok += 1;
    capLens.push(cap.length);
  } else {
    missing += 1;
  }
}

capLens.sort((a, b) => a - b);
function pct(q) {
  if (!capLens.length) return 0;
  const idx = Math.round((capLens.length - 1) * q);
  return capLens[Math.min(idx, capLens.length - 1)];
}

console.log(`file: ${file}`);
console.log(`lines_total: ${total}`);
console.log(`empty_skipped: ${emptySkip}`);
console.log(`json_parse_errors: ${parseErr}`);
console.log(`rows_missing_image_or_caption: ${missing}`);
console.log(`rows_ok: ${ok}`);
if (capLens.length) {
  console.log(
    `caption_len_chars: min=${capLens[0]} p50=${pct(0.5)} p90=${pct(0.9)} p99=${pct(0.99)} max=${capLens[capLens.length - 1]}`
  );
}
