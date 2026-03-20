#!/usr/bin/env node

/**
 * SDX prompt linting for JSONL manifests.
 * Checks:
 * - empty/too-short captions
 * - token length heuristic (distinct-token set size)
 * - positive/negative token overlap (case-insensitive, alnum-only tokenization)
 *
 * Usage:
 *   node native/js/sdx-promptlint.mjs manifest.jsonl [--min-caption-len-chars N] [--max-caption-tokens N]
 *        [--top-overlap-tokens N] [--fail-on-overlap]
 */

import fs from "node:fs";

function tokenizeNormalized(text) {
  // Lightweight tokenizer: keep alnum, lowercase, treat everything else as delimiter.
  // This matches the intent of the rust version closely.
  const out = [];
  let cur = "";
  for (const ch of String(text ?? "")) {
    const code = ch.codePointAt(0);
    const isAlphaNum =
      (code >= 48 && code <= 57) || // 0-9
      (code >= 65 && code <= 90) || // A-Z
      (code >= 97 && code <= 122); // a-z
    if (isAlphaNum) {
      cur += ch.toLowerCase();
    } else if (cur) {
      out.push(cur);
      cur = "";
    }
  }
  if (cur) out.push(cur);
  return out;
}

function parseArgs(argv) {
  const args = {
    minCaptionLenChars: 0,
    maxCaptionTokens: 0,
    topOverlapTokens: 10,
    failOnOverlap: false,
  };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--min-caption-len-chars") args.minCaptionLenChars = Number(argv[++i] ?? 0);
    else if (a === "--max-caption-tokens") args.maxCaptionTokens = Number(argv[++i] ?? 0);
    else if (a === "--top-overlap-tokens") args.topOverlapTokens = Number(argv[++i] ?? 10);
    else if (a === "--fail-on-overlap") args.failOnOverlap = true;
    else if (a.startsWith("-")) throw new Error(`Unknown flag: ${a}`);
    else positional.push(a);
  }
  if (positional.length < 1) throw new Error("Missing FILE argument");
  return { file: positional[0], ...args };
}

function main() {
  const { file, minCaptionLenChars, maxCaptionTokens, topOverlapTokens, failOnOverlap } = parseArgs(
    process.argv.slice(2),
  );

  const raw = fs.readFileSync(file, { encoding: "utf-8" });
  const lines = raw.split(/\r?\n/);

  let totalLines = 0;
  let parseErrors = 0;
  let emptyCaptionRows = 0;
  let jsonMissingCaptionOrNeg = 0;
  let rowsOverMaxTokens = 0;

  let okRows = 0;
  let captionTokenSetSum = 0;
  let captionTokenSetMin = Number.POSITIVE_INFINITY;
  let captionTokenSetMax = 0;

  let overlapRows = 0;
  let maxOverlapDistinctTokens = 0;
  const overlapTokenCounts = new Map(); // token -> count across overlap occurrences

  for (const line of lines) {
    totalLines += 1;
    const t = String(line).trim();
    if (!t) continue;

    let obj;
    try {
      obj = JSON.parse(t);
    } catch {
      parseErrors += 1;
      continue;
    }

    const caption =
      (typeof obj.caption === "string" ? obj.caption : null) ??
      (typeof obj.text === "string" ? obj.text : null) ??
      "";
    const neg =
      (typeof obj.negative_caption === "string" ? obj.negative_caption : null) ??
      (typeof obj.negative_prompt === "string" ? obj.negative_prompt : null) ??
      (typeof obj.negative_text === "string" ? obj.negative_text : null) ??
      "";

    if (!caption || (minCaptionLenChars > 0 && caption.length < minCaptionLenChars)) {
      emptyCaptionRows += 1;
      continue;
    }

    if (!("caption" in obj) && !("text" in obj)) {
      // keep behavior consistent with rust: missing caption keys are treated as invalid rows
      jsonMissingCaptionOrNeg += 1;
    }

    const posTokens = tokenizeNormalized(caption);
    const negTokens = tokenizeNormalized(neg);

    const posSet = new Set(posTokens);
    const negSet = new Set(negTokens);

    const posTokLen = posSet.size;
    okRows += 1;
    captionTokenSetSum += posTokLen;
    captionTokenSetMin = Math.min(captionTokenSetMin, posTokLen);
    captionTokenSetMax = Math.max(captionTokenSetMax, posTokLen);

    if (maxCaptionTokens > 0 && posTokLen > maxCaptionTokens) {
      rowsOverMaxTokens += 1;
    }

    if (posSet.size > 0 && negSet.size > 0) {
      let overlap = 0;
      for (const tok of posSet) {
        if (negSet.has(tok)) {
          overlap += 1;
          overlapTokenCounts.set(tok, (overlapTokenCounts.get(tok) ?? 0) + 1);
        }
      }
      if (overlap > 0) {
        overlapRows += 1;
        maxOverlapDistinctTokens = Math.max(maxOverlapDistinctTokens, overlap);
      }
    }
  }

  const items = Array.from(overlapTokenCounts.entries()).sort((a, b) => b[1] - a[1]);
  const top = topOverlapTokens > 0 ? items.slice(0, topOverlapTokens) : items;

  console.log(`promptlint: file ${file}`);
  console.log(`lines_total: ${totalLines}`);
  console.log(`json_parse_errors: ${parseErrors}`);
  console.log(`empty_caption_rows: ${emptyCaptionRows}`);
  console.log(`rows_over_max_tokens: ${rowsOverMaxTokens}`);
  console.log(`rows_ok: ${okRows}`);
  if (okRows > 0) {
    const avg = captionTokenSetSum / okRows;
    console.log(`caption_token_count(set_size): avg=${avg.toFixed(2)} min=${captionTokenSetMin} max=${captionTokenSetMax}`);
  }
  console.log(`pos_neg_overlap_rows: ${overlapRows}`);
  console.log(`pos_neg_overlap_max_distinct_tokens: ${maxOverlapDistinctTokens}`);
  if (top.length > 0) {
    console.log(
      `top_overlap_tokens: ${top.map(([tok, cnt]) => `${tok}(${cnt})`).join(", ")}`,
    );
  }

  if (failOnOverlap && overlapRows > 0) process.exit(1);
}

main();

