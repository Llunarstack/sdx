//! SDX manifest JSONL tools (aligns with Python dataset key conventions).

use clap::{Parser, Subcommand};
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::collections::{HashMap, HashSet};

#[derive(Parser, Debug)]
#[command(name = "sdx-jsonl-tools", version, about = "SDX JSONL manifest validate / stats")]
struct Cli {
    #[command(subcommand)]
    cmd: SubCmd,
}

#[derive(Subcommand, Debug)]
enum SubCmd {
    /// Print line counts, JSON errors, missing fields, caption length summary
    Stats {
        #[arg(value_name = "FILE")]
        path: PathBuf,
    },
    /// Exit non-zero if any line is invalid JSON or missing required keys (after filters).
    /// Use max_caption_len=0 to disable the upper bound.
    Validate {
        #[arg(value_name = "FILE")]
        path: PathBuf,
        #[arg(long, default_value_t = 0)]
        min_caption_len: usize,
        #[arg(long, default_value_t = 0)]
        max_caption_len: usize,
    },
    /// Lint captions for prompt adherence issues: empty captions, pos/neg overlap, token length.
    /// Exits non-zero if `--fail-on-overlap` and any row has pos/neg token overlap.
    PromptLint {
        #[arg(value_name = "FILE")]
        path: PathBuf,
        #[arg(long, default_value_t = 0)]
        min_caption_len_chars: usize,
        #[arg(long, default_value_t = 0)]
        max_caption_tokens: usize,
        #[arg(long, default_value_t = 10)]
        top_overlap_tokens: usize,
        #[arg(long, default_value_t = false)]
        fail_on_overlap: bool,
    },
    /// Print one image path per line (`image_path` / `path` / `image`) for pipeline / shell use.
    ImagePaths {
        #[arg(value_name = "FILE")]
        path: PathBuf,
        /// Emit a path for every JSON row (duplicates allowed). Default: unique paths only.
        #[arg(long, default_value_t = false)]
        all_rows: bool,
        /// Sort lexicographically before printing (stable for diffs).
        #[arg(long, default_value_t = false)]
        sort: bool,
    },
    /// Print duplicate image paths (count > 1) — useful before dataset dedup.
    DupImagePaths {
        #[arg(value_name = "FILE")]
        path: PathBuf,
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
}

fn image_key(v: &Value) -> Option<String> {
    if let Some(s) = v.get("image_path").and_then(|x| x.as_str()) {
        return Some(s.to_string());
    }
    if let Some(s) = v.get("path").and_then(|x| x.as_str()) {
        return Some(s.to_string());
    }
    v.get("image").and_then(|x| x.as_str()).map(|s| s.to_string())
}

fn caption_key(v: &Value) -> Option<String> {
    if let Some(s) = v.get("caption").and_then(|x| x.as_str()) {
        return Some(s.trim().to_string());
    }
    v.get("text").and_then(|x| x.as_str()).map(|s| s.trim().to_string())
}

fn negative_caption_key(v: &Value) -> Option<String> {
    if let Some(s) = v.get("negative_caption").and_then(|x| x.as_str()) {
        return Some(s.trim().to_string());
    }
    if let Some(s) = v.get("negative_prompt").and_then(|x| x.as_str()) {
        return Some(s.trim().to_string());
    }
    if let Some(s) = v.get("negative_text").and_then(|x| x.as_str()) {
        return Some(s.trim().to_string());
    }
    None
}

fn tokenize_normalized(text: &str) -> Vec<String> {
    // Lightweight tokenizer: keep alnum only, lowercased, treat everything else as delimiter.
    // This is good enough for overlap/conflict detection and token count heuristics.
    let mut cur = String::new();
    let mut out: Vec<String> = Vec::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else {
            if !cur.is_empty() {
                out.push(cur.clone());
                cur.clear();
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn cmd_prompt_lint(
    path: &PathBuf,
    min_caption_len_chars: usize,
    max_caption_tokens: usize,
    top_overlap_tokens: usize,
    fail_on_overlap: bool,
) -> io::Result<i32> {
    let mut total_lines = 0u64;
    let mut parse_err = 0u64;
    let mut empty_caption_rows = 0u64;
    let mut rows_over_max_tokens = 0u64;

    let mut ok_rows = 0u64;
    let mut caption_tokens_sum = 0u64;
    let mut caption_tokens_min = u64::MAX;
    let mut caption_tokens_max = 0u64;

    let mut overlap_rows = 0u64;
    let mut max_overlap_token_count = 0u64;
    let mut overlap_token_counts: HashMap<String, u64> = HashMap::new();

    for line in read_lines(path)? {
        let line = line?;
        total_lines += 1;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(t) {
            Ok(x) => x,
            Err(_) => {
                parse_err += 1;
                continue;
            }
        };

        let cap = match caption_key(&v) {
            Some(c) => c,
            None => continue,
        };
        if cap.is_empty() || (min_caption_len_chars > 0 && cap.len() < min_caption_len_chars) {
            empty_caption_rows += 1;
            continue;
        }

        let neg = negative_caption_key(&v).unwrap_or_else(|| "".to_string());
        // If negative is missing/empty, we still count token stats, but overlap is not checked.
        let tokens_pos = tokenize_normalized(&cap);
        let tokens_neg = tokenize_normalized(&neg);

        let pos_set: HashSet<String> = tokens_pos.into_iter().collect();
        let neg_set: HashSet<String> = tokens_neg.into_iter().collect();

        let caption_tok_len = pos_set.len() as u64;
        ok_rows += 1;
        caption_tokens_sum += caption_tok_len;
        caption_tokens_min = caption_tokens_min.min(caption_tok_len);
        caption_tokens_max = caption_tok_len.max(caption_tokens_max);

        if max_caption_tokens > 0 && caption_tok_len > max_caption_tokens as u64 {
            rows_over_max_tokens += 1;
        }

        if !pos_set.is_empty() && !neg_set.is_empty() {
            let overlap: Vec<&String> = pos_set.intersection(&neg_set).collect();
            if !overlap.is_empty() {
                overlap_rows += 1;
                let n = overlap.len() as u64;
                max_overlap_token_count = max_overlap_token_count.max(n);
                for tok in overlap {
                    if top_overlap_tokens == 0 {
                        continue;
                    }
                    *overlap_token_counts.entry(tok.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    let mut overlap_items: Vec<(String, u64)> = overlap_token_counts.into_iter().collect();
    overlap_items.sort_by(|a, b| b.1.cmp(&a.1));
    if overlap_items.len() > top_overlap_tokens && top_overlap_tokens > 0 {
        overlap_items.truncate(top_overlap_tokens);
    }

    println!("promptlint: file {}", path.display());
    println!("lines_total: {}", total_lines);
    println!("json_parse_errors: {}", parse_err);
    println!("empty_caption_rows: {}", empty_caption_rows);
    println!("rows_over_max_tokens: {}", rows_over_max_tokens);
    println!("rows_ok: {}", ok_rows);
    if ok_rows > 0 {
        let avg = caption_tokens_sum as f64 / ok_rows as f64;
        println!(
            "caption_token_count(set_size): avg={:.2} min={} max={}",
            avg,
            caption_tokens_min,
            caption_tokens_max
        );
    }
    println!("pos_neg_overlap_rows: {}", overlap_rows);
    println!("pos_neg_overlap_max_distinct_tokens: {}", max_overlap_token_count);
    if !overlap_items.is_empty() {
        print!("top_overlap_tokens: ");
        for (i, (tok, cnt)) in overlap_items.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}({})", tok, cnt);
        }
        println!();
    }

    if fail_on_overlap && overlap_rows > 0 {
        return Ok(1);
    }
    Ok(0)
}

fn read_lines(path: &PathBuf) -> io::Result<Box<dyn Iterator<Item = io::Result<String>>>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    Ok(Box::new(reader.lines()))
}

fn cmd_stats(path: &PathBuf) -> io::Result<i32> {
    let mut total_lines = 0u64;
    let mut empty_skipped = 0u64;
    let mut parse_err = 0u64;
    let mut missing = 0u64;
    let mut ok = 0u64;
    let mut cap_lens = Vec::<usize>::new();

    for line in read_lines(path)? {
        let line = line?;
        total_lines += 1;
        let t = line.trim();
        if t.is_empty() {
            empty_skipped += 1;
            continue;
        }
        let v: Value = match serde_json::from_str(t) {
            Ok(x) => x,
            Err(_) => {
                parse_err += 1;
                continue;
            }
        };
        let img = image_key(&v);
        let cap = caption_key(&v);
        match (img, cap) {
            (Some(_), Some(c)) if !c.is_empty() => {
                ok += 1;
                cap_lens.push(c.len());
            }
            _ => missing += 1,
        }
    }

    cap_lens.sort_unstable();
    let p = |q: f64| -> usize {
        if cap_lens.is_empty() {
            return 0;
        }
        let idx = ((cap_lens.len() as f64 - 1.0) * q).round() as usize;
        cap_lens[idx.min(cap_lens.len() - 1)]
    };

    println!("file: {}", path.display());
    println!("lines_total: {total_lines}");
    println!("empty_skipped: {empty_skipped}");
    println!("json_parse_errors: {parse_err}");
    println!("rows_missing_image_or_caption: {missing}");
    println!("rows_ok: {ok}");
    if !cap_lens.is_empty() {
        println!(
            "caption_len_chars: min={} p50={} p90={} p99={} max={}",
            cap_lens[0],
            p(0.5),
            p(0.9),
            p(0.99),
            cap_lens[cap_lens.len() - 1]
        );
    }
    Ok(0)
}

fn cmd_image_paths(path: &PathBuf, all_rows: bool, sort: bool) -> io::Result<i32> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in read_lines(path)? {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(t) {
            Ok(x) => x,
            Err(_) => continue,
        };
        let Some(img) = image_key(&v) else {
            continue;
        };
        if all_rows {
            out.push(img);
        } else if seen.insert(img.clone()) {
            out.push(img);
        }
    }
    if sort {
        out.sort();
    }
    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());
    for p in out {
        writeln!(w, "{p}")?;
    }
    Ok(0)
}

fn cmd_dup_image_paths(path: &PathBuf, top: usize) -> io::Result<i32> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    for line in read_lines(path)? {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(t) {
            Ok(x) => x,
            Err(_) => continue,
        };
        let Some(img) = image_key(&v) else {
            continue;
        };
        *counts.entry(img).or_insert(0) += 1;
    }
    let mut dups: Vec<(String, u64)> = counts.into_iter().filter(|(_, c)| *c > 1).collect();
    dups.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if top > 0 && dups.len() > top {
        dups.truncate(top);
    }
    println!("dup_image_paths: file {}", path.display());
    println!("duplicate_distinct_paths: {}", dups.len());
    for (p, c) in dups {
        println!("{c}\t{p}");
    }
    Ok(0)
}

fn cmd_validate(path: &PathBuf, min_cap: usize, max_cap: usize) -> io::Result<i32> {
    let mut bad = 0u64;
    for line in read_lines(path)? {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(t) {
            Ok(x) => x,
            Err(_) => {
                bad += 1;
                continue;
            }
        };
        let img = image_key(&v);
        let cap = caption_key(&v);
        let ok = match (&img, &cap) {
            (Some(_), Some(c)) if !c.is_empty() => {
                let n = c.len();
                if min_cap > 0 && n < min_cap {
                    false
                } else if max_cap > 0 && n > max_cap {
                    false
                } else {
                    true
                }
            }
            _ => false,
        };
        if !ok {
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!("validate: failed — {bad} bad or filtered rows (non-empty lines only)");
        return Ok(1);
    }
    println!("validate: ok");
    Ok(0)
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.cmd {
        SubCmd::Stats { path } => cmd_stats(&path).unwrap_or_else(|e| {
            eprintln!("stats: {e}");
            2
        }),
        SubCmd::Validate {
            path,
            min_caption_len,
            max_caption_len,
        } => cmd_validate(&path, min_caption_len, max_caption_len).unwrap_or_else(|e| {
            eprintln!("validate: {e}");
            2
        }),
        SubCmd::PromptLint {
            path,
            min_caption_len_chars,
            max_caption_tokens,
            top_overlap_tokens,
            fail_on_overlap,
        } => cmd_prompt_lint(
            &path,
            min_caption_len_chars,
            max_caption_tokens,
            top_overlap_tokens,
            fail_on_overlap,
        )
        .unwrap_or_else(|e| {
            eprintln!("promptlint: {e}");
            2
        }),
        SubCmd::ImagePaths {
            path,
            all_rows,
            sort,
        } => cmd_image_paths(&path, all_rows, sort).unwrap_or_else(|e| {
            eprintln!("image-paths: {e}");
            2
        }),
        SubCmd::DupImagePaths { path, top } => cmd_dup_image_paths(&path, top).unwrap_or_else(|e| {
            eprintln!("dup-image-paths: {e}");
            2
        }),
    };
    std::process::exit(code);
}
