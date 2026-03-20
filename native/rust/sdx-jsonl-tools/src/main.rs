//! SDX manifest JSONL tools (aligns with Python dataset key conventions).

use clap::{Parser, Subcommand};
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

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
    };
    std::process::exit(code);
}
