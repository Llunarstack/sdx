use clap::{Parser, Subcommand};
use image::RgbImage;
use serde_json::json;
use std::collections::VecDeque;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "sdx-image-metrics")]
#[command(about = "Image metrics and count heuristics for SDX workflows")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Print JSON stats: mean_luma, clip_ratio, laplacian_var.
    Stats {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 2)]
        clip_low: u8,
        #[arg(long, default_value_t = 253)]
        clip_high: u8,
    },
    /// Count connected components on thresholded luma.
    CountBlobs {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 140)]
        threshold: u8,
        #[arg(long, default_value_t = 16)]
        min_area: usize,
        #[arg(long, default_value_t = 0)]
        max_area: usize,
    },
}

#[inline]
fn luma_u8(rgb: [u8; 3]) -> u8 {
    let r = rgb[0] as u32;
    let g = rgb[1] as u32;
    let b = rgb[2] as u32;
    ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8
}

fn as_luma_vec(img: &RgbImage) -> Vec<u8> {
    img.pixels()
        .map(|p| luma_u8([p[0], p[1], p[2]]))
        .collect::<Vec<u8>>()
}

fn laplacian_variance(luma: &[u8], w: usize, h: usize) -> f64 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut n = 0usize;
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            let c = luma[idx] as f64;
            let l = luma[idx - 1] as f64;
            let r = luma[idx + 1] as f64;
            let u = luma[idx - w] as f64;
            let d = luma[idx + w] as f64;
            let lap = -4.0 * c + l + r + u + d;
            sum += lap;
            sumsq += lap * lap;
            n += 1;
        }
    }
    if n == 0 {
        return 0.0;
    }
    let mean = sum / n as f64;
    (sumsq / n as f64 - mean * mean).max(0.0)
}

fn count_components(luma: &[u8], w: usize, h: usize, threshold: u8, min_area: usize, max_area: usize) -> usize {
    let mut vis = vec![false; luma.len()];
    let mut q = VecDeque::<usize>::new();
    let mut count = 0usize;
    let area_min = min_area.max(1);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if vis[idx] || luma[idx] > threshold {
                continue;
            }
            vis[idx] = true;
            q.push_back(idx);
            let mut area = 0usize;
            while let Some(cur) = q.pop_front() {
                area += 1;
                let cy = cur / w;
                let cx = cur - cy * w;
                let nbs = [
                    (cy.wrapping_sub(1), cx, cy > 0),
                    (cy + 1, cx, cy + 1 < h),
                    (cy, cx.wrapping_sub(1), cx > 0),
                    (cy, cx + 1, cx + 1 < w),
                ];
                for (ny, nx, ok) in nbs {
                    if !ok {
                        continue;
                    }
                    let ni = ny * w + nx;
                    if vis[ni] || luma[ni] > threshold {
                        continue;
                    }
                    vis[ni] = true;
                    q.push_back(ni);
                }
            }
            if area >= area_min && (max_area == 0 || area <= max_area) {
                count += 1;
            }
        }
    }
    count
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Stats {
            image,
            clip_low,
            clip_high,
        } => {
            let dyn_img = image::open(&image).map_err(|e| format!("failed to open image {}: {}", image.display(), e))?;
            let rgb = dyn_img.to_rgb8();
            let (w_u32, h_u32) = rgb.dimensions();
            let w = w_u32 as usize;
            let h = h_u32 as usize;
            let luma = as_luma_vec(&rgb);
            let n = luma.len().max(1);
            let sum: u64 = luma.iter().map(|&v| v as u64).sum();
            let clipped = luma
                .iter()
                .filter(|&&v| v <= clip_low || v >= clip_high)
                .count();
            let out = json!({
                "image": image.to_string_lossy(),
                "width": w,
                "height": h,
                "mean_luma": (sum as f64) / (n as f64),
                "clip_ratio": (clipped as f64) / (n as f64),
                "laplacian_var": laplacian_variance(&luma, w, h),
            });
            println!("{}", out);
        }
        Commands::CountBlobs {
            image,
            threshold,
            min_area,
            max_area,
        } => {
            let dyn_img = image::open(&image).map_err(|e| format!("failed to open image {}: {}", image.display(), e))?;
            let rgb = dyn_img.to_rgb8();
            let (w_u32, h_u32) = rgb.dimensions();
            let w = w_u32 as usize;
            let h = h_u32 as usize;
            let luma = as_luma_vec(&rgb);
            let c = count_components(&luma, w, h, threshold, min_area, max_area);
            let out = json!({
                "image": image.to_string_lossy(),
                "width": w,
                "height": h,
                "threshold": threshold,
                "min_area": min_area,
                "max_area": max_area,
                "components": c,
            });
            println!("{}", out);
        }
    }
    Ok(())
}
