// Release-mode clap parsing can overflow the default 1 MiB Windows main-thread stack; give the CLIs 8 MiB.
fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if os == "windows" {
        let arg = if env == "msvc" { "/STACK:8388608" } else { "-Wl,--stack,8388608" };
        println!("cargo:rustc-link-arg-bins={arg}");
    }
}
