//! Small shared helpers.

/// Parse an env-var override, falling back to `default`. The FL_* knobs
/// (unit counts, combat scale, camera pose, ...) all go through here.
pub fn env_or<T: std::str::FromStr + Copy>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
