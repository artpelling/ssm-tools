use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BLAS_LIB");

    // 1. Explicit override — highest priority, used in CI / wheel builds.
    if let Ok(dir) = env::var("BLAS_LIB_DIR") {
        let lib = env::var("BLAS_LIB").unwrap_or_else(|_| "openblas".into());
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib={lib}");
        return;
    }

    // 2. pkg-config — handles system OpenBLAS (apt/brew) and conda environments.
    for name in ["openblas", "cblas", "blas"] {
        if pkg_config::probe_library(name).is_ok() {
            return;
        }
    }

    // 3. macOS Accelerate — always available on macOS, provides full CBLAS.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    // 4. numpy LP64 BLAS via Python introspection — useful when pkg-config
    //    is not configured for conda/system numpy (e.g. PKG_CONFIG_PATH unset).
    //    maturin sets PYTHON_SYS_EXECUTABLE; fall back to PYO3_PYTHON or python3.
    if let Some((dir, lib)) = find_numpy_lp64_blas() {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib={lib}");
        return;
    }

    panic!(
        "\n\nNo BLAS library found. Options:\
        \n  - Linux:  apt install libopenblas-dev   (or dnf/pacman equivalent)\
        \n  - macOS:  brew install openblas  then  export PKG_CONFIG_PATH=$(brew --prefix openblas)/lib/pkgconfig\
        \n  - conda:  conda install openblas\
        \n  - Manual: set BLAS_LIB_DIR to the directory containing your BLAS library\
        \n            and optionally BLAS_LIB to the library name (default: openblas)\n"
    );
}

/// Query the active Python interpreter for numpy's BLAS configuration.
/// Returns `None` if numpy is absent, uses ILP64 (USE64BITINT / ilp64 name),
/// or does not expose a usable lib directory.
fn find_numpy_lp64_blas() -> Option<(String, String)> {
    let python = env::var("PYTHON_SYS_EXECUTABLE")
        .or_else(|_| env::var("PYO3_PYTHON"))
        .unwrap_or_else(|_| "python3".into());

    // numpy >= 1.25: show_config(mode='dicts') exposes structured BLAS metadata.
    // ILP64 guard: skip if the openblas config string contains USE64BITINT or
    // the BLAS name contains 'ilp64' (e.g. openblas-ilp64, scipy-openblas64).
    let script = r#"
import sys
try:
    import numpy as np
    cfg = np.show_config(mode='dicts')
    blas = cfg['Build Dependencies']['blas']
    name = blas.get('name', '').lower()
    openblas_cfg = blas.get('openblas configuration', '')
    if 'ilp64' in name or 'USE64BITINT' in openblas_cfg:
        sys.exit(1)
    lib_dir = blas.get('lib directory', '')
    libs = blas.get('link libraries', [])
    if blas.get('found') and lib_dir and libs:
        print(lib_dir)
        print(libs[0])
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
"#;

    let out = Command::new(&python).args(["-c", script]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut lines = stdout.trim().lines();
    Some((lines.next()?.to_string(), lines.next()?.to_string()))
}
