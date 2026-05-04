# BLAS Configuration

The Rust extension uses **LP64 CBLAS** (`cblas_sgemv` / `cblas_dgemv`). A compatible BLAS library must be present at build time.

## How the build script selects a BLAS library

`build.rs` tries each source in order and stops at the first one that works:

| Step | Source | When it applies |
|------|--------|-----------------|
| 1 | `BLAS_LIB_DIR` env var | explicit override, CI, wheel builds |
| 2 | `pkg-config` (`openblas` → `cblas` → `blas`) | system/apt/brew/conda |
| 3 | macOS Accelerate framework | any macOS, automatic |
| 4 | numpy LP64 BLAS via Python introspection | conda without `PKG_CONFIG_PATH` configured |
| — | build error with instructions | nothing found |

## Platform-specific setup

### Linux

Install OpenBLAS development headers and the shared library:

```sh
# Debian / Ubuntu
sudo apt install libopenblas-dev

# Fedora / RHEL
sudo dnf install openblas-devel

# Arch Linux
sudo pacman -S openblas
```

Then build normally — `pkg-config` will locate it automatically:

```sh
uv run maturin develop   # or: cargo build
```

### macOS

Apple's **Accelerate** framework (always installed) is used automatically — no
extra packages are required. If you prefer OpenBLAS (e.g. for benchmarking):

```sh
brew install openblas
export PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig"
uv run maturin develop
```

The `export` is needed because Homebrew installs OpenBLAS in a non-standard
prefix that `pkg-config` does not search by default. Add it to your shell
profile (`~/.zshrc`, `~/.bash_profile`) to make it permanent.

### Windows

> **Note:** Windows is not tested by the developer. The instructions below are
> provided on a best-effort basis and may require adjustment for your setup.
> Bug reports and corrections are welcome.

Windows does not ship a system BLAS. The recommended approach is to point
`BLAS_LIB_DIR` at a pre-built OpenBLAS distribution (available from the
[OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases)):

```powershell
$env:BLAS_LIB_DIR = "C:\openblas\lib"
$env:BLAS_LIB    = "openblas"          # matches libopenblas.lib / openblas.dll
uv run maturin develop
```

Alternatively, install OpenBLAS via vcpkg or conda and ensure the library
directory is on `BLAS_LIB_DIR`.

## Manual override

Set these environment variables to bypass auto-detection entirely:

| Variable | Description | Default |
|----------|-------------|---------|
| `BLAS_LIB_DIR` | Directory containing the BLAS library (`.so` / `.dylib` / `.lib`) | — |
| `BLAS_LIB` | Library name passed to the linker (without `lib` prefix or extension) | `openblas` |

Examples:

```sh
# Use a custom OpenBLAS installation
export BLAS_LIB_DIR=/opt/myopenblas/lib
uv run maturin develop

# Use a different library (e.g. reference BLAS named libblas.so)
export BLAS_LIB_DIR=/usr/lib/x86_64-linux-gnu
export BLAS_LIB=blas
uv run maturin develop

# One-shot without exporting
BLAS_LIB_DIR=/opt/myopenblas/lib uv run maturin develop
```
