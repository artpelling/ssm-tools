# BLAS Configuration

The Rust extension uses **LP64 CBLAS** (`cblas_sgemv` / `cblas_dgemv`) for the
inner solver loop. A compatible BLAS library must be present at build time.

> **LP64 vs ILP64.** Standard BLAS uses 32-bit integers for matrix dimensions
> (LP64). Some distributions ship an ILP64 variant with 64-bit integers and
> different symbol names — most notably `scipy-openblas64`, which is the default
> BLAS bundled with numpy ≥ 2.0 on PyPI. ILP64 is **not compatible** and is
> detected and skipped automatically.

---

## How the build script selects a BLAS library

`build.rs` tries each source in order and stops at the first one that works:

| Step | Source | When it applies |
|------|--------|-----------------|
| 1 | `BLAS_LIB_DIR` env var | explicit override, CI, wheel builds |
| 2 | `pkg-config` (`openblas` → `cblas` → `blas`) | system/apt/brew/conda |
| 3 | macOS Accelerate framework | any macOS, automatic |
| 4 | numpy LP64 BLAS via Python introspection | conda without `PKG_CONFIG_PATH` configured |
| — | build error with instructions | nothing found |

---

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
maturin develop   # or: cargo build
```

### macOS

Apple's **Accelerate** framework (always installed) is used automatically — no
extra packages are required. If you prefer OpenBLAS (e.g. for benchmarking):

```sh
brew install openblas
export PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig"
maturin develop
```

The `export` is needed because Homebrew installs OpenBLAS in a non-standard
prefix that `pkg-config` does not search by default. Add it to your shell
profile (`~/.zshrc`, `~/.bash_profile`) to make it permanent.

### conda

OpenBLAS installed into the active conda environment is picked up via
`pkg-config` without any extra configuration:

```sh
conda install openblas
maturin develop
```

If `pkg-config` is not available in the environment, the build script falls
back to querying numpy's BLAS configuration directly.

### Windows

Windows does not ship a system BLAS. The recommended approach is to point
`BLAS_LIB_DIR` at a pre-built OpenBLAS distribution (available from the
[OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases)):

```powershell
$env:BLAS_LIB_DIR = "C:\openblas\lib"
$env:BLAS_LIB    = "openblas"          # matches libopenblas.lib / openblas.dll
maturin develop
```

Alternatively, install OpenBLAS via vcpkg or conda and ensure the library
directory is on `BLAS_LIB_DIR`.

---

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
maturin develop

# Use a different library (e.g. reference BLAS named libblas.so)
export BLAS_LIB_DIR=/usr/lib/x86_64-linux-gnu
export BLAS_LIB=blas
maturin develop

# One-shot without exporting
BLAS_LIB_DIR=/opt/myopenblas/lib maturin develop
```

---

## Building PyPI wheels

End-users installing a pre-built wheel do not need BLAS installed — it is
bundled inside the wheel. The recommended CI workflow:

1. **Build** with `maturin build --release`, pointing `BLAS_LIB_DIR` at a
   static or relocatable OpenBLAS build.
2. **Repair** the wheel to embed all shared library dependencies:
   - Linux: [`auditwheel repair`](https://github.com/pypa/auditwheel)
   - macOS: [`delocate`](https://github.com/matthew-brett/delocate) (not needed
     for Accelerate — it is part of the OS)
   - Windows: [`delvewheel`](https://github.com/adang1345/delvewheel)

On macOS, Accelerate is always present and no bundling is required.

### Example (Linux, GitHub Actions)

```yaml
- name: Build wheel
  env:
    BLAS_LIB_DIR: /opt/OpenBLAS/lib
    BLAS_LIB: openblas
  run: maturin build --release --out dist

- name: Repair wheel
  run: auditwheel repair dist/*.whl -w dist/
```
