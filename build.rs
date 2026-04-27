fn main() {
    // Link against the system OpenBLAS for fast BLAS-1/2 routines.
    println!("cargo:rustc-link-lib=openblas");
}
