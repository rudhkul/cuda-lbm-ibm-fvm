# 3D Two-Phase VOF Solver in CUDA

`voftrial1.cu` is a single-file CUDA C++ solver for three-dimensional, incompressible two-phase flow using the Volume of Fluid (VOF) method. It produced the 3D two-phase recording on [rudhlab.com](https://rudhlab.com/#simulations).

This is an educational demonstration. It has not been validated or verified.

## Method

* **Interface capturing:** directionally split flux-corrected transport (FCT) of the volume fraction, one sweep per direction.
* **Momentum:** explicit projection method.
* **Pressure Poisson equation:** Jacobi iteration on the GPU.
* **Default case:** a 200 × 200 × 200 grid in a 0.1 m cube; a block of liquid (1000 kg/m³) in one corner collapsing under gravity into a lighter phase (50 kg/m³), a 3D dam break, with surface tension σ = 0.007 N/m and Δt = 4 × 10⁻⁶ s.

## Build and run

Requires the CUDA toolkit and an NVIDIA GPU.

```
nvcc -O3 -o 3dvof_cuda voftrial1.cu
./3dvof_cuda --nstep 100 --max_steps 20000
```

* `--nstep N`: write output every N steps (default 100).
* `--max_steps N`: stop after N steps; 0 (the default) runs until Ctrl+C.
* `-ic 1`: initial condition. Only 1 is implemented.

## Output

Legacy VTK files of the volume fraction in `output/step-NNNNN.vtk`, readable in ParaView.

## License

MIT. See [LICENSE](LICENSE).
