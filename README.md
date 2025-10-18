# SPH Fluid Simulation (`main.cpp`)

This folder contains a 2D Smoothed Particle Hydrodynamics (SPH) demo implemented using GLFW and immediate-mode OpenGL for visualization. It simulates particles with density/pressure and viscosity forces computed using common SPH kernels and a uniform grid neighbourhood search.

Features
- ~1000 particles by default (tunable constants at top of `main.cpp`).
- Cell-based neighbourhood search for performance.
- Mouse/keyboard hooks: left-click applies interactive actions (see code); right-click opens console options for setting velocities or adding blocks.

Build & run
- Requires GLFW and OpenGL. Link with -lglfw -lGL (or -lglfw3 -lopengl32 on Windows).

Example (Windows MSYS2 / mingw):

```powershell
g++ "main.cpp" -o "sph.exe" -lglfw3 -lopengl32 -lgdi32 -std=c++17 -pthread
.\\sph.exe
```

Notes and tuning
- Key constants are at the top of `main.cpp`: particle count, smoothing radius, viscosity, gravity, time step, and max speed.
- The program uses an explicit integrator and applies boundary handling and simple block obstacles.
- The rendering uses GL_POINTS with `glPointSize` — adjust `resolution` and `glPointSize` for display scaling.

Suggested improvements
- Add a CMakeLists.txt and command-line flags for configuration.
- Implement an adaptive time step or semi-implicit integrator for better stability.
- Replace immediate-mode rendering with a VBO/texture blit for faster rendering of many particles.
