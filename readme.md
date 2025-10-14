## 🚀 CUDA Extension Build Process (Personal Setup Summary)

This project demonstrates how to **build and use a custom CUDA `.so` extension** in PyTorch safely inside a **non-root, conda-based environment**.

**Process overview:**

1. **Install CUDA locally** (no sudo)

   * Extracted `cuda_12.1.0_530.30.02_linux.run` to `~/cuda-12.1` using

     ```bash
     sh cuda_12.1.0_530.30.02_linux.run --toolkit --installpath=$HOME/cuda-12.1
     ```
   * This keeps CUDA user-local and avoids system conflicts.

2. **Use conda’s compiler and runtime**

   * Compiling inside the active `gpu_env` ensures the extension links against the same `libstdc++.so.6` used by Python.
   * Prevents the common `GLIBCXX_x.x.xx not found` import error.

3. **Build with `setup.py`**

   * The script explicitly sets:

     * `CUDA_HOME` → `~/cuda-12.1`
     * Compiler paths → conda-provided `x86_64-conda-linux-gnu-g++/gcc`
     * `rpath` → `$CONDA_PREFIX/lib` so the `.so` finds the correct C++ runtime at import time
     * `TORCH_CUDA_ARCH_LIST` → `8.6` for target GPU (RTX 30xx)

4. **Compile the extension**

   ```bash
   python setup.py build_ext --inplace
   ```

5. **Load and run in Python**

   ```python
   import torch, dot_cuda_ext
   # use dot_cuda_ext.dot_forward(...)
   ```

**Result:**
✅ A self-contained CUDA extension (`dot_cuda_ext.so`) that builds and runs smoothly without root access or library conflicts.