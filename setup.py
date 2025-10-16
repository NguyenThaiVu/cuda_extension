import os
import pathlib
os.environ["CUDA_HOME"] = "/home/tnguyen10/cuda-12.1"
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ["PATH"]
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

conda = os.environ.get("CONDA_PREFIX", "")
cuda  = os.environ.get("CUDA_HOME", "")

os.environ["CUDACXX"]   = str(pathlib.Path(cuda) / "bin" / "nvcc")
os.environ["CC"]        = str(pathlib.Path(conda) / "bin" / "x86_64-conda-linux-gnu-cc")
os.environ["CXX"]       = str(pathlib.Path(conda) / "bin" / "x86_64-conda-linux-gnu-c++")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

setup(
    name="torch_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="torch_cuda_ext",
            sources=["torch_extension.cu"],
            include_dirs=["/home/tnguyen10/Desktop/deep_learning_research/test_cuda_lib/cutlass/include",\
                "/home/tnguyen10/Desktop/deep_learning_research/test_cuda_lib/cutlass/tools/util/include"],
            extra_compile_args={"cxx": ["-O3"],\
                "nvcc": ["-O3", "-std=c++17", "-gencode=arch=compute_86,code=sm_86"]},
            # Make the loader prefer your conda C++ runtime (no LD_LIBRARY_PATH needed)
            extra_link_args=[f"-Wl,-rpath,{conda}/lib"] if conda else [],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
