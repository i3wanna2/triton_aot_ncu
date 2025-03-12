import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy as np
import glob
import os
import subprocess
import sys
import tempfile
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import include_dir, library_dirs

tmp_dir = os.getcwd() 
kernel_path = os.getcwd() + "/triton_kernel.py"
sig = "*fp16, *fp16, 7200, 7200, *fp16, *fp16, 2, 16, 16" #argments of kernel, keep same with main.c and triton kernel
kernel_name = "kernel"
grid = "202500, 1, 1" 
warp_num = 4
os.environ["LD_LIBRARY_PATH"] = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ncu_command = f"LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]} CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]} ncu --clock-control none --target-processes all --set full --print-summary per-kernel -f -o profile ./test"
ncu_command = f"LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]} CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]} ncu --clock-control none --target-processes all --set full --print-summary per-kernel ./test > profile.csv 2>&1"



def _compile_kernel(dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")
    try:
        result = subprocess.run(
            [
                sys.executable,
                compiler_path,
                "-n",
                kernel_name,
                "--signature",
                signature,
                "--out-name",
                out_name,
                "-o",
                out_path,
                "-w",
                str(num_warps),
                "-g",
                grid,
                kernel_path,
            ],
            check=True,
            cwd=dir,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise


def compile_aot_kernels(dir, kernel_path, sig, name, grid, warp_num):
    _compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name=name,
        out_name=name,
        out_path=name,
        num_warps=warp_num,
        grid=grid,
        kernel_path=kernel_path,
    )

def link_aot_kernels(dir):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")
    h_files = glob.glob(os.path.join(dir, "*.h"))
    subprocess.run([sys.executable, linker_path] + h_files + ["-o", "kernel"], check=True, cwd=dir)

def gen_kernel_library(dir, libname):
    c_files = glob.glob(os.path.join(dir, "*.c"))
    subprocess.run(
        ["gcc"] + c_files + ["-I", include_dir[0], "-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))

    command = ["gcc", *o_files, "-shared", "-o", libname]
    for lib_dir in library_dirs():
        command.extend(["-L", lib_dir])
    subprocess.run(command, check=True, cwd=dir)


def gen_test_bin(dir, exe="test", algo_id=0):
    command = ["gcc", "main.c"]
    for inc_dir in include_dir:
        command.extend(["-I", inc_dir])
    for lib_dir in library_dirs():
        command.extend(["-L", lib_dir])
    command.extend(["-l", "cuda", "-L", dir, "-l", "kernel", "-o", exe])
    subprocess.run(command, check=True, cwd=dir)

compile_aot_kernels(tmp_dir, kernel_path, sig, kernel_name,grid,warp_num)
link_aot_kernels(tmp_dir)

# compile test case
os.system("mv main.c main.cp")
gen_kernel_library(tmp_dir, "libkernel.so")
os.system("mv main.cp main.c")
gen_test_bin(tmp_dir)


os.system(ncu_command)
    
