# triton_aot_ncu
a implement for triton kernel to get binary file and use ncu to profile.
replace your kernel in triton_kernel.py, change the argements in main.c and test.py, then run test.py.

clean.sh: clean the temporary file while generate binary file.
test.py: will generate AOT kernel, build a binary file called test, and the use ncu command for profile.
main.c: main function for triton kernel.
triton_kernel.py: a triton kernel example.