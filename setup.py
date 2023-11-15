from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rz_linear',
    ext_modules=[CUDAExtension(
        'rz_linear',
        ['rz_linear_kernel.cu'])],
    py_modules=['RzLinear'],
    cmdclass={'build_ext': BuildExtension}
)
