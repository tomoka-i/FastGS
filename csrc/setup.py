from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fastgs_cuda',
    ext_modules=[
        CUDAExtension(
            name='fastgs_cuda',
            sources=['compute_score.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)