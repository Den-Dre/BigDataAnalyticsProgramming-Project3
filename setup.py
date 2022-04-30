# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
#                          DO NOT MODIFY THIS FILE!                           #
#                                                                             #
###############################################################################

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import numpy as np

copt =  {
    "msvc": ["/Ox", "/Og"],
    "mingw32" : ["-O0", "-march=native"],  # O3
    "unix" : ["-O0", "-march=native", "-std=c++11"],  # 03
}
lopt = {
    "mingw32" : ["-lstdc++"],
    "unix" : ["-lstdc++"]
}

# https://stackoverflow.com/a/5192738/14999786
class pqnn_build_ext(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        print("DEBUG", c)
        if c in copt:
           for e in self.extensions:
               e.extra_compile_args = copt[c]
        if c in lopt:
            for e in self.extensions:
                e.extra_link_args = lopt[c]
        build_ext.build_extensions(self)

knnmodule = Extension(
    "prod_quan_nn",
    sources = [
        "bindings.cpp",
        "prod_quan_nn.cpp"
    ]
)

setup(name = "BdapKnn",
      version = "0.1",
      description = "(c) BDAP team",
      ext_modules = [knnmodule],
      include_dirs=[np.get_include()],
      cmdclass = {"build_ext": pqnn_build_ext})
