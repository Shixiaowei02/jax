python build/build.py --enable_cuda
pip uninstall -y jaxlib
pip install /shixiaowei02/Paddle-CINN/jax/dist/jaxlib-0.1.76-cp37-none-manylinux2010_x86_64.whl
