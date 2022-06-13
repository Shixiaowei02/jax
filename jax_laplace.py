import numpy as np
from jax import numpy as jnp
from jax import jit, grad, hessian, vmap, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import (Dense, Tanh)
import paddlescience as psci
import time

from cupy.cuda import nvtx
from cupy.cuda import profiler
from cupy.cuda import runtime

def Network(num_outs):
    return stax.serial(
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(50), Tanh,
        Dense(num_outs)
    )

# Analytical solution
def LaplaceRecSolution(x, y, k=1.0):
    if (k == 0.0):
        return x * y
    else:
        return np.cos(k * x) * np.cosh(k * y)


# Generate analytical Solution using Geometry points
def GenSolution(xy, bc_index):
    sol = np.zeros((len(xy), 1)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    for i in range(len(xy)):
        sol[i] = LaplaceRecSolution(xy[i][0], xy[i][1])
    for i in range(len(bc_index)):
        bc_value[i][0] = sol[bc_index[i]]
    return [sol, bc_value]


if __name__ == "__main__":
  rng_key = random.PRNGKey(0)

  batch_size = 101 * 101
  num_ins = 2
  num_outs = 1
  num_epochs = 2010

  input_shape = (batch_size, num_ins)

  init_func, predict_func = Network(num_outs)
  _, init_params = init_func(rng_key, input_shape)

  # Geometry
  geo = psci.geometry.Rectangular(
      space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
  geo = geo.discretize(space_nsteps=(101, 101))
  golden, bc_value = GenSolution(geo.space_domain, geo.bc_index)
  # save golden
  psci.visu.save_vtk(geo, golden, 'golden_laplace_2d')
  np.save("./golden_laplace_2d.npy", golden)
  inputs = jnp.array(geo.space_domain)
  bc_index = jnp.array(geo.bc_index)
  bc_value = jnp.array(bc_value)

  def laplace_eq_loss(params, inputs):
    def pde_func(params, inputs):
        hes = hessian(predict_func, argnums=1)(params, inputs)
        return hes[0][0][0] + hes[0][1][1]
    pde_vfunc = vmap(pde_func, [None, 0], 0)
    pde_v = pde_vfunc(params, inputs)
    return jnp.linalg.norm(pde_v, ord=2)

  def loss(params, inputs):
    outputs = predict_func(params, inputs)
    eq_loss = laplace_eq_loss(params, inputs)
    bc_loss = jnp.linalg.norm(outputs[bc_index] - bc_value, ord=2)
    return eq_loss + bc_loss

  opt_init, opt_update, get_params = optimizers.adam(step_size=0.001)

  @jit
  def update(i, opt_state, inputs):
    params = get_params(opt_state)
    total_loss = loss(params, inputs)
    opt_state = opt_update(i, grad(loss)(params, inputs), opt_state)
    return total_loss, opt_state

  opt_state = opt_init(init_params)
  begin = time.time()
  for i in range(10):
    total_loss, opt_state = update(i, opt_state, inputs)
    print("num_epoch: ", i, " loss: ", total_loss, ' eq_loss: ')
  total_loss.block_until_ready()
  mid = time.time()

  for i in range(num_epochs - 10):
    total_loss, opt_state = update(i + 10, opt_state, inputs)
    # if (i + 1) % 100 == 0:
    print("num_epoch: ", i + 10, " loss: ", total_loss, ' eq_loss: ')
  total_loss.block_until_ready()
  end = time.time()

  profile_epoch = 0
  if profile_epoch > 0:
    runtime.deviceSynchronize()
    profiler.start()
    for step in range(profile_epoch):
      nvtx.RangePush("Epoch " + str(step))
      total_loss, opt_state = update(num_epochs + step, opt_state, inputs)
      # print("num_epoch: ", num_epochs + step, " totol_loss: ", total_loss)
      nvtx.RangePop()
    total_loss.block_until_ready()
    profiler.stop()

  trained_params = get_params(opt_state)
  rslt = np.array(predict_func(trained_params, inputs), copy=False)
  psci.visu.save_vtk(geo, rslt, 'rslt_laplace_2d')
  np.save('./rslt_laplace_2d.npy', rslt)

  # Calculate diff and l2 relative error
  diff = rslt - golden
  psci.visu.save_vtk(geo, diff, 'diff_laplace_2d')
  np.save('./diff_laplace_2d.npy', diff)
  root_square_error = np.linalg.norm(diff, ord=2)
  mean_square_error = root_square_error * root_square_error / geo.get_domain_size(
  )
  print("mean_sqeare_error: ", mean_square_error)
  print("first 9 epoch time: ", mid-begin)
  print("2000 epoch(10~2010) time: ", end-mid)