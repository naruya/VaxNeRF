# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Evaluation script for Nerf."""
import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
from jax import device_put
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from nerf import datasets
from nerf import models
from nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()

# LPIPS_TFHUB_PATH = "@neural-rendering/lpips/distance/1"


# def compute_lpips(image1, image2, model):
#   """Compute the LPIPS metric."""
#   # The LPIPS model expects a batch dimension.
#   return model(
#       tf.convert_to_tensor(image1[None, Ellipsis]),
#       tf.convert_to_tensor(image2[None, Ellipsis]))[0]


def render_fn(model, voxel, len_inpc, len_inpf, variables, key_0, key_1, rays):
  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates "speckle" artifacts.
  return jax.lax.all_gather(
      model.apply(variables, key_0, key_1, rays, voxel, len_inpc, len_inpf, False)[0],
      axis_name="batch")


def main(unused_argv):
  # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
  # LPIPS computation or dataset loading.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  rng = random.PRNGKey(20200823)

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  dataset = datasets.get_dataset("test", FLAGS)
  if FLAGS.dataset == "nsvf":
    utils.update_flags(FLAGS, no_nf=False)

  rng, key = random.split(rng)
  model, init_variables = models.get_model(key, dataset.peek(), FLAGS)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables

  if not FLAGS.voxel_path == "":
    voxel = device_put(jnp.load(FLAGS.voxel_path).astype(jnp.float32))
  else:
    voxel = None

  # lpips_model = tf_hub.load(LPIPS_TFHUB_PATH)

  # pmap over only the data input.
  render_pfn = jax.pmap(
      functools.partial(render_fn, model, voxel, int(FLAGS.len_inpc*2), int(FLAGS.len_inpf*1.5)),
      axis_name="batch",
      in_axes=(None, None, None, 0),
      donate_argnums=(3,))

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  # last_step = 0
  out_dir = path.join(FLAGS.train_dir,
                      "path_renders" if FLAGS.render_path else "test_preds")
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.train_dir, "eval"))
  steps = [
    10000,
    20000,
    30000,
    40000,
    50000,
    60000,
    70000,
    80000,
    90000,
    100000,
    200000,
    300000,
    400000,
    500000,
    600000,
    700000,
    800000,
    900000,
    1000000,
  ]
  stepi = 0
  while True:
    step = steps[stepi]
    print("step:", step)
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state, step)
    # step = int(state.optimizer.state.step)
    # if step <= last_step:
    #   continue
    if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
    psnr_values = []
    ssim_values = []
    # lpips_values = []
    if not FLAGS.eval_once:
      showcase_index = np.random.randint(0, dataset.size)
    for idx in range(dataset.size):
      print(f"Evaluating {idx+1}/{dataset.size}")
      batch = next(dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, state.optimizer.target),
          batch["rays"],
          rng,
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk)
      if jax.host_id() != 0:  # Only record via host 0.
        continue
      if not FLAGS.eval_once and idx == showcase_index:
        showcase_color = pred_color
        showcase_disp = pred_disp
        showcase_acc = pred_acc
        if not FLAGS.render_path:
          showcase_gt = batch["pixels"]
      if not FLAGS.render_path:
        psnr = utils.compute_psnr(((pred_color - batch["pixels"])**2).mean())
        ssim = ssim_fn(pred_color, batch["pixels"])
        # lpips = compute_lpips(pred_color, batch["pixels"], lpips_model)
        print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
        psnr_values.append(float(psnr))
        ssim_values.append(float(ssim))
        # lpips_values.append(float(lpips))
      if FLAGS.save_output:
        utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
        utils.save_img(pred_disp[Ellipsis, 0],
                       path.join(out_dir, "disp_{:03d}.png".format(idx)))
    if (not FLAGS.eval_once) and (jax.host_id() == 0):
      summary_writer.image("pred_color", showcase_color, step)
      summary_writer.image("pred_disp", showcase_disp, step)
      summary_writer.image("pred_acc", showcase_acc, step)
      if not FLAGS.render_path:
        summary_writer.scalar("psnr", np.mean(np.array(psnr_values)), step)
        summary_writer.scalar("ssim", np.mean(np.array(ssim_values)), step)
        # summary_writer.scalar("lpips", np.mean(np.array(lpips_values)), step)
        summary_writer.image("target", showcase_gt, step)
    if FLAGS.save_output and (not FLAGS.render_path) and (jax.host_id() == 0):
      with utils.open_file(path.join(out_dir, f"psnrs_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in psnr_values]))
      with utils.open_file(path.join(out_dir, f"ssims_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in ssim_values]))
      # with utils.open_file(path.join(out_dir, f"lpips_{step}.txt"), "w") as f:
      #   f.write(" ".join([str(v) for v in lpips_values]))
      with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(psnr_values))))
      with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(ssim_values))))
      # with utils.open_file(path.join(out_dir, "lpips.txt"), "w") as f:
      #   f.write("{}".format(np.mean(np.array(lpips_values))))
    if FLAGS.eval_once:
      break
    if int(step) >= FLAGS.max_steps:
      break
    # last_step = step
    stepi += 1


if __name__ == "__main__":
  app.run(main)
