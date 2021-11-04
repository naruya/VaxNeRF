# python visualhull.py --config configs/demo --data_dir data/nerf_synthetic/lego --vh_test

from absl import app
from absl import flags
from jax import config
from nerf import utils
from nerf import datasets

import jax
from jax import jit
from jax import device_put
import jax.numpy as jnp
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


FLAGS = flags.FLAGS
utils.define_flags()
flags.DEFINE_string("voxel_dir", "data/voxel", "voxel data directory.")
flags.DEFINE_string("vh_save", "shape", "Save type ('shape' of 'color')")
flags.DEFINE_bool("vh_test", False, "If True, test the result of visual hull")
config.parse_flags_with_absl()

# larger size requires larger images
t_n, t_f = FLAGS.near, FLAGS.far
vsize = 400               # voxel size
rsize = (t_f - t_n) / 2.  # real size
# t_c = (t_f + t_n) / 2.  # center


@jit
def digitize(p):
    p = jnp.round((p+rsize) * (vsize/(rsize*2)))
    return jnp.clip(p.astype(jnp.int16), 0, vsize-1)
    # requires more memory
    # return jnp.digitize(p, jnp.linspace(t_n-t_c, t_f-t_c, vsize)).astype(jnp.int16)


@jit
def carve_voxel(o, d, mask):
    voxel_si = jnp.zeros([vsize, vsize, vsize]).astype(jnp.uint8)
    mask = device_put(mask)

    t_all = jnp.linspace(t_n, t_f, vsize+1)
    ray_p = digitize(o[:,:,None] + d[:,:,None] * t_all[None, None, :, None])

    mask = jnp.repeat(mask[:,:,None].astype(jnp.uint8), vsize+1, axis=2)
    voxel_si = voxel_si.at[ray_p[:,:,:,1], ray_p[:,:,:,0], ray_p[:,:,:,2]].set(mask)
    return voxel_si


@jit
def paint_voxel(o, d, img, voxel_c, voxel_t, voxel_s):
    voxel_ci = jnp.zeros([vsize, vsize, vsize, 3]).astype(jnp.float32)
    voxel_ti = jnp.zeros([vsize, vsize, vsize]).astype(jnp.int16) + (vsize+1)

    t_all = jnp.linspace(t_n, t_f, vsize+1)    
    ray_p = digitize(o[:,:,None] + d[:,:,None] * t_all[None, None, :, None])

    ti = jnp.argmax(voxel_s[ray_p[:,:,:,1], ray_p[:,:,:,0], ray_p[:,:,:,2]], axis=2)
    ray_p_ti = jnp.sum((ray_p * jnp.eye(vsize+1)[ti][:,:,:,None]),axis=2).astype(jnp.int16)
    voxel_ci = voxel_ci.at[ray_p_ti[:,:,1], ray_p_ti[:,:,0], ray_p_ti[:,:,2]].set(img)
    voxel_ti = voxel_ti.at[ray_p_ti[:,:,1], ray_p_ti[:,:,0], ray_p_ti[:,:,2]].set(ti)
    voxel_c = jnp.where((voxel_t > voxel_ti)[...,None], voxel_ci, voxel_c)
    voxel_t = jnp.minimum(voxel_t, voxel_ti)
    return voxel_t, voxel_c


@jit
def render_voxel(voxel_s, voxel_c, o, d):
    t_all = jnp.linspace(t_n, t_f, vsize+1)
    ray_p = digitize(o[:,:,None] + d[:,:,None] * t_all[None, None, :, None])

    ti = jnp.argmax(voxel_s[ray_p[:,:,:,1], ray_p[:,:,:,0], ray_p[:,:,:,2]], axis=2)
    ray_p_ti = jnp.sum((ray_p * jnp.eye(vsize+1)[ti][:,:,:,None]),axis=2).astype(jnp.int16)
    img = voxel_c[ray_p_ti[:,:,1], ray_p_ti[:,:,0], ray_p_ti[:,:,2]]
    return img


class PureDataset(datasets.dataset_dict[FLAGS.dataset]):
    def start(self):
        pass


def main(unused_argv):
    if FLAGS.config is not None:
        utils.update_flags(FLAGS)

    target = FLAGS.data_dir.split("/")[-1]
    os.makedirs(os.path.join(FLAGS.voxel_dir, target), exist_ok=True)

    dataset = PureDataset("train", FLAGS)
    dataset.images = dataset.images.reshape(-1,800,800,3)
    dataset.rays = dataset.rays._replace(origins = dataset.rays.origins.reshape(-1,800,800,3))
    dataset.rays = dataset.rays._replace(directions = dataset.rays.directions.reshape(-1,800,800,3))
    dataset.rays = dataset.rays._replace(viewdirs = dataset.rays.viewdirs.reshape(-1,800,800,3))

    # shape 
    voxel_s = np.zeros([vsize, vsize, vsize]).astype(np.bool)

    for idx in tqdm(range(dataset.size)):
        o = dataset.rays.origins[idx]
        d = dataset.rays.directions[idx]
        img = dataset.images[idx]
        mask = np.sum(img != 1., axis=2) != 0
        # remove whiteout
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3)))
        # dilation (It makes appearance worse, but recommended for voxel initialization)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3,3)), iterations=1)
        voxel_s += carve_voxel(o, d, mask).block_until_ready()

    voxel_s = (voxel_s == 100).astype(jnp.uint8)

    if FLAGS.vh_save == "shape":
        print(voxel_s.dtype, voxel_s.shape)
        np.save(os.path.join(FLAGS.voxel_dir, target, "voxel.npy"), voxel_s)
        print("done!")


    if FLAGS.vh_save == "color" or FLAGS.vh_test:
        # color
        voxel_c = device_put(jnp.ones([vsize, vsize, vsize, 3]).astype(jnp.float32))  # white
        voxel_t = device_put(jnp.zeros([vsize, vsize, vsize]).astype(jnp.int16) + (vsize+1))

        for idx in tqdm(range(dataset.size)):
            o = dataset.rays.origins[idx]
            d = dataset.rays.directions[idx]
            img = dataset.images[idx]
            output = paint_voxel(o, d, img, voxel_c, voxel_t, voxel_s)
            jax.tree_map(lambda x: x.block_until_ready(), output)
            voxel_t, voxel_c = output

        if FLAGS.vh_save == "color":
            print(voxel_c.dtype, voxel_c.shape)
            np.save(os.path.join(FLAGS.voxel_dir, target, "voxel.npy"), voxel_c)
            print("done!")


    if FLAGS.vh_test:
        # test
        N = 5
        plt.figure(figsize=(40,8))
        for i in range(N):
            o = dataset.rays.origins[i]
            d = dataset.rays.directions[i]
            frame = render_voxel(voxel_s, voxel_c, o, d)
            plt.subplot(1,N,i+1); plt.imshow(frame)
        plt.savefig(os.path.join(FLAGS.voxel_dir, target, "voxel.png"))
        plt.close()

        # import moviepy.editor as mpy
        # test_dataset = PureDataset("test", FLAGS)
        # test_dataset.images = test_dataset.images.reshape(-1,800,800,3)
        # test_dataset.rays = test_dataset.rays._replace(origins = test_dataset.rays.origins.reshape(-1,800,800,3))
        # test_dataset.rays = test_dataset.rays._replace(directions = test_dataset.rays.directions.reshape(-1,800,800,3))
        # test_dataset.rays = test_dataset.rays._replace(viewdirs = test_dataset.rays.viewdirs.reshape(-1,800,800,3))
        # frames = []
        # for i in tqdm(range(test_dataset.size)):
        #     o = test_dataset.rays.origins[i]
        #     d = test_dataset.rays.directions[i]
        #     frame = render_voxel(voxel_s, voxel_c, o, d).block_until_ready()
        #     frames.append(frame)
        # frames = [(np.array(frame) * 255.).astype(np.uint8) for frame in frames]
        # clip = mpy.ImageSequenceClip(frames, fps=10)
        # clip.write_gif(os.path.join(FLAGS.voxel_dir, target, "voxel.gif"))

        print("test done!")


if __name__ == "__main__":
    app.run(main)