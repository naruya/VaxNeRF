from absl import app
from absl import flags
from jax import config
from nerf import utils
from nerf import datasets

import jax
from jax import jit
from jax import device_put
from functools import partial
import jax.numpy as jnp
import flax
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


@partial(jit, static_argnums=(1,2,))
def digitize(p, rsize, vsize):
    p = jnp.round((p+rsize) * (vsize/(rsize*2)))
    return jnp.clip(p.astype(jnp.uint16), 0, vsize-1)

# somehow this requires more memory
# @partial(jit, static_argnums=(1,2,3,))
# def digitize(p, t_n, t_f, vsize):
#   return jnp.digitize(p + (t_n + t_f) / 2., jnp.linspace(t_n, t_f, vsize-1))


@partial(jit, static_argnums=(5,6,7,8,))
def carve_voxel(o, d, mask, voxel_s, voxel_r, rsize, vsize, t_n, t_f):
    # shape and ray counter
    voxel_si = jnp.zeros([vsize, vsize, vsize]).astype(jnp.uint16)
    voxel_ri = jnp.zeros([vsize, vsize, vsize]).astype(jnp.uint16)

    t_all = jnp.linspace(t_n, t_f, vsize+1)
    ray_p = digitize(o[:,:,None] + d[:,:,None] * t_all[None, None, :, None], rsize, vsize)

    # make a silhouette cone
    mask = jnp.repeat(mask[:,:,None].astype(jnp.uint8), vsize+1, axis=2)
    voxel_si = voxel_si.at[ray_p[:,:,:,0], ray_p[:,:,:,1], ray_p[:,:,:,2]].set(mask)
    voxel_ri = voxel_si.at[ray_p[:,:,:,0], ray_p[:,:,:,1], ray_p[:,:,:,2]].set(jnp.ones_like(mask))
    voxel_s = voxel_s + voxel_si
    voxel_r = voxel_r + voxel_ri
    return voxel_s, voxel_r


@partial(jit, static_argnums=(6,7,8,9,))
def paint_voxel(o, d, img, voxel_c, voxel_t, voxel_s, rsize, vsize, t_n, t_f):
    # color and closest distance
    voxel_ci = jnp.zeros([vsize, vsize, vsize, 3]).astype(jnp.float32)
    voxel_ti = jnp.zeros([vsize, vsize, vsize]).astype(jnp.uint16) + (vsize+1)

    t_all = jnp.linspace(t_n, t_f, vsize+1)
    ray_p = digitize(o[:,:,None] + d[:,:,None] * t_all[None, None, :, None], rsize, vsize)

    # intersection of ray and voxel
    ti = jnp.argmax(voxel_s[ray_p[:,:,:,0], ray_p[:,:,:,1], ray_p[:,:,:,2]], axis=2)
    ray_p_ti = jnp.sum((ray_p * jnp.eye(vsize+1)[ti][:,:,:,None]),axis=2).astype(jnp.uint16)
    voxel_ci = voxel_ci.at[ray_p_ti[:,:,0], ray_p_ti[:,:,1], ray_p_ti[:,:,2]].set(img)
    voxel_ti = voxel_ti.at[ray_p_ti[:,:,0], ray_p_ti[:,:,1], ray_p_ti[:,:,2]].set(ti)

    # not update voxel_c at voxel_ti == 0
    voxel_ci = jnp.where((voxel_ti == 0)[...,None], voxel_ci*0, voxel_ci)
    voxel_ti = jnp.where(voxel_ti == 0, voxel_ti*0 + (vsize+1), voxel_ti)
    voxel_c = jnp.where((voxel_ti < voxel_t)[...,None], voxel_ci, voxel_c)
    voxel_t = jnp.minimum(voxel_ti, voxel_t)
    return voxel_t, voxel_c


@partial(jit, static_argnums=(4,5,6,7,))
def render_voxel(voxel_s, voxel_c, o, d, rsize, vsize, t_n, t_f):
    t_all = jnp.linspace(t_n, t_f, vsize+1)
    ray_p = digitize(o[:,:,None] + d[:,:,None] * t_all[None, None, :, None], rsize, vsize)

    # intersection of ray and voxel
    ti = jnp.argmax(voxel_s[ray_p[:,:,:,0], ray_p[:,:,:,1], ray_p[:,:,:,2]], axis=2)
    ray_p_ti = jnp.sum((ray_p * jnp.eye(vsize+1)[ti][:,:,:,None]),axis=2).astype(jnp.uint16)
    img = voxel_c[ray_p_ti[:,:,0], ray_p_ti[:,:,1], ray_p_ti[:,:,2]]
    return img


# denoising sphere
def get_sphere(vsize, margin=5):
    voxel_sp = np.zeros([vsize, vsize, vsize]).astype(np.uint8)
    for z in range(margin, vsize-margin):
        z = z - vsize//2
        zr = int((((vsize//2-margin)**2 - z**2) ** 0.5))
        cv2.circle(voxel_sp[z + vsize//2], (vsize//2, vsize//2), zr, (1,), thickness=-1)
    return voxel_sp.astype(np.bool)


def visualhull(FLAGS, dataset, test_dataset=None):
    os.makedirs(FLAGS.voxel_dir, exist_ok=True)

    # larger size requires larger images
    t_n, t_f = FLAGS.near, FLAGS.far
    vsize = FLAGS.vsize
    rsize = (t_f - t_n) / 2.  # real size
    # t_c = (t_f + t_n) / 2.  # center

    ### shape
    voxel_s = device_put(jnp.zeros([vsize, vsize, vsize]).astype(jnp.uint16))
    voxel_r = device_put(jnp.zeros([vsize, vsize, vsize]).astype(jnp.uint16))

    for idx in tqdm(range(dataset.size)):
        o = dataset.rays.origins[idx]
        d = dataset.rays.directions[idx]
        img = dataset.images[idx]

        if FLAGS.alpha_bkgd:
            mask = img[Ellipsis, 3] > 0
            img = img[Ellipsis, :3]
        else:
            # get silhouette and remove whiteout
            mask = np.sum(img, axis=2) != 3
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3)))

        dil = FLAGS.dilation
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((dil,dil)), iterations=1)
        output = carve_voxel(o, d, mask, voxel_s, voxel_r, rsize, vsize, t_n, t_f)
        jax.tree_map(lambda x: x.block_until_ready(), output)
        voxel_s, voxel_r = output

    voxel_s = ((voxel_s >= (voxel_r * FLAGS.thresh)) * (voxel_s > 0.)).astype(jnp.uint8)
    voxel_s = voxel_s * get_sphere(FLAGS.vsize, FLAGS.margin)

    if FLAGS.pooling > 0:

        class Pool(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x):
                k = FLAGS.pooling
                x = flax.linen.max_pool(x, (k,k,k), strides=None, padding='SAME')
                return x

        model = Pool()
        key = jax.random.split(jax.random.PRNGKey(0), 1)[0]
        params = model.init(key, voxel_s[...,None])
        voxel_s = model.apply(params, voxel_s[...,None])[...,0]

    np.save(os.path.join(FLAGS.voxel_dir, "voxel.npy"), voxel_s)
    print(voxel_s.dtype, voxel_s.shape, "\nshape done!")

    if not FLAGS.test:
      return None

    ### color
    voxel_c = device_put(jnp.ones([vsize, vsize, vsize, 3]).astype(jnp.float32))
    voxel_t = device_put(jnp.zeros([vsize, vsize, vsize]).astype(jnp.uint16) + (vsize+1))

    for idx in tqdm(range(dataset.size)):
        o = dataset.rays.origins[idx]
        d = dataset.rays.directions[idx]
        img = dataset.images[idx, Ellipsis, :3]

        output = paint_voxel(o, d, img, voxel_c, voxel_t, voxel_s, rsize, vsize, t_n, t_f)
        jax.tree_map(lambda x: x.block_until_ready(), output)
        voxel_t, voxel_c = output

    np.save(os.path.join(FLAGS.voxel_dir, "voxel_color.npy"), voxel_c)
    print(voxel_c.dtype, voxel_c.shape, "\ncolor done!")

    ### test
    N=20
    di = test_dataset.size // N
    plt.figure(figsize=(200,40))
    for i in range(N):
        o = test_dataset.rays.origins[i*di]
        d = test_dataset.rays.directions[i*di]
        frame = render_voxel(voxel_s, voxel_c, o, d, rsize, vsize, t_n, t_f)
        plt.subplot(6,N,i+1+N*0); plt.imshow(frame)

    for i in range(N):
        frame = test_dataset.images[i*di,Ellipsis,:3]
        plt.subplot(6,N,i+1+N*1); plt.imshow(frame)

    voxel_c_red = voxel_c*0 + jnp.array([1.,0.,0.]) * voxel_s[:,:,:,None]
    pred_masks = []
    for i in range(N):
        o = test_dataset.rays.origins[i*di]
        d = test_dataset.rays.directions[i*di]
        frame = render_voxel(voxel_s, voxel_c_red, o, d, rsize, vsize, t_n, t_f)
        pred_masks.append(frame)
        plt.subplot(6,N,i+1+N*2); plt.imshow(frame)

    masks = []
    for i in range(N):
        if FLAGS.alpha_bkgd:
            mask = (test_dataset.images[i*di,Ellipsis,3:] > 0).astype(np.float32)
        else:
            mask = np.sum(test_dataset.images[i*di,Ellipsis,:3], axis=2) != 3
            mask = (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3))))[:,:,None]
        mask = mask * np.array([1.,0.,0.])
        masks.append(mask)
        plt.subplot(6,N,i+1+N*3); plt.imshow(mask)

    for i in range(N):
        plt.subplot(6,N,i+1+N*4); plt.imshow(np.clip(masks[i] - pred_masks[i], 0, 1))

    for i in range(N):
        plt.subplot(6,N,i+1+N*5); plt.imshow(np.clip(pred_masks[i] - masks[i], 0, 1))

    plt.savefig(os.path.join(FLAGS.voxel_dir, "voxel.png"))
    # plt.show()
    plt.close()

    if not FLAGS.save_gif:
        return None

    import moviepy.editor as mpy
    frames = []
    for i in tqdm(range(test_dataset.size)):
        o = test_dataset.rays.origins[i]
        d = test_dataset.rays.directions[i]
        frame = render_voxel(voxel_s, voxel_c, o, d, rsize, vsize, t_n, t_f).block_until_ready()
        frames.append(frame)
    frames = [(np.array(frame) * 255.).astype(np.uint8) for frame in frames]
    clip = mpy.ImageSequenceClip(frames, fps=20)
    clip.write_gif(os.path.join(FLAGS.voxel_dir, "voxel.gif"))

    print("test done!")


def main(unused_argv):
    utils.update_flags(FLAGS, no_nf=True)

    if FLAGS.alpha_bkgd:
        FLAGS.num_rgb_channels = 4
    else:
        assert FLAGS.thresh < 1., "thresh < 1. is recommended"

    class PureDataset(datasets.dataset_dict[FLAGS.dataset]):
        def start(self):
            pass

    dataset = PureDataset("train", FLAGS)
    dataset.images = dataset.images.reshape(-1,800,800,FLAGS.num_rgb_channels)
    dataset.rays = dataset.rays._replace(
        origins=dataset.rays.origins.reshape(-1,800,800,3))
    dataset.rays = dataset.rays._replace(
        directions=dataset.rays.directions.reshape(-1,800,800,3))
    dataset.rays = dataset.rays._replace(
        viewdirs=dataset.rays.viewdirs.reshape(-1,800,800,3))

    if FLAGS.test:
        test_dataset = PureDataset("test", FLAGS)
        test_dataset.images = test_dataset.images.reshape(-1,800,800,FLAGS.num_rgb_channels)
        test_dataset.rays = test_dataset.rays._replace(
            origins=test_dataset.rays.origins.reshape(-1,800,800,3))
        test_dataset.rays = test_dataset.rays._replace(
            directions=test_dataset.rays.directions.reshape(-1,800,800,3))
        test_dataset.rays = test_dataset.rays._replace(
            viewdirs=test_dataset.rays.viewdirs.reshape(-1,800,800,3))
    else:
        test_dataset = None

    utils.update_flags(FLAGS, no_nf=not "nsvf" in FLAGS.config.lower())

    visualhull(FLAGS, dataset, test_dataset)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    utils.define_flags()
    flags.DEFINE_integer("vsize", 400, "voxel size")
    flags.DEFINE_integer("dilation", 7, "dilation size")
    flags.DEFINE_integer("pooling", 0, "pooling size")
    flags.DEFINE_float("thresh", 1., "threshold")
    flags.DEFINE_integer("margin", 40, "margin")
    flags.DEFINE_bool("test", False, "do test or not")
    config.parse_flags_with_absl()
    app.run(main)
