DATA_ROOT=$1

DATA_FOLDER="nerf_synthetic"
SCENES="lego chair drums ficus hotdog materials mic ship"

for scene in $SCENES; do
  echo $scene
  python visualhull.py \
    --config configs/blender \
    --data_dir "$DATA_ROOT"/"$DATA_FOLDER"/"$scene" \
    --voxel_dir "$DATA_ROOT"/voxel/"$scene" \
    --dilation 7 \
    --pooling 7 \
    --thresh 1. \
    --alpha_bkgd \
    --test
done


DATA_FOLDER="Synthetic_NSVF"
SCENES="Bike Palace Robot Toad Wineholder"

for scene in $SCENES; do
  echo $scene
  python visualhull.py \
    --config configs/nsvf \
    --data_dir "$DATA_ROOT"/"$DATA_FOLDER"/"$scene" \
    --voxel_dir "$DATA_ROOT"/voxel/"$scene" \
    --dilation 11 \
    --pooling 7 \
    --thresh 1. \
    --alpha_bkgd \
    --test
done

SCENES="Lifestyle Spaceship Steamtrain"

for scene in $SCENES; do
  echo $scene
  python visualhull.py \
    --config configs/nsvf \
    --data_dir "$DATA_ROOT"/"$DATA_FOLDER"/"$scene" \
    --voxel_dir "$DATA_ROOT"/voxel/"$scene" \
    --dilation 11 \
    --pooling 7 \
    --thresh 0.9 \
    --test
done
