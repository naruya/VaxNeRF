DATA_ROOT=$1
arg1=$2
arg2=$3

DATA_FOLDER="nerf_synthetic"
SCENES="lego chair drums ficus hotdog materials mic ship"

for scene in $SCENES; do
  echo $scene
  python visualhull.py \
    --config configs/demo \
    --data_dir "$DATA_ROOT"/"$DATA_FOLDER"/"$scene" \
    --voxel_dir "$DATA_ROOT"/voxel/"$scene" \
    --dilation $arg1 \
    --pooling $arg2 \
    --thresh 1. \
    --alpha_bkgd \
    --test
done
