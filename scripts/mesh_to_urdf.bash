# Path where the slider mesh is located
root_dir="/home/hong/ws/stable-pushnet-datagen/src/data/dish_mesh"

# Absolute path to this script
SCRIPT_PATH=$(readlink -f "$0")
# Absolute path to the directory where this script is located
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
# Path where the asset (slider urdf) will be located
asset_dir="$SCRIPT_DIR/assets/dish_urdf"

# Mesh extensions
mesh_exts=(".stl" ".obj")

# Iterate over the array and run the Python script with different arguments
for mesh_ext in "${mesh_exts[@]}"
do
    python3 mesh_to_urdf.py --root $root_dir --target $asset_dir --extension $mesh_ext
done