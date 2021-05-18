# run from root directory
# bash scripts/core/5_create_complete_floorplan_images.sh

input_directory='./data/geodata'
output_directory='./data/images/complete_floorplan'
module='complete_floorplan'
step='generate_images'
n_jobs=-1

python ./main/core.py --module=complete_floorplan --step=generate_images --input_directory=./data/geodata --output_directory=./data/images/complete_floorplan --n_jobs=-1
