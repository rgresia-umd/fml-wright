# run from root directory
# bash scripts/core/3_create_structure_images.sh

input_directory='./data/geodata'
output_directory='./data/images/structure'
module='structure_plan'
step='generate_images'

python ./main/core.py --module=structure_plan --step=generate_images --input_directory=./data/geodata --output_directory=./data/images/structure
