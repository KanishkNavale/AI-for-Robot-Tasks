urdf2rai.py panda_arm_hand.urdf > z.1.g
sed 's/package:\/\/franka_description\/meshes/meshes/g' z.1.g > z.2.g
sed 's/\.dae/.ply/g' z.2.g > z.3.g

# DELETE the axis = [0 0 0] but by hand!!

kinEdit -file z.3.g -cleanOnly
mv z.g z.panda.g
