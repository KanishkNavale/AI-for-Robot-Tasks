body base_link {
}
shape visual base_link_1 (base_link) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/visual/base_link.stl'
color:[0.4 0.4 0.4 1.0]
colorName:
visual
}
shape collision base_link_0 (base_link) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/collision/base_link.stl'
contact:-2
}
body link_1 {
}
shape visual link_1_1 (link_1) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/visual/link_1.stl'
color:[0.96 0.76 0.13 1.0]
colorName:
visual
}
shape collision link_1_0 (link_1) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/collision/link_1.stl'
contact:-2
}
body link_2 {
}
shape visual link_2_1 (link_2) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/visual/link_2.stl'
color:[0.96 0.76 0.13 1.0]
colorName:
visual
}
shape collision link_2_0 (link_2) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/collision/link_2.stl'
contact:-2
}
body link_3 {
}
shape visual link_3_1 (link_3) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/visual/link_3.stl'
color:[0.96 0.76 0.13 1.0]
colorName:
visual
}
shape collision link_3_0 (link_3) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/collision/link_3.stl'
contact:-2
}
body link_4 {
}
shape visual link_4_1 (link_4) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/visual/link_4.stl'
color:[0.96 0.76 0.13 1.0]
colorName:
visual
}
shape collision link_4_0 (link_4) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/collision/link_4.stl'
contact:-2
}
body link_5 {
}
shape visual link_5_1 (link_5) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/visual/link_5.stl'
color:[0.96 0.76 0.13 1.0]
colorName:
visual
}
shape collision link_5_0 (link_5) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/collision/link_5.stl'
contact:-2
}
body link_6 {
}
shape visual link_6_1 (link_6) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/visual/link_6.stl'
color:[0.15 0.15 0.15 1.0]
colorName:
visual
}
shape collision link_6_0 (link_6) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://fanuc_description/meshes/collision/link_6.stl'
contact:-2
}
body tool0 {
}
body base {
}
joint joint_1 (base_link link_1) {  
type:hingeX
axis:[0 0 1]
A:<t(0 0 0.450) E(0 0 0)>
limits:[-3.14 3.14]
ctrl_limits:[3.67 0 1]
}
joint joint_2 (link_1 link_2) {  
type:hingeX
axis:[0 1 0]
A:<t(0.150 0 0) E(0 0 0)>
limits:[-1.57 2.79]
ctrl_limits:[3.32 0 1]
}
joint joint_3 (link_2 link_3) {  
type:hingeX
axis:[0 -1 0]
A:<t(0 0 0.600) E(0 0 0)>
limits:[-3.14 4.61]
ctrl_limits:[3.67 0 1]
}
joint joint_4 (link_3 link_4) {  
type:hingeX
axis:[-1 0 0]
A:<t(0 0 0.200) E(0 0 0)>
limits:[-3.31 3.31]
ctrl_limits:[6.98 0 1]
}
joint joint_5 (link_4 link_5) {  
type:hingeX
axis:[0 -1 0]
A:<t(0.640 0 0) E(0 0 0)>
limits:[-3.31 3.31]
ctrl_limits:[6.98 0 1]
}
joint joint_6 (link_5 link_6) {  
type:hingeX
axis:[-1 0 0]
A:<t(0.100 0 0) E(0 0 0)>
limits:[-6.28 6.28]
ctrl_limits:[10.47 0 1]
}
joint joint_6 ool0 (link_6 tool0) {  
type:rigid
A:<t(0 0 0) E(3.1415926535 -1.570796327 0)>
}
joint base_link ase (base_link base) {  
type:rigid
A:<t(0 0 0.450) E(0 0 0)>
} 