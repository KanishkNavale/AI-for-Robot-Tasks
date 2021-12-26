panda_link0 	{  }
panda_link0_1(panda_link0) 	{  shape:mesh  mesh:'meshes/visual/link0.ply' color:[.9 .9 .9] }

panda_link0>panda_joint1(panda_link0) 	{  Q:<0 0 0.14 0.707107 0 -0.707107 0> }
deform1(panda_link0>panda_joint1) { shape:marker, size:[.05] }
panda_joint1(deform1) 	{  joint:hingeX ctrl_H:1 limits=[  -2.8973 2.8973 2.175 87 1  ]  ctrl_limits:[ 2.175 87 1 ] }
panda_link1_1(panda_joint1) 	{  shape:mesh  mesh:'meshes/visual/link1.ply' Q:<.193 0 0 -0.707107 0 -0.707107 0>  color:[.9 .9 .9] }

panda_link1>panda_joint2(panda_joint1) 	{  Q:<.193 0 0 -0.707107 0 0 -0.707107> }
deform2(panda_link1>panda_joint2) { shape:marker, size:[.05] }
panda_joint2(deform2) 	{  joint:hingeX ctrl_H:1 limits=[  -1.7628 1.7628 2.175 87 1  ]  ctrl_limits:[ 2.175 87 1 ] }
panda_link2_1(panda_joint2) 	{  shape:mesh  mesh:'meshes/visual/link2.ply' Q:<0 0 0 -0.707107 0 -0.707107 0>  color:[.9 .9 .9] }

panda_link2>panda_joint3(panda_joint2) 	{   Q:<0 -0.19 0 -0.707107 0 0 0.707107> }
deform3(panda_link2>panda_joint3) { shape:marker, size:[.05] }
panda_joint3(deform3) 	{  joint:hingeX ctrl_H:1 limits=[  -2.8973 2.8973 2.175 87 1  ]  ctrl_limits:[ 2.175 87 1 ] }
panda_link3_1(panda_joint3) 	{  shape:mesh  mesh:'meshes/visual/link3.ply' Q:<.126 0 0 -0.707107 0 -0.707107 0>  color:[.9 .9 .9] }

panda_link3>panda_joint4(panda_joint3) 	{  Q:<.126 0 -0.0825 -0.707107 0 0 0.707107> }
deform4(panda_link3>panda_joint4){ shape:marker, size:[.05] }
panda_joint4(deform4) 	{  joint:hingeX ctrl_H:1 limits=[  -3.0718 -0.0698 2.175 87 1  ]  ctrl_limits:[ 2.175 87 1 ] }
panda_link4_1(panda_joint4) 	{  shape:mesh  mesh:'meshes/visual/link4.ply' Q:<0 0 0 -0.707107 0 -0.707107 0>  color:[.9 .9 .9] }

panda_link4>panda_joint5(panda_joint4) 	{  Q:<0 0.13 0.0825 -0.707107 0 0 -0.707107> }
deform5(panda_link4>panda_joint5) { shape:marker, size:[.05] }
panda_joint5(deform5) 	{  joint:hingeX ctrl_H:1 limits=[  -2.8973 2.8973 2.61 12 1  ]  ctrl_limits:[ 2.61 12 1 ] }
panda_link5_1(panda_joint5) 	{  shape:mesh  mesh:'meshes/visual/link5.ply' Q:<.254 0 0 -0.707107 0 -0.707107 0>  color:[.9 .9 .9] }

panda_link5>panda_joint6(panda_joint5) 	{  Q:<0.254 0 0 -0.707107 0 0 0.707107> }
deform6(panda_link5>panda_joint6){ shape:marker, size:[.05] }
panda_joint6(deform6) 	{  joint:hingeX ctrl_H:1 limits=[  -0.0175 3.7525 2.61 12 1  ]  ctrl_limits:[ 2.61 12 1 ] }
panda_link6_1(panda_joint6) 	{  shape:mesh  mesh:'meshes/visual/link6.ply' Q:<0 0 0 -0.707107 0 -0.707107 0>  color:[.9 .9 .9] }

panda_link6>panda_joint7(panda_joint6) 	{  Q:<1.95399e-17 0 -0.088 -0.707107 0 0 0.707107> }
deform7(panda_link6>panda_joint7){ shape:marker, size:[.05] }
panda_joint7(deform7) 	{  joint:hingeX ctrl_H:1 limits=[  -2.8973 2.8973 2.61 12 1  ]  ctrl_limits:[ 2.61 12 1 ] }
panda_link7_1(panda_joint7) 	{  shape:mesh  mesh:'meshes/visual/link7.ply' Q:<0 0 0 -0.707107 0 -0.707107 0>  color:[.9 .9 .9] }

panda_link7>panda_joint8(panda_joint7) 	{  Q:<0.107 0 2.37588e-17 -0.707107 0 -0.707107 0> }
panda_joint8(panda_link7>panda_joint8) 	{  joint:rigid ctrl_H:1 }

panda_link8>panda_hand_joint(panda_joint8) 	{  Q:<0 0 0 0.92388 0 0 -0.382683> }
panda_hand_joint(panda_link8>panda_hand_joint) 	{  joint:rigid ctrl_H:1 }
panda_hand_1(panda_hand_joint) 	{  shape:mesh  mesh:'meshes/visual/hand.ply'  color:[.9 .9 .9] }
panda_hand>panda_finger_joint1(panda_hand_joint) 	{  Q:<0 0 0.0584 0.707107 0 0 0.707107> }
panda_hand>panda_finger_joint2(panda_hand_joint) 	{  Q:<0 0 0.0584 0.707107 0 0 -0.707107> }
panda_finger_joint1(panda_hand>panda_finger_joint1) 	{  joint:transX ctrl_H:1 limits=[  0 0.04 0.2 20 1  ]  ctrl_limits:[ 0.2 20 1 ] }
panda_finger_joint2(panda_hand>panda_finger_joint2) 	{  joint:transX ctrl_H:1 limits=[  0 0.04 0.2 20 1  ] mimic:(panda_finger_joint1)  ctrl_limits:[ 0.2 20 1 ] }
panda_leftfinger_1(panda_finger_joint1) 	{  shape:mesh  mesh:'meshes/visual/finger.ply' Q:<0 0 0 -0.707107 0 0 0.707107>  color:[.9 .9 .9] }
panda_rightfinger_1(panda_finger_joint2) 	{  shape:mesh  mesh:'meshes/visual/finger.ply' Q:<0 0 0 -0.707107 0 0 0.707107>  color:[.9 .9 .9] }


## collision models

#panda_coll1(panda_joint1)	{ shape:sphere color:[1.,1.,1.,.2] size:[.1] Q:<t(-.1 0 0)>, contact:1 }
#panda_coll2(panda_joint2)	{ shape:sphere color:[1.,1.,1.,.2] size:[.1] Q:<t(0 0 0)>, contact:1  }
#panda_coll3(panda_joint4)	{ shape:sphere color:[1.,1.,1.,.2] size:[.1] Q:<t(0 0 0)>, contact:1  }
#panda_coll4(panda_joint6)	{ shape:sphere color:[1.,1.,1.,.2] size:[.1] Q:<t(0 0 0)>, contact:1  }

#panda_coll4(panda_joint2)	{ shape:capsule color:[1.,1.,1.,.2] size:[.12 .08] Q:<d(90 0 1 0) t(0 0 .0)>, noVisual, contact:-2  }
#panda_coll5(panda_joint4)	{ shape:capsule color:[1.,1.,1.,.2] size:[.12 .08] Q:<d(90 0 1 0) t(0 0 .0)>, noVisual, contact:-2  }
#panda_coll6(panda_joint6)	{ shape:capsule color:[1.,1.,1.,.2] size:[.1 .07] Q:<d(90 0 1 0) t(0 .0 -.04)>, noVisual, contact:-2  }
#panda_coll7(panda_joint7)	{ shape:capsule color:[1.,1.,1.,.2] size:[.1 .07] Q:<d(90 0 1 0) t(0 .0 .01)>, noVisual, contact:-2  }

#panda_coll_hand(panda_hand_joint)	{ shape:capsule color:[1.,1.,1.,.2] size:[.15 .06] Q:<d(90 1 0 0) t(0 .01 .0)>, noVisual, contact:-2  }

#panda_coll_finger1(panda_finger_joint1)	{ shape:capsule color:[1.,1.,1.,.2] size:[.03 .015] Q:<d(0 1 0 0) t(.015 .0 .03)>, noVisual, contact:-2  }
#panda_coll_finger2(panda_finger_joint2)	{ shape:capsule color:[1.,1.,1.,.2] size:[.03 .015] Q:<d(0 1 0 0) t(.015 .0 .03)>, noVisual, contact:-2  }


        
## zero position

Edit panda_joint1 { q= 0.0 }
Edit panda_joint2 { q= -1. }
Edit panda_joint3 { q= 0. }
Edit panda_joint4 { q= -2.}
Edit panda_joint5 { q= -0. }
Edit panda_joint6 { q= 2. }
Edit panda_joint7 { q= 0.0 }
Edit panda_finger_joint1 { q=.05 }
