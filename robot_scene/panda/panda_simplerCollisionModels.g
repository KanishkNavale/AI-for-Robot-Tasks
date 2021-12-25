Delete panda_link0_0
Delete panda_link1_0
Delete panda_link2_0
Delete panda_link3_0
Delete panda_link4_0
Delete panda_link5_0
Delete panda_link6_0
Delete panda_link7_0
Delete panda_hand_0
Delete panda_leftfinger_0
Delete panda_rightfinger_0

frame panda_coll0(panda_link0)	{ shape:capsule color:[1.,1.,1.,.2] size:[.1 .1] Q:<t(-.04 .0 .03) d(90 0 1 0)>, noVisual, contact:-2  }

frame panda_coll1(panda_joint1)	{ shape:capsule color:[1.,1.,1.,.2] size:[.2 .08] Q:<d(90 0 1 0) t(0 0 -.15)>, noVisual, contact:-2  }
frame panda_coll2(panda_joint3)	{ shape:capsule color:[1.,1.,1.,.2] size:[.2 .08] Q:<d(90 0 1 0) t(0 0 -.15)>, noVisual, contact:-2  }
frame panda_coll3(panda_joint5)	{ shape:capsule color:[1.,1.,1.,.2] size:[.22 .08] Q:<d(90 0 1 0) t(0 .02 -.2)>, noVisual, contact:-2  }

frame panda_coll4(panda_joint2)	{ shape:capsule color:[1.,1.,1.,.2] size:[.12 .08] Q:<d(90 0 1 0) t(0 0 .0)>, noVisual, contact:-2  }
frame panda_coll5(panda_joint4)	{ shape:capsule color:[1.,1.,1.,.2] size:[.12 .08] Q:<d(90 0 1 0) t(0 0 .0)>, noVisual, contact:-2  }
frame panda_coll6(panda_joint6)	{ shape:capsule color:[1.,1.,1.,.2] size:[.1 .07] Q:<d(90 0 1 0) t(0 .0 -.04)>, noVisual, contact:-2  }
frame panda_coll7(panda_joint7)	{ shape:capsule color:[1.,1.,1.,.2] size:[.1 .07] Q:<d(90 0 1 0) t(0 .0 .01)>, noVisual, contact:-2  }

frame panda_coll_hand(panda_hand_joint)	{ shape:capsule color:[1.,1.,1.,.2] size:[.15 .06] Q:<d(90 1 0 0) t(0 .01 .0)>, noVisual, contact:-2  }

frame panda_coll_finger1(panda_finger_joint1)	{ shape:capsule color:[1.,1.,1.,.2] size:[.03 .015] Q:<d(0 1 0 0) t(.015 .0 .03)>, noVisual, contact:-2  }
frame panda_coll_finger2(panda_finger_joint2)	{ shape:capsule color:[1.,1.,1.,.2] size:[.03 .015] Q:<d(0 1 0 0) t(.015 .0 .03)>, noVisual, contact:-2  }

