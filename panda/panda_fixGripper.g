
Include: '../panda/panda.g'

Edit panda_joint2 { q= -.5 }
Edit panda_joint4 { q= -2 }

## delete original gripper

Delete panda_hand_0
Delete panda_leftfinger_0
Delete panda_rightfinger_0

#Include: 'gripper.g'

gripper (panda_joint7){ Q:<d(-90 0 1 0) d(135 0 0 1) t(0 0 -.155)> }
gripperCenter (gripper){ shape:marker, size:[.03], color:[.9 .9 .9], Q:<t(0 0 -.055)> }

Edit panda_finger_joint1{ Q:<t(.05 0 0)> joint:rigid }
Edit panda_finger_joint2{ Q:<t(.05 0 0)> joint:rigid }

finger1(panda_finger_joint1){ Q:<t(.018 0 .035)> contact: -2, shape:capsule, size:[.02, .02], color:[.9 .9 .9 .5] }
finger2(panda_finger_joint2){ Q:<t(.018 0 .035)>contact: -2, shape:capsule, size:[.02, .02], color:[.9 .9 .9 .5]}
        
