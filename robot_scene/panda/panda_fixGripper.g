## this should be the default panda we use on the real system
#  with NO dofs for the gripper

Include: 'panda.g'

# modify default home pose
Edit panda_joint2 { q= -.5 }
Edit panda_joint4 { q= -2 }

# delete original gripper
Delete panda_hand_0
Delete panda_leftfinger_0
Delete panda_rightfinger_0

# kill rigid hand joints
Edit panda_joint8 { joint:none }
Edit panda_hand_joint { joint:none }
        
# make fingers part of the gripper link
Edit panda_finger_joint1{ Q:<t(.05 0 0)> joint:none }
Edit panda_finger_joint2{ Q:<t(.05 0 0)> joint:none }

