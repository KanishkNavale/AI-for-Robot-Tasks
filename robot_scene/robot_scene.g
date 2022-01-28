# Call in the Solidworks Assembly ---------------------------------------------------
world {}

base (world){
    shape:mesh, color:[0.25098, 0.25098, 0.25098, 1], mesh:'meshes/base.STL'
    Q:<t(0.0 0.0 0.02)>, fixed, contact, logical:{ }
}

pedestal (base){
    shape:mesh, color:[1, 1, 1, 1], mesh:'meshes/pedestal.STL'
    Q:[0, -0.57, 0.05, 0.707105, 0.707108, 0, 0],
    joint:rigid
}



camera_stand (base){
    shape:mesh, color:[1, 1, 1, 1], mesh:'meshes/camera_stand.STL'
    Q:[0, 0.525, 0.05, 0, 0, 0.707108, 0.707105],
    joint:rigid
}


# Add the robot ---------------------------------------------------------------------
Include 'panda/panda_fixGripper.g'

Edit gripper {
    contact:-2
}

joint (pedestal panda_link0){
    joint:rigid Q:<t(0 0.59 -0.05) E(-1.5707 1.5707 0)>
}

# Add the camera --------------------------------------------------------------------
camera(world){
    Q:<t(0.0 0.0 2.05)>,
    shape:marker, size:[.1],
    focalLength:0.895, width:640, height:360, zRange:[.5 2]
}

