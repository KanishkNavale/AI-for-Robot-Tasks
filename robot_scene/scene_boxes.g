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

table (base){
    shape:mesh, color:[1, 1, 1, 1], mesh:'meshes/table.STL'
    Q:[0, 0, 0.91, 0.707105, 0.707108, 0, 0],
    joint:rigid
}

camera_stand (base){
    shape:mesh, color:[1, 1, 1, 1], mesh:'meshes/camera_stand.STL'
    Q:[0, 0.525, 0.05, 0, 0, 0.707108, 0.707105],
    joint:rigid
}

# Add Collision Models --------------------------------------------------------------

base (base){
    shape:mesh, color:[1, 0, 0, 1], mesh:'meshes/base.STL'
    Q:<d(0 0 1 0) t(0 0 .0)>, noVisual, contact:-2 
}

pedestal (pedestal){
    shape:mesh, color:[1, 0, 0, 1], mesh:'meshes/pedestal.STL'
    Q:<d(0 0 1 0) t(0 0 .0)>, noVisual, contact:-2  
}

table (table){
    shape:mesh, color:[1, 0, 0, 1], mesh:'meshes/table.STL'
    Q:<d(0 0 1 0) t(0 0 .0)>, noVisual, contact:-2 
}

camera_stand (camera_stand){
    shape:mesh, color:[1, 0, 0, 1], mesh:'meshes/camera_stand.STL'
    Q:<d(0 0 1 0) t(0 0 .0)>, noVisual, contact:-2 
}

### Boxes
box1 (table)	{  shape:ssBox, size:[0.2, 0.01, 0.2, 0.01],color:[1,0,0], mass:0.2, Q:<t(-0.15 0.04 -0.15)> }

box2 (table)	{  shape:ssBox, size:[0.2, 0.01, 0.2, 0.01],color:[0,0,1], mass:0.2, Q:<t(0.15 0.04 -0.15)> }


# Add the cameras ---------------------------------------------------------------------
camera1 (world){
     shape:ssBox, size:[0.2, 0.1, 0.1, 0.01],
     color:[1,0,0], mass:0.2, Q:<t(0. 0. 1.)>
}
