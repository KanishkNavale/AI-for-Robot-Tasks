base 	{ , mass:65.965, inertiaTensor:[15.345, 2.2841e-17, 5.5511e-16, 18.762, 8.2978e-17, 3.4445] }
base_1 (base) 	{ , shape:mesh, color:[0.25098, 0.25098, 0.25098, 1], mesh:'meshes/base.STL' }
base_0 (base) 	{ , shape:mesh, color:[0.8, 0.2, 0.2, 0.5], mesh:'meshes/base.STL', contact, fixed, logical:{}, friction:.1}
pedestal (base) 	{  Q:[0, -0.57, 0.05, 0.707105, 0.707108, 0, 0], joint:rigid }
table (base) 	{  Q:[0, 0, 0.91, 0.707105, 0.707108, 0, 0], joint:rigid }
base>camera_stand (base) 	{  Q:[0, 0.525, 0.05, -2.59734e-06, -2.59735e-06, 0.707108, 0.707105] }
pedestal (pedestal) 	{ , mass:24.425, inertiaTensor:[1.0652, -1.2821e-17, 4.2196e-18, 0.31302, -7.2506e-17, 1.0652] }
table (table) 	{ , mass:7.0683, inertiaTensor:[0.19397, 3.4959e-17, -2.0705e-18, 0.18317, 3.3108e-18, 0.19397] }
camera_stand (base>camera_stand) 	{ , joint:rigid }
pedestal_1 (pedestal) 	{ , shape:mesh, color:[1, 1, 1, 1], mesh:'meshes/pedestal.STL' }
pedestal_0 (pedestal) 	{ , shape:mesh, color:[0.8, 0.2, 0.2, 0.5], mesh:'meshes/pedestal.STL', contact:-2 }
table_1 (table) 	{ , shape:mesh, color:[1, 1, 1, 1], mesh:'meshes/table.STL' }
table_0 (table) 	{ , shape:mesh, color:[0.8, 0.2, 0.2, 0.5], mesh:'meshes/table.STL', contact:-2 }
camera_stand (camera_stand) 	{ , mass:0, inertiaTensor:[0, 0, 0, 0, 0, 0] }
camera_stand_1 (camera_stand) 	{ , shape:mesh, color:[1, 1, 1, 1], mesh:'meshes/camera_stand.STL' }
camera_stand_0 (camera_stand) 	{ , shape:mesh, color:[0.8, 0.2, 0.2, 0.5], mesh:'meshes/camera_stand.STL', contact:-2 }

Include 'panda/panda_fixGripper.g'
joint (pedestal_0 panda_link0){ joint:rigid Q:<t(0 0.57 0)> }