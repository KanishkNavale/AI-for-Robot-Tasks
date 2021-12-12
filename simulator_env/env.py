import libry as lry

# Load the robot file
K = lry.Config()
D = K.view()
K.clear()
K.addFile('fanuc_description/fanuc.g')
