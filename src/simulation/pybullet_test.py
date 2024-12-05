import pybullet as p
physicsClient = p.connect(p.DIRECT)
p.setGravity(0,0,-10)
planeId = p.loadURDF("src/simulation/data/plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("src/simulation/data/r2d2.urdf",cubeStartPos, cubeStartOrientation)
p.stepSimulation()
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()