from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
import os, sys

#Usage python simulate.py pdbfile xmlfile

def describe_state(state: openmm.State, name: str = "State"):
    max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
    print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
          f"with maximum force {round(max_force, 2)} kJ/(mol nm)")

def write_structure(sim: Simulation, pdb_fn: str):
    with open(pdb_fn, 'w') as f:
        PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
    print(f'Wrote: {pdb_fn}')
#Simulation Parameters
n_total, n_dcd, n_std, timestep, temp = 10000000, 2000, 100, 0.5*femtosecond, 300*kelvin
#Load PDB File
pdb = PDBFile(sys.argv[1])
#Load XML file
with open(sys.argv[2]) as f:
    sys = XmlSerializer.deserialize(f.read())
#Define Integrator
integrator = LangevinIntegrator(temp, 1/picosecond, timestep)
#Assign Barostate
sys.addForce(MonteCarloBarostat(1.0*bar, temp, 100))
#Define Simulation and set initial positions
simulation = Simulation(pdb.topology, sys, integrator)
simulation.context.setPositions(pdb.positions)
#Write out the energy and structure of the initial state (for testing purposes)
describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Original state")
write_structure(simulation, 'pre_minimized.pdb')
#Minimize the energy
simulation.minimizeEnergy()
#WRite out again
describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Minimized state")
write_structure(simulation, 'test_translate_minimized.pdb')
#Set initial velocities
simulation.context.setVelocitiesToTemperature(temp)
#Standard Out reporter
SDR = StateDataReporter('test_join.stdout', n_std, step=True, time=True,
                        potentialEnergy=True, temperature=True, remainingTime=True,
                        totalSteps=n_total, separator='     ')
simulation.reporters.append(SDR)
#Trajectory Reporter (You need this to keep your coordinates!)
DCR = DCDReporter('test_trajectory.dcd', n_dcd)
simulation.reporters.append(DCR)
#Run the simulation
simulation.step(n_total)
