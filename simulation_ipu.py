import schnetpack
import torch
from schnetpack import properties
from schnetpack.md import Simulator, UniformInit, System
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.simulation_hooks import MoleculeStream, PropertyStream, FileLogger
from schnetpack.model import AtomisticModel
from schnetpack.ipu_modules import IPUCalculator, MultiTempLangevinThermostat

from utils.create_model import create_model

from utils.molecule_conversion import get_moleculekit_obj, moleculekit2ase

run_on_ipu = True

if torch.cuda.is_available():
    cuda = True
    print("Cuda is available. Try to run model on GPU")
else:
    print("Cuda is not available. If no IPU run is configured, the model runs on CPU.")
    cuda = False

#create model parameter:
n_atom_basis = 32
energy_key = "energy"
forces_key = "forces"
n_rbf = 20
cutoff = 5.
n_neighbors = 5
n_batches = 1

#other parameter
cutoff = 5.
cutoff_shell = 2.
model_path = "train_chignolin/best_inference_model"
device = "cpu"
time_step = 0.5
#pdb_file = "data/chignolin_cln025.pdb"
pdb_file = "data/albumin.pdb"
initial_temperature = 300
log_file = "sim_chognolin_log.log"
energy_log_file = "chignolin_energy_log.log"


def deactivate_postprocessing(model: AtomisticModel) -> AtomisticModel:
    """
    This is a copy from the Calcuator class of SchNet. When using a cacluator
    :param model:
    :return:
    """
    if hasattr(model, "postprocessors"):
        for pp in model.postprocessors:
            if isinstance(pp, schnetpack.transform.AddOffsets):
                print("Found `AddOffsets` postprocessing module...")
                print(
                    "Constant offset of {:20.11f} per atom  will be removed...".format(
                        pp.mean.detach().cpu().numpy()
                    )
                )
    model.do_postprocessing = False
    return model


def main():

    # get molecule object
    mk_mol = get_moleculekit_obj(pdb_file)
    ase_mol = moleculekit2ase(mk_mol)
    n_atoms = ase_mol.get_number_of_atoms()

    # create model
    model = create_model(
        n_atom_basis=n_atom_basis,
        n_rbf=n_rbf,
        k_neighbors=n_neighbors,
        n_atoms=n_atoms,
        n_batches=n_batches,
        rbf_cutoff=cutoff,
        n_interactions=3,
        constant_batch_size=False
    )

    model.to(torch.float32)
    model.eval()
    model = deactivate_postprocessing(model)

    if cuda:
        model.to("cuda:0")




    # now create Simulation environment
    md_calculator = IPUCalculator(
        model,
        "forces",  # force key
        "kcal/mol",  # energy units
        "Angstrom",  # length units
        energy_key="energy",  # name of potential energies
        required_properties=[],  # additional properties extracted from the model
        run_on_ipu=run_on_ipu,
        n_atoms=n_atoms,
        n_molecules=n_batches,
        n_neighbors=n_neighbors
    )
    md_system = System()
    md_system.load_molecules(
        ase_mol,
        n_batches,
        position_unit_input="Angstrom"
    )

    md_initializer = UniformInit(
        initial_temperature,
        remove_center_of_mass=True,
        remove_translation=True,
        remove_rotation=True,
    )
    md_initializer.initialize_system(md_system)

    md_integrator = VelocityVerlet(time_step)

    thermostat = MultiTempLangevinThermostat(torch.linspace(100, 800, n_batches), 10)

    buffer_size = 100
    # Set up data streams to store positions, momenta and the energy
    data_streams = [
        MoleculeStream(store_velocities=True),
        PropertyStream(target_properties=[properties.energy]),
    ]

    file_logger = FileLogger(
        energy_log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=1,  # logging frequency
        precision=32,  # floating point precision used in hdf5 database
    )

    simulator_hooks = [
        thermostat,
        file_logger
    ]

    md_simulator = Simulator(
        md_system,
        md_integrator,
        md_calculator,
        simulator_hooks=simulator_hooks
    )

    md_simulator.simulate(10)


if __name__ == '__main__':
    main()
