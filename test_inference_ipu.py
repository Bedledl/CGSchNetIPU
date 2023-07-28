import schnetpack
import torch
import poptorch
from schnetpack.model import AtomisticModel

from utils.create_model import create_model

#create model parameter:
n_atom_basis = 20#32
energy_key = "energy"
forces_key = "forces"
n_rbf = 20
cutoff = 5.
coords = "data/chignolin_ca_coords.npy"
forces = "data/chignolin_ca_forces.npy"
embeddings = "data/chignolin_ca_embeddings.npy"
n_neighbors = 5
n_atoms = 10# TODO only for chgnolin
n_batches = 1

#other parameter
cutoff = 5.
cutoff_shell = 2.
model_path = "train_chignolin/best_inference_model"
device = "cpu"
time_step = 0.5
pdb_file = "data/chignolin_cln025.pdb"
topology = "data/chignolin_ca_top.psf"
coordinates = "data/chignolin_ca_initial_coords.xtc"
forcefield = "data/chignolin_priors_fulldata.yaml"
initial_temperature = 300
log_file = "sim_chognolin_log.log"


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
    model = create_model(
        n_atom_basis=n_atom_basis,
        n_rbf=n_rbf,
        k_neighbors=n_neighbors,
        n_atoms=n_atoms,
        n_batches=n_batches,
        rbf_cutoff=cutoff,
        n_interactions=3,
        constant_batch_size=True
    )

    model.to(torch.float32)
    model.eval()
    model = deactivate_postprocessing(model)

    ipu_model = poptorch.inferenceModel(model)

    example_input_in = {'_atomic_numbers': torch.tensor([4, 4, 5, 8, 6, 13, 2, 13, 7, 4]),
                        '_n_atoms': torch.tensor([10]),
                        '_positions': torch.tensor([[0.5590, 0.0547, 0.3513],
                                                    [0.2491, -0.2769, -0.1848],
                                                    [-0.2806, -0.0103, 0.1212],
                                                    [-0.4281, -0.3007, 0.2916],
                                                    [-0.8361, -0.1767, 0.1711],
                                                    [-0.5605, 0.0261, -0.2235],
                                                    [-0.2587, -0.1896, -0.3444],
                                                    [-0.1374, 0.1214, -0.3597],
                                                    [0.4958, 0.1250, -0.2471],
                                                    [0.2307, 0.3432, 0.1813]]),
                        '_idx_m': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        '_cell': torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]),
                        '_pbc': torch.tensor([[False, False, False]]),
                        '_offsets': torch.tensor([[0, 0, 0]]).repeat(10 * n_neighbors, 1),
                        '_n_molecules': 1}
    print(ipu_model(example_input_in))


if __name__ == '__main__':
    main()
