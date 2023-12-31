import schnetpack as spk
import schnetpack.transform as trn
import schnetpack.ipu_modules as schnet_ipu_modules


def create_model(
        n_atom_basis,
        n_rbf,
        k_neighbors,
        n_atoms,
        n_batches,
        rbf_cutoff=5.,
        n_interactions=3,
        constant_batch_size=True,
        energy_key="energy",
        forces_key="forces"
):
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=energy_key)
    pred_forces = spk.atomistic.Forces(energy_key=energy_key, force_key=forces_key)

    neighbor_distance = schnet_ipu_modules.KNNNeighborTransform(k_neighbors, n_batches, n_atoms)

    pairwise_distance = schnet_ipu_modules.PairwiseDistancesIPU()

    radial_basis = schnet_ipu_modules.GaussianRBFIPU(n_rbf=n_rbf, cutoff=rbf_cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=schnet_ipu_modules.DummyCutoff(rbf_cutoff),
        activation=schnet_ipu_modules.ShiftedSoftplusIPU(),
        n_neighbors=k_neighbors,
    )

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[trn.CastTo32(), neighbor_distance, pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
        ]
    )

    return nnpot
