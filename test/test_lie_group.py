import theseus as th
import torch
import torchlie as lie
import transformations as tf
from rich import print


def test_lie_group():
    rand_se3 = lie.SE3.rand()
    print(rand_se3)
    id = lie.SE3.identity()
    print(id)

    # get rotation from lie
    print(lie.SE3.rand(5).shape)

    # exp map
    v = torch.rand(6)
    print(v.shape)
    print(lie.SE3.exp(v))


def test_rotation_translation():
    # start with random homogeneous matrix
    HT = torch.from_numpy(tf.random_rotation_matrix())
    HT[:3, 3] = torch.rand(3)
    print("HT:\n", HT)
    print("HT[:3, :] shape: ", HT[:3, :].shape)

    # convert to lie group
    HT_lie = th.SE3(tensor=HT[:3, :].unsqueeze(0))

    # turn into tangent space
    HT_lie_vec = th.SE3.log_map(HT_lie)

    # recover
    HT_lie_recovered = th.SE3.exp_map(HT_lie_vec)

    print("HT_lie_recovered:\n", HT_lie_recovered)


def test_optimizer_on_lie_group():
    # a translation target transform
    Th_target = lie.SE3.exp(torch.tensor([1.0, 0.0, 0.0, torch.pi / 2.0, 0.0, 0.0]))

    # some point in R^3
    p = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    ).unsqueeze(0)
    p_prime = Th_target @ p

    print("p prime: ", p_prime)

    print("p shape: ", p.shape)
    print("p_prime shape: ", p_prime.shape)

    # setup optimization variables
    p_var = th.Variable(p, name="p")
    p_prime_var = th.Variable(p_prime, name="p_prime")
    Th_vec_var = th.Vector(dof=6, name="Th_vec")

    # specify cost function
    def cost_fn(optim_vars, aux_vars):
        Th_vec = optim_vars[0]
        p, p_prime = aux_vars

        Th = th.SE3.exp_map(Th_vec.tensor)

        p_prime_hat = (
            p.tensor @ Th.tensor[:, :3, :3].transpose(-1, -2) + Th.tensor[:, :3, 3]
        )

        err = torch.norm(torch.sub(p_prime_hat, p_prime.tensor), dim=-1)
        return err

    # setup objective
    objective = th.Objective()
    cost_fn = th.AutoDiffCostFunction(
        optim_vars=[Th_vec_var],
        err_fn=cost_fn,
        dim=6,
        aux_vars=[p_var, p_prime_var],
        cost_weight=th.ScaleCostWeight(1.0),
        name="cost_fn",
    )
    objective.add(cost_fn)
    optimizer = th.LevenbergMarquardt(objective, max_iteration=50, step_size=1.0)

    layer = th.TheseusLayer(optimizer=optimizer)

    with torch.no_grad():
        output, info = layer.forward(
            input_tensors={
                "Th_vec": torch.zeros(6).unsqueeze(0)
            },  # zeros as initial guess
            optimizer_kwargs={"track_best_solution": True, "verbose": True},
        )

    print("best solution:\n", info.best_solution["Th_vec"])
    print("target solution:\n", Th_target.log())


if __name__ == "__main__":
    # test_lie_group()
    # test_rotation_translation()
    test_optimizer_on_lie_group()
