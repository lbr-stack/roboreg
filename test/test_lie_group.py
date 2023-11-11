import theseus as th
import torch
import torchlie as lie


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


def test_optimizer_on_lie_group():
    # a translation target transform
    Th_target = lie.SE3.exp(torch.tensor([0.4, 0.5, 0.6, 0.0, 0.0, 0.0]))

    # some point in R^3
    p = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
    p_prime = Th_target @ p

    # setup optimization variables
    p_var = th.Variable(p, name="p")
    p_prime_var = th.Variable(p_prime, name="p_prime")
    Th_vec_var = th.Vector(dof=6, name="Th_vec")

    # specify cost function
    def cost_fn(optim_vars, aux_vars):
        Th_vec = optim_vars[0]
        p, p_prime = aux_vars
        Th = lie.SE3.exp(Th_vec.tensor)

        p_prime_hat = Th @ p.tensor
        err = p_prime_hat - p_prime.tensor
        return err

    # setup objective
    objective = th.Objective()
    cost_fn = th.AutoDiffCostFunction(
        optim_vars=[Th_vec_var],
        err_fn=cost_fn,
        dim=3,
        aux_vars=[p_var, p_prime_var],
        cost_weight=th.ScaleCostWeight(1.0),
        name="cost_fn",
    )
    objective.add(cost_fn)
    optimizer = th.LevenbergMarquardt(objective, max_iteration=15, step_size=1.0)

    layer = th.TheseusLayer(optimizer=optimizer)

    with torch.no_grad():
        output, info = layer.forward(
            input_tensors={
                "Th_vec": torch.zeros(6).unsqueeze(0)
            },  # zeros as initial guess
            optimizer_kwargs={"track_best_solution": True, "verbose": True},
        )

    print("best solution: ", info.best_solution)
    print(output)


if __name__ == "__main__":
    # test_lie_group()
    test_optimizer_on_lie_group()
