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


if __name__ == "__main__":
    test_lie_group()
