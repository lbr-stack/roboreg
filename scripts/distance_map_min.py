import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import theseus as th
import torch
import torchlie as lie
import transformations as tf
from rich import print

from roboreg.util import generate_o3d_robot, normalized_distance_transform

# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    prefix = "test/data/high_res"

    # camera intrinsics
    width = 960
    height = 540
    intrinsic_matrix = np.array(
        [
            [533.9981079101562, 0.0, 478.0845642089844],
            [0.0, 533.9981079101562, 260.9956970214844],
            [0.0, 0.0, 1.0],
        ]
    )

    # load robot and generate render given homogeneous transform
    HT_base_cam = np.load(
        os.path.join(prefix, "HT_hydra_robust.npy")
    )  # base frame (reference / world) -> camera (we take this as the initial guess)

    # static transforms
    HT_cam_optical = tf.quaternion_matrix([0.5, -0.5, 0.5, -0.5])  # camera -> optical

    # base to optical frame
    HT_base_optical = HT_base_cam @ HT_cam_optical  # base frame -> optical
    HT_optical_base = np.linalg.inv(HT_base_optical)

    robot = generate_o3d_robot()

    # load data
    visualize = False

    def plot_render(idx: int):
        mask = cv2.imread(os.path.join(prefix, f"mask_{idx}.png"), cv2.IMREAD_GRAYSCALE)
        norm_dist = normalized_distance_transform(mask)
        joint_state = np.load(os.path.join(prefix, f"joint_state_{idx}.npy"))

        # render
        robot.set_joint_positions(joint_state)
        render = robot.render(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=HT_optical_base,
            width=width,
            height=height,
        )
        threshold = 50
        binary_render = np.where(
            cv2.cvtColor(render, cv2.COLOR_RGB2GRAY) > threshold, 255, 0
        ).astype(np.uint8)

        # visualize
        cv2.imshow("mask", mask)
        cv2.imshow("norm_dist", norm_dist)
        cv2.imshow("binary_render", binary_render)
        cv2.imshow("diff", np.abs(binary_render - mask))
        cv2.waitKey(0)

    if visualize:
        for idx in range(7):
            plot_render(idx)

    # "shadow-cast": we are not interested in the actual render, but rather the projection of the mesh into the image
    # we therefore do not care where points from the mesh are sampled from
    print("Sampling points from mesh...")
    idx = 2
    joint_state = np.load(os.path.join(prefix, f"joint_state_{idx}.npy"))
    robot.set_joint_positions(joint_state)
    pcds = robot.sample_point_clouds(number_of_points_per_link=1000)
    # print("Visualizeing sampled points...")
    # plot_render(idx)

    # point clouds to torch
    print("Converting point clouds to torch...")
    pcds = [
        torch.from_numpy(np.array(pcd.points)).to(device=device, dtype=torch.float32)
        for pcd in pcds
    ]
    pcd = torch.concatenate(pcds, dim=0)
    print(f"Got point cloud of shape: {pcd.shape}.")

    # turn all to torch
    HT_optical_base = torch.from_numpy(HT_optical_base).to(
        device=device, dtype=torch.float32
    )

    HT_base_optical = torch.from_numpy(HT_base_optical).to(
        device=device, dtype=torch.float32
    )

    HT_base_cam = torch.from_numpy(HT_base_cam).to(device=device, dtype=torch.float32)
    HT_cam_base = torch.linalg.inv(HT_base_cam)

    intrinsic_matrix = torch.from_numpy(intrinsic_matrix).to(
        device=device, dtype=torch.float32
    )

    # project onto image plane (i.e. render)
    print("Projecting points onto image plane...")
    print("HT_base_cam:\n", HT_base_cam)
    print("HT_base_optical:\n", HT_base_optical)
    print("HT_optical_base:\n", HT_optical_base)

    HT_optical_base[0, 3] += 0.3  # shift a little along x-axis to increase error
    # HT_optical_base[1, 3] += 0.3  # shift a little along x-axis to increase error
    HT_optical_base_lie = th.SE3(tensor=HT_optical_base[:3, :].unsqueeze(0))
    HT_optical_base_lie_vec = th.SE3.log_map(HT_optical_base_lie)

    HT_optical_base_lie_recovered = th.SE3.exp_map(HT_optical_base_lie_vec)

    print("HT_optical_base_lie:\n", HT_optical_base_lie)
    print("HT_optical_base_lie_recovered:\n", HT_optical_base_lie_recovered)

    print("pcd shape: ", pcd.shape)
    print("HT_optical_base_lie_recovered shape: ", HT_optical_base_lie_recovered.shape)

    # p_prime_hat = (
    #     p.tensor @ Th.tensor[:, :3, :3].transpose(-1, -2) + Th.tensor[:, :3, 3]
    # )

    projected_pcd = torch.matmul(
        pcd @ HT_optical_base_lie_recovered[0, :3, :3].transpose(-2, -1)
        + HT_optical_base_lie_recovered[0, :3, 3],
        intrinsic_matrix.T,
    )
    projected_pcd = projected_pcd / projected_pcd[:, 2].unsqueeze(-1)

    plot_scatter = False
    if plot_scatter:
        print(f"Got projected point cloud of shape: {projected_pcd.shape}.")

        print("Visualizing projected points...")

        plt.scatter(
            projected_pcd[0, :].cpu().numpy(),
            -1 * projected_pcd[1, :].cpu().numpy() + height,
        )
        plt.xlim([0, width])
        plt.ylim([0, height])
        plt.show()

    # sample distance map @ projected_pcd
    uv_pcd = projected_pcd[:, :2]

    # these need to be normalized [-1 , 1]
    uv_pcd[:, 0] = (uv_pcd[:, 0] / width) * 2 - 1
    uv_pcd[:, 1] = (uv_pcd[:, 1] / height) * 2 - 1

    # nor
    mask = cv2.imread(os.path.join(prefix, f"mask_{idx}.png"), cv2.IMREAD_GRAYSCALE)
    norm_dist = normalized_distance_transform(mask)

    norm_dist = torch.from_numpy(norm_dist).to(device=device, dtype=torch.float32)

    values = torch.nn.functional.grid_sample(
        norm_dist.unsqueeze(0).unsqueeze(0), uv_pcd.unsqueeze(0).unsqueeze(0)
    )

    print("values shape: ", values.shape)
    print("values min:   ", values.min())
    print("values max:   ", values.max())
    print("values mean:  ", values.mean())

    # plot values via scatter of uv_pcd and values at location
    plot_distance_scatter = True
    if plot_distance_scatter:
        print("Visualizing distance map sampling...")
        plt.scatter(
            uv_pcd[:, 0].cpu().numpy(),
            -1 * uv_pcd[:, 1].cpu().numpy(),
            c=values.squeeze(0).squeeze(0).cpu().numpy(),
        )
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.legend()
        plt.show()

    ## optim vars are rotation R in R^3 and translation t in R^3
    ## auxiliary vars are distance map d_map, intrinsic matrix K, and point cloud pcd

    ### exp(Th)p where Th in se(3) and p in R^3
    ### optimization yields Th vector in R^6 -> exp(Th) in SE(3)

    ## not differentiable?????
    def error_fn(optim_vars, aux_vars):
        Th_vec = optim_vars[0]
        K, d_map, pcd = aux_vars

        # project pcd
        Th = th.SE3.exp_map(Th_vec.tensor)
        pcd_proj = torch.matmul(
            pcd.tensor @ Th.tensor[:, :3, :3].transpose(-2, -1) + Th.tensor[:, :3, 3],
            K.tensor.transpose(-1, -2),
        )

        # normalize
        pcd_proj = pcd_proj / pcd_proj[..., 2].unsqueeze(-1)

        # sample distance map @ projected_pcd
        pcd_proj_norm = pcd_proj[..., :2]

        # these need to be normalized [-1 , 1]
        pcd_proj_norm[..., 0] = (pcd_proj_norm[..., 0] / width) * 2 - 1
        pcd_proj_norm[..., 1] = (pcd_proj_norm[..., 1] / height) * 2 - 1

        # sample from distance map at projections
        d = torch.nn.functional.grid_sample(
            d_map.tensor.unsqueeze(0),
            pcd_proj_norm.unsqueeze(0),
        ).squeeze(0, 1)

        # return error
        return d

    K_var = th.Variable(intrinsic_matrix.unsqueeze(0), name="K")
    d_map_var = th.Variable(norm_dist.unsqueeze(0), name="d_map")
    pcd_var = th.Variable(pcd.unsqueeze(0), name="pcd")
    Th_vec = th.Vector(dof=6, name="Th_vec")

    objective = th.Objective()
    cost_fn = th.AutoDiffCostFunction(
        optim_vars=[Th_vec],
        err_fn=error_fn,
        dim=pcd.shape[0],
        cost_weight=th.ScaleCostWeight(1.0),
        aux_vars=[K_var, d_map_var, pcd_var],
        name="cost_fn",
    )
    objective.add(cost_fn)
    optimizer = th.LevenbergMarquardt(objective, max_iteration=15, step_size=1.0)

    layer = th.TheseusLayer(optimizer=optimizer)
    layer.to(device=device)

    # HT_optical_base[
    #     0, 3
    # ] += 0.1  # shift a little along x-axis to increase error on initial guess

    # HT_optical_base[
    #     1, 3
    # ] += 0.1  # shift a little along y-axis to increase error on initial guess

    print("-----------------------------------------------")

    Th = lie.SE3(HT_optical_base[:3, :])
    input = {
        "K": K_var.tensor,
        "d_map": d_map_var.tensor,
        "pcd": pcd_var.tensor,
        "Th_vec": Th.log().unsqueeze(0),
    }

    print("input: ", input)

    optimizer.objective.update(input)
    jacobians, error = cost_fn.jacobians()
    print(jacobians)
    # non-zeros
    print("non-zeros: ", jacobians[0].nonzero().sum())

    print("test")

    # with torch.no_grad():
    #     output, info = layer.forward(
    #         input, optimizer_kwargs={"track_best_solution": True, "verbose": True}
    #     )  # runs entire optimization

    # print("best solution: ", info.best_solution)
    # print(output)
    # print(info)
