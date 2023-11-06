import copy

import faiss
import faiss.contrib.torch_utils
import numpy as np
import open3d as o3d
import torch


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw([source_temp, target_temp])
    o3d.visualization.draw_plotly([source_temp, target_temp])


def point_to_point_icp(source, target, threshold, trans_init):
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation, "\n")
    draw_registration_result(source, target, reg_p2p.transformation)


def point_to_plane_icp(source, target, threshold, trans_init):
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation, "\n")
    draw_registration_result(source, target, reg_p2l.transformation)


pcd_data = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(pcd_data.paths[0])
target = o3d.io.read_point_cloud(pcd_data.paths[1])
threshold = 0.02
trans_init = np.asarray(
    [
        [0.862, 0.4, -0.507, 0.5],
        [-0.139, 0.967, -0.215, 0.7],
        [0.487, 0.255, 0.835, -1.4],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
draw_registration_result(source, target, trans_init)
print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init
)
print(evaluation, "\n")


print(f"Running PyTorch version: {torch.__version__}")

torchdevice = torch.device("cpu")
if torch.cuda.is_available():
    torchdevice = torch.device("cuda")
    print("Default GPU is " + torch.cuda.get_device_name(torch.device("cuda")))
# torchdevice = xm.xla_device()
# torchdevice = torch.device('cpu')
print("Running on " + str(torchdevice))


# Get points as torch tensors (num_points, 3)
sourcepoints = torch.tensor(
    np.asarray(source.points), device=torchdevice, dtype=torch.float32
)
targetpoints = torch.tensor(
    np.asarray(target.points), device=torchdevice, dtype=torch.float32
)
# Convert input point clouds structures to
# padded tensors of shape (N, P, 3)
# Xt = torch.tensor(np.asarray(source.points), device=torchdevice).unsqueeze(0)
# Yt = torch.tensor(np.asarray(target.points), device=torchdevice).unsqueeze(0)

# Get unit normals for the target
targetnormals = torch.tensor(
    np.asarray(target.normals), device=torchdevice, dtype=torch.float32
)
# ncheck = torch.linalg.vector_norm(targetnormals, dim=1)
# print(ncheck[0:10])

numsourcepoints = sourcepoints.shape[0]
numtargetpoints = targetpoints.shape[0]
# num_points_X = Xt.shape[1]
# num_points_Y = Yt.shape[1]
# num_points_X = torch.tensor(Xt.shape[1], device=torchdevice)
# num_points_Y = torch.tensor(Yt.shape[1], device=torchdevice)

sourcecrossmats = torch.zeros(
    (numsourcepoints, 3, 3), device=torchdevice, dtype=torch.float32
)
sourcecrossmats[:, 0, 1] = -sourcepoints[:, 2]
sourcecrossmats[:, 0, 2] = sourcepoints[:, 1]
sourcecrossmats[:, 1, 0] = sourcepoints[:, 2]
sourcecrossmats[:, 1, 2] = -sourcepoints[:, 0]
sourcecrossmats[:, 2, 0] = -sourcepoints[:, 1]
sourcecrossmats[:, 2, 1] = sourcepoints[:, 0]


max_iterations = 100
max_inner_iterations = 3
threshold_sq = threshold**2


# index = torchpq.index.IVFPQIndex(
#  d_vector=3,
#  n_subvectors=64,
#  n_cells=1024,
#  initial_size=2048,
#  distance="euclidean",
# )

# trainset = torch.randn(d_vector, n_data, device="cuda:0")
# index.train(trainset)

# Exact search
# TODO: Try approximate nearest neighbor with FAISS
res = faiss.StandardGpuResources()
flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = 0
index = faiss.GpuIndexFlatL2(res, 3, flat_config)

# ivff_config = faiss.GpuIndexIVFFlatConfig()
# ivff_config.device = 0
# index = faiss.GpuIndexIVFFlat(res, 3, 100, faiss.METRIC_L2, ivff_config)

# index = faiss.index_factory(3, "IVF16384,Flat")
# co = faiss.GpuClonerOptions()
# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
# co.useFloat16 = True
# index = faiss.index_cpu_to_gpu(res, 0, index, co)

index.train(targetpoints)
index.add(targetpoints)

ptsourceidx = torch.arange(numsourcepoints, device=torchdevice)

Rot = torch.tensor(trans_init[0:3, 0:3], device=torchdevice, dtype=torch.float32)
Trans = torch.tensor(trans_init[0:3, 3], device=torchdevice, dtype=torch.float32)

dTh = torch.zeros((4, 4), device=torchdevice, dtype=torch.float32)

# print(Rot,Trans)

trans_id = np.eye(4)
trans_curr = copy.deepcopy(trans_id)
# trsfpoints = sourcepoints @ Rot.T + Trans

# source_temp = copy.deepcopy(source)
# source_temp.points = o3d.utility.Vector3dVector(trsfpoints.cpu().numpy())
# draw_registration_result(source_temp, target, trans_id)

# the main loop over ICP iterations
# TODO implement a stopping criteria beyond number of iterations
for iteration in range(max_iterations):
    print(f"Alignment at it #{iteration}")
    trans_curr[0:3, 0:3] = Rot.cpu().numpy()
    trans_curr[0:3, 3] = Trans.cpu().numpy()
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_curr
    )
    print(evaluation, "\n")

    # Xt_nn_points = pytorch3d.ops.knn_points(
    #    Xt, Yt, lengths1=num_points_X, lengths2=num_points_Y, K=1, return_nn=True
    #    ).knn[:, :, 0, :]
    # cdist = torch.cdist(Yt - Yt.mean(axis=0),
    #                    Xt - Xt.mean(axis=0))
    # mindists, argmins = torch.min(cdist, axis=1)
    # topk_values, topk_ids = index.topk(queryset, k=1)

    trsfpoints = sourcepoints @ Rot.T + Trans

    # source_temp = copy.deepcopy(source)
    # source_temp.points = o3d.utility.Vector3dVector(trsfpoints.cpu().numpy())
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #           source_temp, target, threshold, trans_id)
    # print(evaluation, "\n")

    distances, matchindices = index.search(trsfpoints, 1)
    mask = distances.squeeze() < threshold_sq
    print(f"# of matches: {mask.sum().cpu().numpy()}")

    selectptsourceidx = ptsourceidx[mask]
    selectpttargetidx = matchindices[mask].squeeze()

    selsourcecrossmats = sourcecrossmats[selectptsourceidx, :, :]
    seltrsfpts = trsfpoints[selectptsourceidx, :]
    seltargpts = targetpoints[selectpttargetidx, :]

    seltarnorms = targetnormals[selectpttargetidx, :]

    # dcheck = torch.nn.functional.pairwise_distance(seltrsfpts,seltargpts,eps=0)
    # print(dcheck[0:10]**2)
    # print(distances[mask].squeeze()[0:10])

    # TODO implement a stopping criteria beyond number of iterations
    for inner_iteration in range(max_inner_iterations):
        Ar = torch.matmul(seltarnorms.unsqueeze(1), Rot.unsqueeze(0))
        Al = -torch.matmul(Ar, selsourcecrossmats)
        A = torch.cat((Al.squeeze(), Ar.squeeze()), dim=1)
        B = torch.linalg.vecdot(seltarnorms, seltargpts - seltrsfpts)

        print(f"Init residuals={torch.sum(torch.square(B))}")

        # TODO, profile LSTSQ solvers and maybe switch to Cholesky for speed
        dTh_vec, resid, rank, singvals = torch.linalg.lstsq(A, B)
        print(f"residuals={resid.cpu().numpy()}")

        # TODO use closed form expression for exponential
        dTh[0, 1] = -dTh_vec[2]
        dTh[0, 2] = dTh_vec[1]
        dTh[1, 0] = dTh_vec[2]
        dTh[1, 2] = -dTh_vec[0]
        dTh[2, 0] = -dTh_vec[1]
        dTh[2, 1] = dTh_vec[0]

        dTh[0, 3] = dTh_vec[3]
        dTh[1, 3] = dTh_vec[4]
        dTh[2, 3] = dTh_vec[5]
        print(dTh_vec.cpu().numpy())
        # print(dTh.cpu().numpy())

        exp_dTh = torch.linalg.matrix_exp(dTh)
        # print(exp_dTh.cpu().numpy())
        R1R2T1 = Rot @ exp_dTh[0:3, :]
        Rot = R1R2T1[:, 0:3]
        Trans = R1R2T1[:, 3] + Trans
        # print(Rot.cpu().numpy())
        # print(Trans.cpu().numpy())

        trsfpoints = sourcepoints @ Rot.T + Trans
        seltrsfpts = trsfpoints[selectptsourceidx, :]

source_temp = copy.deepcopy(source)
source_temp.points = o3d.utility.Vector3dVector(trsfpoints.cpu().numpy())
draw_registration_result(source_temp, target, trans_id)
