import torch
import math
from sklearn.utils.extmath import cartesian
import numpy as np
from torch.nn import Module
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


from src.utils.utils import _assert_no_grad
from src.utils.contact_utils import (
    batch_mesh_contains_points, 
    batch_pairwise_dist, 
    batch_index_select, 
    masked_mean_loss, 
    load_contacts)
from src.utils.cam_utils import verts_transfer_cam, project_hand_2D
from chamfer_distance import ChamferDistance
from manotorch.manolayer import ManoLayer, MANOOutput
from geomloss import SamplesLoss
from scipy.signal import find_peaks
import cv2


def find_closest_points_torch(source_points, target_points):
    """Finds closest point correspondences using broadcasting (can be memory-intensive)."""
    # Expand dimensions for broadcasting
    source_points_expanded = source_points[:, None, :]  # (N, 1, 3)
    target_points_expanded = target_points[None, :, :]  # (1, M, 3)

    # Calculate pairwise distances
    distances = torch.norm(source_points_expanded - target_points_expanded, dim=2)  # (N, M)

    # Find the indices of the closest points in the target point cloud
    _, indices = torch.min(distances, dim=1)
    return distances[torch.arange(source_points.shape[0]), indices], indices 


def find_closest_points_with_normals(source_points, target_points, 
                                     source_normals, target_normals, 
                                     alpha=0.8, normal_threshold=0.1):
    """
    Finds closest point correspondences considering normals and removes source points with negative dot product.

    Args:
        source_points: (N, 3) PyTorch tensor of source points.
        target_points: (M, 3) PyTorch tensor of target points.
        source_normals: (N, 3) PyTorch tensor of source normals.
        target_normals: (M, 3) PyTorch tensor of target normals.
        alpha: Weight for balancing position and normal alignment.

    Returns:
        filtered_source_points: Filtered source points with positive normal alignment.
        filtered_distances: Tensor of distances for the closest points.
        filtered_indices: Indices of the closest points in the target set.
    """
    # Compute dot products between source and target normals
    src_norm_expanded = source_normals[:, None, :]
    tgt_norm_expanded = target_normals[None, :, :]
    normal_dot_product = torch.sum(src_norm_expanded * tgt_norm_expanded, dim=2)

    # Filter out source points with negative dot product
    valid_mask = torch.max(normal_dot_product, dim=1).values > normal_threshold

    # Apply mask to source points and normals
    filt_src_pts = source_points[valid_mask]
    filt_src_norm = source_normals[valid_mask]

    # Expand dimensions for broadcasting
    filt_src_pts_expanded = filt_src_pts[:, None, :]
    tgt_pts_expanded = target_points[None, :, :]

    # Calculate positional distances
    pos_distances = torch.norm(filt_src_pts_expanded - tgt_pts_expanded, dim=2)

    # Calculate normal alignment (1 - dot product)
    filt_src_norm_expanded = filt_src_norm[:, None, :]
    normal_alignment = 1 - torch.abs(torch.sum(filt_src_norm_expanded * tgt_norm_expanded, dim=2))

    # Combine positional and normal distances
    combined_score = alpha * pos_distances + (1 - alpha) * normal_alignment

    # Find the indices of the closest points in the target point cloud
    filt_dist, filt_idx = torch.min(combined_score, dim=1)
    
    return filt_src_pts, filt_src_norm, filt_dist, filt_idx

def icp_with_scale(src_points, src_norm,
                   tgt_points, tgt_norm,
                   fix_scale=False,
                   fix_R=False,
                   fix_t=False,
                   max_iterations=200, tolerance=1e-4, device='cpu'):
    """
    Performs ICP alignment with scale estimation using PyTorch.

    Args:
        source_points: (N, 3) PyTorch tensor of source points.
        target_points: (M, 3) PyTorch tensor of target points.
        max_iterations: Maximum number of ICP iterations.
        tolerance: Convergence tolerance.
        device: Device to run on ('cpu' or 'cuda').

    Returns:
        R: Estimated rotation matrix (3x3).
        t: Estimated translation vector (3,).
        s: Estimated scale factor.
    """
    
    # src_points, mask = statistical_outlier_removal(src_points)
    # src_norm = src_norm[mask]
    # tgt_points, mask = statistical_outlier_removal(tgt_points)
    # tgt_norm = tgt_norm[mask]

    # Move tensors to the specified device
    src_points = src_points.to(device)
    tgt_points = tgt_points.to(device)

    # Initialize R, t, and s
    R = torch.eye(3, device=device)
    t = torch.zeros((3,), device=device)
    scale = 1.0

    for _ in range(max_iterations):
        src_points, src_norm, distances, indices = find_closest_points_with_normals(
                src_points, tgt_points, src_norm, tgt_norm
            )

        centroid_source = src_points.mean(dim=0)
        centroid_target = tgt_points[indices].mean(dim=0)
        centered_source = src_points - centroid_source
        centered_target = tgt_points[indices] - centroid_target

        if fix_scale is False:
            var_source = (centered_source ** 2).sum()
            cov_ST = (centered_source * centered_target).sum()
            scale = cov_ST / var_source

        if fix_R is False:
            H = (centered_source.T @ centered_target)
            U, _, V = torch.linalg.svd(H)
            R = V @ U.T

        if fix_t is False:
            t = centroid_target - scale * R @ centroid_source

        scaled_source_points = scale * src_points
        transformed_source_points = (R @ scaled_source_points.T).T + t

        mean_error = distances.mean()
        if mean_error < tolerance:
            break

    return R, t, scale, transformed_source_points, mean_error

def statistical_outlier_removal(points, k=20, z_score_threshold=0.5):
    """
    Remove statistical outliers from a point cloud.

    Args:
        points: (N, 3) PyTorch tensor or NumPy array of 3D points.
        k: Number of nearest neighbors to consider.
        z_score_threshold: Threshold for z-score to identify outliers.

    Returns:
        filtered_points: Point cloud with outliers removed.
        inlier_mask: Boolean mask indicating which points are inliers.
    """
    # Convert to NumPy array if input is a PyTorch tensor
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points_np)
    distances, _ = nbrs.kneighbors(points_np)

    # Calculate mean distance to k-nearest neighbors for each point
    mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude the first distance (always 0)

    # Calculate z-scores
    std_dev = np.std(mean_distances)
    z_scores = np.abs((mean_distances - np.mean(mean_distances)) / std_dev)

    # Identify inliers
    inlier_mask = z_scores < z_score_threshold

    # Filter points
    filtered_points = points_np[inlier_mask]

    # Convert back to PyTorch tensor if input was a tensor
    if isinstance(points, torch.Tensor):
        filtered_points = torch.from_numpy(filtered_points).to(points.device)
        inlier_mask = torch.from_numpy(inlier_mask).to(points.device)

    return filtered_points, inlier_mask

def compute_h2o_sdf_loss(
    obj_sdf,
    hand_pts,
    contact_zones,
    contact_thresh=0,
    penetr_thresh=5e-3,
):
    origin = obj_sdf["origin"]
    scale = obj_sdf["scale"]
    voxel = obj_sdf["voxel"]
    
    voxel = voxel.permute([2,1,0]) # original voxel is zyx, convert it to xyz
    
    D, H, W = voxel.shape
    hand_pts = hand_pts.squeeze()
    B = hand_pts.shape[0]
    
    voxel = voxel[None, None, :, :, :] #[1, 1, D, H, W]  xyz
    voxel = voxel.expand([B,-1,-1,-1,-1]) #[V, 1, D, H, W]  xyz
    query_grids = (hand_pts - origin) * scale # [V, 3]    
    query_grids = query_grids[:, None, None, None, :] # [V, 1, 1, 1, 3]
    
    # xyz = query_grids.cpu().squeeze()
    # xyz = (xyz - (-1))/2 * 64
    # x,y,z = int(xyz[0, 0]), int(xyz[0, 1]), int(xyz[0, 2])
    
    dist = F.grid_sample(voxel, query_grids, padding_mode="border")
    dist = dist.squeeze()
    
    # print(dist)
    inside_mask = dist < -penetr_thresh  # Boolean mask for points inside the mesh
    penetr_loss = masked_mean_loss(-dist, inside_mask)
    
    contact_dist = dist[contact_zones]
    outside_mask = (contact_dist > contact_thresh) 
    contact_loss = masked_mean_loss(contact_dist - contact_thresh, outside_mask)
    
    return penetr_loss, contact_loss

def compute_obj_contact_loss(
    obj_contact_pts,
    hand_verts,
    hand_contact_zone=None
):
    if hand_contact_zone is not None:
        hand_contact_pts = hand_verts[:, hand_contact_zone]
    else:
        hand_contact_pts = hand_verts
    dists = batch_pairwise_dist(hand_contact_pts, obj_contact_pts)
    mins12, min12idxs = torch.min(dists, 1)
    
    obj2hand_dists = torch.sqrt(mins12)
    contact_loss = torch.mean(obj2hand_dists)
    
    return contact_loss
    
    

def compute_penetr_loss(
    hand_verts,
    obj_verts_pt,
    obj_faces,
):
    """
    SDF is > 0 inside and < 0 outside mesh.
    """
    dists = batch_pairwise_dist(hand_verts, obj_verts_pt)
    mins12, min12idxs = torch.min(dists, 1)
    mins21, min21idxs = torch.min(dists, 2)

    # Get obj triangle positions
    obj_triangles = obj_verts_pt[:, obj_faces]
    exterior = batch_mesh_contains_points(
        hand_verts.detach(), obj_triangles.detach()
    )
    penetr_mask = ~exterior
    results_close = batch_index_select(obj_verts_pt, 1, min21idxs)
    dist_h2o = torch.norm(results_close - hand_verts, 2, 2)
    penetr_loss = masked_mean_loss(dist_h2o, penetr_mask)
    
    # dist_h2o[exterior] *= -1
    return penetr_loss

def remove_mask_elements_by_depth(mask, depth_map):
    """find the cluster with larger depth values"""
    min_d = depth_map[depth_map>0].min()
    max_d = depth_map[depth_map>0].max()
    histogram, bin_edges = np.histogram(depth_map.flatten(), range=(min_d, max_d), bins=10) # Only use mask True value to calculate histogram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find prominent peaks in the histogram (you can adjust parameters)
    peaks, _ = find_peaks(histogram, distance=10, prominence=5)

    # Use the first peak (smallest depth) as the threshold
    if len(peaks) > 0:
        depth_threshold = bin_edges[peaks[0]+1]

        # Remove mask elements with depth > depth_threshold
        mask[depth_map > depth_threshold] = False

    return mask

def filter_by_depth_kmeans(mask, depth_map, side, n_clusters=3):
    if np.count_nonzero(mask > 0) < n_clusters: 
        return None
    
    masked_depth = depth_map[mask > 0].reshape(-1, 1)  # Only consider depth values within the mask region
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(masked_depth)
    
    cluster_centers = kmeans.cluster_centers_  # Get the depth values for each cluster center
    if side == 'back':
        best_cluster = np.argmax(cluster_centers)  # Find the cluster with the largest center depth
    else:
        best_cluster = np.argmin(cluster_centers)  # Find the cluster with the nearest center depth
        
    # Keep only the pixels belonging to the best cluster
    original_mask_indices = np.where(mask > 0)
    best_cluster_mask = np.zeros_like(mask, dtype=np.uint8)
    best_cluster_mask[original_mask_indices] = (labels == best_cluster).astype(np.uint8)

    return best_cluster_mask

def compute_obj_contact(
    side,
    mask:torch.Tensor, #['H', 'W'] mask for obj and hand contact
    verts:torch.Tensor, # [num_vertices, 3]
    faces:torch.Tensor, # [num_faces, 3]
    normals:torch.Tensor, # [num_vertices, 3]
    rast:torch.Tensor, #['N', 'H', 'W', 4]
    skipped_face_ids = None,
    num_sample = 1000
):
    # <---------- find object contact id ------------>
    
    # rasterizer output in order (u, v, z/w, triangle_id)
    # Field triangle_id is the triangle index, offset by one. 
    # Pixels where no triangle was rasterized will receive a zero in all channels.
    # tri_id = rast[..., 3] - 1 
    
    if side == 'obj_front':
        n_layer = 0
    else:
        n_layer = 1
    
    u = rast[n_layer, ..., 0]
    v = rast[n_layer, ..., 1]
    depth = rast[n_layer, ..., 2]
    
    tri = rast[n_layer, ..., 3] - 1 #['H', 'W']
    contact_mask = ((mask>0) & (tri != -1))
    if side == 'obj_front':
        contact_mask = filter_by_depth_kmeans(contact_mask.cpu().numpy(), depth.cpu().numpy(), 'front')
    else:
        contact_mask = filter_by_depth_kmeans(contact_mask.cpu().numpy(), depth.cpu().numpy(), 'back')
        
    if contact_mask is None:
        return None, None, None
    
    contact_mask = torch.tensor(contact_mask)
    
    p_id = torch.nonzero(contact_mask) # contact pixel id
    tri_id = tri[p_id[:, 0], p_id[:, 1]] 
    
    if skipped_face_ids is not None:
        skipped_face_ids = torch.tensor(skipped_face_ids).to(tri_id)
        mask = ~torch.isin(tri_id, skipped_face_ids)
        tri_id = tri_id[mask]
        p_id = p_id[mask]
    
    verts_id = faces[tri_id.cpu().long()] # [num_tri, 3]
    verts_id = verts_id.cpu().long()
    # verts_id = torch.unique(verts_id).cpu().long()
    
    u = u[p_id[:, 0], p_id[:, 1]][:,None]
    v = v[p_id[:, 0], p_id[:, 1]][:,None]
    
    pts = u * verts[verts_id[:,0]] + v*verts[verts_id[:,1]] + (1-u-v)*verts[verts_id[:,2]]
    contact_normals = u * normals[verts_id[:,0]] + v*normals[verts_id[:,1]] + (1-u-v)*normals[verts_id[:,2]]
    
    random_indices = torch.randperm(pts.shape[0])[:num_sample]
    pts = pts[random_indices]
    contact_normals = contact_normals[random_indices]
    p_id = p_id[random_indices]
    
    return pts, contact_normals, contact_mask, p_id


def compute_hand_contact(
    side,
    mask:torch.Tensor, #['H', 'W'] mask for obj and hand contact
    verts:torch.Tensor, # [num_vertices, 3]
    faces:torch.Tensor, # [num_faces, 3]
    normals:torch.Tensor, # [num_vertices, 3]
    rast:torch.Tensor, #['N', 'H', 'W', 4]
    skipped_face_ids: list,
    num_sample = 1000
):
    # <---------- find hand contact id ------------>
    # rasterizer output in order (u, v, z/w, triangle_id)
    # Field triangle_id is the triangle index, offset by one. 
    # Pixels where no triangle was rasterized will receive a zero in all channels.
    # tri_id = rast[..., 3] - 1 
    N, H, W, _ = rast.shape
    multi_tri = rast[:, ..., 3] - 1 #[N, H, W]
    multi_u = rast[:, ..., 0]
    multi_v = rast[:, ..., 1]
    
    skipped_face_ids = torch.tensor(skipped_face_ids).to(multi_tri)
    exclude_mask = torch.isin(multi_tri, skipped_face_ids) # Mask: True for faces to exclude
    multi_tri[exclude_mask] = -1 # Set skipped faces to -1
    valid_mask = (multi_tri != -1)
    exists = valid_mask.any(dim=0)  # True if any valid triangle exists at [h, w]
    
    if side == 'hand_front':
        first_valid_index = valid_mask.float().argmax(dim=0)
        valid_index = torch.where(exists, first_valid_index, -1)  # Set invalid positions to -1
    elif side == 'hand_back':
        reversed_valid_mask = valid_mask.flip(dims=(0,))
        last_index_reversed = reversed_valid_mask.float().argmax(dim=0)  # Shape: [H, W]
        last_valid_index = N - 1 - last_index_reversed  # Convert reversed indices to original indices
        valid_index = torch.where(exists, last_valid_index, -1)  # Set invalid positions to -1
        
    contact_mask = (valid_index != -1) & (mask > 0)
    p_id = torch.nonzero(contact_mask) # contact pixel id
    tri_id = multi_tri[valid_index[contact_mask],p_id[:, 0], p_id[:, 1]] 
    verts_id = faces[tri_id.cpu().long()] # [num_tri, 3]
    verts_id = verts_id.cpu().long()
    # verts_id = torch.unique(verts_id).cpu().long()
    
    u = multi_u[valid_index[contact_mask],p_id[:, 0], p_id[:, 1]]
    v = multi_v[valid_index[contact_mask],p_id[:, 0], p_id[:, 1]]
    
    u = u[:,None]
    v = v[:,None]
    
    pts = u * verts[verts_id[:,0]] + v*verts[verts_id[:,1]] + (1-u-v)*verts[verts_id[:,2]]
    contact_normals = u * normals[verts_id[:,0]] + v*normals[verts_id[:,1]] + (1-u-v)*normals[verts_id[:,2]]
    
    print(pts.shape) #torch.Size([2987, 3])
    print(p_id.shape) #torch.Size([2987, 2])
    print(contact_normals.shape) #torch.Size([2987, 3])
    # exit()
    ## TODO: Determine the correspondence between contact points using pixel IDs.
    
    random_indices = torch.randperm(pts.shape[0])[:num_sample]
    pts = pts[random_indices]
    contact_normals = contact_normals[random_indices]
    p_id = p_id[random_indices]
    
    
    return pts, contact_normals, contact_mask , p_id
    
def compute_depth_loss(
    hand_mask:torch.Tensor, #['H', 'W']
    object_mask:torch.Tensor, #['H', 'W']
    hand_depth:torch.Tensor, #['N', 'H', 'W']
    object_depth:torch.Tensor, #['N', 'H', 'W']
)->torch.Tensor:
    nearest_obj_depth = object_depth[0]
    farthest_obj_depth, _ = torch.max(object_depth, dim=0)
    nearest_hand_depth = hand_depth[0]
    farthest_hand_depth, _ = torch.max(hand_depth, dim=0)
    obj_front_mask = (object_mask > 0)
    hand_front_mask = (hand_mask > 0)
    
    zero_tensor = torch.zeros_like(farthest_obj_depth)
    mask = (farthest_obj_depth > nearest_hand_depth).detach()
    loss_obj_front = masked_mean_loss((farthest_obj_depth -nearest_hand_depth)**2, obj_front_mask&mask)
    mask = (nearest_hand_depth > nearest_obj_depth).detach()
    loss_hand_front = masked_mean_loss((nearest_hand_depth-nearest_obj_depth)**2, hand_front_mask&mask)
    
    h2o_near = hand_depth - nearest_obj_depth[None] #[N, H, W]
    h2o_far = hand_depth - farthest_obj_depth[None]
    
    interior = torch.nonzero((h2o_near>0)&(h2o_far<0), as_tuple=False).detach() #[num_indices, 3]
    outerior = torch.nonzero((h2o_near<0)|(h2o_far>0), as_tuple=False).detach() #[num_indices', 3]
    
    i_h2o_n = h2o_near[interior[:,0], interior[:,1], interior[:,2]] #[num_indices,] all >0
    i_h2o_f = h2o_far[interior[:,0], interior[:,1], interior[:,2]] #[num_indices,] all <0
    i_h2o_f = torch.abs(i_h2o_f)
    
    mask = (i_h2o_n<i_h2o_f).detach()
    loss_pntr = masked_mean_loss(i_h2o_n**2, mask) + masked_mean_loss(i_h2o_f**2, ~mask)
    
    o_h2o_n = h2o_near[outerior[:,0], outerior[:,1], outerior[:,2]]
    o_h2o_f = h2o_far[outerior[:,0], outerior[:,1], outerior[:,2]]
    
    mask = (torch.abs(o_h2o_n)<torch.abs(o_h2o_f)).detach()
    loss_contact = masked_mean_loss(o_h2o_n**2, mask) + masked_mean_loss(o_h2o_f**2, ~mask)
    
    # print(loss_contact.item(), loss_obj_front.item(), loss_hand_front.item())
    
    loss_contact = loss_contact + loss_obj_front + loss_hand_front
    
    return loss_contact, loss_pntr

def compute_contact_loss(
    hand_verts_pt,
    hand_faces,
    obj_verts_pt,
    obj_faces,
    contact_thresh=5,
    contact_mode="dist_sq",
    collision_thresh=10,
    collision_mode="dist_sq",
    contact_target="all",
    contact_sym=False,
    contact_zones="all",
):
    # obj_verts_pt = obj_verts_pt.detach()
    # hand_verts_pt = hand_verts_pt.detach()
    
    dists = batch_pairwise_dist(hand_verts_pt, obj_verts_pt)
    mins12, min12idxs = torch.min(dists, 1)
    mins21, min21idxs = torch.min(dists, 2)

    # Get obj triangle positions
    obj_triangles = obj_verts_pt[:, obj_faces]
    exterior = batch_mesh_contains_points(
        hand_verts_pt.detach(), obj_triangles.detach()
    )
    penetr_mask = ~exterior
    results_close = batch_index_select(obj_verts_pt, 1, min21idxs)

    if contact_target == "all":
        anchor_dists = torch.norm(results_close - hand_verts_pt, 2, 2)
    elif contact_target == "obj":
        anchor_dists = torch.norm(results_close - hand_verts_pt.detach(), 2, 2)
    elif contact_target == "hand":
        anchor_dists = torch.norm(results_close.detach() - hand_verts_pt, 2, 2)
    else:
        raise ValueError(
            "contact_target {} not in [all|obj|hand]".format(contact_target)
        )
    if contact_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            contact_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            contact_vals = ((results_close - hand_verts_pt.detach()) ** 2).sum(
                2
            )
        elif contact_target == "hand":
            contact_vals = ((results_close.detach() - hand_verts_pt) ** 2).sum(
                2
            )
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
        below_dist = mins21 < (contact_thresh ** 2)
    elif contact_mode == "dist":
        # Use distance to penalize contact
        contact_vals = anchor_dists
        below_dist = mins21 < contact_thresh
    elif contact_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        contact_vals = contact_thresh * torch.tanh(
            anchor_dists / contact_thresh
        )
        # All points are taken into account
        below_dist = torch.ones_like(mins21).byte()
    else:
        raise ValueError(
            "contact_mode {} not in [dist_sq|dist|dist_tanh]".format(
                contact_mode
            )
        )
    if collision_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            collision_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            collision_vals = (
                (results_close - hand_verts_pt.detach()) ** 2
            ).sum(2)
        elif contact_target == "hand":
            collision_vals = (
                (results_close.detach() - hand_verts_pt) ** 2
            ).sum(2)
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
    elif collision_mode == "dist":
        # Use distance to penalize collision
        collision_vals = anchor_dists
    elif collision_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        collision_vals = collision_thresh * torch.tanh(
            anchor_dists / collision_thresh
        )
    else:
        raise ValueError(
            "collision_mode {} not in "
            "[dist_sq|dist|dist_tanh]".format(collision_mode)
        )

    missed_mask = below_dist & exterior
    if contact_zones == "tips":
        tip_idxs = [745, 317, 444, 556, 673]
        tips = torch.zeros_like(missed_mask)
        tips[:, tip_idxs] = 1
        missed_mask = missed_mask & tips
    elif contact_zones == "zones":
        _, contact_zones = load_contacts(
            "assets/contact_zones.pkl"
        )
        contact_matching = torch.zeros_like(missed_mask)
        for zone_idx, zone_idxs in contact_zones.items():
            min_zone_vals, min_zone_idxs = mins21[:, zone_idxs].min(1)
            cont_idxs = mins12.new(zone_idxs)[min_zone_idxs]
            # For each batch keep the closest point from the contact zone
            contact_matching[
                [torch.range(0, len(cont_idxs) - 1).long(), cont_idxs.long()]
            ] = 1
        missed_mask = missed_mask & contact_matching
    elif contact_zones == "all":
        missed_mask = missed_mask
    else:
        raise ValueError(
            "contact_zones {} not in [tips|zones|all]".format(contact_zones)
        )

    # Apply losses with correct mask
    missed_loss = masked_mean_loss(contact_vals, missed_mask)
    penetr_loss = masked_mean_loss(collision_vals, penetr_mask)
    if contact_sym:
        obj2hand_dists = torch.sqrt(mins12)
        sym_below_dist = mins12 < contact_thresh
        sym_loss = masked_mean_loss(obj2hand_dists, sym_below_dist)
        missed_loss = missed_loss + sym_loss
    # print('penetr_nb: {}'.format(penetr_mask.sum()))
    # print('missed_nb: {}'.format(missed_mask.sum()))
    max_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).max(1)[0].mean()
    )
    mean_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).mean(1).mean()
    )
    contact_info = {
        "attraction_masks": missed_mask,
        "repulsion_masks": penetr_mask,
        "contact_points": results_close,
        "min_dists": mins21,
    }
    metrics = {
        "max_penetr": max_penetr_depth,
        "mean_penetr": mean_penetr_depth,
    }
    return missed_loss, penetr_loss, contact_info, metrics


def anatomy_loss(mano_output:MANOOutput, axisFK, anatomyLoss):
    hand_verts_curr = mano_output.verts
    axisFK = axisFK.to(hand_verts_curr.device)
    
    T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
    T_g_a, R, ee = axisFK(T_g_p)  # ee (B, 16, 3)

    loss = anatomyLoss(ee)
    return loss

def soft_iou_loss(pred, gt, smooth=1e-6):
    """
    Compute the Soft IoU value as a differentiable function.
    pred and gt should be tensors with probabilities [0, 1].
    """
    intersection = (pred * gt).sum()
    total = (pred + gt).sum()
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def compute_nonzero_distance_2d(mask1, mask2, max_num=2000):
    if not isinstance(mask1, torch.Tensor):
        mask1 = torch.tensor(mask1)
    if not isinstance(mask2, torch.Tensor):
        mask2 = torch.tensor(mask2)
    
    nonzero1 = torch.nonzero(mask1, as_tuple=False).float()
    nonzero2 = torch.nonzero(mask2, as_tuple=False).float()
    
    if len(nonzero1) > max_num:
        indices1 = torch.randperm(len(nonzero1))[:max_num]
        nonzero1 = nonzero1[indices1]
    
    if len(nonzero2) > max_num:
        indices2 = torch.randperm(len(nonzero2))[:max_num]
        nonzero2 = nonzero2[indices2]
    
    if nonzero1.numel() == 0 or nonzero2.numel() == 0:
        return None  # or return float('inf')
    
    H, W = mask1.shape
    scale = torch.tensor([H, W], device=mask1.device)
    nonzero1 = nonzero1 / scale
    nonzero2 = nonzero2 / scale
    
    distances = torch.cdist(nonzero1, nonzero2)
    
    return torch.min(distances)

def mask_to_2dcoords(mask:torch.Tensor, tgt_mask:torch.Tensor, num_to_select=20000):
    device = mask.device
    region = torch.nonzero(mask).reshape(-1,2)
    min_x, min_y = torch.min(region[1]), torch.min(region[0])
    max_x, max_y = torch.max(region[1]), torch.max(region[0])
    
    x_range = torch.arange(min_x, max_x+1)
    y_range = torch.arange(min_y, max_y+1)
    
    x_coords, y_coords = torch.meshgrid(x_range, y_range, indexing="ij")
    coords = torch.stack([x_coords, y_coords], dim=-1).float().to(device)
    flat_coords = coords.reshape(-1, 2)
    new_mask = mask[min_y:max_y+1, min_x:max_x+1]
    flat_mask = new_mask.reshape(-1)
    
    if flat_coords.shape[0] > num_to_select:
        indices = torch.randperm(flat_coords.shape[0])
        flat_coords = flat_coords[indices[:num_to_select]]
        flat_mask = flat_mask[indices[:num_to_select]]
        
    
    
    return flat_coords, flat_mask


def one_side_chamfer_dist_loss(pred_mask, tgt_mask):
    pred_coord, flat_mask = mask_to_2dcoords(pred_mask, tgt_mask)
    tgt_pts = torch.nonzero(tgt_mask).reshape(-1,2).float()
    
    
    dist = torch.cdist(pred_coord.unsqueeze(0), tgt_pts.unsqueeze(0)).squeeze() # [pred_num, tgt_num]
    min_dist, _ = torch.min(dist, dim=-1) # [pred_num]
    
    min_dist = flat_mask * min_dist # don't consider points in the masked region
    
    loss = torch.mean(min_dist) # one sided chamfer distance
    return loss

def chamfer_dist_loss(pred_mask, tgt_mask):
    if torch.nonzero(tgt_mask).shape[0] == 0 or torch.nonzero(pred_mask).shape[0]==0:
        return 0
    img_scale = max(pred_mask.shape[0], pred_mask.shape[1])
    pred_coord, flat_mask = mask_to_2dcoords(pred_mask, tgt_mask)
    pred_coord = pred_coord.unsqueeze(0)
    tgt_pts = torch.nonzero(tgt_mask).reshape(-1,2).float().unsqueeze(0)
    if torch.nonzero(flat_mask).shape[0] == 0:
        return 0
    
    dist1 = torch.cdist(pred_coord, tgt_pts).squeeze() # [pred_num, tgt_num]
    min_dist1, _ = torch.min(dist1, dim=-1) # [pred_num] 
    min_dist1 = flat_mask * min_dist1 # only consider points in the masked region
    
    dist2 = dist1[flat_mask>0]
    min_dist2,_ = torch.min(dist2, dim=0) # [tgt_num]
    
    loss = 0.5 * torch.mean(min_dist1) + 0.5 * torch.mean(min_dist2)
    loss /= img_scale
    return loss

def occlusion_loss(pred_mask, tgt_mask, obj_mask):
    obj_region = torch.nonzero(obj_mask).reshape(-1,2)
    min_x, min_y = torch.min(obj_region[1]), torch.min(obj_region[0])
    max_x, max_y = torch.max(obj_region[1]), torch.max(obj_region[0])
    
    pred_mask = pred_mask[min_y: max_y, min_x:max_x]
    tgt_mask = tgt_mask[min_y: max_y, min_x:max_x]
    
    # loss = torch.nn.functional.binary_cross_entropy(pred_mask, tgt_mask)
    loss = chamfer_dist_loss(pred_mask, tgt_mask)
    return loss


sinkhorn_loss = SamplesLoss('sinkhorn')
def compute_sinkhorn_loss(pred_mask, gt_mask, max_num = 1000):
    assert gt_mask.sum() > 0, "gt mask shouldn't be empty"
    reg_loss = torch.abs(pred_mask.sum() - gt_mask.sum()) / gt_mask.sum() 
    # print("reg_loss: ", reg_loss.item())
    
    # Convert masks to "point clouds" with weights
    pred_pts = pred_mask.nonzero(as_tuple=False).float()
    pred_w = pred_mask[pred_mask != 0]
    
    pred_num = pred_pts.shape[0]
    if pred_num > max_num:
        indices = torch.randperm(pred_num)[:max_num]
        pred_pts = pred_pts[indices]
        pred_w = pred_w[indices]
        pred_num = max_num

    gt_pts = gt_mask.nonzero(as_tuple=False).float()
    gt_w = gt_mask[gt_mask != 0]
    
    gt_num = gt_pts.shape[0]
    if gt_num > max_num:
        indices = torch.randperm(gt_num)[:max_num]
        gt_pts = gt_pts[indices]
        gt_w = gt_w[indices]
        gt_num = max_num

    # Normalize weights
    pred_w /= pred_num
    gt_w /= gt_num
    
    if pred_w.numel() == 0 or pred_pts.numel() == 0:
        # Handle empty inputs explicitly
        main_loss = 0
        # print("main_loss: ", 0)
    else:
        # Compute Sinkhorn loss
        main_loss = sinkhorn_loss(pred_w.contiguous(), pred_pts.contiguous(), 
                                gt_w.contiguous(), gt_pts.contiguous())
        main_loss /= gt_num
        # print("main_loss: ", main_loss.item())
    
    loss = main_loss + 10 * reg_loss
    
    return loss

def compute_sinkhorn_loss_rgb(pred_img, gt_img, max_num=1000):
    pred_pts_geom = pred_img.nonzero(as_tuple=False).float()
    pred_pts_color = pred_img[pred_pts_geom[:,0].long(), pred_pts_geom[:,1].long()]
    pred_pts = torch.cat([pred_pts_geom, pred_pts_color], dim=1)
    
    pred_num = pred_pts.shape[0]
    if pred_num > max_num:
        indices = torch.randperm(pred_num)[:max_num]
        pred_pts = pred_pts[indices]
        pred_num = max_num
    
    gt_pts_geom = gt_img.nonzero(as_tuple=False).float()
    gt_pts_color = gt_img[gt_pts_geom[:,0].long(), gt_pts_geom[:,1].long()]
    gt_pts = torch.cat([gt_pts_geom, gt_pts_color], dim=1)
    
    gt_num = gt_pts.shape[0]
    
    if gt_num > max_num:
        indices = torch.randperm(gt_num)[:max_num]
        gt_pts = gt_pts[indices]
        gt_num = max_num
    
    loss = sinkhorn_loss(pred_pts.contiguous(), gt_pts.contiguous())
    loss /= gt_num
    
    return loss


def moment_based_comparison(mask1, mask2):
    """
    Compare shapes using Hu Moments, which are invariant to scale and translation.
    
    Args:
    mask1, mask2: 2D boolean numpy arrays representing the masks
    
    Returns:
    float: A measure of difference between the shapes (lower means more similar)
    """
    # Check if masks are empty
    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        print("Zero!")
        return float('inf')  # Return infinity for empty masks
    
    moments1 = cv2.moments(mask1.astype(np.uint8))
    moments2 = cv2.moments(mask2.astype(np.uint8))
    
    print(moments1, moments2)
    
    hu_moments1 = cv2.HuMoments(moments1).flatten()
    hu_moments2 = cv2.HuMoments(moments2).flatten()
    
    print(hu_moments1, hu_moments2)
    
    
    return np.sum(np.abs(np.log(hu_moments1 + 1e-10) - np.log(hu_moments2 + 1e-10)))



# copied from https://github.com/jkxing/DROT
# class DROTLossFunction(Module):
#     def __init__(self, resolution, renderer, device, settings, debug, num_views, logger):
#         super().__init__()
#         self.num_views = num_views
        
#         self.match_weight = settings.get("matching_weight",1.0)
#         self.matchings_record=[0 for i in range(num_views)]
#         self.matchings = [[] for i in range(num_views)]
#         self.rasts = [[] for i in range(num_views)]
#         self.rgb_weight = [self.match_weight for i in range(num_views)]
#         self.matching_interval = settings.get("matching_interval",0)
#         self.renderer = renderer
#         self.device = device
#         self.resolution = resolution[0]
#         self.debug=debug
#         self.logger = logger
#         self.step = -1
#         #Matcher setting
#         self.matcher_type=settings.get("matcher","Sinkhorn")
#         self.matcher = None
#         self.loss = SamplesLoss("sinkhorn", blur=0.01)

#         #normal image grid, used for pixel position completion
#         x = torch.linspace(0, 1, self.resolution)
#         y = torch.linspace(0, 1, self.resolution)
#         pos = torch.meshgrid(x, y)
#         self.pos = torch.cat([pos[1][..., None], pos[0][..., None]], dim=2).to(self.device)[None,...].repeat(num_views,1,1,1)
#         self.pos_np = self.pos[0].clone().cpu().numpy().reshape(-1,2)
    
#     def visualize_point(self, res, title, view):#(N,5) (r,g,b,x,y)
#         res = res.detach().cpu().numpy()
#         X = res[...,3:]
#         #need install sklearn
#         nbrs = None#NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
#         distances, indices = nbrs.kneighbors(self.pos_np)
#         distances = np.exp(-distances*self.resolution)
#         img = np.sum(res[indices,:3]*distances[...,None],axis = 1)
#         img = img/np.sum(distances,axis = 1)[...,None]
#         img = img.reshape(self.resolution, self.resolution, 3)
#         self.logger.add_image(title+"_"+str(view), img, self.step)

#     #unused currently
#     def rgb_match_weight(self, view=0):
#         return self.rgb_weight[view]

#     def match_Sinkhorn(self, haspos, render_point_5d, gt_rgb, view):
#         h,w = render_point_5d.shape[1:3]
#         target_point_5d = torch.zeros((haspos.shape[0], h, w, 5), device=self.device)
#         target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1)
#         target_point_5d[..., 3:] = render_point_5d[...,3:].clone().detach()
#         target_point_5d = target_point_5d.reshape(-1, h*w, 5)
#         render_point_5d_match = render_point_5d.clone().reshape(-1,h*w,5)
#         render_point_5d_match.clamp_(0.0,1.0)
#         render_point_5d_match[...,:3] *= self.rgb_match_weight(view)
#         target_point_5d[...,:3] = target_point_5d[...,:3]*self.rgb_match_weight(view)
#         pointloss = self.loss(render_point_5d_match, target_point_5d)*self.resolution*self.resolution
#         [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match])
#         g[...,:3]/=self.rgb_match_weight(view)
#         return (render_point_5d-g.reshape(-1,h,w,5)).detach()
    
#     def get_loss(self, render_res, gt_rgb, view):
#         haspos = render_res["msk"]
#         render_pos = (render_res["pos"]+1.0)/2.0
#         render_rgb = render_res["images"][...,:3]
#         render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
#         render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
#         match_point_5d = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)
#         disp = match_point_5d-render_point_5d
#         loss = torch.mean(disp**2)
#         return loss
    
#     def forward(self, gt, iteration=-1, scene=None, view=0):
#         self.step=iteration

#         new_match = ((self.matchings_record[view] % (self.matching_interval+1))==0)

#         if new_match:
#             render_res = self.renderer.render(scene, view=view, DcDt=False)
#             self.rasts[view] = render_res["rasts"]
#         else:
#             render_res = self.renderer.render(scene, rasts_list = self.rasts[view], view=view, DcDt=False)
        
#         self.matchings_record[view] += 1
#         haspos = render_res["msk"]
#         render_pos = (render_res["pos"]+1.0)/2.0
#         render_rgb = render_res["images"]
#         render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
#         render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
#         gt_rgb=gt["images"][view:view+1]
#         if new_match:
#             if self.matcher_type=="Sinkhorn":
#                 self.matchings[view] = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)

#         match_point_5d = self.matchings[view]
#         disp = match_point_5d-render_point_5d
#         loss = torch.mean(disp**2)

#         if self.debug:
#             self.visualize_point(match_point_5d.reshape(-1,5),title="match",view=view)
        
#         return loss, render_res
            

            