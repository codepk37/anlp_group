import torch
import numpy as np
import open3d as o3d
import trimesh as tr
from pathlib import Path
import tyro
from tqdm import tqdm

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.exporter.exporter_utils import generate_point_cloud

class LERFBaker:
    """A class to handle the one-time pre-computation of LERF embeddings."""

    def __init__(self, config_path: Path):
        print("Loading LERF pipeline for baking...")
        _, self.pipeline, _, _ = eval_setup(config_path)
        self.model = self.pipeline.model
        self.model.eval()
        self.device = self.pipeline.model.device

        dp_outputs = self.pipeline.datamanager.train_dataparser_outputs
        self.applied_transform = np.eye(4)
        if hasattr(dp_outputs, 'dataparser_transform'):
            transform_3x4 = dp_outputs.dataparser_transform.numpy()
            scale = dp_outputs.dataparser_scale
            transform_4x4 = np.eye(4)
            transform_4x4[:3, :] = transform_3x4
            inv_transform = np.linalg.inv(transform_4x4)
            inv_scale_mat = np.diag([1/scale, 1/scale, 1/scale, 1])
            self.applied_transform = inv_transform @ inv_scale_mat
        print("LERFWrapper initialized for baking.")

    def create_pointcloud(self) -> o3d.geometry.PointCloud:
        """Generates a dense point cloud from the LERF model."""
        orig_num_rays_per_batch = self.pipeline.datamanager.train_pixel_sampler.num_rays_per_batch
        self.pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 30000

        pcd = generate_point_cloud(
            pipeline=self.pipeline,
            remove_outliers=True,
            std_ratio=0.1,
            depth_output_name="depth",
            num_points=500000,
        )

        self.pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = orig_num_rays_per_batch
        
        pcd.points = o3d.utility.Vector3dVector(
            tr.transformations.transform_points(np.asarray(pcd.points), self.applied_transform)
        )
        return pcd

    @torch.no_grad()
    def bake_embeddings(self, output_path: Path):
        """
        Generates a point cloud and computes multi-scale CLIP embeddings for each point,
        then saves them to a file. This version is memory-efficient.
        """
        print("Generating a dense point cloud for baking...")
        pcd = self.create_pointcloud()
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        points_torch = torch.from_numpy(points).float().to(self.device)

        scales = torch.linspace(0.0, self.model.config.max_scale, self.model.config.n_scales).to(self.device)
        num_points = points_torch.shape[0]
        num_scales = len(scales)
        embedding_dim = self.model.lerf_field.clip_net.n_output_dims
        
        # *** FIX: Create the large array in CPU RAM using NumPy ***
        all_embeddings_np = np.zeros((num_points, num_scales, embedding_dim), dtype=np.float16)

        batch_size = 2**14
        print(f"Baking multi-scale embeddings for {num_points} points...")
        for i in tqdm(range(0, num_points, batch_size)):
            batch_points = points_torch[i : i + batch_size]
            
            positions = self.model.lerf_field.spatial_distortion(batch_points)
            positions = (positions + 2.0) / 4.0

            encoded_points = [e(positions.view(-1, 3)) for e in self.model.lerf_field.clip_encs]
            encoded_points = torch.cat(encoded_points, dim=-1)

            # This temporary tensor is small and stays on the GPU
            batch_scale_embeddings_gpu = torch.zeros((batch_points.shape[0], num_scales, embedding_dim), device=self.device, dtype=torch.float16)

            for j, scale in enumerate(scales):
                scaled_encoded_points = torch.cat([encoded_points, torch.full_like(encoded_points[..., :1], scale)], dim=-1)
                clip_features = self.model.lerf_field.clip_net(scaled_encoded_points)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                batch_scale_embeddings_gpu[:, j, :] = clip_features.half()
            
            # *** FIX: Move the completed batch results to CPU and store in the NumPy array ***
            all_embeddings_np[i : i + batch_points.shape[0]] = batch_scale_embeddings_gpu.cpu().numpy()

        print(f"Saving baked data to {output_path}...")
        np.savez_compressed(
            output_path,
            points=points,
            colors=colors,
            embeddings=all_embeddings_np
        )
        print("Baking complete!")

def main(config_path: Path, output_path: Path):
    """
    Main function to run the embedding baking process.
    """
    baker = LERFBaker(config_path)
    baker.bake_embeddings(output_path)

if __name__ == "__main__":
    tyro.cli(main)