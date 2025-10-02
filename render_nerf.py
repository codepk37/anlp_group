import torch
import numpy as np
import tyro
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
import torchvision
from typing import Tuple

# Import necessary components from nerfstudio
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras

CONSOLE = Console(force_terminal=True)

def get_look_at_transform_manual(
    camera_pos: torch.Tensor, target_pos: torch.Tensor, up_vector: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the camera-to-world (c2w) transformation matrix in the (3, 4)
    format required by nerfstudio.
    """
    forward = target_pos - camera_pos
    forward = forward / torch.linalg.norm(forward)

    right = torch.cross(forward, up_vector)
    if torch.linalg.norm(right) < 1e-6:
        temp_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        if torch.allclose(forward.abs(), temp_up.abs()):
            temp_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        right = torch.cross(forward, temp_up)
    right = right / torch.linalg.norm(right)

    up = torch.cross(right, forward)
    up = up / torch.linalg.norm(up)

    # --- THIS IS THE CRITICAL FIX ---
    # Create the 3x4 rotation + translation matrix directly.
    c2w = torch.zeros((3, 4))
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward # Z-axis points away from the scene
    c2w[:3, 3] = camera_pos
    # --------------------------------
    
    return c2w


@dataclass
class RenderFromCheckpoint:
    """
    Renders a single image from a trained nerfstudio model checkpoint.
    """

    load_config: Path
    """Path to the config YAML file for the LERF model."""

    output_path: Path
    """Path to save the rendered RGB image (e.g., 'render.png')."""

    camera_position: Tuple[float, float, float] = (.46,0,.28)#(0.0, 0.0, 0.5)
    """The 3D position (x, y, z) of the camera."""

    look_at: Tuple[float, float, float] = (.46,0,-.18)#(0.0, 0.0, 0.0) 
    """The 3D point (x, y, z) the camera should look at (the origin)."""

    up_vector: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    """The 'up' direction for the camera. (0, 1, 0) is standard."""

    image_height: int = 1024
    """Height of the rendered image."""

    image_width: int = 1024
    """Width of the rendered image."""

    focal_length: float = 1111.0
    """Focal length of the virtual camera. A common default."""

    def main(self) -> None:
        """Main function to orchestrate the rendering process."""
        
        CONSOLE.print(f"Loading LERF model from checkpoint specified in: {self.load_config}")
        _, pipeline, _, _ = eval_setup(self.load_config)
        model = pipeline.model.to("cuda")
        model.eval()

        CONSOLE.print(f"Creating camera at position {self.camera_position}, looking at {self.look_at}")
        
        camera_pos_tensor = torch.tensor(self.camera_position, dtype=torch.float32)
        look_at_tensor = torch.tensor(self.look_at, dtype=torch.float32)
        up_vector_tensor = torch.tensor(self.up_vector, dtype=torch.float32)

        c2w = get_look_at_transform_manual(
            camera_pos=camera_pos_tensor,
            target_pos=look_at_tensor,
            up_vector=up_vector_tensor,
        )

        camera = Cameras(
            camera_to_worlds=c2w[None, ...],  # Add a batch dimension -> shape becomes (1, 3, 4)
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
            width=self.image_width,
            height=self.image_height,
        ).to(model.device)

        CONSOLE.print("Rendering image...")
        with torch.no_grad():
            ray_bundle = camera.generate_rays(camera_indices=0)
            outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle)
        
        rgb_image = outputs['rgb']

        CONSOLE.print(f"Saving rendered image to: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rgb_image_chw = rgb_image.permute(2, 0, 1)
        torchvision.utils.save_image(rgb_image_chw, self.output_path)
        
        CONSOLE.rule(f"[bold green]Success! Image saved to: {self.output_path}")

def entrypoint():
    """Entrypoint for use with tyro.cli()."""
    tyro.cli(RenderFromCheckpoint).main()

if __name__ == "__main__":
    entrypoint()


#run
# python /scratch/codepk37/anlpnerf/lerf/render_single_image.py     --load-config /scratch/codepk37/anlpnerf/lerf/outputs/final_scene2ttc/lerf/2025-09-30_102444/config.yml     --output-path /scratch/codepk37/anlpnerf/lerf/renders/my_render.png
