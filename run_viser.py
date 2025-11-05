import time
import numpy as np
import open3d as o3d
import matplotlib
import viser
from pathlib import Path
import torch
import tyro
import open_clip # You may need to install this: pip install open_clip_torch
from tqdm import tqdm

# =================================================================================
# MAIN VISUALIZATION SCRIPT
# =================================================================================

def main(embedding_path: Path):
    """
    Main function to launch the Viser visualization using pre-baked LERF embeddings.
    """
    # 1. Load the pre-baked data into CPU RAM
    print(f"Loading pre-baked embeddings from {embedding_path}...")
    baked_data = np.load(embedding_path)
    points = baked_data["points"]
    colors = baked_data["colors"]
    # Keep embeddings in RAM as a NumPy array
    embeddings_np = baked_data["embeddings"]
    print("Embeddings loaded into CPU RAM.")

    # 2. Setup OpenCLIP model for encoding text queries
    print("Initializing OpenCLIP model for text queries...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    clip_model = clip_model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    print("CLIP model ready.")

    # 3. Prepare the viser server
    server = viser.ViserServer()

    # 4. Add GUI elements
    gui_query = server.gui.add_text("LERF Query", initial_value="a duck")
    gui_view_mode = server.gui.add_dropdown(
        "View Mode",
        ("LERF Point Cloud", "Relevancy Map", "Both"),
        initial_value="LERF Point Cloud",
    )
    gui_relevancy_threshold = server.gui.add_slider(
        "Relevancy Threshold", min=0.0, max=1.0, step=0.01, initial_value=0.5
    )
    gui_compute_relevancy = server.gui.add_button("Compute Relevancy")

    # Shared state for visualization
    state = {
        "scene_pcd_node": None,
        "relevancy_pcd_node": None,
        "raw_relevancies": None,
    }

    # 5. Display the initial world pointcloud
    state["scene_pcd_node"] = server.add_point_cloud(
        name="/scene_pointcloud",
        points=points,
        colors=colors,
        point_size=0.005,
        visible=True,
    )
    print("Scene point cloud added to Viser.")

    def update_relevancy_display():
        if state["raw_relevancies"] is None:
            return

        threshold = gui_relevancy_threshold.value
        mask = (state["raw_relevancies"] > threshold).squeeze()
        
        visible_points = points[mask]
        visible_relevancies = state["raw_relevancies"][mask]

        if state["relevancy_pcd_node"] is not None:
            state["relevancy_pcd_node"].remove()
            state["relevancy_pcd_node"] = None

        if visible_points.shape[0] == 0:
            return

        norm_scores = (visible_relevancies - visible_relevancies.min()) / (visible_relevancies.max() - visible_relevancies.min() + 1e-8)
        relevancy_colors = matplotlib.colormaps['jet'](norm_scores.squeeze())[:, :3]

        state["relevancy_pcd_node"] = server.add_point_cloud(
            name="/relevancy_pointcloud",
            points=visible_points,
            colors=relevancy_colors,
            point_size=0.005,
        )
        state["relevancy_pcd_node"].visible = gui_view_mode.value in ["Relevancy Map", "Both"]

    @gui_compute_relevancy.on_click
    @torch.no_grad()
    def _(_):
        """Callback to compute relevancy scores using pre-baked embeddings in batches."""
        gui_compute_relevancy.disabled = True
        print(f"Computing relevancy for query: '{gui_query.value}'")
        
        # 1. Encode text query (very fast)
        text = tokenizer([gui_query.value]).to(device)
        text_feat = clip_model.encode_text(text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

        # 2. *** FIX: Process embeddings in batches to avoid OOM error ***
        batch_size = 2**16
        num_points = embeddings_np.shape[0]
        all_max_similarities = []
        
        print("Calculating similarity in batches...")
        for i in tqdm(range(0, num_points, batch_size)):
            # Move one batch of embeddings to the GPU
            batch_embeddings_gpu = torch.from_numpy(embeddings_np[i:i+batch_size]).float().to(device)
            
            # Compute similarity for the batch
            similarity = torch.einsum('psd,cd->ps', batch_embeddings_gpu, text_feat)
            
            # Find max similarity across scales for the batch and move to CPU
            max_similarity_batch, _ = torch.max(similarity, dim=1)
            all_max_similarities.append(max_similarity_batch.cpu())

            # Clear GPU memory for the next batch
            del batch_embeddings_gpu, similarity, max_similarity_batch
            torch.cuda.empty_cache()

        # 3. Concatenate results from all batches
        state["raw_relevancies"] = torch.cat(all_max_similarities).numpy()
        
        # 4. Update the display
        update_relevancy_display()
        
        print("Relevancy map updated instantly.")
        gui_compute_relevancy.disabled = False

    @gui_relevancy_threshold.on_update
    def _(_):
        update_relevancy_display()

    @gui_view_mode.on_update
    def _(_):
        show_scene = gui_view_mode.value in ["LERF Point Cloud", "Both"]
        show_relevancy = gui_view_mode.value in ["Relevancy Map", "Both"]

        if state["scene_pcd_node"] is not None:
            state["scene_pcd_node"].visible = show_scene
        if state["relevancy_pcd_node"] is not None:
            state["relevancy_pcd_node"].visible = show_relevancy

    print("\nOpen the Viser link in your browser and refresh the page.")
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)