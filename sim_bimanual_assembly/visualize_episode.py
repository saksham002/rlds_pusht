import io
import argparse
import os
import h5py
import numpy as np
import cv2
import pdb
import wandb
import io
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import jax
import jax.numpy as jnp

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize an HDF5 episode and log to wandb.")
    parser.add_argument(
        "--episode_file_path",
        type=str,
        required=True,
        help="Path to the HDF5 episode file."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode."
    )
    parser.add_argument(
        "--value_function_checkpoint",
        type=str,
        default=None,
        help="Path to value function checkpoint (optional). If provided, plots value function alongside rewards."
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA network for value function (default: False)."
    )
    return parser.parse_args()

def create_composite_frame(images, decoded_images_dict, rewards, step, total_steps, values=None):
    """Creates a single video frame with subplots for images, rewards, and optionally values."""
    fig, axs = plt.subplots(2, 3, figsize = (15, 10))
    fig.suptitle(f"Episode Visualization - Step {step+1}/{total_steps}", fontsize=16)

    # Image keys and their positions in the plot
    image_keys = [
        "obses/images/left/top", "obses/images/left/wrist",
        "obses/images/right/top", "obses/images/right/wrist"
    ]
    ax_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for ax_pos, key in zip(ax_positions, image_keys):
        row, col = ax_pos
        ax = axs[row, col]
        
        # Use pre-decoded images if available, otherwise decode from HDF5
        if decoded_images_dict is not None and key in decoded_images_dict and step < len(decoded_images_dict[key]):
            img_decoded = decoded_images_dict[key][step]
            ax.imshow(img_decoded)
            ax.set_title(key)
        elif key in images and len(images[key]) > step:
            # Fallback: decode image
            pdb.set_trace()
            img_encoded = images[key][step]
            img_decoded = cv2.imdecode(img_encoded, 1)
            ax.imshow(img_decoded)
            ax.set_title(key)
        else:
            ax.set_title(f"{key} (No Data)")
        ax.axis('off')

    # Reward plot (top right)
    ax_reward = axs[0, 2]
    ax_reward.plot(range(step + 1), rewards[ : step + 1], color = 'r', label = 'Reward')
    ax_reward.set_xlim(0, total_steps)
    min_reward = np.min(rewards) if rewards.size > 0 else 0
    max_reward = np.max(rewards) if rewards.size > 0 else 1
    ax_reward.set_ylim(min_reward - 0.1, max_reward + 0.1)
    ax_reward.set_title("Rewards")
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True)
    ax_reward.legend()
    
    # Value function plot (bottom right) - only if values are provided
    ax_value = axs[1, 2]
    if values is not None and len(values) > 0:
        ax_value.plot(range(step + 1), values[ : step + 1], color = 'b', label = 'Value')
        ax_value.set_xlim(0, total_steps)
        min_value = np.min(values) if values.size > 0 else 0
        max_value = np.max(values) if values.size > 0 else 1
        ax_value.set_ylim(min_value - 0.1, max_value + 0.1)
        ax_value.set_title("Value Function")
        ax_value.set_xlabel("Step")
        ax_value.set_ylabel("Value")
        ax_value.grid(True)
        ax_value.legend()
    else:
        ax_value.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Convert plot to numpy array using an in-memory buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    # Decode the PNG buffer to a numpy array
    frame = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_COLOR)
    # OpenCV decodes to BGR, so convert to RGB for correct colors
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.close(fig)
    return frame

def main():
    """Main function to generate and log visualization."""
    args = parse_args()
    episode_path = args.episode_file_path
    debug = args.debug
    value_checkpoint = args.value_function_checkpoint
    use_ema = args.use_ema

    if not os.path.exists(episode_path):
        print(f"Error: File not found at {episode_path}")
        return

    # Generate wandb run name from file path
    episode_dir = os.path.basename(os.path.dirname(episode_path))
    episode_filename = os.path.basename(episode_path)
    episode_number = episode_filename.replace('episode_', '').replace('.hdf5', '')
    wandb_run_name = f"{episode_dir}_{episode_number}"

    # Load value function if checkpoint is provided
    value_fn = None
    aug_agent = None
    value_agent = None

    try:
        # Initialize wandb
        if not debug:
            wandb.init(
                project="jflow_rl",
                group="sim_bimanual_data",
                name=wandb_run_name,
            )
            print(f"Wandb run initiated with name: {wandb_run_name}")

        # Read HDF5 file
        with h5py.File(episode_path, 'r') as f:
            image_keys = [
                "obses/images/left/top", "obses/images/left/wrist",
                "obses/images/right/top", "obses/images/right/wrist"
            ]
            images = {key: f[key][:] for key in image_keys if key in f}
            rewards = f['rewards/rewards'][:] if 'rewards/rewards' in f else np.array([])
            
            # Load states if value function is provided
            if value_checkpoint is not None:
                # Read state data
                state_data = []
                state_keys = [
                    'obses/state/left/relative2_tcp_pose',
                    'obses/state/left/relative2_tcp_vel',
                    'obses/state/left/wrist_tcp_vel',
                    'obses/state/left/gripper_pos',
                    'obses/state/right/relative2_tcp_pose',
                    'obses/state/right/relative2_tcp_vel',
                    'obses/state/right/wrist_tcp_vel',
                    'obses/state/right/gripper_pos',
                ]
                for key in state_keys:
                    comp = f[key]
                    comp_rank = len(np.shape(comp))
                    if comp_rank == 1:
                        state_data.append(np.expand_dims(comp, axis = -1))
                    elif comp_rank == 2:
                        # Already 2D, use as is
                        state_data.append(comp)
                    else:
                        raise ValueError(f"Unexpected dimension for state component: {comp_rank}")

                
                states = np.concatenate(state_data, axis = -1)
        
        num_steps = len(rewards)
        if num_steps == 0:
            print("No steps to visualize.")
            return
        
        # Decode all images first (for both visualization and value computation)
        print("Decoding images...")
        decoded_images_dict = {}
        image_keys = [
            "obses/images/left/top", "obses/images/left/wrist",
            "obses/images/right/top", "obses/images/right/wrist"
        ]
        for key in image_keys:
            if key in images:
                decoded_images_dict[key] = []
                for i in tqdm(range(num_steps), desc=f"Decoding {key}"):
                    img_encoded = images[key][i]
                    img_decoded = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_COLOR)
                    decoded_images_dict[key].append(img_decoded)
        
        # Compute values if value function is provided
        values = None
        if value_checkpoint is not None:
            print(f"Loading value function from {value_checkpoint}...")
            # Import necessary modules
            from bc_flowmatch.configs.double_insert_rl_config import RL_XArm_Config
            from bc_flowmatch.agents.value_functions import ValueFunctionLearning
            from bc_flowmatch.agents.image_augmentation_agent import ImageAugmentationAgent
            
            # Load value function
            value_agent = ValueFunctionLearning.load_checkpoint(
                filename=value_checkpoint,
                seed=0,
                is_state_only=False,
            )
            
            # Select network based on use_ema flag
            if use_ema:
                value_fn = value_agent.v_ema_network
                print("Using EMA network for value function")
            else:
                value_fn = value_agent.v_network
                print("Using regular network for value function")
            
            # Initialize config and get eval augmentations
            config = RL_XArm_Config(ckpt_dir = "", device = "cpu")
            eval_augmentations = config.dataset_config.eval_image_fn
            camera_names = config.dataset_config.camera_names

            eval_augmentations = [eval_augmentations[cam_name] for cam_name in camera_names]
            
            # Initialize ImageAugmentationAgent
            device = jax.devices("gpu")[0]
            aug_agent = ImageAugmentationAgent(
                augmentation_chains=eval_augmentations,
                input_shape=(256, 224, 224, 3),
                device=device,
            )
            
            print("Value function and augmentation agent initialized")

            print("Computing value function...")
            values = []
            
            # Batch processing parameters
            batch_size = 256
            key = jax.random.PRNGKey(0)
            
            # Prepare all decoded images in camera order
            all_decoded_images = []
            for i in range(num_steps):
                step_images = []
                for camera_name in camera_names:
                    hdf5_key = f"obses/images/{camera_name}"
                    step_images.append(decoded_images_dict[hdf5_key][i])
                all_decoded_images.append(np.stack(step_images, axis = 0))
            all_decoded_images = np.stack(all_decoded_images, axis = 0)  # Shape: (num_steps, K, H, W, C)
            
            # Process in batches
            for batch_start in tqdm(range(0, num_steps, batch_size), desc="Computing values (batched)"):
                batch_end = min(batch_start + batch_size, num_steps)
                current_batch_size = batch_end - batch_start
                
                # Get batch of images
                batch_images = all_decoded_images[batch_start : batch_end, None, :, :, :, :]  # (B, 1, K, H, W, C)
                
                # Create batch dict
                batch = {'observation.image': batch_images}
                
                # Apply augmentations
                key, subkey = jax.random.split(key)
                augmented_batch = aug_agent.augment_batch(batch, subkey)
                
                # Get augmented images (shape: (B, 1, K, C, H, W))
                augmented_images = augmented_batch['observation.image']
                
                # Prepare states (add history dimension)
                # State shape: (B, 1, state_dim)
                batch_states = states[batch_start : batch_end][:, None, :]
                
                # Normalize states
                normalized_states = value_agent.normalize_fn(batch_states, value_agent.state_norm)
                
                # Compute values for batch
                key, subkey = jax.random.split(key)
                batch_keys = jax.random.split(subkey, current_batch_size)
                
                # Vectorized value computation
                batch_values = jax.vmap(value_fn)(augmented_images, normalized_states, batch_keys)
                
                # Append to values list
                values.extend(batch_values.flatten().tolist())
            
            values = np.array(values)
            print(f"Computed {len(values)} values")

        # Generate all frames for the video
        print("Generating video frames...")
        video_frames = []
        for i in tqdm(range(num_steps)):
            frame = create_composite_frame(images, decoded_images_dict, rewards, i, num_steps, values)
            video_frames.append(frame)

        # Save video to disk and log to wandb
        if video_frames:
            video_frames = np.array(video_frames)
            
            # Create videos directory if it doesn't exist
            videos_dir = "videos_neg" if "neg" in value_checkpoint else "videos"
            os.makedirs(videos_dir, exist_ok = True)
            
            # Generate video filename from episode path
            episode_dir = os.path.basename(os.path.dirname(episode_path))
            episode_filename = os.path.basename(episode_path)
            episode_number = episode_filename.replace('episode_', '').replace('.hdf5', '')
            video_filename = f"{episode_dir}_{episode_number}.mp4"
            video_path = os.path.join(videos_dir, video_filename)
            
            # Save MP4 video to disk
            print(f"Saving MP4 video to {video_path}...")
            imageio.mimsave(video_path, video_frames, format='mp4', fps=60, codec='libx264', quality=8)
            print(f"Video saved successfully to {video_path}")
            
            # Log GIF video to wandb
            if not debug:
                print("Logging GIF video to wandb...")
                gif_buffer = io.BytesIO()
                imageio.mimsave(gif_buffer, video_frames, format='GIF', fps=60)
                gif_buffer.seek(0)
                wandb.log({"episode_visualization": wandb.Video(gif_buffer, format="gif", fps=60)})
                print("GIF video logged successfully to wandb.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if wandb.run:
            wandb.finish()
            print("Wandb run finished.")

if __name__ == "__main__":
    main()
