import numpy as np
import matplotlib.pyplot as plt

def device_to_camera(
    P_device: np.ndarray, 
    extrinsic_matrix: np.ndarray
):
    """Convert device coordinates to camera coordinates."""
    assert(P_device.shape == (3,))
    assert(extrinsic_matrix.shape == (4, 4))
    P_device_hom = np.append(P_device, 1)
    P_camera_hom = np.dot(extrinsic_matrix, P_device_hom)
    return P_camera_hom[:3]

def camera_to_image(P_camera, intrinsic_matrix):
    """Convert camera coordinates to image coordinates."""
    P_image_homogeneous = np.dot(intrinsic_matrix, P_camera)
    P_image = P_image_homogeneous[:2] / P_image_homogeneous[2]
    return P_image

def plot_trajectory_on_image(
    frame, trajectory, extrinsic_matrix, intrinsic_matrix, marker="o", color="red"
):
    # Convert device coordinates to camera coordinates
    future_pos_camera = np.array(
        [device_to_camera(p, extrinsic_matrix) for p in trajectory]
    )
    # Keep only points in front of the camera (z > 0)
    future_pos_camera = future_pos_camera[future_pos_camera[:, 2] > 0]

    # Convert camera coordinates to image coordinates
    future_pos_image = np.array(
        [camera_to_image(p, intrinsic_matrix) for p in future_pos_camera]
    )

    # Filter out points that are outside the image bounds
    image_height, image_width = frame.shape[:2]
    future_pos_image = future_pos_image[
        (future_pos_image[:, 0] >= 0) & (future_pos_image[:, 0] < image_width) &
        (future_pos_image[:, 1] >= 0) & (future_pos_image[:, 1] < image_height)
    ]

    plt.imshow(frame)
    plt.axis("off")
    if len(future_pos_image) > 0:
        plt.plot(
            future_pos_image[:, 0], future_pos_image[:, 1],
            marker=marker, color=color, linestyle="solid", linewidth=1, markersize=3,
        )
    plt.show()

plot_trajectory_on_image(
    frame=frame,
    trajectory=trajectory,
    extrinsic_matrix=extrinsic_matrix,
    intrinsic_matrix=intrinsic_matrix,
    marker="o",
    color="red"
)