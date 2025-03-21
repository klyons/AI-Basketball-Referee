#imnport libraries
#  https://www.youtube.com/watch?v=5ypQIUbpA7c&t=599s

# Enable mixed precision for faster inference and potentially better results
torch.set_float32_matmul_precision('medium')

# Optionally, fine-tune the model on a custom dataset if available
# This can improve depth perception for specific use cases
# Example: model.train() with a dataset of labeled depth images
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import os
import cv2

import numpy as np
import open3d as o3d

#globals
plot_heatmap = True

# grab image from video
video_path = "pitching_motion.mp4"
image_path = "data/myimage.jpg"

if not os.path.exists(image_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(image_path, frame)
    else:
        raise RuntimeError("Failed to extract the first frame from the video.")


feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")



image = Image.open("data/myimage.jpg")
print(image.size)
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32

new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

## prepareing the image for the model

inputs = feature_extractor(images=image, return_tensors = "pt")

#get the prediction from the model

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

#post processing
pad = 16
output = predicted_depth.squeeze(0).cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))


if plot_heatmap:
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(image)
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].imshow(output, cmap="plasma")
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.pause(10)


#prepareing depth image for open3d
width, height = image.size
depth_image = (output * 255 / np.max(output)).astype(np.uint8)
image = np.array(image)

depth_o3d = o3d.geometry.Image(depth_image)
color_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)

# create the open3d object camera angle

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 525, 525, width // 2, new_height // 2)

pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
o3d.visualization.draw_geometries([pcd_raw])

# post processing

cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
pcd = pcd_raw.select_by_index(ind)

pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

#surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_marching_cubes(pcd, depth=20, n_threads=4)[0]

rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=mesh.get_center(0, 0, 0))

o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

#draw uniform mesh
mesh_uniform = mesh.paint_uniform_color([0.8, 0.7, 0.8])
mesh_uniform.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_uniform], mesh_show_back_face=True)

