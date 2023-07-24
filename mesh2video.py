import os

import cv2
import pyglet
import trimesh


def capture_frame(window, filename):
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    image_data = buffer.get_image_data()
    data = image_data.get_data("RGB", image_data.width * 3)
    image = cv2.flip(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB), 0)
    cv2.imwrite(filename, image)


def generate_video_from_meshes(folder_path, output_video_path):
    files = sorted(
        [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(".obj")
        ]
    )
    frames = []

    for file in files:
        # Load the mesh using trimesh
        mesh = trimesh.load_mesh(file)

        # Create a scene for rendering
        scene = mesh.scene()

        # Get a pyglet window for visualization
        window = pyglet.window.Window()

        # Set the camera view (modify as needed)
        scene.set_camera(angles=[0.5, 1.5, 0.5])

        # Render the scene
        img = scene.show(
            flags={"mesh": True}, return_scene=False, resolution=[800, 600]
        )

        # Capture the frame
        filename = os.path.join("temp_frames", os.path.basename(file) + ".png")
        frames.append(filename)
        capture_frame(window, filename)
        window.close()

    # Compile the frames into a video
    video = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, (800, 600)
    )

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        video.write(frame)
        os.remove(frame_path)  # Clean up the temporary frame

    video.release()


if __name__ == "__main__":
    generate_video_from_meshes("tiktok_meshes", "output_video.avi")
