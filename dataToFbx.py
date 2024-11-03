import bpy
import os
import numpy as np
from mathutils import Euler, Vector


def npy_to_fbx(npy_path, fbx_path):
    try:
        # Ensure we're in the correct context
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                override = bpy.context.copy()
                override['area'] = area
                break

        # Clear the scene
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # Load data
        print("Loading motion data...")
        motion_data = np.load(npy_path)
        print(f"Motion data shape: {motion_data.shape}")

        # Get dimension information
        num_frames, num_features = motion_data.shape

        # Assuming the pose data is 104-dimensional as in demo.py
        assert num_features == 104, "Expected 104 features for pose data"

        print(f"Number of frames: {num_frames}")
        print(f"Number of features: {num_features}")

        # Create armature
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.context.active_object
        armature.name = "Motion_Armature"

        # Get edit bones
        edit_bones = armature.data.edit_bones

        # Remove default bone
        for bone in edit_bones:
            edit_bones.remove(bone)

        # Create bones based on SMPL model structure
        bone_names = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
            'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        ]

        for i, name in enumerate(bone_names):
            bone = edit_bones.new(name)
            bone.head = (0, 0, i * 0.1)
            bone.tail = (0, 0, (i * 0.1) + 0.05)

        # Set up bone hierarchy
        edit_bones['L_Hip'].parent = edit_bones['Pelvis']
        edit_bones['R_Hip'].parent = edit_bones['Pelvis']
        edit_bones['Spine1'].parent = edit_bones['Pelvis']
        edit_bones['L_Knee'].parent = edit_bones['L_Hip']
        edit_bones['R_Knee'].parent = edit_bones['R_Hip']
        edit_bones['Spine2'].parent = edit_bones['Spine1']
        edit_bones['L_Ankle'].parent = edit_bones['L_Knee']
        edit_bones['R_Ankle'].parent = edit_bones['R_Knee']
        edit_bones['Spine3'].parent = edit_bones['Spine2']
        edit_bones['L_Foot'].parent = edit_bones['L_Ankle']
        edit_bones['R_Foot'].parent = edit_bones['R_Ankle']
        edit_bones['Neck'].parent = edit_bones['Spine3']
        edit_bones['L_Collar'].parent = edit_bones['Spine3']
        edit_bones['R_Collar'].parent = edit_bones['Spine3']
        edit_bones['Head'].parent = edit_bones['Neck']
        edit_bones['L_Shoulder'].parent = edit_bones['L_Collar']
        edit_bones['R_Shoulder'].parent = edit_bones['R_Collar']
        edit_bones['L_Elbow'].parent = edit_bones['L_Shoulder']
        edit_bones['R_Elbow'].parent = edit_bones['R_Shoulder']
        edit_bones['L_Wrist'].parent = edit_bones['L_Elbow']
        edit_bones['R_Wrist'].parent = edit_bones['R_Elbow']
        edit_bones['L_Hand'].parent = edit_bones['L_Wrist']
        edit_bones['R_Hand'].parent = edit_bones['R_Wrist']

        # Switch to pose mode
        bpy.ops.object.mode_set(mode='POSE')

        # Set animation frame range
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = num_frames - 1

        # Create keyframes for each frame
        print("Creating keyframes...")
        for frame_idx in range(num_frames):
            bpy.context.scene.frame_set(frame_idx)
            frame_data = motion_data[frame_idx]

            for bone_idx, bone_name in enumerate(bone_names):
                bone = armature.pose.bones[bone_name]

                # Calculate data indices (3 values per joint: x, y, z rotation)
                start_idx = bone_idx * 3
                end_idx = start_idx + 3

                try:
                    # Get rotation data
                    rotation = frame_data[start_idx:end_idx]

                    # Set bone rotation
                    bone.rotation_mode = 'XYZ'
                    bone.rotation_euler = Euler(rotation)

                    # Insert keyframe
                    bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
                except Exception as e:
                    print(f"Error at frame {frame_idx}, bone {bone_name}")
                    print(f"Rotation data: {rotation}")
                    raise e

            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}/{num_frames}")

        # Select armature
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)

        # Export FBX
        print("Exporting to FBX...")
        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            use_selection=True,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            add_leaf_bones=False,
            global_scale=0.01
        )

        print(f"Successfully exported to {fbx_path}")

    except Exception as e:
        print(f"Error in npy_to_fbx: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# ... existing code ...

def main():
    npy_path = r"D:\Work\virtualCAS3\dataset\GQS883\scene01_body_pose.npy"
    fbx_path = r"D:\Work\virtualCAS3\output_animation.fbx"

    if not os.path.exists(npy_path):
        print(f"Error: Input file {npy_path} not found!")
        return

    os.makedirs(os.path.dirname(fbx_path), exist_ok=True)

    try:
        npy_to_fbx(npy_path, fbx_path)
    except Exception as e:
        print(f"Error occurred during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
