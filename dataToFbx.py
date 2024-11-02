import bpy
import os
import numpy as np
from mathutils import Euler, Vector


def npy_to_fbx(npy_path, fbx_path):
    try:
        # 确保在正确的上下文中执行
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                override = bpy.context.copy()
                override['area'] = area
                break

        # 清除场景
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # 加载数据
        print("Loading motion data...")
        motion_data = np.load(npy_path)
        print(f"Motion data shape: {motion_data.shape}")

        # 获取维度信息
        num_frames = len(motion_data)
        data_points_per_frame = motion_data.shape[1]
        num_bones = data_points_per_frame // 6

        print(f"Number of frames: {num_frames}")
        print(f"Data points per frame: {data_points_per_frame}")
        print(f"Calculated number of bones: {num_bones}")

        # 创建骨骼armature
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.context.active_object
        armature.name = "Motion_Armature"

        # 获取编辑骨骼
        edit_bones = armature.data.edit_bones

        # 删除默认骨骼
        for bone in edit_bones:
            edit_bones.remove(bone)

        # 创建骨骼
        for i in range(num_bones):
            bone = edit_bones.new(f'Bone_{i}')
            bone.head = (0, 0, i * 0.2)
            bone.tail = (0, 0, (i * 0.2) + 0.1)
            if i > 0:
                bone.parent = edit_bones[i - 1]

        # 切换到姿态模式
        bpy.ops.object.mode_set(mode='POSE')

        # 设置动画帧范围
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = num_frames - 1

        # 为每一帧创建关键帧
        print("Creating keyframes...")
        for frame_idx in range(num_frames):
            bpy.context.scene.frame_set(frame_idx)
            frame_data = motion_data[frame_idx]

            for bone_idx in range(num_bones):
                bone = armature.pose.bones[f'Bone_{bone_idx}']

                # 计算数据索引
                pos_start = bone_idx * 6
                pos_end = pos_start + 3
                rot_start = pos_end
                rot_end = rot_start + 3

                try:
                    # 获取位置和旋转数据
                    position = frame_data[pos_start:pos_end]
                    rotation = frame_data[rot_start:rot_end]

                    # 设置骨骼变换
                    bone.location = Vector((
                        float(position[0]),
                        float(position[1]),
                        float(position[2])
                    ))

                    bone.rotation_euler = Euler((
                        float(rotation[0]),
                        float(rotation[1]),
                        float(rotation[2])
                    ))

                    # 插入关键帧
                    bone.keyframe_insert(data_path="location", frame=frame_idx)
                    bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
                except Exception as e:
                    print(f"Error at frame {frame_idx}, bone {bone_idx}")
                    print(f"Position data: {position}")
                    print(f"Rotation data: {rotation}")
                    raise e

            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}/{num_frames}")

        # 选择armature
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)

        # 导出FBX
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