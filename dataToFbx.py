import bpy
import numpy as np
from mathutils import Euler, Vector

def npy_to_fbx(npy_path, fbx_path):
    # 清除当前场景中的所有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # 加载npy数据
    motion_data = np.load(npy_path)
    
    # 创建骨骼armature
    bpy.ops.object.armature_add()
    armature = bpy.context.active_object
    armature.name = "Motion_Armature"
    
    # 进入编辑模式创建骨骼
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    
    # 创建骨骼层级结构
    num_bones = motion_data.shape[1]  # 获取骨骼数量
    for i in range(num_bones):
        bone = edit_bones.new(f'Bone_{i}')
        bone.head = (0, 0, i * 0.2)  # 设置骨骼头部位置
        bone.tail = (0, 0, (i * 0.2) + 0.1)  # 设置骨骼尾部位置
        
        # 设置骨骼父子关系
        if i > 0:
            bone.parent = edit_bones[i-1]
    
    # 切换到姿态模式
    bpy.ops.object.mode_set(mode='POSE')
    
    # 设置动画帧范围
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(motion_data) - 1
    
    # 为每一帧创建关键帧
    for frame_idx, frame_data in enumerate(motion_data):
        bpy.context.scene.frame_set(frame_idx)
        
        # 设置骨骼位置和旋转
        for bone_idx, bone_data in enumerate(frame_data):
            bone = armature.pose.bones[f'Bone_{bone_idx}']
            
            # 使用Vector和Euler确保数据类型正确
            location = Vector(bone_data[:3].tolist())
            rotation = Euler(bone_data[3:6].tolist())
            
            bone.location = location
            bone.rotation_euler = rotation
            
            # 插入关键帧
            bone.keyframe_insert(data_path="location", frame=frame_idx)
            bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
    
    # 选择armature
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    
    # 导出FBX
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        add_leaf_bones=False
    )