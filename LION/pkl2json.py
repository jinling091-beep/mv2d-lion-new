import pickle
import json
import numpy as np

# ====================== 只改这两行路径 ======================
pkl_file_path = "/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/LION/output/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim/default/eval/epoch_36/val/default/result.pkl"
json_file_path = "/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/LION/output/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim/default/eval/epoch_36/val/default/result.pkl.json"
# ============================================================

# ✅✅✅ 完整版转换函数：覆盖所有你pkl里的类型（numpy+Quaternion+Box+所有特殊类型）✅✅✅
def convert_numpy_to_python(obj):
    # 处理 numpy 所有基础类型
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # 处理 None
    elif obj is None:
        return None
    
    # ✅ 重点新增：处理 Quaternion 四元数对象 → 转为字典存储核心属性
    elif str(type(obj)) == "<class 'nuscenes.eval.common.utils.Quaternion'>":
        return {"w": obj.w, "x": obj.x, "y": obj.y, "z": obj.z}
    
    # ✅ 重点新增：处理 Box 3D框对象 → 转为字典存储核心属性
    elif str(type(obj)) == "<class 'nuscenes.eval.common.utils.Box'>":
        return {
            "center": obj.center.tolist(),
            "wlh": obj.wlh.tolist(),
            "orientation": {"w": obj.orientation.w, "x": obj.orientation.x, "y": obj.orientation.y, "z": obj.orientation.z},
            "velocity": obj.velocity.tolist(),
            "name": obj.name
        }
    
    # ✅ 兜底：处理其他所有自定义类/未知对象 → 统一转成字典
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    
    # 其他Python原生类型，原样返回
    return obj

try:
    # 读取pickle
    with open(pkl_file_path, 'rb') as f_pkl:
        pkl_data = pickle.load(f_pkl)
    
    # 写入json
    with open(json_file_path, 'w', encoding='utf-8') as f_json:
        json.dump(pkl_data, f_json, indent=2, ensure_ascii=False, default=convert_numpy_to_python)

    print(f"✅ 转换成功！文件位置：{json_file_path}")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件，请检查路径是否正确 → {pkl_file_path}")
except PermissionError:
    print(f"❌ 错误：权限不足，无法读写文件 → {pkl_file_path}")
except Exception as e:
    print(f"❌ 转换失败，错误信息：{str(e)}")