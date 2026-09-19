import numpy as np
import json
import pickle
from pyquaternion import Quaternion
from tqdm import tqdm
import os

class NuScenesStandaloneEvaluator:
    def __init__(self, max_dist=54.4):
        self.dist_ths = [0.5, 1.0, 2.0, 4.0]
        self.max_dist = max_dist
        # 定义需要跳过特定指标的类别
        self.no_vel_cats = ['traffic_cone', 'barrier']
        self.no_ang_cats = ['traffic_cone', 'barrier']
        self.no_attr_cats = ['traffic_cone', 'barrier']

    def _load_file(self, path):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif ext == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        raise ValueError(f"Unsupported file format: {ext}")

    def _get_yaw(self, q):
        """从四元数获取 Yaw 角"""
        return q.yaw_pitch_roll[0]

    def _global_to_lidar(self, pred_box, info):
        """坐标转换：Global -> Ego -> Lidar"""
        # 1. Global -> Ego
        e2g_t = np.array(info['ego2global_translation'])
        e2g_r = Quaternion(info['ego2global_rotation'])
        pos_global = np.array(pred_box['translation'])
        rot_global = Quaternion(pred_box['rotation'])
        
        pos_ego = e2g_r.inverse.rotate(pos_global - e2g_t)
        rot_ego = e2g_r.inverse * rot_global

        # 2. Ego -> Lidar
        l2e_t = np.array(info['lidar2ego_translation'])
        l2e_r = Quaternion(info['lidar2ego_rotation'])
        
        pos_lidar = l2e_r.inverse.rotate(pos_ego - l2e_t)
        rot_lidar = l2e_r.inverse * rot_ego
        
        # 3. Velocity 转换 (支持 2D 速度输入)
        vel_global = np.array(pred_box.get('velocity', [0.0, 0.0]))
        if len(vel_global) == 2:
            # 补 0 变成 3D 向量进行旋转计算
            vel_global_3d = np.append(vel_global, 0.0)
        else:
            vel_global_3d = vel_global[:3]
            
        # 经过 ego 和 lidar 两级旋转变换到 lidar 系
        vel_lidar_3d = l2e_r.inverse.rotate(e2g_r.inverse.rotate(vel_global_3d))
        
        return pos_lidar, rot_lidar, vel_lidar_3d[:2]

    def _compute_tp_errors(self, p_pos, p_rot, p_vel, p_size, p_attr, gt_geom, gt_vel, gt_attr, category_name):
        """
        在 Lidar 系下计算误差
        category_name: 用于判断是否跳过某些指标
        """
        # 1. ATE (Translation) - 始终计算
        ate = np.linalg.norm(p_pos[:2] - gt_geom[:2])
        
        # # 2. ASE (Scale) - 始终计算
        # p_s, g_s = np.array(p_size), np.array(gt_geom[3:6])
        # ase = 1 - (np.prod(np.minimum(p_s, g_s)) / np.prod(np.maximum(p_s, g_s)))
        # 2. ASE (Scale) - 修正尺寸顺序
        # 预测值 p_size 是 [w, l, h]
        p_s = np.array(p_size) 
        
        # GT 的 gt_geom[3:6] 是 [l, w, h]
        # 我们将其交换为 [w, l, h] 以匹配预测值
        g_s_raw = np.array(gt_geom[3:6])
        g_s = np.array([g_s_raw[1], g_s_raw[0], g_s_raw[2]]) # 交换 0和1 维，变为 [w, l, h]
        
        # 计算 ASE: 1 - min(w,l,h和w',l',h')的体积 / max(w,l,h和w',l',h')的体积
        # 现在 p_s 和 g_s 都是 [w, l, h] 顺序了
        ase = 1 - (np.prod(np.minimum(p_s, g_s)) / np.prod(np.maximum(p_s, g_s)))
        
        # 3. AOE (Orientation/Angular)
        if category_name in self.no_ang_cats:
            aoe = 0.0
        else:
            # 还原 GT 的标准 Yaw: yaw = -special_yaw - pi/2
            # special_yaw = gt_geom[6]
            # g_yaw = -special_yaw - (np.pi / 2)
            g_yaw = gt_geom[6]  # 直接使用标注的 Yaw 角
            p_yaw = self._get_yaw(p_rot)
            aoe = np.abs(p_yaw - g_yaw) % (2 * np.pi)
            aoe = min(aoe, 2 * np.pi - aoe)
        
        # 4. AVE (Velocity)
        if category_name in self.no_vel_cats:
            ave = 0.0
        else:
            # 确保 gt_vel 是 2D，计算 L2 范数误差
            ave = np.linalg.norm(np.array(p_vel[:2]) - np.array(gt_vel[:2]))

        # 5. AAE (Attribute)
        if category_name in self.no_attr_cats:
            aae = 0.0
        else:
            aae = 1.0 if str(p_attr) == str(gt_attr) else 0.0

        return ate, ase, aoe, ave, aae

    def calculate_ap_11point(self, tp, fp, num_gt):
        if num_gt == 0: return 0.0
        tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            ap += np.max(precisions[mask]) if np.any(mask) else 0.0
        return ap / 11.0

    def evaluate(self, result_path, pkl_path):
        results_data = self._load_file(result_path)
        results = results_data['results']

        # 打印 JSON 中检测到的所有类别
        found_in_json = set(p['detection_name'] for preds in results.values() for p in preds)
        print(f"Categories found in JSON: {found_in_json}")     

        infos_data = self._load_file(pkl_path)
        infos = infos_data['infos'] if isinstance(infos_data, dict) and 'infos' in infos_data else infos_data

        found_in_gt = set()
        for info in infos:
            found_in_gt.update(info['gt_names'])
        print(f"Categories found in PKL: {found_in_gt}")
        info_dict = {info['token']: info for info in infos}

        all_categories = sorted(list(set(p['detection_name'] for preds in results.values() for p in preds)))
        stats = {cat: {'aps': [], 'ate': [], 'ase': [], 'aoe': [], 'ave': [], 'aae': []} for cat in all_categories}

        for cat in tqdm(all_categories, desc="Evaluating"):
            # 1. Preds Global -> Lidar
            cat_preds = []
            for token, preds in results.items():
                if token not in info_dict: continue
                for p in preds:
                    if p['detection_name'] == cat:
                        l_pos, l_rot, l_vel = self._global_to_lidar(p, info_dict[token])
                        if np.linalg.norm(l_pos[:2]) <= self.max_dist:
                            p_copy = p.copy()
                            p_copy.update({'l_pos': l_pos, 'l_rot': l_rot, 'l_vel': l_vel})
                            cat_preds.append(p_copy)
            cat_preds = sorted(cat_preds, key=lambda x: x['detection_score'], reverse=True)

            # 2. Count GTs
            total_gts = 0
            for info in info_dict.values():
                for i, name in enumerate(info['gt_names']):
                    if name == cat and np.linalg.norm(info['gt_boxes'][i][:2]) <= self.max_dist:
                        total_gts += 1
            if total_gts == 0: continue


            # 3. Matching
            for dist_th in self.dist_ths:
                tps, fps = np.zeros(len(cat_preds)), np.zeros(len(cat_preds))
                matched_gt_ids = {token: set() for token in info_dict.keys()}

                for i, p in enumerate(cat_preds):
                    info = info_dict[p['sample_token']]
                    gt_idx_list = [j for j, name in enumerate(info['gt_names']) if name == cat]
                    if not gt_idx_list: fps[i] = 1; continue
                    
                    gt_boxes = info['gt_boxes'][gt_idx_list]
                    dists = np.linalg.norm(gt_boxes[:, :2] - p['l_pos'][:2], axis=1)
                    min_idx = np.argmin(dists)

                    if dists[min_idx] < dist_th and min_idx not in matched_gt_ids[p['sample_token']]:
                        tps[i] = 1
                        matched_gt_ids[p['sample_token']].add(min_idx)
                        if dist_th == 0.5:
                            actual_idx = gt_idx_list[min_idx]
                            # 确保获取的是 2D 速度
                            g_vel_all = info.get('gt_velocity', np.zeros((len(info['gt_boxes']), 2)))
                            g_vel = g_vel_all[actual_idx][:2] 
                            g_attr = info.get('gt_attributes', np.zeros(len(info['gt_boxes'])))[actual_idx]
                            
                            # 传入类别名称以进行误差掩码
                            errs = self._compute_tp_errors(p['l_pos'], p['l_rot'], p['l_vel'], p['size'], 
                                                          p.get('attribute', -1), info['gt_boxes'][actual_idx], 
                                                          g_vel, g_attr, cat)
                            for idx, key in enumerate(['ate', 'ase', 'aoe', 'ave', 'aae']):
                                stats[cat][key].append(errs[idx])
                    else: fps[i] = 1
                stats[cat]['aps'].append(self.calculate_ap_11point(tps, fps, total_gts))

        # self._show_final_metrics(stats, all_categories)
        # 最后的显示函数现在应该返回核心指标字典
        return self._show_final_metrics(stats, all_categories)

    def _show_final_metrics(self, stats, categories):
        # 定义 nuScenes 官方建议的固定显示顺序 
        target_order = [
            'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
            'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
        ]
        
        # 实际数据中存在的类别（确保不会因为顺序表里有但数据里没有而报错）
        existing_cats = set(stats.keys())
        
        print(f"\n{'Class':<25} | mAP | ATE | ASE | AOE | AVE | AAE")
        print("-" * 85)
        
        class_aps = []
        class_tps = [] 

        # 按照固定顺序遍历显示 [cite: 19, 34, 40]
        for cat in target_order:
            if cat not in existing_cats or not stats[cat]['aps']: 
                continue
            
            # 计算当前类别的平均 mAP [cite: 19]
            cat_ap = np.mean(stats[cat]['aps'])
            
            display_row = [f"{cat_ap:.3f}"]
            cat_tps_for_nds = []

            # 遍历五个 TP 指标
            for key in ['ate', 'ase', 'aoe', 'ave', 'aae']:
                # 获取该类别的平均误差，若为空则默认 0.0 以便 NDS 补偿 [cite: 3, 16, 17]
                val = np.mean(stats[cat][key]) if stats[cat][key] else 0.0
                cat_tps_for_nds.append(val)
                
                # 判断当前类别在该指标下是否应显示为 nan [cite: 16, 17, 21]
                is_nan = False
                if key == 'aoe' and cat in self.no_ang_cats: is_nan = True
                if key == 'ave' and cat in self.no_vel_cats: is_nan = True
                if key == 'aae' and cat in self.no_attr_cats: is_nan = True
                
                if is_nan:
                    display_row.append(f"{'nan':>4}")
                else:
                    display_row.append(f"{val:.2f}")

            print(f"{cat:<25} | {' | '.join(display_row)}")
            
            # 存储用于全局 NDS 计算的等权重数据 
            class_aps.append(cat_ap)
            class_tps.append(cat_tps_for_nds)
        
        # 处理顺序表之外可能存在的其他类别
        other_cats = [c for c in categories if c not in target_order]
        for cat in other_cats:
            # 此处逻辑与上方一致，略... (为保持代码简洁建议将显示逻辑封装)
            pass

        if class_aps:
            # 先分类别平均，再取全局平均，消除类别样本不平衡影响 [cite: 19, 20, 34]
            mAP = np.mean(class_aps)
            mTP_errors = np.mean(np.array(class_tps), axis=0)
            
            # NDS 公式：1/10 * [5*mAP + sum(1 - min(1, mTP))] [cite: 2, 3]
            # 对于 nan 显示项，其 mTP 为 0.0，故 (1 - 0) = 1.0 满分贡献 [cite: 3, 17, 40]
            tp_score = np.sum(1 - np.minimum(1.0, mTP_errors))
            nds = (5 * mAP + tp_score) / 10
            
            print("-" * 85)
            print(f"Overall mAP: {mAP:.4f} | Overall NDS: {nds:.4f}")
            # 返回字典以便主框架记录
            return {'mAP': mAP, 'NDS': nds}
        return {}

if __name__ == "__main__":
    RES_PATH = "/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/MV2DFusion/test/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep24_nusc/Mon_Jun_15_21_12_34_2026/pts_bbox/results_nusc.json"
    PKL_PATH = "/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_data_space/Special_Dataset_v1/mv2dfusion_temporal_infos_special_val.pkl" 
    print(f"Checking RES_PATH: {os.path.exists(RES_PATH)}")
    print(f"Checking PKL_PATH: {os.path.exists(PKL_PATH)}")
    # if os.path.exists(RES_PATH) and os.path.exists(PKL_PATH):
    #     NuScenesStandaloneEvaluator().evaluate(RES_PATH, PKL_PATH)
    if os.path.exists(RES_PATH) and os.path.exists(PKL_PATH):
        evaluator = NuScenesStandaloneEvaluator()
        print("Starting evaluation...")
        evaluator.evaluate(RES_PATH, PKL_PATH)
    else:
        print("Error: One of the paths does not exist!")