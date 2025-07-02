import os
import argparse
import torch
import numpy as np
import h5py
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from medpy import metric

from networks.net_factory import net_factory
# from myutils.test_patch import test_all_case_2d  # 不需要导入，我们自己实现测试函数

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ACDC', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Root path of the project')
parser.add_argument('--dataset_path', type=str, default='/home/jovyan/work/medical_dataset/ACDC_processed', help='Path to the processed ACDC dataset')
parser.add_argument('--exp', type=str, default='acdc_soss', help='exp_name')
parser.add_argument('--model', type=str, default='corn2d', help='model_name')
parser.add_argument('--max_iteration', type=int, default=20000, help='maximum iteration to train')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--test_final', action='store_true', help='use the best pth or the final pth')
parser.add_argument('--consistency', type=float, default=1.0, help='lambda_c')
parser.add_argument('--consistency_o', type=float, default=0.05, help='lambda_s to balance sim loss')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--memory_num', type=int, default=256, help='num of embeddings per class in memory bank')
parser.add_argument('--embedding_dim', type=int, default=64, help='dim of embeddings to calculate similarity')
parser.add_argument('--num_filtered', type=int, default=12800,
                    help='num of unlabeled embeddings to calculate similarity')

# ACDC特定参数
parser.add_argument('--patch_size', type=tuple, default=(256, 256), help='patch size for 2D ACDC')
parser.add_argument('--num_classes', type=int, default=4, help='number of classes for ACDC: 0-background, 1-LV, 2-RV, 3-MYO')

# 新增参数以匹配训练脚本
parser.add_argument('--num_dfp', type=int, default=8, help='number of dynamic feature pools')
parser.add_argument('--lambda_compact', type=float, default=0.1, help='weight for intra-pool compactness loss')
parser.add_argument('--lambda_separate', type=float, default=0.05, help='weight for inter-pool separation loss')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# 创建多个可能的路径变体，以匹配训练脚本的实际行为
def find_model_snapshot_path():
    """查找实际存在的模型路径"""
    # 可能的路径变体
    path_variants = [
        # 变体1：exp包含模型名（训练时可能的情况）
        "./model/ACDC_{}_{}_{}_dfp{}_memory{}_feat{}_compact{}_separate{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
            FLAGS.exp, FLAGS.model, FLAGS.labelnum, FLAGS.num_dfp, FLAGS.memory_num, FLAGS.embedding_dim,
            FLAGS.lambda_compact, FLAGS.lambda_separate, FLAGS.num_filtered, 
            int(FLAGS.consistency), FLAGS.consistency_rampup, FLAGS.consistency_o,
            FLAGS.max_iteration, FLAGS.seed, FLAGS.model),
        # 变体2：标准格式（整数consistency）
        "./model/ACDC_{}_{}_dfp{}_memory{}_feat{}_compact{}_separate{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
            FLAGS.exp, FLAGS.labelnum, FLAGS.num_dfp, FLAGS.memory_num, FLAGS.embedding_dim,
            FLAGS.lambda_compact, FLAGS.lambda_separate, FLAGS.num_filtered, 
            int(FLAGS.consistency), FLAGS.consistency_rampup, FLAGS.consistency_o,
            FLAGS.max_iteration, FLAGS.seed, FLAGS.model),
        # 变体3：浮点数consistency
        "./model/ACDC_{}_{}_dfp{}_memory{}_feat{}_compact{}_separate{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
            FLAGS.exp, FLAGS.labelnum, FLAGS.num_dfp, FLAGS.memory_num, FLAGS.embedding_dim,
            FLAGS.lambda_compact, FLAGS.lambda_separate, FLAGS.num_filtered, 
            FLAGS.consistency, FLAGS.consistency_rampup, FLAGS.consistency_o,
            FLAGS.max_iteration, FLAGS.seed, FLAGS.model),
        # 变体4：exp包含模型名+浮点数consistency
        "./model/ACDC_{}_{}_{}_dfp{}_memory{}_feat{}_compact{}_separate{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
            FLAGS.exp, FLAGS.model, FLAGS.labelnum, FLAGS.num_dfp, FLAGS.memory_num, FLAGS.embedding_dim,
            FLAGS.lambda_compact, FLAGS.lambda_separate, FLAGS.num_filtered, 
            FLAGS.consistency, FLAGS.consistency_rampup, FLAGS.consistency_o,
            FLAGS.max_iteration, FLAGS.seed, FLAGS.model)
    ]
    
    # 检查哪个路径存在
    for path in path_variants:
        if os.path.exists(path):
            print(f"找到模型路径: {path}")
            return path
    
    # 如果都不存在，打印所有尝试的路径以便调试
    print("未找到匹配的模型路径，尝试的路径包括：")
    for i, path in enumerate(path_variants, 1):
        print(f"变体{i}: {path}")
    
    return None

snapshot_path = find_model_snapshot_path()
if snapshot_path is None:
    print("错误：无法找到模型路径!")
    exit(1)

test_save_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}_predictions/".format(FLAGS.dataset_name, FLAGS.exp,
                                                                                   FLAGS.labelnum, FLAGS.model)

num_classes = FLAGS.num_classes
patch_size = FLAGS.patch_size

# ACDC数据集处理
if FLAGS.dataset_name == "ACDC":
    # 从验证集H5文件中读取数据
    val_h5_path = os.path.join(FLAGS.dataset_path, 'val.h5')
    if not os.path.exists(val_h5_path):
        print(f"错误: 找不到验证集文件 {val_h5_path}")
        exit(1)
    
    # 读取验证集数据
    with h5py.File(val_h5_path, 'r') as f:
        val_images = f['image'][:]  # [N, 1, H, W]
        val_labels = f['label'][:]  # [N, H, W]
        val_patient_ids = [pid.decode('utf-8') for pid in f['patient_id'][:]]
        val_frame_types = [ft.decode('utf-8') for ft in f['frame_type'][:]]
        if 'slice_idx' in f:
            val_slice_indices = f['slice_idx'][:]
        else:
            val_slice_indices = list(range(len(val_images)))
    
    print(f"验证集数据:")
    print(f"  图像数量: {len(val_images)}")
    print(f"  图像形状: {val_images.shape}")
    print(f"  标签形状: {val_labels.shape}")
    print(f"  唯一患者ID: {len(set(val_patient_ids))}")

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print("Model path:", snapshot_path)
print("Test save path:", test_save_path)


def calculate_metric_percase(pred, gt):
    """计算单个样本的指标"""
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    
    if pred.sum() > 0 and gt.sum() > 0:
        # 计算Dice系数
        intersection = (pred & gt).sum()
        dice = (2. * intersection) / (pred.sum() + gt.sum())
        
        # 计算HD95（如果medpy可用）
        try:
            from medpy import metric as medpy_metric
            hd95 = medpy_metric.binary.hd95(pred, gt)
        except:
            hd95 = 0  # 如果计算失败，设为0
        
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0, float('inf')
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0, float('inf')
    else:
        return 1, 0


def test_single_slice(net, image, label, class_num):
    """测试单个2D切片"""
    image = torch.from_numpy(image).unsqueeze(0).float().cuda()  # [1, 1, H, W]
    
    net.eval()
    with torch.no_grad():
        output = net(image)
        if isinstance(output, dict):
            out = output['seg1']  # 使用第一个分割输出
        elif isinstance(output, (list, tuple)):
            out = output[0]  # 使用第一个输出
        else:
            out = output
        
        out = torch.softmax(out, dim=1)
        prediction = torch.argmax(out, dim=1).squeeze(0)
        prediction = prediction.cpu().data.numpy()
    
    return prediction


def test_all_case_acdc(net, val_images, val_labels, val_patient_ids, val_frame_types, 
                      num_classes=4, save_result=True, test_save_path=None, metric_detail=0):
    """测试所有ACDC验证集样本"""
    total_metric = 0.0
    metric_dict = {}
    
    # 按患者分组数据
    patient_data = {}
    for i, pid in enumerate(val_patient_ids):
        if pid not in patient_data:
            patient_data[pid] = {'images': [], 'labels': [], 'frame_types': [], 'indices': []}
        patient_data[pid]['images'].append(val_images[i])
        patient_data[pid]['labels'].append(val_labels[i])
        patient_data[pid]['frame_types'].append(val_frame_types[i])
        patient_data[pid]['indices'].append(i)
    
    print(f"测试 {len(patient_data)} 个患者")
    
    for patient_id, data in tqdm(patient_data.items(), desc="测试患者"):
        patient_images = np.array(data['images'])  # [n_slices, 1, H, W]
        patient_labels = np.array(data['labels'])  # [n_slices, H, W]
        
        # 预测每个切片
        predictions = []
        for i in range(len(patient_images)):
            pred = test_single_slice(net, patient_images[i], patient_labels[i], num_classes)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # [n_slices, H, W]
        
        # 计算每个类别的指标
        patient_metrics = {}
        for class_idx in range(1, num_classes):  # 跳过背景类
            # 二值化预测和真实标签
            pred_binary = (predictions == class_idx).astype(np.uint8)
            gt_binary = (patient_labels == class_idx).astype(np.uint8)
            
            # 计算3D指标（将所有切片合并）
            pred_3d = pred_binary.reshape(-1)
            gt_3d = gt_binary.reshape(-1)
            
            if gt_3d.sum() > 0:  # 只有当真实标签中存在该类别时才计算
                dice, hd95 = calculate_metric_percase(pred_3d, gt_3d)
                patient_metrics[f'class_{class_idx}_dice'] = dice
                patient_metrics[f'class_{class_idx}_hd95'] = hd95
            else:
                patient_metrics[f'class_{class_idx}_dice'] = np.nan
                patient_metrics[f'class_{class_idx}_hd95'] = np.nan
        
        # 计算平均指标
        valid_dices = [v for k, v in patient_metrics.items() if 'dice' in k and not np.isnan(v)]
        patient_avg_dice = np.mean(valid_dices) if valid_dices else 0
        
        metric_dict[patient_id] = patient_metrics
        metric_dict[patient_id]['avg_dice'] = patient_avg_dice
        
        if metric_detail:
            print(f"患者 {patient_id}: 平均Dice = {patient_avg_dice:.4f}")
            for class_idx in range(1, num_classes):
                dice = patient_metrics.get(f'class_{class_idx}_dice', np.nan)
                hd95 = patient_metrics.get(f'class_{class_idx}_hd95', np.nan)
                if not np.isnan(dice):
                    print(f"  类别 {class_idx}: Dice = {dice:.4f}, HD95 = {hd95:.2f}")
        
        # 保存预测结果
        if save_result and test_save_path:
            # 重建3D体积并保存
            patient_dir = os.path.join(test_save_path, patient_id)
            if not os.path.exists(patient_dir):
                os.makedirs(patient_dir)
            
            # 保存为NIFTI格式
            pred_volume = predictions.transpose(1, 2, 0)  # [H, W, n_slices]
            pred_sitk = sitk.GetImageFromArray(pred_volume)
            sitk.WriteImage(pred_sitk, os.path.join(patient_dir, f"{patient_id}_pred.nii.gz"))
            
            gt_volume = patient_labels.transpose(1, 2, 0)  # [H, W, n_slices]
            gt_sitk = sitk.GetImageFromArray(gt_volume)
            sitk.WriteImage(gt_sitk, os.path.join(patient_dir, f"{patient_id}_gt.nii.gz"))
    
    # 计算总体指标
    all_avg_dices = [data['avg_dice'] for data in metric_dict.values() if not np.isnan(data['avg_dice'])]
    overall_avg_dice = np.mean(all_avg_dices) if all_avg_dices else 0
    
    # 计算每个类别的总体指标
    class_metrics = {}
    for class_idx in range(1, num_classes):
        class_dices = [data[f'class_{class_idx}_dice'] for data in metric_dict.values() 
                      if f'class_{class_idx}_dice' in data and not np.isnan(data[f'class_{class_idx}_dice'])]
        class_hd95s = [data[f'class_{class_idx}_hd95'] for data in metric_dict.values() 
                      if f'class_{class_idx}_hd95' in data and not np.isnan(data[f'class_{class_idx}_hd95']) and data[f'class_{class_idx}_hd95'] != float('inf')]
        
        if class_dices:
            class_metrics[f'class_{class_idx}_avg_dice'] = np.mean(class_dices)
            class_metrics[f'class_{class_idx}_std_dice'] = np.std(class_dices)
        if class_hd95s:
            class_metrics[f'class_{class_idx}_avg_hd95'] = np.mean(class_hd95s)
            class_metrics[f'class_{class_idx}_std_hd95'] = np.std(class_hd95s)
    
    print(f"\n=== 总体测试结果 ===")
    print(f"总体平均Dice: {overall_avg_dice:.4f}")
    
    class_names = ['', 'LV', 'RV', 'MYO']  # ACDC类别名称
    for class_idx in range(1, num_classes):
        if f'class_{class_idx}_avg_dice' in class_metrics:
            avg_dice = class_metrics[f'class_{class_idx}_avg_dice']
            std_dice = class_metrics[f'class_{class_idx}_std_dice']
            print(f"{class_names[class_idx]} - Dice: {avg_dice:.4f} ± {std_dice:.4f}")
            
            if f'class_{class_idx}_avg_hd95' in class_metrics:
                avg_hd95 = class_metrics[f'class_{class_idx}_avg_hd95']
                std_hd95 = class_metrics[f'class_{class_idx}_std_hd95']
                print(f"{class_names[class_idx]} - HD95: {avg_hd95:.2f} ± {std_hd95:.2f}")
    
    return overall_avg_dice


def test_calculate_metric():
    """主测试函数"""
    # 确保snapshot_path不为None
    if snapshot_path is None:
        return 'Cannot find model snapshot path!'
        
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    
    # 查找最终迭代的模型文件
    final_iter_path = ''
    if os.path.exists(snapshot_path):
        for file_path in os.listdir(snapshot_path):
            if str(FLAGS.max_iteration) in file_path and file_path.endswith('.pth'):
                final_iter_path = file_path
                break
    
    if final_iter_path == '':
        print(f'错误: 在 {snapshot_path} 中找不到最终迭代的保存模型!')
        return 'Saved checkpoint of the final iteration does not exist!'
    
    save_model_path = os.path.join(snapshot_path, final_iter_path)
    
    try:
        net.load_state_dict(torch.load(save_model_path), strict=False)
        print("模型加载成功: {}".format(save_model_path))
    except Exception as e:
        print(f"模型加载失败: {e}")
        return f'Failed to load model: {e}'
    
    net.eval()

    if FLAGS.dataset_name == "ACDC":
        print(f"开始ACDC测试")
        avg_metric = test_all_case_acdc(net, val_images, val_labels, val_patient_ids, val_frame_types,
                                       num_classes=num_classes, save_result=True, 
                                       test_save_path=test_save_path, metric_detail=FLAGS.detail)
    else:
        print(f"不支持的数据集: {FLAGS.dataset_name}")
        return f'Unsupported dataset: {FLAGS.dataset_name}'
    
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(f"\n最终测试指标: {metric}") 