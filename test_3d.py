import os
import argparse
import torch

from networks.net_factory import net_factory
from myutils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Root path of the project')
parser.add_argument('--dataset_path', type=str, default='/home/jovyan/work/medical_dataset/LA', help='Path to the dataset')
parser.add_argument('--exp', type=str, default='cov_dfp_metric_learning', help='exp_name')
parser.add_argument('--model', type=str, default='corn', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=4, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--test_final', action='store_true', help='use the best pth or the final pth')
parser.add_argument('--consistency', type=float, default=1, help='lambda_c')
parser.add_argument('--consistency_o', type=float, default=0.05, help='lambda_s to balance sim loss')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--memory_num', type=int, default=256, help='num of embeddings per class in memory bank')
parser.add_argument('--embedding_dim', type=int, default=128, help='dim of embeddings to calculate similarity')
parser.add_argument('--num_filtered', type=int, default=12800,
                    help='num of unlabeled embeddings to calculate similarity')

# 新增参数以匹配训练脚本
parser.add_argument('--num_dfp', type=int, default=6, help='number of dynamic feature pools')
parser.add_argument('--lambda_compact', type=float, default=0.1, help='weight for intra-pool compactness loss')
parser.add_argument('--lambda_separate', type=float, default=0.05, help='weight for inter-pool separation loss')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# 修改snapshot_path构建方式，与训练脚本保持一致
snapshot_path = "./model/LA_{}_{}_dfp{}_memory{}_feat{}_compact{}_separate{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}".format(
    FLAGS.exp,
    FLAGS.labelnum,
    FLAGS.num_dfp,
    FLAGS.memory_num,
    FLAGS.embedding_dim,
    FLAGS.lambda_compact,
    FLAGS.lambda_separate,
    FLAGS.num_filtered,
    int(FLAGS.consistency) if FLAGS.consistency == int(FLAGS.consistency) else FLAGS.consistency,
    FLAGS.consistency_rampup,
    FLAGS.consistency_o,
    FLAGS.max_iteration,
    FLAGS.seed)

test_save_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}_predictions/".format(FLAGS.dataset_name, FLAGS.exp,
                                                                                   FLAGS.labelnum, FLAGS.model)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    with open(FLAGS.dataset_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.dataset_path + "/2018LA_Seg_Testing Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                  image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print("Model path:", snapshot_path)
print("Test save path:", test_save_path)


def test_calculate_metric():
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    final_iter_path = ''
    for file_path in os.listdir(snapshot_path):
        if str(FLAGS.max_iteration) in file_path:
            final_iter_path = file_path
            break
    if final_iter_path == '':
        return 'Saved checkpoint of the final iteration does not exist!'
    save_model_path = os.path.join(snapshot_path, final_iter_path)
    net.load_state_dict(torch.load(save_model_path), strict=False)
    print("init weight from {}".format(save_model_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        print(f"Test Begin")
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                                   patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                   save_result=True, test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail, nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
