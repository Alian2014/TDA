import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
'''
get_arguments() 为命令行参数解析函数，从命令行读取和解析参数，返回一个包含所有参数的 args 对象
'''


def get_arguments():
    """Get arguments of the test-time adaptation."""
    # 实例化对象，parser.add_argument 用于向命令行解析器添加参数规则
    parser = argparse.ArgumentParser()
    # 指定配置文件路径（YAML 格式），包含 TDA 的各种设置
    parser.add_argument(
        '--config',
        dest='config',
        required=True,
        help='settings of TDA on specific dataset in yaml format.')
    # 是否开启 Weights & Biases 的日志记录（实验可视化工具）
    parser.add_argument(
        '--wandb-log',
        dest='wandb',
        action='store_true',
        help=
        'Whether you want to log to wandb. Include this flag to enable logging.'
    )
    # 指定要处理的数据集（可能有多个，用 / 分隔）
    parser.add_argument(
        '--datasets',
        dest='datasets',
        type=str,
        required=True,
        help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S"
    )
    # 指定数据集根目录的路径
    parser.add_argument(
        '--data-root',
        dest='data_root',
        type=str,
        default='./dataset/',
        help='Path to the datasets directory. Default is ./dataset/')
    # 指定 CLIP 模型的骨干网络
    parser.add_argument('--backbone',
                        dest='backbone',
                        type=str,
                        choices=['RN50', 'ViT-B/16'],
                        required=True,
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')

    # 根据命令行返回 args，一个包含所有参数的命名空间对象
    args = parser.parse_args()

    return args


'''
update_cache(...)为缓存更新函数，
将某个类别 pred 对应的新样本（包含其特征与损失信息）加入缓存中，并按照容量上限进行替换策略管理

cache (dict):                   # 缓存字典，每个 key 是类别，每个 value 是一个列表，包含 (feature, loss[, prob_map])。
pred (int):                     # 当前样本预测类别。
features_loss (tuple):          # 新样本的特征与损失（可能包含 prob_map），如 (feature, loss) 或 (feature, loss, prob_map)。
shot_capacity (int):            # 每个类别缓存的最大容量。
include_prob_map (bool):        # 是否包含 prob_map 信息。
'''


def update_cache(cache,
                 pred,
                 features_loss,
                 shot_capacity,
                 include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    # 表示这个过程是非训练性的
    with torch.no_grad():
        # 如果 include_prob_map=True，说明 features_loss 中包含 [feature, loss, prob_map]，否则就直接拿全部内容作为缓存元素
        # prob_map.shape == [1, C], feature.shape == [1, D]
        item = features_loss if not include_prob_map else features_loss[:2] + [
            features_loss[2]
        ]
        # 类别 pred 已存在缓存中，尝试加入新样本
        if pred in cache:
            # 如果缓存中该类样本数量小于设定的上限
            if len(cache[pred]) < shot_capacity:
                #  将当前样本（item）添加进缓存字典 cache 中，属于某个类别 pred 的样本列表
                cache[pred].append(item)
            # 如果样本数量已达上限，则判断该样本的熵是否小于缓存中最后一个样本的熵
            elif features_loss[1] < cache[pred][-1][1]:
                # 若该样本熵低，则替换缓存中最后一个样本
                cache[pred][-1] = item
            # 按照熵进行排序，最后一个熵最大
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            # # 第一次遇到该类别，初始化其缓存列表
            cache[pred] = [item]


''' 
compute_cache_logits(...) 是典型的基于缓存（cache）的 logit 计算模块

image_features：                                        # Tensor [1, D]：当前待分类图像的特征（经过 CLIP 编码）
cache：dict：                                           # 缓存数据结构 {class_id: List[(feat, prob, entropy)]}
alpha：float：                                          # 权重系数，控制 cache logits 对最终 logits 的影响程度
beta：float：                                           # 缩放系数，控制特征相似度放大（用于 softmax sharpness）
clip_weights：                                          # Tensor [C, D]	：所有类别的 prompt 特征，用于对齐维度（有时用于 fallback）
neg_mask_thresholds：Tuple[float, float] or None：      # 仅在负缓存启用时，表示用于掩蔽筛选缓存样本的概率/熵范围
'''


def compute_cache_logits(image_features,
                         cache,
                         alpha,
                         beta,
                         clip_weights,
                         neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        # 所有缓存特征图
        cache_keys = []
        # 将 feature 添加到特征图列表
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) &
                             (cache_values < neg_mask_thresholds[1])).type(
                                 torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(
                torch.Tensor(cache_values).to(torch.int64),
                num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []

        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {
                k: pos_cfg[k]
                for k in ['shot_capacity', 'alpha', 'beta']
            }
        if neg_enabled:
            neg_params = {
                k: neg_cfg[k]
                for k in [
                    'shot_capacity', 'alpha', 'beta', 'entropy_threshold',
                    'mask_threshold'
                ]
            }

        #Test-time adaptation
        for i, (images, target) in enumerate(
                tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(
                images, clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(
                loss, clip_weights)

            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss],
                             pos_params['shot_capacity'])

            if neg_enabled and neg_params['entropy_threshold'][
                    'lower'] < prop_entropy < neg_params['entropy_threshold'][
                        'upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map],
                             neg_params['shot_capacity'], True)

            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache,
                                                     pos_params['alpha'],
                                                     pos_params['beta'],
                                                     clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(
                    image_features, neg_cache, neg_params['alpha'],
                    neg_params['beta'], clip_weights,
                    (neg_params['mask_threshold']['lower'],
                     neg_params['mask_threshold']['upper']))

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)
            wandb.log(
                {"Averaged test accuracy": sum(accuracies) / len(accuracies)},
                commit=True)

            if i % 1000 == 0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(
                    sum(accuracies) / len(accuracies)))
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(
            sum(accuracies) / len(accuracies)))
        return sum(accuracies) / len(accuracies)


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP",
                             config=cfg,
                             group=group_name,
                             name=run_name)

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader,
                           clip_model, clip_weights)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()


if __name__ == "__main__":
    main()
