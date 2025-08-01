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
        # 每个缓存对应的类别或概率图
        cache_values = []
        # 将 feature 添加到特征图列表
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                # 如果启用了 neg_mask_thresholds，说明当前你希望使用 item[2] 的概率图（prob_map）进行筛选或加权
                if neg_mask_thresholds:
                    # 缓存值中存入概率图
                    cache_values.append(item[2])
                else:
                    # 否则，就是普通的分类任务，将类别存入缓存
                    cache_values.append(class_index)

        # 拼接所有缓存图像特征为 [N, D]，然后转置为 [D, N]
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        # 如果启用阈值过滤
        if neg_mask_thresholds:
            # torch.cat(..., dim=0) 把它们拼成一个 [N, D] 的 tensor
            cache_values = torch.cat(cache_values, dim=0)
            # 阈值过滤 (val > th0) & (val < th1) 得到 [N, D] 的 mask，过滤缓存中不可靠的特征维度
            # 随后将 bool 类型转成 int8（0 或 1），再转成 float16 并搬到 GPU
            cache_values = (((cache_values > neg_mask_thresholds[0]) &
                             (cache_values < neg_mask_thresholds[1])).type(
                                 torch.int8)).cuda().half()
        else:
            # F.one_hot(...) → [N, C]：将每个类别索引转换成 one-hot 标签
            # .cuda().half()：转为 float16 并搬到 CUDA
            cache_values = (F.one_hot(
                torch.Tensor(cache_values).to(torch.int64),
                num_classes=clip_weights.size(1))).cuda().half()

        # 当前图像特征与 N 个缓存样本的特征向量相乘后得到每个缓存样本与该图像的相似度
        # affinity.shape == [1, N]
        affinity = image_features @ cache_keys
        # beta - beta * affinity = beta * (1 - affinity) 越相似的 affinity 趋近于 1，这项趋近于 0；越不相似的 affinity 趋近于 0，这项趋近于 beta
        # (-1 * ...).exp() = exp(beta * (affinity - 1)) 使得越相似的 affinity 得到的权重越大（接近 1），越不相似的权重接近 0
        # cache_logits = weight @ cache_values 是一个典型的加权平均操作
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # return alpha * cache_logits, affinity
        # 最后输出一个加权后的向量（特征或伪 logit），乘以调节系数 alpha
        # 返回值 shape 保持为 [1, D]
        return alpha * cache_logits


'''
run_test_tda(...) 用于测试 TDA 在分类任务中的准确率

pos_cfg, neg_cfg：                      # 用于构建正负样本缓存的配置
loader：                                # PyTorch 的 DataLoader 对象，加载测试集 
clip_model：                            # CLIP 模型（或其变体），用来提取图像特征
clip_model：                            # CLIP 模型（或其变体），用来提取图像特征
logger：                                # 日志记录器
args, cfg：                             # 其他控制参数或配置对象
'''


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []

        # Unpack all hyperparameters
        # 解包配置中的开关状态,从配置字典 pos_cfg, neg_cfg 中读取是否启用对应缓存机制
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        # 如果启用正缓存，提取 pos_cfg 中的主要参数
        if pos_enabled:
            pos_params = {
                k: pos_cfg[k]
                for k in ['shot_capacity', 'alpha', 'beta']
            }
        # 如果启用负缓存，额外会多两个筛选参数
        if neg_enabled:
            neg_params = {
                k: neg_cfg[k]
                for k in [
                    'shot_capacity', 'alpha', 'beta', 'entropy_threshold',
                    'mask_threshold'
                ]
            }

        #Test-time adaptation
        # 遍历测试集 loader，每次取出一批 images 和其真实标签 target，在遍历同时解包
        '''
        这是一个迭代器包裹写法

        loader：                                # 通常是一个 PyTorch 的 DataLoader，会返回 (images, target) 的 batch。
        tqdm(...)：                             # 将 loader 包裹，显示处理进度。
        desc='Processed test images: '：        # 指定进度条前缀文本
        '''
        for i, (images, target) in enumerate(
                tqdm(loader, desc='Processed test images: ')):
            # 使用目前提供的数据集时，infer_ori_image = True 总成立
            # 获取图像特征与 CLIP 输出
            '''
            image_features:                 # 原图图像特征向量
            clip_logits：                   # 原图的 zero-shot logits
            loss：                          # 原图的 zero-shot 熵
            prob_map：                      # 原图的 zero-shot 概率分布
            pre：                           # 原图的 zero-shot 预测类别
            '''
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(
                images, clip_model, clip_weights)
            # 将 target 移到 GPU 上，get_entropy() 可能是根据 loss 或 clip_logits 计算预测分布的不确定性（熵）
            target, prop_entropy = target.cuda(), get_entropy(
                loss, clip_weights)
            '''
            如果启用 positive cache，就调用 update_cache() 更新缓存
            
            pred:                           # 当前样本原图的预测的类别；
            [image_features, loss]:         # 要缓存的原图图像特征和其对应的 zero-shot 熵（可用于排序）；
            shot_capacity:                  # 每类最多缓存多少个样本；
            '''
            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss],
                             pos_params['shot_capacity'])

            # 如果启用 negative cache 且当前样本的归一化熵（前面由 get_entropy() 得到）在给定的熵筛选区间
            if neg_enabled and neg_params['entropy_threshold'][
                    'lower'] < prop_entropy < neg_params['entropy_threshold'][
                        'upper']:
                # 使用 update_cache 把 [image_features, loss, prob_map] 放入对应 pred 类别的 neg_cache 中
                update_cache(neg_cache, pred, [image_features, loss, prob_map],
                             neg_params['shot_capacity'], True)

            # 初始 logits 仅为 CLIP 输出，用作基础
            final_logits = clip_logits.clone()
            # 加入正缓存得分
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache,
                                                     pos_params['alpha'],
                                                     pos_params['beta'],
                                                     clip_weights)
            # 减去负缓存得分，这样能有效抑制误分类倾向（比如视觉相似但语义不同的类别）
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(
                    image_features, neg_cache, neg_params['alpha'],
                    neg_params['beta'], clip_weights,
                    (neg_params['mask_threshold']['lower'],
                     neg_params['mask_threshold']['upper']))

            # 使用最终融合后的 final_logits 进行分类
            acc = cls_acc(final_logits, target)
            # 将该 batch 的准确率加入 accuracies 列表，供最后汇总
            accuracies.append(acc)

            # wandb.log() 是用来上传一个或多个指标到 Weights & Biases 仪表盘的方法，一般在训练/测试过程中定期调用
            # 在 Weights & Biases 仪表盘中，你会看到：一个名为 "Averaged test accuracy" 的图线或表格。
            # 每次调用 wandb.log(..., commit=True)，它会成为图表中的一个点（step）
            wandb.log(
                {"Averaged test accuracy": sum(accuracies) / len(accuracies)},
                commit=True)

            # 每隔 1000 个测试样本/批次，打印一次当前的平均测试准确率，格式化为保留两位小数的输出
            if i % 1000 == 0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(
                    sum(accuracies) / len(accuracies)))
        # 打印一次最终的平均测试准确率，格式化为保留两位小数的输出
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(
            sum(accuracies) / len(accuracies)))
        return sum(accuracies) / len(accuracies)


def main():
    # 调用 get_arguments()，解析命令行参数
    args = get_arguments()
    # 获取配置文件路径
    config_path = args.config

    # Initialize CLIP model
    # 从 args.backbone 指定路径加载 CLIP 模型和对应图像预处理方法
    clip_model, preprocess = clip.load(args.backbone)
    # 将模型设为推理模式，关闭 dropout 等
    clip_model.eval()

    # Set random seed
    # 设置随机种子（保证实验可复现）
    random.seed(1)
    torch.manual_seed(1)

    # 如果你开启了 wandb 日志（通过命令行参数 --wandb），则执行以下逻辑
    if args.wandb:
        # 当前时间
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        # wandb 分组名，用于将多个数据集下的运行结果统一归入一个组
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    # Run TDA on each dataset
    # 遍历多个数据集并处理，如果你设置了 --datasets imagenet/caltech101, 就会自动拆分成多个子任务
    datasets = args.datasets.split('/')
    # 对数据集路径下每个数据集
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        # 获取设置文件
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        # build_test_data_loader()：加载测试图像数据（可迭代的 DataLoader）、类别名称、prompt 模板
        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess)
        # clip_classifier()：基于类别名和 prompt 模板，使用 CLIP 的 encode_text() 得到每个类别的特征向量（class centers）
        clip_weights = clip_classifier(classnames, template, clip_model)

        # 启动 wandb 日志（每个数据集一个 run）
        if args.wandb:
            '''
            wandb.init()：              # 开始记录一个实验运行，项目名是 ETTA-CLIP。
            config=cfg：                # 将该数据集的配置也记录到仪表盘中。
            group=group_name：          # 每个组可能对应一次完整运行，便于多数据集对比。
            name=run_name：             # 这个子实验的名字，就是当前数据集名
            '''
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP",
                             config=cfg,
                             group=group_name,
                             name=run_name)

        # 运行测试阶段的 TDA 算法，输出该数据集下的平均准确率（acc）
        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader,
                           clip_model, clip_weights)

        if args.wandb:
            # 记录本次测试的准确率
            wandb.log({f"{dataset_name}": acc})
            # 关闭当前 wandb run，释放资源
            run.finish()


if __name__ == "__main__":
    main()
