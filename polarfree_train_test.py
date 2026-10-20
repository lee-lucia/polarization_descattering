import os
import shutil

def sync_dirs(train_dir):
    gt_dir = os.path.join(train_dir, 'gt')
    input_dir = os.path.join(train_dir, 'input')

    # 获取两个目录下所有子文件夹名称集合
    gt_subs = {d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))}
    input_subs = {d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))}

    # 计算交集和差集
    common = gt_subs & input_subs
    gt_only = gt_subs - input_subs
    input_only = input_subs - gt_subs

    # 删除 gt 中多余的文件夹
    for d in gt_only:
        path = os.path.join(gt_dir, d)
        shutil.rmtree(path)
        print(f'已删除 GT 独有文件夹: {path}')

    # 删除 input 中多余的文件夹
    for d in input_only:
        path = os.path.join(input_dir, d)
        shutil.rmtree(path)
        print(f'已删除 INPUT 独有文件夹: {path}')

    print(f'同步完毕，仅保留以下共同文件夹：{sorted(common)}')

if __name__ == '__main__':
    # 将这里改为你的 train 目录路径
    train_root = '/home/tcpd/PolarFree/PolaRGB/train/hard'
    sync_dirs(train_root)
