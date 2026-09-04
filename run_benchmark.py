"""
2026.09.04
论文benchmark数据集复现
"""
import pandas as pd
import numpy as np
from cogamereg import *
import os
import pandas as pd
import numpy as np
from scipy.io import arff

def load_arff_data(file_path):
    """内部函数：读取 .arff 并解码 byte strings"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    try:
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # 解码 byte strings
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except AttributeError:
                    pass
        return df
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return None

def process_dataset_generic(dataframe, target_col=None, label_ratio=0.025, n_train_num=2000, random_seed=42):
    """内部函数：通用的半监督数据切分与预处理逻辑"""
    df = dataframe.copy()
    
    # 1. 确定目标列 (默认最后一列)
    if target_col is None:
        target_col = df.columns[-1]
    
    # X = df.drop(columns=[target_col])
    # y = df[[target_col]]
    
    # # 2. 自动特征编码 (处理类别特征)
    # le = LabelEncoder()
    # for col in X.columns:
    #     if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
    #         X.loc[:, col] = le.fit_transform(X[col])

    # --- 🔴 修改开始：针对性处理类别特征 (One-Hot) ---
    # 找出所有的类别列（object 或 category）
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 只要不是目标列，都做 One-Hot
    cat_cols = [c for c in cat_cols if c != target_col]
    
    if cat_cols:
        # drop_first=True 是为了避免多重共线性（对于线性模型很重要）
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # --- 🔴 修改结束 ---

    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    # 3. 整合并打乱
    all_data_df = pd.concat([X, y], axis=1)
    feature_cols = X.columns.tolist()
    
    all_data_shuffled = all_data_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 4. 动态切分数量
    total_samples = len(all_data_shuffled)
    current_train_num = n_train_num
    
    # 如果数据量不足，自动调整为 80%
    if n_train_num >= total_samples:
        current_train_num = int(total_samples * 0.8)
        print(f"⚠️ 样本不足 ({total_samples})，已调整训练池大小为: {current_train_num}")

    n_test = total_samples - current_train_num
    
    # 5. 切分逻辑
    # 测试集 / 训练池
    test_df = all_data_shuffled.iloc[:n_test]
    train_pool = all_data_shuffled.iloc[n_test:]
    
    # 有标签 / 无标签
    n_labeled_total = int(current_train_num * label_ratio)
    labeled_pool = train_pool.iloc[:n_labeled_total]
    unlabeled_df = train_pool.iloc[n_labeled_total:]
    
    # 验证集 / 训练集 (20% 的有标签数据作为验证)
    n_val = int(n_labeled_total * 0.20)
    # n_val = 0


    # if n_val < 1 and n_labeled_total >= 2: n_val = 1 # 兜底
    
    val_df = labeled_pool.iloc[:n_val]
    train_df = labeled_pool.iloc[n_val:]
    
    # 6. 转换为 Numpy & 标准化 (仅利用训练集统计量)
    data_u = unlabeled_df[feature_cols].values
    data_l = train_df[feature_cols].values
    data_v = val_df[feature_cols].values
    data_t = test_df[feature_cols].values
    
    y_train = train_df[target_col].values.flatten()
    y_val = val_df[target_col].values.flatten()
    y_test = test_df[target_col].values.flatten()
    real = unlabeled_df[target_col].values.flatten()
    
    if len(data_l) > 1:
        mean, std = data_l.mean(axis=0), data_l.std(axis=0)
        std[std == 0] = 1
    else:
        mean, std = np.zeros(len(feature_cols)), np.ones(len(feature_cols))

    data_u = (data_u - mean) / std
    data_l = (data_l - mean) / std
    if len(data_v) > 0: data_v = (data_v - mean) / std
    data_t = (data_t - mean) / std

    # print("-" * 30)
    # print(f"✅ 处理完成 (Label Ratio: {label_ratio*100}%)")
    # print(f"  - 总样本数: {total_samples}")
    # print(f"  - 特征数量: {len(feature_cols)}")
    # print(f"  - 设定训练池大小: {n_train_num}")
    # print(f"  - 自动计算测试集: {n_test}")
    # print(f"  - 有标签池总数: {n_labeled_total}")
    # print(f"    -> 验证集 (20%): {len(val_df)}")
    # print(f"    -> 训练集 (80%): {len(train_df)}")
    # print(f"  - 无标签集: {len(unlabeled_df)}")
    # print("-" * 30)

    # 返回 13 个变量
    return (all_data_df, data_u, data_l, y_train, train_df, real, train_df.index, 
            data_t, y_test, test_df, test_df.index, data_v, y_val)

# =========================================================
# ✅ 核心接口：供您的回归算法直接调用
# =========================================================
def get_data_by_path(file_path, label_ratio=0.05, n_train_num=2000, random_seed=42):
    """
    输入: ARFF 文件路径
    输出: 包含所有预处理后矩阵的元组 (可以直接解包)
    """
    print(f"🚀 正在加载数据集: {file_path}")
    
    # 1. 加载
    df = load_arff_data(file_path)
    if df is None:
        raise ValueError("数据加载失败，请检查路径。")
        
    # 2. 处理 (自动识别最后一列为目标)
    results = process_dataset_generic(df, target_col=None, label_ratio=label_ratio, n_train_num=n_train_num, random_seed=random_seed)
    
    print(f"✅ 数据准备完毕! (Train Labeled: {len(results[3])}, Val: {len(results[12])}, Test: {len(results[8])})")
    
    return results



# ==========================================
#  主流程调用函数
# ==========================================


import numpy as np
# 如果你需要自定义高斯过程的核函数，需要引入:
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ==========================================
#  模型参数配置池 (Model Zoo)
# ==========================================

model_zoo_ablation = {
    # ----------------------------------------
    # 1. 树模型与集成学习 (通常效果最好)
    # ----------------------------------------
    'xgb': {
        'n_estimators': 100,       # 树的数量 (范围: 50-1000)
        'learning_rate': 0.3,      # 学习率 (范围: 0.01-0.3)
        'max_depth': 6,            # 树深 (范围: 3-10，太深容易过拟合)
        'subsample': 1.0,          # 样本采样率 (范围: 0.5-1.0)
        'colsample_bytree': 1.0,   # 特征采样率 (范围: 0.5-1.0)
        'reg_alpha': 0,            # L1 正则化 (范围: 0-1)
        'reg_lambda': 1,           # L2 正则化 (范围: 0-1)
    },

    # # ----------------------------------------
    # # 2. 支持向量机 (适合中小数据集)
    # # ----------------------------------------
    'svr': { # 非线性支持向量机
        'kernel': 'rbf',           # 核函数 ('rbf', 'linear', 'poly')
        'C': 1.0,                  # 惩罚系数 (范围: 0.1 - 1000)
        'epsilon': 0.1,            # 容忍度
        'gamma': 'scale'           # 核系数 ('scale', 'auto')
    },


    # # ----------------------------------------
    # # 3. 神经网络 (适合复杂非线性)
    # # ----------------------------------------
    'mlp': { # 多层感知机
        'hidden_layer_sizes': (100, 50), # 隐藏层结构 (100,) 表示一层100个神经元
        'activation': 'relu',      # 激活函数 ('relu', 'tanh', 'logistic')
        'solver': 'sgd',          # 优化器 ('adam', 'lbfgs'-小数据推荐, 'sgd')
        'alpha': 0.0001,           # L2正则化项
        'learning_rate_init': 0.001,
        'max_iter': 2000            # 最大迭代次数
    },

    # # ----------------------------------------
    # # 4. 线性模型与正则化 (适合高维稀疏数据)
    # # ----------------------------------------
    'ridge': { # 岭回归 (L2正则)
        'alpha': 1.0,              # 正则力度，越大越不易过拟合
        'solver': 'auto'
    },
    
    'linear': { # 普通最小二乘法
        'fit_intercept': True      # 是否计算截距
    },

    # ----------------------------------------
    # 5. 其他 (近邻与高斯过程)
    # ----------------------------------------
    'knn': { # K近邻
        'n_neighbors': 5,          # 邻居数
        'weights': 'uniform',      # 权重 ('uniform', 'distance'-越近越重要)
        'algorithm': 'auto',
        'p': 2                     # 1=曼哈顿距离, 2=欧氏距离
    },

    
}



import os
import csv
import itertools
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

# 屏蔽 sklearn 的收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.metrics import mean_squared_error, r2_score
# 引入上面的类

def run_calibration_experiment(file_path, label_ratio=0.005, runs=5):
    """
    运行原始 COREG 校准实验，用于和论文表格对比。
    默认 0.5% (0.005) 标签率。
    """
    print("="*60)
    print(f"🏁 启动基准校准 (Calibration Mode)")
    print(f"Target Dataset: {file_path}")
    print(f"Label Ratio: {label_ratio} | Runs: {runs}")
    print("="*60)
    
    rmses = []
    
    for i in range(runs):
        seed = 42 + i
        
        # 1. 获取数据 (使用你现有的函数)
        # 注意：这里不需要 TripleNet，我们只取原始特征
        (all_data, data_u, data_l, y_train, train_df, real, train_idx, 
         data_t, y_test, test_df, test_idx, 
         data_v, y_val) = get_data_by_path(file_path, label_ratio=label_ratio, random_seed=seed)
        

        # ours
        model_name = "CoGameReg"
        y_pred = CoGameReg_validation_data_benchmark(
            data_l=data_l, 
            y=y_train,         
            data_u=data_u,       
            data_t=data_t,   
            model_zoo_config=model_zoo_ablation,
            ablation_mode='undecouple_shap',# 可选: 'base', 'decouple_shap', 'undecouple_shap', 'upgrade_weight'
            T=10,              
            s=5,               
            k=3,
            metric='euclidean', 
            threshold=1e-4,
            prefer='mse',
            val_random_state=seed,
            k_per_labeled=3,
            alpha=0.2,
            beta=0.4,
        )

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Run {i+1}/{runs} | RMSE: {rmse:.4f}")
        rmses.append(rmse)
        
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    
    print("-" * 60)
    print(f"📊 Calibration Result for {os.path.basename(file_path)}")
    # print(f"My COREG: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"{model_name}: {mean_rmse:.4f} ± {std_rmse:.4f}")

    print("-" * 60)
    
    return mean_rmse, std_rmse


# import os
# import numpy as np

def main_batch_experiment():
    # 1. 定义数据集路径列表（排除掉你注释掉的那两个）
    data_dir = "data/"
    dataset_files = [
        "abalone.arff", "bank32nh.arff", "elevators.arff", 
        "Folds5x2_pp.arff", "kin8nm.arff", "parkinsons.arff",
        "puma8NH.arff", "space_ga.arff", "wind.arff", "wine_quality.arff"
    ]
       
    
    # 2. 定义标签比例
    ratios = [0.025, 0.05, 0.1]


    
    # 3. 存储所有结果用于最后统一打印
    # 结构: {ratio: {dataset_name: "mean ± std"}}
    all_results = {r: {} for r in ratios}

    for ratio in ratios:
        print(f"\n\n" + "#"*80)
        print(f"### 正在进行 Label Ratio: {ratio} 的全量实验 ###")
        print("#"*80)
        
        for filename in dataset_files:
            file_path = os.path.join(data_dir, filename)
            
            if os.path.exists(file_path):
                # 调用你写好的函数
                # 建议 runs 设为 5 或 10 (30次太慢了，深度学习 5 次通常就够看指标了)
                mean_val, std_val = run_calibration_experiment(
                    file_path, 
                    label_ratio=ratio, 
                    runs=10
                )
                
                # 记录结果，存成 LaTeX 格式方便复制
                dataset_key = filename.replace(".arff", "")
                all_results[ratio][dataset_key] = f"${mean_val:.4f}_{{{{\\pm}}{std_val:.4f}}}$"
            else:
                print(f"⚠️ 找不到文件: {file_path}")

    # 4. 终极大汇总打印
    print("\n\n" + "="*80)
    print("      FINAL EXPERIMENT SUMMARY      ")
    print("="*80)
    
    for ratio in ratios:
        print(f"\n[Label Ratio: {ratio}]")
        print("-" * 30)
        for ds in [f.replace(".arff", "") for f in dataset_files]:
            res = all_results[ratio].get(ds, "N/A")
            print(f"{ds:15s} : {res}")

if __name__ == "__main__":
    main_batch_experiment()