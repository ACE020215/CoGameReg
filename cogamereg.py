"""
2026.03.30
多模型集成学习，利用混合博弈论 (Rank + Kernel MSE + BiGC Consistency) 做归一化权重
"""

from sklearn.utils import resample
from itertools import combinations
import inspect
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

# ------------------------------------------------------------------------------
# 模型库导入 (保持不变)
# ------------------------------------------------------------------------------
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# ------------------------------------------------------------------------------
# 1. 基础工具函数
# ------------------------------------------------------------------------------

def compute_mahalanobis_params(X_labeled, U=None):
    if U is not None:
        X_for_cov = np.vstack([X_labeled, U])
    else:
        X_for_cov = X_labeled
    
    if X_for_cov.shape[1] == 1:
        V = np.array([[np.var(X_for_cov)]])
    else:
        V = np.cov(X_for_cov, rowvar=False)
    
    try:
        reg_lambda = 1e-6
        V = V + np.eye(V.shape[0]) * reg_lambda
        VI = np.linalg.pinv(V)
    except Exception:
        VI = np.linalg.inv(V)
    return {'VI': VI}

def calculate_delta(L, x_u, y_u_pred, k, metric='manhattan', p=2, prefer='mse', U=None):
    if L is None or len(L) < 2: return -np.inf
    
    k_eff = min(k, len(L))
    model_kwargs = {'n_neighbors': k_eff, 'metric': metric}
    
    if metric == 'minkowski': model_kwargs['p'] = p
    elif metric == 'mahalanobis':
        metric_params = compute_mahalanobis_params(L[:, :-1], U)
        model_kwargs['metric_params'] = metric_params
        model_kwargs['algorithm'] = 'brute'

    temp_model = KNeighborsRegressor(**model_kwargs)
    temp_model.fit(L[:, :-1], L[:, -1])

    neighbors_idx = temp_model.kneighbors([x_u], n_neighbors=k_eff, return_distance=False)[0]
    L_neighbors_X = L[neighbors_idx, :-1]
    L_neighbors_y = L[neighbors_idx, -1]

    pred_before = temp_model.predict(L_neighbors_X)
    
    if prefer == 'mse':
        score_before = np.mean((L_neighbors_y - pred_before) ** 2)
    elif prefer == 'r2':
        score_before = r2_score(L_neighbors_y, pred_before)
    else:
        raise ValueError(f"不支持的指标: {prefer}")

    L_augmented = np.vstack([L, np.hstack([x_u, y_u_pred])])
    temp_model_aug = KNeighborsRegressor(**model_kwargs)
    temp_model_aug.fit(L_augmented[:, :-1], L_augmented[:, -1])

    pred_after = temp_model_aug.predict(L_neighbors_X)
    
    if prefer == 'mse':
        score_after = np.mean((L_neighbors_y - pred_after) ** 2)
        delta = score_before - score_after 
    elif prefer == 'r2':
        score_after = r2_score(L_neighbors_y, pred_after)
        if score_before < -10 or score_after < -10: 
            delta = -np.inf
        else:
            delta = score_after - score_before 

    if np.isnan(delta) or np.isinf(delta): delta = -np.inf
    return delta

def initialize_model_zoo(model_zoo_config, random_state=42):
    initialized_models = {}
    
    def safe_init(cls, params):
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            return cls(**params)

        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return cls(**params)

        valid_keys = {
            key
            for key, parameter in signature.parameters.items()
            if key != 'self'
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        invalid_keys = sorted(set(params) - valid_keys)
        if invalid_keys:
            raise TypeError(
                f"{cls.__name__} does not accept parameters: {invalid_keys}"
            )
        return cls(**params)

    for name, config in model_zoo_config.items():
        params = config.copy()
        current_seed = random_state + len(initialized_models)

        if name == 'lsvr':
            params.setdefault('max_iter', 20000)
            params.setdefault('dual', True) 
            model = safe_init(LinearSVR, dict(random_state=current_seed, **params))
            
        elif name == 'mlp':
            params.setdefault('warm_start', True)
            params.setdefault('max_iter', 5000)
            params.setdefault('random_state', current_seed)
            model = safe_init(MLPRegressor, params)
        
        elif name == 'knn':
            model = safe_init(KNeighborsRegressor, params)
        elif name == 'linear':
            model = safe_init(LinearRegression, params)
        elif name == 'ridge':
            model = safe_init(Ridge, dict(random_state=current_seed, **params))
        elif name == 'lasso':
            model = safe_init(Lasso, dict(random_state=current_seed, **params))
        elif name == 'enet':
            model = safe_init(ElasticNet, dict(random_state=current_seed, **params))
        elif name == 'rf':
            model = safe_init(RandomForestRegressor, dict(random_state=current_seed, **params))
        elif name == 'gbr':
            model = safe_init(GradientBoostingRegressor, dict(random_state=current_seed, **params))
        elif name == 'sgd':
            model = safe_init(SGDRegressor, dict(random_state=current_seed, **params))
        elif name == 'gpr':
            model = safe_init(GaussianProcessRegressor, params)
        elif name == 'svr':
            model = safe_init(SVR, params)
        elif name == 'xgb':
            if XGBRegressor is not None:
                model = safe_init(XGBRegressor, params)
            else:
                continue
            
        initialized_models[name] = model
        
    return initialized_models

# ------------------------------------------------------------------------------
# 2. Tri-Game Shapley 权重计算 (Rank + MSE + Consistency)
# ------------------------------------------------------------------------------

def simple_l1_normalize(vec):
    """
    中间过程使用的简单归一化，仅为了保证概率和为1，不进行截断。
    防止在中间步骤过早杀掉模型。
    """
    vec = np.maximum(vec, 0)
    total = np.sum(vec)
    if total == 0: return np.ones(len(vec)) / len(vec)
    return vec / total

# 引入 warnings 库（如果还没有的话）
import warnings
from scipy.stats import spearmanr, ConstantInputWarning

def compute_tri_game_shapley(models_dict, X_val, y_val, alpha=0.6, beta=0.2):
    """
    计算基于三部分博弈的混合 Shapley 权重。
    """
    model_names = list(models_dict.keys())
    n = len(model_names)
    
    # 1. 预计算所有单模型的预测值
    predictions_matrix = np.array([models_dict[name].predict(X_val) for name in model_names])
    
    # --- Part A: Consistency Weight (BiGC Convex Game) ---
    dists = squareform(pdist(predictions_matrix, metric='euclidean'))
    sigma_con = np.median(dists) + 1e-6
    similarity_matrix = np.exp(- (dists**2) / (2 * sigma_con**2))
    np.fill_diagonal(similarity_matrix, 0) 
    
    raw_consistency = np.sum(similarity_matrix, axis=1) 
    # 中间步骤：使用简单L1归一化，保留原始分布
    norm_consistency = simple_l1_normalize(raw_consistency)
    
    # --- Part B & C: Rank & Accuracy Weight (Submodular Game) ---
    baseline_mse = np.var(y_val)
    sigma_mse = baseline_mse * 0.5 + 1e-6 
    y_val_std = np.std(y_val) # 预计算标准差

    # 混合价值函数 V_mix(S)
    def get_mixed_value(subset_indices):
        if not subset_indices: 
            return 0.0
        
        # 快速切片获取子集预测
        sub_preds = predictions_matrix[list(subset_indices)]
        ensemble_pred = np.mean(sub_preds, axis=0)
        
        # --- 修复：安全计算 Rank Value (防止常数预测报错) ---
        pred_std = np.std(ensemble_pred)
        
        if pred_std < 1e-9 or y_val_std < 1e-9:
            rho = 0.0 # 常数无趋势
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConstantInputWarning)
                try:
                    rho, _ = spearmanr(y_val, ensemble_pred)
                    if np.isnan(rho): rho = 0.0
                except:
                    rho = 0.0
        
        val_rank = (1 + rho) / 2.0
        # -----------------------------------------------
        
        # MSE Value
        mse = mean_squared_error(y_val, ensemble_pred)
        val_mse = np.exp(- mse / (2 * sigma_mse**2))
        
        # 线性组合
        total_ab = alpha + beta
        if total_ab == 0: return 0
        
        return alpha * val_rank + beta * val_mse

    # Shapley 值计算
    from math import factorial
    fact = [factorial(i) for i in range(n + 1)]
    shapley_mix = np.zeros(n)
    
    indices = range(n)
    for i in indices:
        others = [x for x in indices if x != i]
        for k in range(len(others) + 1):
            weight = fact[k] * fact[n - k - 1] / fact[n]
            for S in combinations(others, k):
                val_S = get_mixed_value(S)
                val_S_union_i = get_mixed_value(S + (i,))
                shapley_mix[i] += weight * (val_S_union_i - val_S)

    # 中间步骤：简单L1归一化
    norm_mix = simple_l1_normalize(shapley_mix)
    
    # --- Final Fusion (融合三部分) ---
    gamma = 1 - alpha - beta
    final_weights = np.zeros(n)
    total_weight_ab = alpha + beta
    
    for i in range(n):
        w_part1 = norm_mix[i] * total_weight_ab
        w_part2 = norm_consistency[i] * gamma
        final_weights[i] = w_part1 + w_part2
        
    final_weights_smart = final_weights

    
    return dict(zip(model_names, final_weights_smart))

# ------------------------------------------------------------------------------
# 3. 核心算法升级：多模型协同训练 (CoGameReg)
# ------------------------------------------------------------------------------

def build_hybrid_U_prime(X_labeled, X_unlabeled, k_per_labeled, metric, metric_params=None):
    algorithm = 'brute' if metric == 'mahalanobis' else 'auto'
    nbrs = NearestNeighbors(n_neighbors=k_per_labeled, metric=metric, 
                           metric_params=metric_params, algorithm=algorithm)
    nbrs.fit(X_unlabeled)
    distances, indices = nbrs.kneighbors(X_labeled)
    U_prime_idx = np.unique(indices.flatten())
    return X_unlabeled[U_prime_idx], U_prime_idx



def print_weight_change_comparison(initial_weights, final_weights, model_names):
    print("\n" + "=" * 76)
    print("📈 Shapley权重变化对比（初始化 -> 迭代完成后）")
    print("=" * 76)
    print(f"{'Model':<12}{'Initial':>12}{'Final':>12}{'Delta':>13}{'Change(%)':>14}")
    print("-" * 76)

    for name in model_names:
        w0 = float(initial_weights.get(name, 0.0))
        w1 = float(final_weights.get(name, 0.0))
        delta = w1 - w0

        if abs(w0) > 1e-12:
            pct = delta / w0 * 100.0
            pct_str = f"{pct:+.2f}%"
        else:
            pct_str = "inf" if abs(delta) > 1e-12 else "0.00%"

        print(f"{name:<12}{w0:>12.4f}{w1:>12.4f}{delta:+13.4f}{pct_str:>14}")

    print("-" * 76)

    changes = []
    for name in model_names:
        w0 = float(initial_weights.get(name, 0.0))
        w1 = float(final_weights.get(name, 0.0))
        changes.append((name, w1 - w0))

    changes_sorted_desc = sorted(changes, key=lambda x: x[1], reverse=True)
    changes_sorted_asc = sorted(changes, key=lambda x: x[1])

    print("⬆️ 权重提升最多的模型：")
    for name, d in changes_sorted_desc[:3]:
        print(f"  {name:<12} {d:+.4f}")

    print("⬇️ 权重下降最多的模型：")
    for name, d in changes_sorted_asc[:3]:
        print(f"  {name:<12} {d:+.4f}")

    print("=" * 76 + "\n")
    
def CoGameReg_main(
    X_labeled, y_labeled, X_unlabeled,
    model_zoo_config,
    T=10, k=3, s=2,
    selection_ratio=0.3,
    metric='manhattan',
    threshold=0.0,
    prefer='r2',
    val_random_state=42,
    use_pseudo_in_training=True,
    k_per_labeled=3,
    ablation_mode='base',      # 'base', 'decouple_shap', 'undecouple_shap'
    alpha=0.6,
    beta=0.2
):
    print(f"ablation_mode={ablation_mode}, models={len(model_zoo_config)}, metric={metric}")
    print(f"ablation_mode={ablation_mode}, models={len(model_zoo_config)}, metric={metric}, prefer={prefer}, threshold={threshold}", flush=True)


    valid_modes = ['base', 'decouple_shap', 'undecouple_shap']
    if ablation_mode not in valid_modes:
        raise ValueError(f"不支持的 ablation_mode: {ablation_mode}，可选 {valid_modes}")

    rng = np.random.default_rng(val_random_state)

    # --- 1. 数据准备 ---
    X_labeled = np.array(X_labeled)
    y_labeled = np.array(y_labeled).ravel()
    X_unlabeled = np.array(X_unlabeled)

    # --- 2. 初始化基模型：全部 labeled data ---
    models = initialize_model_zoo(model_zoo_config, random_state=val_random_state)
    model_names = list(models.keys())

    L_sets = {}
    for name in model_names:
        Lx, Ly = X_labeled.copy(), y_labeled.copy()
        models[name].fit(Lx, Ly)
        L_sets[name] = (Lx, Ly)

    pseudo_pools = {name: {'X': [], 'y': []} for name in model_names}

    # --- 3. 一次性计算初始 Shapley 权重（只在 labeled 上算一次） ---
    # base：始终均匀权重
    # decouple_shap：训练时均匀权重，但最终输出初始 Shapley
    # undecouple_shap：训练时固定使用初始 Shapley，最终输出也为初始 Shapley
    if ablation_mode == 'base':
        fixed_shap_weights = None
        final_weights = {name: 1.0 / len(models) for name in model_names}
        calc_method = "Uniform"
    else:
        fixed_shap_weights = compute_tri_game_shapley(
            models,
            X_labeled,
            y_labeled,
            alpha=alpha,
            beta=beta,
        )
        final_weights = fixed_shap_weights.copy()

        if ablation_mode == 'decouple_shap':
            calc_method = "Initial Tri-Game Shapley (train uses uniform, final uses initial Shapley)"
        elif ablation_mode == 'undecouple_shap':
            calc_method = "Fixed Tri-Game Shapley (train and final both use initial Shapley)"
        else:
            calc_method = "Fixed Tri-Game Shapley"

    # --- 4. 构建 U' ---
    metric_params = None
    if metric == 'mahalanobis':
        metric_params = compute_mahalanobis_params(X_labeled, X_unlabeled)

    U_prime, U_prime_idx = build_hybrid_U_prime(
        X_labeled, X_unlabeled, k_per_labeled, metric, metric_params
    )

    all_unlabeled_idx = np.arange(len(X_unlabeled))
    U_remaining_idx = np.setdiff1d(all_unlabeled_idx, U_prime_idx)

    print(f"CoGameReg | Models: {len(models)} | Init U': {len(U_prime)}")

    # --- 5. 协同训练循环 ---
    for t in range(1, T + 1):
        if len(U_prime) == 0:
            break

        # 伪标签迭代阶段的集成权重
        if ablation_mode in ['base', 'decouple_shap']:
            iter_weights_dict = {name: 1.0 / len(models) for name in model_names}
            iter_method = "Uniform"
        else:  # 'undecouple_shap'
            iter_weights_dict = fixed_shap_weights.copy()
            iter_method = "Fixed Initial Shapley"

        weights_vec = np.array([iter_weights_dict[name] for name in model_names])

        # 1. 各模型对 U' 的预测
        all_preds_matrix = np.array([models[name].predict(U_prime) for name in model_names])

        # 2. 加权集成预测 -> 作为候选伪标签
        y_u_ensemble = np.average(all_preds_matrix, axis=0, weights=weights_vec)

        candidates_to_remove = []

        for name in model_names:
            base_X, base_y = L_sets[name]
            curr_pseudo_X = pseudo_pools[name]['X']
            curr_pseudo_y = pseudo_pools[name]['y']

            if len(curr_pseudo_X) > 0:
                L_curr_X = np.vstack([base_X, np.vstack(curr_pseudo_X)])
                L_curr_y = np.hstack([base_y, np.hstack(curr_pseudo_y)])
            else:
                L_curr_X, L_curr_y = base_X, base_y

            L_mat = np.hstack([L_curr_X, L_curr_y.reshape(-1, 1)])
            local_candidates = []

            for i in range(len(U_prime)):
                x_u = U_prime[i]
                y_target = y_u_ensemble[i]

                delta = calculate_delta(
                    L_mat, x_u, y_target,
                    k=k,
                    metric=metric,
                    prefer=prefer,
                    U=X_unlabeled
                )

                if delta > threshold:
                    local_candidates.append((i, U_prime_idx[i], x_u, y_target, delta))

            local_candidates.sort(key=lambda x: x[4], reverse=True)

            if len(local_candidates) > 0:
                n_select = max(1, int(len(local_candidates) * selection_ratio))
            else:
                n_select = 0

            top_candidates = local_candidates[:n_select]

            for idx_in_Uprime, global_idx, x_u, y_p, _ in top_candidates:
                pseudo_pools[name]['X'].append(x_u)
                pseudo_pools[name]['y'].append(y_p)
                candidates_to_remove.append(global_idx)

        unique_removed = np.unique(candidates_to_remove)
        print(f"第 {t}/{T} 轮 | iter_weight={iter_method} | 选出 {len(unique_removed)} 个样本加入伪标签集")

        if len(unique_removed) > 0:
            mask = np.isin(U_prime_idx, unique_removed)
            U_prime = U_prime[~mask]
            U_prime_idx = U_prime_idx[~mask]
        else:
            print(f"  [提前结束] 第 {t} 轮没有样本被选中.")
            if t > 5:
                break

        # --- 重训基模型 ---
        for name in model_names:
            base_X, base_y = L_sets[name]
            p_X = pseudo_pools[name]['X']
            p_y = pseudo_pools[name]['y']

            if use_pseudo_in_training and len(p_X) > 0:
                X_train_new = np.vstack([base_X, np.vstack(p_X)])
                y_train_new = np.hstack([base_y, np.hstack(p_y)])
                models[name].fit(X_train_new, y_train_new)
            else:
                models[name].fit(base_X, base_y)

        # --- 补充 U' ---
        target_size = s * len(X_labeled)
        if len(U_prime) < target_size and len(U_remaining_idx) > 0:
            n_take = min(target_size - len(U_prime), len(U_remaining_idx))
            new_indices = rng.choice(U_remaining_idx, size=n_take, replace=False)
            U_prime = np.vstack([U_prime, X_unlabeled[new_indices]])
            U_prime_idx = np.hstack([U_prime_idx, new_indices])
            U_remaining_idx = np.setdiff1d(U_remaining_idx, new_indices)

    # --- 7. 打印最终权重 ---
    print("\n" + "-" * 40)
    print(f"⚖️ 最终模型权重分配 ({calc_method})")
    print("-" * 40)
    sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
    for name, w in sorted_weights:
        bar_len = int(w * 20)
        bar = "█" * bar_len
        print(f"  {name:<10} : {w:.4f} |{bar}")
    print("-" * 40 + "\n")

    return models, final_weights

# ------------------------------------------------------------------------------
# 4. Wrapper 接口
# ------------------------------------------------------------------------------

def CoGameReg_validation_data_benchmark(
    data_l, y, data_u, data_t,
    model_zoo_config,
    ablation_mode='base',
    k_per_labeled=3,
    alpha=0.6,  # Tri-Game Shapley 中 Rank 的权重
    beta=0.2,   # Tri-Game Shapley 中 MSE 的
    **kwargs
):
    print("开始训练模型", flush=True)
    # print('使用模型及参数配置：', flush=True)
    # for model_name, params in model_zoo_config.items():
    #     print(f"{model_name}: {params}", flush=True)
    print(f"k_per_labeled: {k_per_labeled}", flush=True)
    print(f"alpha (Rank 权重): {alpha}", flush=True)
    print(f"beta (MSE 权重): {beta}", flush=True)
    
    
    final_models, final_weights = CoGameReg_main(
        X_labeled=data_l,
        y_labeled=y,
        X_unlabeled=data_u,
        model_zoo_config=model_zoo_config,
        k_per_labeled=k_per_labeled,
        ablation_mode=ablation_mode,
        alpha=alpha,
        beta=beta,
        **kwargs
    )
    print("开始预测数据", flush=True)

    def ensemble_predict(model_dict, weights_dict, X):
        pred = np.zeros(len(X))
        for name, model in model_dict.items():
            w = weights_dict.get(name, 0)
            pred += w * model.predict(X)
        return pred

    y_pred = ensemble_predict(final_models, final_weights, data_t)

    return y_pred
