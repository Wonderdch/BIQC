# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run hyperparameter search and store results with a command-line script."""

import joblib
import numpy as np
import sys
import os
import time
import argparse
import logging
import pandas as pd
from importlib import import_module
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from qml_benchmarks.hyperparam_search_utils import read_data, construct_hyperparameter_grid
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

np.random.seed(42)

logging.getLogger().setLevel(logging.INFO)
logging.info('cpu count:' + str(os.cpu_count()))


def convergence_step_scorer(estimator, X, y):
    """返回模型的收敛步数
    
    Parameters
    ----------
    estimator : object
        训练好的模型实例
    X : array-like
        输入数据（在这个场景下不使用）
    y : array-like
        目标值（在这个场景下不使用）
        
    Returns
    -------
    int
        收敛步数
    """
    # 直接从估计器获取收敛步数
    return int(getattr(estimator, 'converge_step_', estimator.max_steps))


if __name__ == "__main__":
    # print 开始时间
    logging.info(" ".join(["开始时间", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]))
    start_time = time.time()

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with hyperparameter search.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 添加新的命令行参数
    parser.add_argument(
        "--log-prefix",
        help="Prefix for the log file name",
        default="",
        type=str
    )

    parser.add_argument(
        "--classifier-name",
        help="Classifier to run",
    )

    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset",
    )

    parser.add_argument(
        "--results-path", default=".", help="Path to store the experiment results"
    )

    parser.add_argument(
        "--clean",
        help="True or False. Remove previous results if it exists",
        dest="clean",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--hyperparameter-scoring",
        type=list,
        nargs="+",
        default=["accuracy", "roc_auc"],
        help="Scoring for hyperparameter search.",
    )

    parser.add_argument(
        "--hyperparameter-refit",
        type=str,
        default="accuracy",
        help="Refit scoring for hyperparameter search.",
    )

    parser.add_argument(
        "--plot-loss",
        help="True or False. Plot loss history for single fit",
        dest="plot_loss",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel threads to run"
    )

    # Parse the arguments along with any extra arguments that might be model specific
    args, unknown_args = parser.parse_known_args()

    if any(arg is None for arg in [args.classifier_name,
                                   args.dataset_path]):
        msg = "\n================================================================================"
        msg += "\nA classifier from qml.benchmarks.model and dataset path are required. E.g., \n \n"
        msg += "python run_hyperparameter_search \ \n--classifier DataReuploadingClassifier \ \n--dataset-path train.csv\n"
        msg += "\nCheck all arguments for the script with \n"
        msg += "python run_hyperparameter_search --help\n"
        msg += "================================================================================"
        raise ValueError(msg)

    # Add model specific arguments to override the default hyperparameter grid
    hyperparam_grid = construct_hyperparameter_grid(
        hyper_parameter_settings, args.classifier_name
    )
    for hyperparam in hyperparam_grid:
        hp_type = type(hyperparam_grid[hyperparam][0])
        parser.add_argument(f'--{hyperparam}',
                            type=hp_type,
                            nargs="+",
                            default=hyperparam_grid[hyperparam],
                            help=f'{hyperparam} grid values for {args.classifier_name}')

    args = parser.parse_args(unknown_args, namespace=args)

    for hyperparam in hyperparam_grid:
        override = getattr(args, hyperparam)
        if override is not None:
            hyperparam_grid[hyperparam] = override
    logging.info(
        "Running hyperparameter search experiment with the following settings:"
    )
    logging.info(args.log_prefix)
    logging.info(args.classifier_name)
    logging.info(args.dataset_path)
    logging.info(" ".join(args.hyperparameter_scoring))
    logging.info(args.hyperparameter_refit)
    logging.info("Hyperparam grid:" + " ".join(
        [(str(key) + str(":") + str(hyperparam_grid[key])) for key in hyperparam_grid.keys()]))
    logging.info("\n")

    experiment_path = args.results_path
    results_path = os.path.join(experiment_path, "results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ###################################################################
    # Get the classifier, dataset and search methods from the arguments
    ###################################################################
    Classifier = getattr(
        import_module("qml_benchmarks.models"),
        args.classifier_name
    )
    classifier_name = Classifier.__name__

    # Run the experiments save the results
    train_dataset_filename = os.path.join(args.dataset_path)
    X, y = read_data(train_dataset_filename)

    dataset_path_obj = Path(args.dataset_path)
    results_filename_stem = " ".join([Classifier.__name__ + "_" + dataset_path_obj.stem + "_GridSearchCV"])

    if args.log_prefix:
        results_filename_stem = args.log_prefix + "_" + results_filename_stem

    # If we have already run this experiment then continue
    if os.path.isfile(os.path.join(results_path, results_filename_stem + ".csv")):
        if args.clean is False:
            msg = "\n================================================================================="
            msg += "\nResults exist in " + os.path.join(results_path, results_filename_stem + ".csv")
            msg += "\nSpecify --clean True to override results or new --results-path"
            msg += "\n================================================================================="
            logging.warning(msg)
            sys.exit(msg)
        else:
            logging.warning("Cleaning existing results for ",
                            os.path.join(results_path, results_filename_stem + ".csv"))

    ###########################################################################
    # Single fit to check everything works
    ###########################################################################
    first_search_param = {key: value[0] for key, value in hyperparam_grid.items()}

    # 根据 plot_loss 参数决定是否添加绘图标识信息
    plot_identifier = ""
    if args.plot_loss:
        # 构建绘图标识信息
        plot_identifier = f"{args.log_prefix}_{dataset_path_obj.stem}" if args.log_prefix else dataset_path_obj.stem
    first_search_param['plot_identifier'] = plot_identifier

    classifier = Classifier(**first_search_param)

    a = time.time()
    classifier.fit(X, y)
    b = time.time()
    acc_train = classifier.score(X, y)
    logging.info(" ".join(
        [classifier_name,
         "Dataset path",
         args.dataset_path,
         "Train acc:",
         str(acc_train),
         "Time single run",
         str(b - a)])
    )

    if hasattr(classifier, "n_qubits_"):
        logging.info(" ".join(["Num qubits", f"{classifier.n_qubits_}"]))

    # 添加收敛步数评分器
    scoring_dict = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'step': convergence_step_scorer
    }

    ###########################################################################
    # Hyperparameter search
    # 自动计算 mean_fit_time（平均训练时间） 和 mean_score_time（平均推理时间）
    ###########################################################################
    gs = GridSearchCV(estimator=Classifier(plot_identifier=plot_identifier),
                      param_grid=hyperparam_grid,
                      scoring=scoring_dict,
                      refit=args.hyperparameter_refit,  # "accuracy" 以 accuracy 为标准进行模型选择
                      verbose=3,
                      n_jobs=args.n_jobs).fit(
        X, y
    )

    logging.info("Best hyperparams")
    logging.info(gs.best_params_)

    # 创建方法特定的文件夹
    method_folder = os.path.join(results_path, args.classifier_name)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    # 处理 GridSearchCV 结果
    df = pd.DataFrame.from_dict(gs.cv_results_)
    # 1. 将 param_xxxx 改为 xxxx
    param_columns = [col for col in df.columns if col.startswith('param_')]
    rename_param_dict = {col: col.replace('param_', '') for col in param_columns}
    df = df.rename(columns=rename_param_dict)
    # 2. 移除 rank_xxxx 列
    df = df.loc[:, ~df.columns.str.startswith('rank_')]
    # 3. 移除 split*_test_step 列
    df = df.loc[:, ~df.columns.str.match(r'split\d+_test_step')]
    # 4. 保存结果
    df.to_csv(os.path.join(method_folder, results_filename_stem + ".csv"))
    best_df = pd.DataFrame(list(gs.best_params_.items()), columns=['hyperparameter', 'best_value'])
    # Save best hyperparameters to a CSV file
    best_df.to_csv(os.path.join(method_folder, results_filename_stem + '-best-hyperparameters.csv'), index=False)

    # 存储最佳模型
    model_path = os.path.join(method_folder, results_filename_stem + ".joblib")
    joblib.dump(gs.best_estimator_, model_path)
    logging.info(f"Best model saved to: {model_path}")

    # print 结束时间
    logging.info(" ".join(["结束时间", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]))

    # print 总耗时
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("总耗时: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
