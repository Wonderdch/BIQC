import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import train

jax.config.update("jax_enable_x64", True)


class CNN(nn.Module):
    """The convolutional neural network used for classification."""
    output_channels: tuple  # 使用类属性声明
    kernel_shape: int  # 使用类属性声明

    def setup(self):
        """替代 __init__ 的 Flax 推荐方式"""
        # 在这里可以初始化任何需要的组件
        pass

    @nn.compact
    def __call__(self, x):
        # 检查是否是 hidden_states 模式
        is_hidden_states = False
        if isinstance(x, tuple) and len(x) > 1:
            data, mode = x
            if mode == 'hidden_states':
                # 提取特征部分
                x = data
                is_hidden_states = True

        x = nn.Conv(
            features=self.output_channels[0], kernel_size=(self.kernel_shape, self.kernel_shape)
        )(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(
            features=self.output_channels[1], kernel_size=(self.kernel_shape, self.kernel_shape)
        )(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=self.output_channels[1] * 2)(x)
        # 不能直接返回 CNN 的输出，而必须过一个线性层，再进行 t-SNE 可视化才比较正常
        if is_hidden_states:
            return x
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


def construct_cnn(output_channels, kernel_shape):
    """构造 CNN 模型实例"""
    return CNN(output_channels=output_channels, kernel_shape=kernel_shape)


class ConvolutionalNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            kernel_shape=3,
            # output_channels=[32, 64],
            output_channels=[16, 32],
            learning_rate=0.001,
            convergence_interval=200,
            max_steps=10000,
            batch_size=32,
            max_vmap=None,
            jit=True,
            random_state=42,
            scaling=1.0,
            plot_identifier=''
    ):
        # attributes that do not depend on data
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.scaling = scaling
        self.jit = jit
        self.kernel_shape = kernel_shape
        self.output_channels = output_channels
        self.convergence_interval = convergence_interval
        self.batch_size = batch_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.plot_identifier = plot_identifier

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.scaler = None  # data scaler will be fitted on training data

        self.converge_step = self.max_steps

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def initialize(self, n_features, classes=None):
        """Initialize attributes that depend on the number of features and the class labels.
        Args:
            classes (array-like): class labels that the classifier expects
        """
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        # initialise the model
        self.cnn = construct_cnn(self.output_channels, self.kernel_shape)
        self.forward = self.cnn

        # create dummy data input to initialise the cnn
        height = int(jnp.sqrt(n_features))
        X0 = jnp.ones(shape=(1, height, height, 1))
        self.initialize_params(X0)

    def initialize_params(self, X):
        # initialise the trainable parameters
        self.params_ = self.cnn.init(self.generate_key(), X)

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.initialize(X.shape[1], classes=np.unique(y))

        y = jnp.array(y, dtype=int)

        # scale input data
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.transform(X)

        # 在这里调用 print_model_info 方法
        # self.print_model_info()

        def loss_fn(params, X, y):
            y = jax.nn.relu(y)  # convert to 0,1 labels
            vals = self.forward.apply(params, X)[:, 0]
            loss = jnp.mean(optax.sigmoid_binary_cross_entropy(vals, y))
            return loss

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        optimizer = optax.adam
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

        return self

    def predict(self, X):
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        # get probabilities of y=1
        p1 = jax.nn.sigmoid(self.forward.apply(self.params_, X)[:, 0])
        predictions_2d = jnp.c_[1 - p1, p1]
        return predictions_2d

    def transform(self, X):
        """
        If scaler is initialized, transform the inputs.

        Put into NCHW format. This assumes square images.
        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        if self.scaler is None:
            # if the model is unfitted, initialise the scaler here
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        X = self.scaler.transform(X) * self.scaling

        # reshape data to square array
        X = jnp.array(X)
        height = int(jnp.sqrt(X.shape[1]))
        X = jnp.reshape(X, (X.shape[0], height, height, 1))

        return X

    def print_model_info(self):
        if self.params_ is None:
            print("CNN 模型尚未初始化。请先调用 fit() 方法。")
            return

        print("模型架构:")
        print(self.cnn)

        total_params = sum(p.size for p in jax.tree_leaves(self.params_))
        print(f'\n可训练参数总数：{total_params}')

        print("\n每层可训练参数数量:")

        def print_params(name, params, indent=''):
            if isinstance(params, dict):
                for sub_name, sub_param in params.items():
                    print_params(f"{name}.{sub_name}", sub_param, indent + '  ')
            else:
                print(f"{indent}{name}: {params.size} 参数")

        for name, layer_params in self.params_.items():
            print_params(name, layer_params)

    def __getstate__(self):
        """序列化时的状态处理"""
        state = self.__dict__.copy()
        # 保存必要的参数
        if 'forward' in state:
            state['output_channels'] = self.output_channels
            state['kernel_shape'] = self.kernel_shape
            state['params_'] = self.params_
            del state['forward']
        return state

    def __setstate__(self, state):
        """反序列化时的状态恢复"""
        self.__dict__.update(state)
        if 'output_channels' in state and 'kernel_shape' in state:
            # 重新构造模型
            self.forward = construct_cnn(
                state['output_channels'],
                state['kernel_shape']
            )
