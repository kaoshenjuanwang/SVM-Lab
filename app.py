from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.svm import SVC
import json

app = Flask(__name__)
CORS(app)

class CustomSVM:
    def __init__(self, kernel='linear', C=1.0, degree=3, gamma='scale', coef0=0.0):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.svm = None
        self.intermediate_states = []
        
    def fit(self, X, y):
        try:
            # 创建SVC模型
            self.svm = SVC(
                kernel=self.kernel,
                C=self.C,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0,
                random_state=42
            )
            
            # 训练模型并记录中间状态
            X = np.array(X)
            y = np.array(y)
            
            # 生成网格点用于可视化
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            step = 0.2
            xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                                np.arange(y_min, y_max, step))
            
            # 记录中间状态
            n_steps = 10
            subset_sizes = np.linspace(max(2, len(X)//2), len(X), n_steps, dtype=int)
            
            self.intermediate_states = []  # 清空之前的状态

            for i, subset_size in enumerate(subset_sizes):
                # 确保包含正负样本
                pos_indices = np.where(y == 1)[0]
                neg_indices = np.where(y == 0)[0]
                
                n_pos = min(len(pos_indices), subset_size // 2)
                n_neg = min(len(neg_indices), subset_size - n_pos)
                
                selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
                selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
                
                indices = np.concatenate([selected_pos, selected_neg])
                X_subset = X[indices]
                y_subset = y[indices]
                
                # 训练当前子集
                self.svm.fit(X_subset, y_subset)
                
                # 计算决策边界
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                Z = self.svm.predict(grid_points)
                Z = Z.reshape(xx.shape)
                
                # 获取支持向量
                support_vectors = self.svm.support_vectors_
                
                # 计算当前步骤的准确率
                y_pred = self.svm.predict(X)
                current_accuracy = float(np.mean(y_pred == y))

                # 生成更详细的描述
                descriptions = [
                    "初始化训练集，选择部分数据点开始训练",
                    "扩大训练集，寻找初步的决策边界",
                    "优化支持向量的选择，调整边界位置",
                    "增加更多数据点，提升模型泛化能力",
                    "细化决策边界，平衡正负类别",
                    "进一步优化支持向量位置",
                    "扩大训练集至接近完整数据",
                    "微调决策边界和间隔",
                    "最终优化支持向量",
                    "使用全部数据完成训练"
                ]

                # 保存当前状态（确保所有numpy类型都转换为Python原生类型）
                state = {
                    'iteration': int(i + 1),
                    'support_vectors': support_vectors.tolist(),
                    'Z': Z.tolist(),
                    'xx': xx.tolist(),
                    'yy': yy.tolist(),
                    'description': descriptions[i],
                    'subset_size': int(subset_size),
                    'total_size': int(len(X)),
                    'accuracy': float(current_accuracy),
                    'n_support_vectors': int(len(support_vectors))
                }
                self.intermediate_states.append(state)
            
            # 最终训练使用全部数据
            self.svm.fit(X, y)
            
            return self
            
        except Exception as e:
            print(f"训练错误: {str(e)}")
            raise

    def predict(self, X):
        if self.svm is None:
            raise ValueError("模型尚未训练")
        return self.svm.predict(X)

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        
        # 基本数据验证
        if not data or 'X' not in data or 'y' not in data:
            return jsonify({'success': False, 'message': '缺少训练数据'}), 400
        
        X = np.array(data['X'])
        y = np.array(data['y'])
        
        # 验证数据点数量
        if len(X) < 2:
            return jsonify({'success': False, 'message': '需要至少两个数据点'}), 400
        
        if len(X) != len(y):
            return jsonify({'success': False, 'message': '特征和标签数量不匹配'}), 400
            
        # 验证是否有不同类别的数据点
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return jsonify({'success': False, 'message': '需要至少两个不同类别的数据点'}), 400

        # 处理gamma参数
        gamma = data.get('gamma', 'scale')
        if gamma == 'manual':
            gamma = float(data.get('gamma_value', 1.0))
            
        # 创建和训练模型
        model = CustomSVM(
            kernel=data.get('kernel', 'linear'),
            C=float(data.get('C', 1.0)),
            degree=int(data.get('degree', 3)),
            gamma=gamma,
            coef0=float(data.get('coef0', 0.0))
        )
        
        # 训练模型
        model.fit(X, y)
        
        # 生成网格点进行可视化
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        step = 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                            np.arange(y_min, y_max, step))
        
        # 预测网格点的类别
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 计算准确率
        y_pred = model.predict(X)
        accuracy = float(np.mean(y_pred == y))
        
        # 确保所有numpy类型都转换为Python原生类型
        response_data = {
            'success': True,
            'X': X.tolist(),
            'y': y.tolist(),
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'Z': Z.tolist(),
            'support_vectors': model.svm.support_vectors_.tolist(),
            'accuracy': float(accuracy),
            'intermediate_states': model.intermediate_states
        }
        
        # 使用json.dumps确保数据可以被序列化
        return app.response_class(
            response=json.dumps(response_data),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        print(f"训练错误: {str(e)}")
        return jsonify({'success': False, 'message': f'训练失败: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True) 