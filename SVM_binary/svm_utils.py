import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas as pd
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_name, file_path=None):
    """加载数据集
    
    参数:
        dataset_name: 数据集名称，可以是 "线性可分数据集"、"非线性可分数据集1"、"非线性可分数据集2" 或 "自定义数据集"
        file_path: 当dataset_name为"自定义数据集"时，指定数据文件对象
    """
    if dataset_name == "线性可分数据集":
        # 生成线性可分的数据集
        X, y = datasets.make_blobs(
            n_samples=100,
            centers=2,  # 确保生成两个类别
            n_features=2,
            random_state=42,
            cluster_std=1.0,
            center_box=(-10.0, 10.0),  # 设置中心点的范围
            shuffle=True  # 打乱数据
        )
        feature_names = ['特征1', '特征2']
        target_names = ['负例', '正例']
        
    elif dataset_name == "非线性可分数据集1（月牙形）":
        # 生成非线性可分的数据集（月牙形）
        n_samples = 300  # 增加样本数
        np.random.seed(42)
        
        # 生成两个月牙形
        n_samples_per_class = n_samples // 2
        
        # 第一个月牙（类别0）- 上方月牙
        theta1 = np.linspace(0.4*np.pi, 1.6*np.pi, n_samples_per_class)  # 控制开口范围
        r1 = np.random.normal(4, 0.3, n_samples_per_class)  # 基础半径
        noise1 = np.random.normal(0, 0.3, (n_samples_per_class, 2))  # 二维噪声
        
        # 生成基础圆形
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)
        
        # 添加偏移和噪声
        X1 = np.column_stack([x1, y1 + 2]) + noise1  # 向左偏移4个单位，向上偏移4个单位
        
        # 第二个月牙（类别1）- 下方月牙
        theta2 = np.linspace(1.4*np.pi, 2.6*np.pi, n_samples_per_class)  # 控制开口范围
        r2 = np.random.normal(4, 0.3, n_samples_per_class)  # 基础半径
        noise2 = np.random.normal(0, 0.3, (n_samples_per_class, 2))  # 二维噪声
        
        # 生成基础圆形
        x2 = r2 * np.cos(theta2)
        y2 = r2 * np.sin(theta2)
        
        # 添加偏移和噪声
        X2 = np.column_stack([x2, y2 - 2]) + noise2  # 向右偏移4个单位，向下偏移4个单位
        
        # 合并数据
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples_per_class, dtype=int), 
                      np.ones(n_samples_per_class, dtype=int)])
        
        # 打乱数据
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        feature_names = ['特征1', '特征2']
        target_names = ['负例', '正例']
    
    elif dataset_name == "非线性可分数据集2（椭圆形）":
        # 生成椭圆形非线性可分数据集
        n_samples = 300  # 增加样本数
        np.random.seed(42)
        
        # 生成内圈数据点（负类）
        n_inner = n_samples // 2
        theta_inner = np.linspace(0, 2*np.pi, n_inner)
        r_inner = np.random.normal(1, 0.1, n_inner)
        x_inner = 3 * r_inner * np.cos(theta_inner)  # 增大椭圆的尺寸
        y_inner = 2 * r_inner * np.sin(theta_inner)  # 使椭圆更扁
        
        # 生成外圈数据点（正类）
        n_outer = n_samples - n_inner
        theta_outer = np.linspace(0, 2*np.pi, n_outer)
        r_outer = np.random.normal(2, 0.15, n_outer)
        x_outer = 6 * r_outer * np.cos(theta_outer)  # 增大外圈椭圆的尺寸
        y_outer = 4 * r_outer * np.sin(theta_outer)  # 使外圈椭圆更扁
        
        # 添加椭圆变形
        X_inner = np.column_stack([x_inner, y_inner])
        X_outer = np.column_stack([x_outer, y_outer])
        
        # 添加噪声
        X_inner += np.random.normal(0, 0.2, X_inner.shape)
        X_outer += np.random.normal(0, 0.3, X_outer.shape)
        
        # 合并数据
        X = np.vstack([X_inner, X_outer])
        y = np.hstack([np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)])
        
        # 打乱数据
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        feature_names = ['特征1', '特征2']
        target_names = ['负例', '正例']
    
    elif dataset_name == "自定义数据集":
        if file_path is None:
            raise ValueError("自定义数据集需要提供文件")
        try:
            # 读取文件内容
            content = file_path.read().decode('utf-8')
            
            # 根据文件扩展名选择不同的读取方式
            file_extension = file_path.name.lower().split('.')[-1]
            if file_extension == 'csv':
                df = pd.read_csv(StringIO(content))
            elif file_extension == 'txt':
                # 尝试不同的分隔符
                for sep in [',', '\t', ' ']:
                    try:
                        df = pd.read_csv(StringIO(content), sep=sep)
                        # 如果成功读取并且列数大于1，说明找到了正确的分隔符
                        if df.shape[1] > 1:
                            break
                    except:
                        continue
                else:
                    raise ValueError("无法识别TXT文件的分隔符，请确保使用逗号、制表符或空格分隔")
            else:
                raise ValueError("不支持的文件格式，请使用CSV或TXT文件")
            
            # 基本数据验证
            if df.empty:
                raise ValueError("数据集为空")
            if df.shape[1] < 2:
                raise ValueError("数据集至少需要一个特征列和一个目标变量列")
            if df.isnull().any().any():
                # 移除包含缺失值的行
                df = df.dropna()
                if df.empty:
                    raise ValueError("移除缺失值后数据集为空")
            
            # 分离特征和标签
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1].tolist()
            
            # 检查特征是否为数值型
            if not all(np.issubdtype(X.dtype, np.number) for X in df.iloc[:, :-1].values.T):
                raise ValueError("所有特征必须是数值型")
            
            # 异常值检测和处理（使用IQR方法）
            def remove_outliers(X):
                # 计算每个特征的IQR
                Q1 = np.percentile(X, 25, axis=0)
                Q3 = np.percentile(X, 75, axis=0)
                IQR = Q3 - Q1
                
                # 定义异常值的界限
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 创建掩码标识非异常值
                mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
                return mask
            
            # 应用异常值检测
            normal_samples_mask = remove_outliers(X)
            X = X[normal_samples_mask]
            y = y[normal_samples_mask]
            
            if len(X) < 10:  # 设置最小样本数阈值
                raise ValueError("移除异常值后样本数太少，请检查数据质量")
            
            # 处理标签
            unique_classes = np.unique(y)
            if len(unique_classes) != 2:
                raise ValueError(f"自定义数据集必须只包含两个类别，当前包含 {len(unique_classes)} 个类别")
            
            # 标签格式转换
            if set(unique_classes) == {1, -1}:
                # 将-1转换为0
                y = np.where(y == -1, 0, 1)
            elif set(unique_classes) == {1, 2}:
                # 将2转换为0
                y = np.where(y == 2, 0, 1)
            elif not set(unique_classes) == {0, 1}:
                # 其他情况，将较小的值映射为0，较大的值映射为1
                min_label = np.min(unique_classes)
                y = np.where(y == min_label, 0, 1)
            
            # 确保转换后仍然有两个类别
            final_unique_classes = np.unique(y)
            if len(final_unique_classes) != 2:
                raise ValueError("标签转换后类别数量不正确，请检查数据")
            if not set(final_unique_classes) == {0, 1}:
                raise ValueError("标签转换失败，请检查数据格式")
            
            target_names = ['负例', '正例']
            
            # 输出数据处理信息
            print(f"数据处理完成:")
            print(f"- 原始样本数: {len(normal_samples_mask)}")
            print(f"- 移除异常值后样本数: {len(X)}")
            print(f"- 原始标签值: {unique_classes}")
            print(f"- 标签已统一转换为: [0, 1]")
            print(f"- 各类别样本数: 类别0: {np.sum(y == 0)}, 类别1: {np.sum(y == 1)}")
            
        except UnicodeDecodeError:
            raise ValueError("无法解码文件，请确保文件使用UTF-8编码")
        except pd.errors.EmptyDataError:
            raise ValueError("文件为空")
        except pd.errors.ParserError:
            raise ValueError("无法解析文件，请检查文件格式")
        except Exception as e:
            raise ValueError(f"无法加载自定义数据集: {str(e)}")
    else:
        raise ValueError("不支持的数据集")
    
    return X, y, feature_names, target_names

def visualize_pca_process(X, feature_names):
    """可视化PCA降维过程"""
    # 创建PCA对象并拟合数据
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 获取主成分和解释方差比
    components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # 创建子图，设置为垂直布局（4行1列）
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            '<b>原始特征空间分布</b>',
            '<b>主成分解释方差比</b>',
            '<b>特征投影方向</b>',
            '<b>降维后的数据分布</b>'
        ),
        specs=[[{'type': 'scatter3d'}],
               [{'type': 'bar'}],
               [{'type': 'scatter'}],
               [{'type': 'scatter'}]],
        vertical_spacing=0.15,  # 增加垂直间距，避免标题重叠
        row_heights=[0.35, 0.18, 0.23, 0.24]  # 微调每个子图的高度比例
    )
    
    # 1. 原始特征空间分布（使用前三个特征）
    if X.shape[1] >= 3:
        # 计算数据范围以设置合适的坐标轴范围
        x_range = X[:, 0].max() - X[:, 0].min()
        y_range = X[:, 1].max() - X[:, 1].min()
        z_range = X[:, 2].max() - X[:, 2].min()
        
        # 计算中心点
        x_center = (X[:, 0].max() + X[:, 0].min()) / 2
        y_center = (X[:, 1].max() + X[:, 1].min()) / 2
        z_center = (X[:, 2].max() + X[:, 2].min()) / 2
        
        # 设置坐标轴范围，给予一定的边距
        margin = 0.2
        x_min = x_center - x_range * (1 + margin)
        x_max = x_center + x_range * (1 + margin)
        y_min = y_center - y_range * (1 + margin)
        y_max = y_center + y_range * (1 + margin)
        z_min = z_center - z_range * (1 + margin)
        z_max = z_center + z_range * (1 + margin)
        
        fig.add_trace(
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color='#1f77b4',
                    colorscale='Blues'
                ),
                name='原始数据点'
            ),
            row=1, col=1
        )
        
        # 设置3D图的视角和样式
        fig.update_scenes(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2.5, y=2.5, z=1.2)  # 调整视角，使图形更扁平
            ),
            aspectmode='manual',
            aspectratio=dict(
                x=2.0,  # 增加x轴的比例
                y=1.5,  # 增加y轴的比例
                z=0.8   # 减小z轴的比例，使图形更扁平
            ),
            xaxis=dict(
                range=[x_min, x_max],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title=dict(
                    text="特征1",
                    font=dict(size=12, color='#2c3e50')
                )
            ),
            yaxis=dict(
                range=[y_min, y_max],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title=dict(
                    text="特征2",
                    font=dict(size=12, color='#2c3e50')
                )
            ),
            zaxis=dict(
                range=[z_min, z_max],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title=dict(
                    text="特征3",
                    font=dict(size=12, color='#2c3e50')
                )
            )
        )
    
    # 2. 主成分解释方差比
    fig.add_trace(
        go.Bar(
            x=['PC1', 'PC2'],
            y=explained_variance_ratio,
            marker_color=['#2ecc71', '#e74c3c'],
            name='解释方差比',
            text=[f'{v:.1%}' for v in explained_variance_ratio],
            textposition='outside',
            width=0.6  # 减小柱状图宽度
        ),
        row=2, col=1
    )
    
    # 3. 特征投影方向
    max_component = max(abs(components.min()), abs(components.max()))
    
    for i, (feature, pc1, pc2) in enumerate(zip(feature_names, components[0], components[1])):
        scale = 0.8
        pc1_norm = pc1 * scale / max_component
        pc2_norm = pc2 * scale / max_component
        
        fig.add_trace(
            go.Scatter(
                x=[0, pc1_norm],
                y=[0, pc2_norm],
                mode='lines+markers+text',
                name=feature,
                text=['', feature],
                textposition='top center',
                line=dict(
                    color=px.colors.qualitative.Set2[i % 8],
                    width=2
                ),
                marker=dict(size=[6, 8])
            ),
            row=3, col=1
        )
    
    # 4. 降维后的数据分布
    fig.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=X_pca[:, 0],
                colorscale='Viridis',
                opacity=0.7,
                showscale=True,
                colorbar=dict(
                    title='PC1值',
                    thickness=15,
                    len=0.3,
                    y=0.1,
                    titleside='right'
                )
            ),
            name='降维后的数据'
        ),
        row=4, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=1800,  # 增加总高度
        width=1000,   # 增加总宽度
        showlegend=False,
        title=dict(
            text='<b>PCA降维过程可视化</b>',
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='#2c3e50')
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=100, b=60)  # 增加边距
    )
    
    # 更新每个子图的样式
    for row in range(1, 5):
        # 更新X轴
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='#e0e0e0',
            title_font=dict(size=12),
            title_standoff=15,
            row=row,
            col=1
        )
        # 更新Y轴
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='#e0e0e0',
            title_font=dict(size=12),
            title_standoff=15,
            row=row,
            col=1
        )
    
    # 设置子图标题的字体和位置
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=14, color='#2c3e50')
        i['y'] = i['y'] + 0.02  # 微调标题位置，避免重叠
    
    # 更新坐标轴标题
    fig.update_xaxes(title_text="主成分", row=2, col=1)
    fig.update_yaxes(title_text="解释方差比", row=2, col=1)
    
    fig.update_xaxes(title_text="第一主成分方向", row=3, col=1)
    fig.update_yaxes(title_text="第二主成分方向", row=3, col=1)
    
    fig.update_xaxes(title_text="第一主成分", row=4, col=1)
    fig.update_yaxes(title_text="第二主成分", row=4, col=1)
    
    return fig, X_pca, components

def train_svm_with_visualization(X, y, kernel='rbf', C=1.0, gamma='scale', degree=3, visualization_callback=None):
    """
    训练SVM模型并可视化训练过程，展示分类超平面和支持向量的迭代过程
    """
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 初始化SVM模型
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    
    # 创建网格点用于绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # 获取每个类别的索引
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    # 确保每个类别至少有一个样本
    min_samples_per_class = max(3, len(X) // 20)  # 至少使用3个样本或总样本的5%
    
    # 训练模型并可视化过程
    n_steps = 20  # 训练步数
    for step in range(n_steps):
        # 计算当前步骤每个类别应该使用的样本数
        current_samples_per_class = max(
            min_samples_per_class,
            int(len(X) * (step + 1) / n_steps / 2)  # 除以2是因为有两个类别
        )
        
        # 从每个类别中随机选择样本，但保持之前选择的样本
        if step == 0:
            selected_class_0 = np.random.choice(
                class_0_indices,
                size=min(current_samples_per_class, len(class_0_indices)),
                replace=False
            )
            selected_class_1 = np.random.choice(
                class_1_indices,
                size=min(current_samples_per_class, len(class_1_indices)),
                replace=False
            )
        else:
            # 在剩余样本中选择新的样本
            remaining_class_0 = np.setdiff1d(class_0_indices, selected_class_0)
            remaining_class_1 = np.setdiff1d(class_1_indices, selected_class_1)
            
            additional_samples_0 = min(
                current_samples_per_class - len(selected_class_0),
                len(remaining_class_0)
            )
            additional_samples_1 = min(
                current_samples_per_class - len(selected_class_1),
                len(remaining_class_1)
            )
            
            if additional_samples_0 > 0:
                new_samples_0 = np.random.choice(
                    remaining_class_0,
                    size=additional_samples_0,
                    replace=False
                )
                selected_class_0 = np.concatenate([selected_class_0, new_samples_0])
            
            if additional_samples_1 > 0:
                new_samples_1 = np.random.choice(
                    remaining_class_1,
                    size=additional_samples_1,
                    replace=False
                )
                selected_class_1 = np.concatenate([selected_class_1, new_samples_1])
        
        # 合并选择的样本
        selected_indices = np.concatenate([selected_class_0, selected_class_1])
        X_subset = X_scaled[selected_indices]
        y_subset = y[selected_indices]
        
        # 训练模型
        clf.fit(X_subset, y_subset)
        
        # 获取支持向量（需要转换回原始空间）
        support_vectors_scaled = clf.support_vectors_
        support_vectors = scaler.inverse_transform(support_vectors_scaled)
        
        # 为网格点创建缩放版本
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_scaled = scaler.transform(grid_points)
        
        # 计算决策函数值
        Z = clf.decision_function(grid_points_scaled)
        Z = Z.reshape(xx.shape)
        
        # 计算当前准确率
        y_pred = clf.predict(X_scaled)
        accuracy = np.mean(y_pred == y)
        
        # 获取决策函数的范围
        decision_values = clf.decision_function(X_scaled)
        decision_min = np.min(decision_values)
        decision_max = np.max(decision_values)
        
        # 生成当前步骤的描述
        kernel_descriptions = {
            'linear': '线性核函数将在原始特征空间中寻找最优分类超平面',
            'rbf': 'RBF 核函数通过高斯径向基函数将数据映射到高维空间',
            'poly': f'{degree}次多项式核函数将数据映射到更高维的特征空间',
            'sigmoid': 'Sigmoid 核函数模拟神经网络的激活函数特性'
        }
        
        description = f"""
        第{step + 1}步训练：
        • 类别1使用了{len(selected_class_0)}个样本，类别2使用了{len(selected_class_1)}个样本
        • {kernel_descriptions[kernel]}
        • 当前支持向量数量为{len(support_vectors)}个
        • 惩罚参数C={C}控制模型对错误分类的容忍度
        • {'gamma参数=' + str(gamma) if gamma != 'scale' else 'gamma自适应缩放'}
        """
        
        # 创建统计信息字典
        stats = {
            'accuracy': accuracy,
            'n_support_vectors': len(support_vectors),
            'decision_min': decision_min,
            'decision_max': decision_max,
            'description': description
        }
        
        # 创建可视化
        fig = go.Figure()
        
        # 绘制决策边界的颜色区块
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            colorscale='RdBu',
            showscale=True,
            opacity=0.5,
            contours=dict(
                start=-3,
                end=3,
                size=0.5
            ),
            name='决策函数值'
        ))
        
        # 绘制等高线
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            colorscale='RdBu',
            showscale=False,
            opacity=0.2,
            contours=dict(
                showlines=True,
                coloring='lines',
                start=-3,
                end=3,
                size=0.5
            ),
            line=dict(
                width=1
            ),
            name='等高线'
        ))
        
        # 添加决策边界（Z = 0）的粗线条
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            showscale=False,
            contours=dict(
                coloring=None,
                showlines=True,
                start=0,
                end=0,
                size=1,
                type='constraint'
            ),
            line=dict(
                width=3,
                color='black',
                dash='solid'
            ),
            name='分类超平面'
        ))
        
        # 绘制数据点
        for label in np.unique(y):
            mask = y == label
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=label,
                    colorscale='Viridis',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=f'类别 {label + 1}',
                showlegend=True
            ))
        
        # 绘制支持向量
        fig.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            marker=dict(
                size=12,
                symbol='circle-open',
                color='black',
                line=dict(width=2)
            ),
            name='支持向量',
            showlegend=True
        ))
        
        # 更新布局
        fig.update_layout(
            title=f'训练进度: {step + 1}/{n_steps}',
            xaxis_title='特征1',
            yaxis_title='特征2',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            width=800,
            height=600
        )
        
        # 调用回调函数，传递图、进度和统计信息
        if visualization_callback:
            progress = (step + 1) / n_steps
            visualization_callback(fig, progress, stats)
        
        # 添加短暂延迟以便观察
        time.sleep(0.5)
    
    return clf

def reduce_to_2d(X):
    """使用PCA将数据降维到2D"""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d, pca

def plot_classification_process(clf, X, current_labels, X_2d, pca, support_vectors=None, title=None):
    """绘制分类过程，展示分类超平面和支持向量"""
    # 将标题文本分成多行
    if title:
        title_lines = [line for line in title.split('\n') if not line.startswith("迭代")]
    else:
        title_lines = ["分类过程可视化"]
    
    # 创建图形
    fig = go.Figure()
    
    # 如果模型已训练且有足够的已标记样本，绘制决策边界
    if clf is not None and hasattr(clf, 'support_') and len(np.where(current_labels != -1)[0]) >= 2:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            colorscale=['#e74c3c', '#2ecc71'],  # 红色和绿色表示两个类别
            showscale=False,
            opacity=0.3,
            contours=dict(showlines=False)
        ))
        
        # 绘制决策函数值
        if hasattr(clf, 'decision_function'):
            Z_decision = clf.decision_function(grid_points)
            Z_decision = Z_decision.reshape(xx.shape)
            
            # 绘制决策函数等高线
            fig.add_trace(go.Contour(
                x=np.arange(x_min, x_max, 0.1),
                y=np.arange(y_min, y_max, 0.1),
                z=Z_decision,
                colorscale='RdBu',
                showscale=True,
                opacity=0.2,
                contours=dict(
                    showlines=True,
                    coloring='lines',
                    start=-3,
                    end=3,
                    size=0.5
                )
            ))
            
            # 添加决策边界（Z_decision = 0）的粗线条
            levels = np.array([0])  # 只显示决策边界
            fig.add_trace(go.Contour(
                x=np.arange(x_min, x_max, 0.1),
                y=np.arange(y_min, y_max, 0.1),
                z=Z_decision,
                showscale=False,
                contours=dict(
                    coloring=None,
                    showlines=True,
                    start=0,
                    end=0,
                    size=1,
                    type='constraint'
                ),
                line=dict(
                    width=3,
                    color='black'
                ),
                name='分类超平面'
            ))
    
    # 添加数据点
    colors = ['#808080', '#e74c3c', '#2ecc71']  # 灰色用于未分类，红色和绿色用于两个类别
    
    # 首先绘制未标记的点
    unlabeled_mask = current_labels == -1
    if np.any(unlabeled_mask):
        fig.add_trace(go.Scatter(
            x=X[unlabeled_mask, 0],
            y=X[unlabeled_mask, 1],
            mode='markers',
            name='未分类',
            marker=dict(
                size=8,
                color=colors[0],
                symbol='circle',
                line=dict(color='white', width=1)
            )
        ))
    
    # 绘制已标记的点
    for label in [0, 1]:
        mask = current_labels == label
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=f'类别 {label + 1}',
                marker=dict(
                    size=8,
                    color=colors[label + 1],
                    symbol='circle',
                    line=dict(color='white', width=1)
                )
            ))
    
    # 绘制支持向量
    if support_vectors is not None and len(support_vectors) > 0:
        sv_points = X[support_vectors]
        sv_labels = current_labels[support_vectors]
        
        for label in [0, 1]:
            mask = sv_labels == label
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=sv_points[mask, 0],
                    y=sv_points[mask, 1],
                    mode='markers',
                    name=f'支持向量(类别 {label + 1})',
                    marker=dict(
                        size=12,
                        color='rgba(0,0,0,0)',
                        symbol='circle',
                        line=dict(
                            color=colors[label + 1],
                            width=2
                        )
                    )
                ))
    
    # 创建训练信息的注释文本
    info_text = "<br>".join([
        f"<b>{line}</b>" for line in title_lines
    ])
    
    # 设置布局
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=120, t=100, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        xaxis=dict(
            title=dict(
                text="特征1",
                font=dict(size=12),
                standoff=15
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=False
        ),
        yaxis=dict(
            title=dict(
                text="特征2",
                font=dict(size=12),
                standoff=15
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=False
        ),
        annotations=[
            dict(
                text=info_text,
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0,
                y=1.1,
                font=dict(size=10)
            )
        ]
    )
    
    return fig, title_lines

def plot_confusion_matrix(y_true, y_pred, target_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    # 确保target_names的长度与混淆矩阵维度匹配
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(target_names) > len(unique_labels):
        target_names = target_names[:len(unique_labels)]
    elif len(target_names) < len(unique_labels):
        target_names = [f"类别{i+1}" for i in range(len(unique_labels))]
    
    fig = px.imshow(
        cm,
        labels=dict(x="预测类别", y="真实类别", color="数量"),
        x=target_names,
        y=target_names,
        color_continuous_scale='Viridis',
        text_auto=True
    )
    
    fig.update_layout(
        title=dict(
            text="混淆矩阵",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        xaxis_title="预测类别",
        yaxis_title="真实类别",
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14)
    )
    
    return fig

def visualize_3d_svm(X, y, kernel='rbf', C=1.0, gamma='scale'):
    """在3D空间中可视化SVM分类结果"""
    # 创建SVM模型并训练
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X, y)
    
    # 获取支持向量
    support_vectors = clf.support_vectors_
    
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    
    # 计算决策函数值作为z轴
    if kernel == 'rbf':
        gamma_value = 1.0 / (X.shape[1] * X.var()) if gamma == 'scale' else gamma
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        zz = clf.decision_function(grid_points).reshape(xx.shape)
    else:
        zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # 创建3D图
    fig = go.Figure()
    
    # 添加决策曲面
    fig.add_trace(go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale='RdBu',
        opacity=0.8,
        name='决策曲面',
        showscale=True,
        colorbar=dict(
            title='决策函数值',
            thickness=15,
            len=0.5
        )
    ))
    
    # 添加原始数据点
    colors = ['#e74c3c', '#2ecc71']  # 红色和绿色
    for label in [0, 1]:
        mask = y == label
        z_points = clf.decision_function(X[mask])
        
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0],
            y=X[mask, 1],
            z=z_points,
            mode='markers',
            marker=dict(
                size=6,
                color=colors[label],
                opacity=0.8
            ),
            name=f'类别 {label + 1}'
        ))
    
    # 添加支持向量
    sv_z = clf.decision_function(support_vectors)
    fig.add_trace(go.Scatter3d(
        x=support_vectors[:, 0],
        y=support_vectors[:, 1],
        z=sv_z,
        mode='markers',
        marker=dict(
            size=8,
            color='black',
            symbol='circle',
            line=dict(color='white', width=1)
        ),
        name='支持向量'
    ))
    
    # 添加决策边界平面（z=0）
    z_plane = np.zeros_like(xx)
    fig.add_trace(go.Surface(
        x=xx,
        y=yy,
        z=z_plane,
        opacity=0.5,
        showscale=False,
        colorscale=[[0, 'rgb(200,200,200)'], [1, 'rgb(200,200,200)']],
        name='决策边界'
    ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='SVM在3D空间的分类可视化',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='特征1',
            yaxis_title='特征2',
            zaxis_title='决策函数值',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig, clf

def evaluate_model(clf, X_test, y_test):
    """评估模型性能"""
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred 