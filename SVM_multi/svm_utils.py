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

def load_dataset(dataset_name, file_path=None):
    """加载数据集
    
    参数:
        dataset_name: 数据集名称，可以是 "鸢尾花数据集"、"Glass数据集"、"Wine数据集" 或 "自定义数据集"
        file_path: 当dataset_name为"自定义数据集"时，指定数据文件路径
    """
    if dataset_name == "鸢尾花数据集":
        data = datasets.load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == "Glass数据集":
        # 从UCI加载Glass数据集
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        try:
            response = requests.get(url)
            data = pd.read_csv(StringIO(response.text), header=None)
            # 删除ID列（第一列）
            data = data.iloc[:, 1:]
            # 分离特征和标签
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values - 1  # 标签从1开始，转换为从0开始
            
            feature_names = [
                'Refractive Index',
                'Sodium',
                'Magnesium',
                'Aluminum',
                'Silicon',
                'Potassium',
                'Calcium',
                'Barium',
                'Iron'
            ]
            
            target_names = [f'Type {i}' for i in range(1, 8)]  # Glass数据集有7种类型
            
        except Exception as e:
            raise ValueError(f"无法加载Glass数据集: {str(e)}")
            
    elif dataset_name == "Wine数据集":
        # 直接使用scikit-learn的Wine数据集
        wine = datasets.load_wine()
        X = wine.data
        y = wine.target
        feature_names = wine.feature_names
        target_names = wine.target_names
            
    elif dataset_name == "自定义数据集":
        if file_path is None:
            raise ValueError("自定义数据集需要提供文件路径")
        try:
            # 尝试读取CSV文件
            df = pd.read_csv(file_path)
            # 假设最后一列是目标变量
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1].tolist()
            target_names = [f"类别{i+1}" for i in range(len(np.unique(y)))]
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
    
    # 创建四个独立的图表
    figures = []
    
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
        
        # 设置更合适的边距比例
        margin = 0.1  # 减小边距比例
        x_min = x_center - x_range * (1 + margin)
        x_max = x_center + x_range * (1 + margin)
        y_min = y_center - y_range * (1 + margin)
        y_max = y_center + y_range * (1 + margin)
        z_min = z_center - z_range * (1 + margin)
        z_max = z_center + z_range * (1 + margin)
        
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode='markers',
                marker=dict(
                    size=6,  # 增大点的大小
                    opacity=0.8,  # 增加不透明度
                    color='#1f77b4',
                    colorscale='Blues'
                ),
                name='原始数据点'
            )
        )
        
        # 优化3D图的视角和样式
        fig1.update_scenes(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.2)  # 调整视角距离
            ),
            aspectmode='cube',  # 使用立方体模式确保三个轴的比例相同
            xaxis=dict(
                range=[x_min, x_max],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title=dict(text="特征1", font=dict(size=12, color='#2c3e50')),
                showspikes=False  # 关闭轴线上的刺度显示
            ),
            yaxis=dict(
                range=[y_min, y_max],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title=dict(text="特征2", font=dict(size=12, color='#2c3e50')),
                showspikes=False
            ),
            zaxis=dict(
                range=[z_min, z_max],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title=dict(text="特征3", font=dict(size=12, color='#2c3e50')),
                showspikes=False
            )
        )
        
        fig1.update_layout(
            title=dict(
                text='<b>原始特征空间分布</b>',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=16, color='#2c3e50')
            ),
            height=600,  # 增加图表高度
            width=800,
            margin=dict(l=0, r=0, t=80, b=0),  # 减小边距以增大图表显示区域
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=False  # 隐藏图例，因为只有一种点
        )
        figures.append(fig1)
    
    # 2. 主成分解释方差比
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=['PC1', 'PC2'],
            y=explained_variance_ratio,
            marker_color=['#2ecc71', '#e74c3c'],
            name='解释方差比',
            text=[f'{v:.1%}' for v in explained_variance_ratio],
            textposition='outside',
            width=0.4  # 减小柱状图宽度
        )
    )
    
    fig2.update_layout(
        title=dict(
            text='<b>主成分解释方差比</b>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color='#2c3e50')
        ),
        height=500,  # 增加图表高度
        width=700,   # 增加图表宽度
        margin=dict(l=50, r=50, t=100, b=50),  # 增加顶部边距
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title="主成分",
            title_font=dict(size=12),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="解释方差比",
            title_font=dict(size=12),
            tickfont=dict(size=12),
            range=[0, 1.1],  # 扩大y轴范围以确保标签显示完整
            tickformat='.0%'  # 将刻度格式设置为百分比
        ),
        showlegend=False,
        bargap=0.5  # 增加柱状图之间的间距
    )
    figures.append(fig2)
    
    # 3. 特征投影方向
    fig3 = go.Figure()
    max_component = max(abs(components.min()), abs(components.max()))
    
    for i, (feature, pc1, pc2) in enumerate(zip(feature_names, components[0], components[1])):
        scale = 0.8
        pc1_norm = pc1 * scale / max_component
        pc2_norm = pc2 * scale / max_component
        
        fig3.add_trace(
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
            )
        )
    
    fig3.update_layout(
        title=dict(
            text='<b>特征投影方向</b>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color='#2c3e50')
        ),
        height=500,
        width=600,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis_title="第一主成分方向",
        yaxis_title="第二主成分方向",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    figures.append(fig3)
    
    # 4. 降维后的数据分布
    fig4 = go.Figure()
    
    # 计算颜色范围
    color_min = np.floor(X_pca[:, 0].min())  # 向下取整
    color_max = np.ceil(X_pca[:, 0].max())   # 向上取整
    
    fig4.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=X_pca[:, 0],
                cmin=color_min,    # 设置颜色最小值
                cmax=color_max,    # 设置颜色最大值
                colorscale=[
                    [0, '#1f77b4'],    # 深蓝色
                    [0.25, '#2ecc71'], # 绿色
                    [0.5, '#f1c40f'],  # 黄色
                    [0.75, '#e67e22'], # 橙色
                    [1, '#e74c3c']     # 红色
                ],
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='PC1值',
                        font=dict(size=14)
                    ),
                    thickness=25,      # 增加颜色条宽度
                    len=0.75,         # 增加颜色条长度
                    y=0.5,
                    yanchor='middle',
                    titleside='right',
                    outlinewidth=1,
                    outlinecolor='#888',
                    tickfont=dict(size=12),
                    ticks='outside',
                    tickwidth=2,
                    ticklen=8,
                    dtick=1.0,        # 设置刻度间隔为1
                    tick0=color_min    # 设置起始刻度
                )
            ),
            name='降维后的数据'
        )
    )
    
    fig4.update_layout(
        title=dict(
            text='<b>降维后的数据分布</b>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color='#2c3e50')
        ),
        height=500,
        width=600,
        margin=dict(l=50, r=100, t=80, b=50),  # 增加右边距以容纳更宽的颜色条
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title="第一主成分",
            gridcolor='#f0f0f0',
            zerolinecolor='#888',
            zerolinewidth=1
        ),
        yaxis=dict(
            title="第二主成分",
            gridcolor='#f0f0f0',
            zerolinecolor='#888',
            zerolinewidth=1
        ),
        showlegend=False
    )
    figures.append(fig4)
    
    return figures, X_pca, components

def train_svm_with_visualization(X, y, kernel='rbf', C=1.0, gamma='auto', degree=3, visualization_callback=None):
    """训练SVM模型，展示真实的分类过程"""
    # 确保C是浮点数
    C = float(C)
    # gamma可以是'auto'或'scale'或浮点数
    if isinstance(gamma, str) and gamma not in ['auto', 'scale']:
        raise ValueError("gamma必须是'auto'、'scale'或浮点数")
    elif not isinstance(gamma, str):
        gamma = float(gamma)
    
    # 首先划分训练集和测试集（使用更小的测试集比例）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 将训练数据降维到2D用于可视化
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    
    # 初始化模型
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    
    # 初始化所有点为未分类状态
    current_labels = np.full(len(X_train), -1)  # -1 表示未分类
    
    # 记录错误分类的点
    misclassified_indices = []
    current_accuracy = 0.0
    best_accuracy = 0.0
    best_clf = None
    
    # 获取每个类别的样本索引
    unique_classes = np.unique(y_train)
    class_indices = {label: np.where(y_train == label)[0] for label in unique_classes}
    
    # 开始训练过程
    i = 0
    max_iterations = 100  # 增加最大迭代次数
    
    while np.any(current_labels == -1) and i < max_iterations:
        # 如果是第一次迭代，随机选择每个类别的一个点
        if i == 0:
            for label in unique_classes:
                idx = np.random.choice(class_indices[label])
                current_labels[idx] = y_train[idx]
        else:
            labeled_indices = np.where(current_labels != -1)[0]
            if len(labeled_indices) >= len(unique_classes):
                X_train_labeled = X_train[labeled_indices]
                y_train_labeled = y_train[labeled_indices]
                
                # 训练模型
                clf.fit(X_train_labeled, y_train_labeled)
                
                # 计算当前准确率（使用所有已标记的点）
                current_predictions = clf.predict(X_train)
                current_accuracy = np.mean(current_predictions[labeled_indices] == y_train[labeled_indices])
                
                # 更新最佳模型
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_clf = clf
                
                # 更新错误分类的点
                misclassified_indices = labeled_indices[current_predictions[labeled_indices] != y_train[labeled_indices]]
                
                # 从未标记的样本中选择新样本
                unlabeled_indices = np.where(current_labels == -1)[0]
                if len(unlabeled_indices) > 0:
                    # 计算到已标记点的距离
                    labeled_points = X_train_2d[labeled_indices]
                    unlabeled_points = X_train_2d[unlabeled_indices]
                    
                    # 计算到最近已标记点的距离
                    distances = np.zeros(len(unlabeled_points))
                    for j, point in enumerate(unlabeled_points):
                        point_distances = np.sqrt(np.sum((labeled_points - point) ** 2, axis=1))
                        distances[j] = np.min(point_distances)
                    
                    # 计算决策函数值（不确定性）
                    if hasattr(clf, 'decision_function'):
                        uncertainties = np.abs(clf.decision_function(X_train[unlabeled_indices]))
                        if len(uncertainties.shape) > 1:
                            uncertainties = np.min(np.abs(uncertainties), axis=1)
                    else:
                        uncertainties = np.zeros(len(unlabeled_indices))
                    
                    # 综合考虑距离和不确定性
                    scores = distances + 1.0 / (uncertainties + 1e-10)
                    
                    # 选择得分最高的点（每次只选择1-2个点）
                    n_select = min(2, len(unlabeled_indices))
                    selected_indices = unlabeled_indices[np.argsort(scores)[-n_select:]]
                    current_labels[selected_indices] = y_train[selected_indices]
        
        # 计算当前进度
        progress = np.sum(current_labels != -1) / len(X_train)
        
        # 生成详细的迭代信息
        iteration_info = "\n".join([
            f"已标记样本：{np.sum(current_labels != -1)}个 / 总计：{len(X_train)}个",
            f"当前训练准确率：{current_accuracy:.2%}",
            f"最佳训练准确率：{best_accuracy:.2%}",
            f"错误分类点数：{len(misclassified_indices)}个",
            f"支持向量数量：{len(clf.support_) if hasattr(clf, 'support_') else 0}个",
            f"当前参数：C={C}, gamma={gamma}"
        ])
        
        # 如果提供了可视化回调函数，调用它
        if visualization_callback:
            # 创建当前步骤的决策边界图
            result = plot_classification_process(
                clf if len(np.where(current_labels != -1)[0]) >= len(unique_classes) else None,
                X_train, current_labels, X_train_2d, pca,
                misclassified_indices=misclassified_indices,
                title=iteration_info
            )
            visualization_callback(result, progress)
        
        # 添加短暂延迟以便观察
        time.sleep(0.5)
        i += 1
    
    return best_clf if best_clf is not None else clf

def reduce_to_2d(X):
    """使用PCA将数据降维到2D"""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d, pca

def plot_classification_process(clf, X, current_labels, X_2d, pca, misclassified_indices=None, title=None):
    """绘制分类过程，区分已标记和未标记样本，突出显示错误分类的点"""
    # 创建图形
    fig = go.Figure()
    
    # 计算数据范围并添加边距
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    
    # 如果模型已训练且有足够的已标记样本，绘制决策边界
    if clf is not None and hasattr(clf, 'support_') and len(np.unique(current_labels[current_labels != -1])) >= len(np.unique(current_labels[current_labels >= 0])):
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_original = pca.inverse_transform(grid_points)
        
        Z = clf.predict(grid_points_original)
        Z = Z.reshape(xx.shape)
        
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            colorscale='Viridis',
            showscale=False,
            opacity=0.3,
            contours=dict(showlines=False)
        ))
    
    # 添加数据点
    colors = ['#808080'] + list(px.colors.qualitative.Set1)  # 灰色用于未分类的点
    
    # 首先绘制未标记的点
    unlabeled_mask = current_labels == -1
    if np.any(unlabeled_mask):
        fig.add_trace(go.Scatter(
            x=X_2d[unlabeled_mask, 0],
            y=X_2d[unlabeled_mask, 1],
            mode='markers',
            name='未分类',
            marker=dict(
                size=8,
                color=colors[0],
                symbol='circle',
                line=dict(color='white', width=1)
            )
        ))
    
    # 创建标签到颜色的映射
    unique_labels = np.unique(current_labels[current_labels != -1])
    label_to_color = {label: colors[i + 1] for i, label in enumerate(unique_labels)}
    
    # 然后绘制已标记的点
    for i, label in enumerate(unique_labels):
        mask = current_labels == label
        if misclassified_indices is not None:
            mask = mask & ~np.isin(np.arange(len(current_labels)), misclassified_indices)
        
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode='markers',
                name=f'类别 {label + 1}',
                marker=dict(
                    size=8,
                    color=colors[i + 1],
                    symbol='circle',
                    line=dict(color='white', width=1)
                )
            ))
    
    # 如果有错误分类的点，用黑色实心点显示
    if misclassified_indices is not None and len(misclassified_indices) > 0:
        fig.add_trace(go.Scatter(
            x=X_2d[misclassified_indices, 0],
            y=X_2d[misclassified_indices, 1],
            mode='markers',
            name='分类错误',
            marker=dict(
                size=8,
                color='black',
                symbol='circle'
            )
        ))
    
    # 如果模型已训练，添加支持向量
    if clf is not None and hasattr(clf, 'support_vectors_'):
        sv_indices = clf.support_
        sv_points = X_2d[sv_indices]
        sv_labels = current_labels[sv_indices]
        
        for label in unique_labels:
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
                            color=label_to_color[label],
                            width=2
                        )
                    )
                ))
    
    # 固定画布大小和布局
    fig.update_layout(
        width=800,  # 固定宽度
        height=600, # 固定高度
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bordercolor='#E2E2E2',
            borderwidth=1
        ),
        margin=dict(l=50, r=150, t=50, b=50),  # 调整边距以适应图例
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title=dict(
                text="第一主成分",
                font=dict(size=12),
                standoff=15
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#E2E2E2',
            zerolinewidth=1,
            range=[x_min, x_max]  # 固定坐标轴范围
        ),
        yaxis=dict(
            title=dict(
                text="第二主成分",
                font=dict(size=12),
                standoff=15
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#E2E2E2',
            zerolinewidth=1,
            range=[y_min, y_max]  # 固定坐标轴范围
        )
    )
    
    return fig, title

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

def evaluate_model(clf, X_test, y_test):
    """评估模型性能"""
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred 