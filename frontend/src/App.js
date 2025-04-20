/* @charset "UTF-8"; */
import React, { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';
import './App.css';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/contrib/auto-render.min.js';

// 添加 KaTeX 字体 CDN
const katexFonts = `
  @font-face {
    font-family: 'KaTeX_AMS';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_AMS-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_AMS-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Caligraphic';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Caligraphic-Bold.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Caligraphic-Bold.woff') format('woff');
    font-weight: 700;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Caligraphic';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Caligraphic-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Caligraphic-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Fraktur';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Fraktur-Bold.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Fraktur-Bold.woff') format('woff');
    font-weight: 700;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Fraktur';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Fraktur-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Fraktur-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Main';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-Bold.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-Bold.woff') format('woff');
    font-weight: 700;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Main';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-BoldItalic.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-BoldItalic.woff') format('woff');
    font-weight: 700;
    font-style: italic;
  }
  @font-face {
    font-family: 'KaTeX_Main';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-Italic.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-Italic.woff') format('woff');
    font-weight: 400;
    font-style: italic;
  }
  @font-face {
    font-family: 'KaTeX_Main';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Main-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Math';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Math-BoldItalic.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Math-BoldItalic.woff') format('woff');
    font-weight: 700;
    font-style: italic;
  }
  @font-face {
    font-family: 'KaTeX_Math';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Math-Italic.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Math-Italic.woff') format('woff');
    font-weight: 400;
    font-style: italic;
  }
  @font-face {
    font-family: 'KaTeX_SansSerif';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_SansSerif-Bold.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_SansSerif-Bold.woff') format('woff');
    font-weight: 700;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_SansSerif';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_SansSerif-Italic.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_SansSerif-Italic.woff') format('woff');
    font-weight: 400;
    font-style: italic;
  }
  @font-face {
    font-family: 'KaTeX_SansSerif';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_SansSerif-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_SansSerif-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Script';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Script-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Script-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Size1';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size1-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size1-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Size2';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size2-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size2-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Size3';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size3-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size3-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Size4';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size4-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Size4-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
  @font-face {
    font-family: 'KaTeX_Typewriter';
    src: url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Typewriter-Regular.woff2') format('woff2'),
         url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/KaTeX_Typewriter-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
  }
`;

// 创建样式标签并添加到文档头部
const style = document.createElement('style');
style.textContent = katexFonts;
document.head.appendChild(style);

// 添加图片组件
const ImageComponent = ({ src, alt, className }) => {
  return (
    <div className={`image-container ${className || ''}`}>
      <img src={src} alt={alt} className="content-image" />
    </div>
  );
};

const WelcomePage = ({ setCurrentMode }) => {
  return (
    <div className="welcome-container">
      <div className="welcome-header">
        <h1>欢迎使用 SVM 教学平台</h1>
        <p>探索支持向量机的奥秘，体验交互式学习之旅</p>
      </div>
      <div className="welcome-features">
        <div className="feature-card">
          <h3>基础知识</h3>
          <p>了解 SVM 的基本概念和原理</p>
        </div>
        <div className="feature-card">
          <h3>动态演示</h3>
          <p>实时可视化 SVM 的训练过程</p>
        </div>
        <div className="feature-card">
          <h3>拓展内容</h3>
          <p>深入了解 SVM 的知识迁移</p>
        </div>
      </div>
      <div className="welcome-actions">
        <button className="start-button" onClick={() => setCurrentMode('basic')}>
          开始学习
        </button>
      </div>
      <div className="copyright">
        ©2025 基于 SVM 的仿真实验与理论教学一体化交互平台 | 由大连理工大学 MindForge 团队开发
      </div>
    </div>
  );
};

function App() {
  const [currentMode, setCurrentMode] = useState('welcome');
  // 数据及其他状态
  const [data, setData] = useState([]);
  const [labels, setLabels] = useState([]);
  const [visualization, setVisualization] = useState(null);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('info');
  const [currentLabel, setCurrentLabel] = useState(1);
  const [isTraining, setIsTraining] = useState(false);
  const [kernels, setKernels] = useState([]);
  const [kernelParams, setKernelParams] = useState({});
  const [selectedKernel, setSelectedKernel] = useState("linear");
  const [selectedParams, setSelectedParams] = useState({
    C: 1.0,
    gamma: 'manual', // 默认设为manual
    gamma_value: 1.0, // 默认值设为1
    degree: 3,
    coef0: 0
  });
  const [teachingStep, setTeachingStep] = useState(0);
  const [showTeaching, setShowTeaching] = useState(false);
  const [animationStates, setAnimationStates] = useState([]);
  const [currentAnimationFrame, setCurrentAnimationFrame] = useState(0);
  const [animationFrame, setAnimationFrame] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const animationTimer = useRef(null);
  const plotRef = useRef(null);
  const lastClickTime = useRef(0);
  // 添加手动加点相关的状态
  const [showManualAddModal, setShowManualAddModal] = useState(false);
  const [manualInputText, setManualInputText] = useState('');
  // 添加滑动框加点相关的状态
  const [showSliderAddModal, setShowSliderAddModal] = useState(false);
  const [sliderX1, setSliderX1] = useState(0);
  const [sliderX2, setSliderX2] = useState(0);
  const [sliderLabel, setSliderLabel] = useState(1);
  const [previewPoint, setPreviewPoint] = useState(null);
  const [plotLayout, setPlotLayout] = useState({
    width: 800,
    height: 600,
    title: '训练结果',
    xaxis: {
      title: 'X轴',
      gridcolor: '#eee',
      zerolinecolor: '#969696',
      range: [-10, 10],
      autorange: true,
      fixedrange: false,
      dtick: 1,  // 主要刻度间隔
      tick0: 0,  // 起始刻度
      minor: {
        dtick: 0.2,  // 次要刻度间隔
        ticklen: 4,  // 次要刻度长度
        tickcolor: '#999'  // 次要刻度颜色
      }
    },
    yaxis: {
      title: 'Y轴',
      gridcolor: '#eee',
      zerolinecolor: '#969696',
      range: [-10, 10],
      autorange: true,
      fixedrange: false,
      dtick: 1,  // 主要刻度间隔
      tick0: 0,  // 起始刻度
      minor: {
        dtick: 0.2,  // 次要刻度间隔
        ticklen: 4,  // 次要刻度长度
        tickcolor: '#999'  // 次要刻度颜色
      }
    },
    plot_bgcolor: '#fff',
    paper_bgcolor: '#fff',
    margin: { l: 50, r: 50, t: 50, b: 50 },
    showlegend: true,
    legend: {
      x: 1,
      y: 1,
      bgcolor: '#fff',
      bordercolor: '#969696'
    },
    updatemenus: [{
      type: 'buttons',
      showactive: false,
      x: 0.1,
      y: 1.1,
      buttons: [{
        label: '播放',
        method: 'animate',
        args: [null, {
          fromcurrent: true,
          frame: { duration: 1000, redraw: true },
          transition: { duration: 500 }
        }]
      }]
    }]
  });

  // 添加基础知识导航状态
  const [activeBasicSection, setActiveBasicSection] = useState('introduction');

  // 基础知识导航项
  const basicSections = [
    { id: 'introduction', title: 'SVM 简介' },
    { id: 'core_concepts', title: '核心概念' },
    { id: 'kernel_functions', title: '核函数详解' },
    { id: 'parameters', title: '参数说明' },
    { id: 'applications', title: '应用场景' },
    { id: 'advantage', title: '优缺点分析' },
    { id: 'implementation', title: '实现细节' },
    { id: 'formula', title: '公式推导' }
  ];

  // 添加分类模式状态
  const [classificationMode, setClassificationMode] = useState('binary'); // 'binary' 或 'multi'

  const [plotData, setPlotData] = useState([]);
  const [trainingResults, setTrainingResults] = useState(null);

  const [animationInterval, setAnimationInterval] = useState(1); // 默认1秒

  const [activeTab, setActiveTab] = useState('basics'); // 默认显示基础知识

  // 修改消息状态
  const [messages, setMessages] = useState([]);

  // 添加显示消息的函数
  const showMessage = (text, type) => {
    const id = Date.now();
    setMessages(prev => [...prev, { id, text, type }]);
    
    // 2秒后自动移除消息
    setTimeout(() => {
      setMessages(prev => prev.filter(msg => msg.id !== id));
    }, 2000);
  };

  useEffect(() => {
    axios.get('http://localhost:5000/api/kernels')
      .then(response => {
        if (response.data.status === 'success') {
          setKernels(response.data.kernels);
          setKernelParams(response.data.params);
        }
      })
      .catch(error => {
        // 移除错误提示
        console.log('Loading kernels failed:', error);
      });
  }, []);

  const handlePlotClick = (event) => {
    if (isTraining) return;
    
    const plotElement = document.querySelector('.js-plotly-plot');
    if (!plotElement) return;
    
    const rect = plotElement.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    
    // 转换坐标到数据范围
    const xRange = plotElement._fullLayout.xaxis.range;
    const yRange = plotElement._fullLayout.yaxis.range;
    const dataX = xRange[0] + (clickX / rect.width) * (xRange[1] - xRange[0]);
    const dataY = yRange[0] + (clickY / rect.height) * (yRange[1] - yRange[0]);
    
    setData([...data, [dataX, dataY]]);
    setLabels([...labels, currentLabel]);
    
    // 更新坐标轴范围
    const newRange = calculateRange([...data, [dataX, dataY]]);
    setPlotLayout(prev => ({
      ...prev,
      xaxis: {
        ...prev.xaxis,
        range: newRange[0]
      },
      yaxis: {
        ...prev.yaxis,
        range: newRange[1]
      }
    }));
  };

  const readFile = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          const lines = content.split('\n');
          const data = [];
          const labels = [];
          
          lines.forEach((line, index) => {
            const trimmedLine = line.trim();
            if (!trimmedLine) return;
            
            // 处理CSV文件
            if (file.name.endsWith('.csv')) {
              const values = trimmedLine.split(',').map(v => v.trim());
              if (values.length >= 3) {  // 确保有x, y, label三个值
                const x = parseFloat(values[0]);
                const y = parseFloat(values[1]);
                const label = parseFloat(values[2]);
                if (!isNaN(x) && !isNaN(y) && !isNaN(label)) {
                  data.push([x, y]);
                  labels.push(label > 0 ? 1 : 0);  // 将标签转换为1或0
                }
              }
            }
            // 处理TXT文件
            else if (file.name.endsWith('.txt')) {
              const values = trimmedLine
                .replace(/\t/g, ' ')
                .replace(/\s+/g, ' ')
                .split(' ')
                .filter(v => v !== '');
              
              if (values.length >= 3) {  // 确保有x, y, label三个值
                const x = parseFloat(values[0]);
                const y = parseFloat(values[1]);
                const label = parseFloat(values[2]);
                if (!isNaN(x) && !isNaN(y) && !isNaN(label)) {
                  data.push([x, y]);
                  labels.push(label > 0 ? 1 : 0);  // 将标签转换为1或0
                }
              }
            }
          });
          
          if (data.length > 0) {
            resolve({ data, labels });
          } else {
            reject(new Error('文件中没有有效的数据点'));
          }
        } catch (error) {
          reject(new Error(`文件解析错误: ${error.message}`));
        }
      };
      
      reader.onerror = () => {
        reject(new Error('文件读取失败'));
      };
      
      reader.readAsText(file);
    });
  };

  const handleFileUpload = async (event) => {
    // 创建一个隐藏的文件输入元素
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.csv,.txt';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    // 监听文件选择事件
    fileInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) {
        setMessage('请选择文件');
        setMessageType('error');
        return;
      }

      try {
        const fileData = await readFile(file);
        if (!fileData || !fileData.data || !Array.isArray(fileData.data)) {
          throw new Error('文件格式不正确');
        }

        setData(fileData.data);
        setLabels(fileData.labels);
        showMessage('数据加载成功', 'success');
      } catch (error) {
        console.error('文件处理错误:', error);
        showMessage(`文件处理错误: ${error.message}`, 'error');
      } finally {
        // 清理文件输入元素
        document.body.removeChild(fileInput);
      }
    };

    // 触发文件选择对话框
    fileInput.click();
  };

  // 修改后的 handleKernelChange：设置选中的核函数及其默认参数
  const handleKernelChange = (kernel) => {
    setSelectedKernel(kernel);
    // 重置核函数参数
    const defaultParams = {
      linear: { C: 1.0 },
      poly: { C: 1.0, degree: 3, gamma: 'scale', coef0: 0.0 },
      rbf: { C: 1.0, gamma: 1.0 },
      sigmoid: { C: 1.0, gamma: 1.0, coef0: 0.0 }
    };
    setSelectedParams(defaultParams[kernel]);
  };

  // 修改后的 handleParamChange：更新 selectedParams 状态
  const handleParamChange = (param, value) => {
    setSelectedParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const setKernel = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/set_kernel', {
        kernel: selectedKernel,
        params: selectedParams
      });
      setMessage(response.data.message);
      setVisualization(null);
    } catch (error) {
      setMessage(error.response?.data?.message || 'Error setting kernel');
    }
  };

  const startAnimation = () => {
    if (!visualization?.intermediate_states) return;

    // 清除之前的计时器
    if (animationTimer.current) {
      clearInterval(animationTimer.current);
    }

    setIsAnimating(true);
    setCurrentAnimationFrame(0);

    // 使用设置的间隔时间
    const timer = setInterval(() => {
      setCurrentAnimationFrame(prev => {
        if (prev >= (visualization.intermediate_states.length - 1)) {
          clearInterval(timer);
          setIsAnimating(false);
          return prev;
        }
        return prev + 1;
      });
    }, animationInterval * 1000); // 转换为毫秒

    animationTimer.current = timer;
  };

  const stopAnimation = () => {
    if (animationTimer.current) {
      clearInterval(animationTimer.current);
      animationTimer.current = null;
    }
    setIsAnimating(false);
    // 重置到初始状态
    setCurrentAnimationFrame(0);
  };

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      if (animationTimer.current) {
        clearInterval(animationTimer.current);
      }
    };
  }, []);

  const trainModel = async () => {
    if (data.length < 2) {
      showMessage('请至少添加两个数据点（正例和负例各一个）', 'error');
      return;
    }

      setIsTraining(true);
    showMessage('正在训练模型...', 'info');

    try {
      const requestData = {
        X: data.map(p => [p[0], p[1]]),
        y: labels,
        kernel: selectedKernel,
        C: selectedParams.C,
        degree: selectedParams.degree,
        gamma: selectedParams.gamma,
        coef0: selectedParams.coef0
      };

      const response = await fetch('http://127.0.0.1:5000/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        showMessage(result.message || '训练失败', 'error');
        return;
      }

      setVisualization({
        X: result.X,
        y: result.y,
        xx: result.xx,
        yy: result.yy,
        Z: result.Z,
        support_vectors: result.support_vectors,
        intermediate_states: result.intermediate_states || [],
        accuracy: result.accuracy
      });

      showMessage(`训练完成！准确率: ${(result.accuracy * 100).toFixed(2)}%`, 'success');

      if (result.intermediate_states && result.intermediate_states.length > 0) {
        setAnimationStates(result.intermediate_states);
        setCurrentAnimationFrame(0);
        startAnimation();
      }
    } catch (error) {
      console.error('训练错误:', error);
      showMessage(`训练失败: ${error.message}`, 'error');
    } finally {
      setIsTraining(false);
    }
  };

  const calculateRange = (points) => {
    if (points.length === 0) return [-10, 10];
    const allX = points.map(p => p[0]);
    const allY = points.map(p => p[1]);
    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);
    return [
      [minX - 1, maxX + 1],
      [minY - 1, maxY + 1]
    ];
  };

  const renderKernelControls = () => {
    return (
      <div className="sidebar">
        <div className="parameter-panel">
          <div className="classification-mode">
            <h3 className="classification-title">分类模式</h3>
            <div className="classification-buttons">
              <button 
                className={`mode-toggle-button ${classificationMode === 'binary' ? 'active' : ''}`}
                onClick={() => setClassificationMode('binary')}
              >
                二分类
              </button>
              <button 
                className={`mode-toggle-button ${classificationMode === 'multi' ? 'active' : ''}`}
                onClick={() => setClassificationMode('multi')}
              >
                多分类
              </button>
            </div>
          </div>
          <h3 className="panel-title">核函数与参数设置</h3>
          <div className="kernel-section">
            <div className="kernel-buttons">
              <button 
                className={`kernel-button ${selectedKernel === 'linear' ? 'active' : ''}`}
                onClick={() => handleKernelChange('linear')}
              >
                线性核
              </button>
              <button 
                className={`kernel-button ${selectedKernel === 'poly' ? 'active' : ''}`}
                onClick={() => handleKernelChange('poly')}
              >
                多项式核
              </button>
              <button 
                className={`kernel-button ${selectedKernel === 'rbf' ? 'active' : ''}`}
                onClick={() => handleKernelChange('rbf')}
              >
                RBF核
              </button>
              <button 
                className={`kernel-button ${selectedKernel === 'sigmoid' ? 'active' : ''}`}
                onClick={() => handleKernelChange('sigmoid')}
              >
                Sigmoid核
              </button>
            </div>
          </div>
    
          <div className="parameters-section">
            {/* 惩罚参数 C */}
            <div className="param-group">
              <label>惩罚参数 C:</label>
              <div className="param-input">
                <input 
                  type="range" 
                  min="0.01" 
                  max="100" 
                  step="0.01"
                  value={selectedParams.C} 
                  onChange={(e) => handleParamChange('C', parseFloat(e.target.value))}
                />
                <input 
                  type="number" 
                  min="0.1" 
                  max="100" 
                  step="0.1"
                  value={selectedParams.C} 
                  onChange={(e) => handleParamChange('C', parseFloat(e.target.value))}
                />
              </div>
            </div>
    
            {/* gamma 参数 */}
            {(selectedKernel === 'rbf' || selectedKernel === 'poly' || selectedKernel === 'sigmoid') && (
              <div className="param-group">
                <label>gamma:</label>
                <div className="param-input">
                  <select 
                    value={selectedParams.gamma} 
                    onChange={(e) => handleParamChange('gamma', e.target.value)}
                  >
                    <option value="scale">scale</option>
                    <option value="auto">auto</option>
                    <option value="manual">手动设置</option>
                  </select>
                </div>
                  {selectedParams.gamma === 'manual' && (
                  <div className="gamma-manual">
                    <div className="param-input">
                      <input 
                        type="range" 
                        min="0.01" 
                        max="10" 
                        step="0.01"
                        value={selectedParams.gamma_value} 
                        onChange={(e) => handleParamChange('gamma_value', parseFloat(e.target.value))}
                      />
                      <input 
                        type="number" 
                        min="0.01" 
                        max="10" 
                        step="0.01"
                        value={selectedParams.gamma_value} 
                        onChange={(e) => handleParamChange('gamma_value', parseFloat(e.target.value))}
                      />
                    </div>
                    </div>
                  )}
              </div>
            )}
    
            {/* degree 参数 - 只在 Poly 核函数时显示 */}
            {selectedKernel === 'poly' && (
                <div className="param-group">
                  <label>多项式次数:</label>
                  <div className="param-input">
                    <input 
                      type="range" 
                      min="2" 
                      max="10" 
                      step="1"
                      value={selectedParams.degree} 
                    onChange={(e) => handleParamChange('degree', parseInt(e.target.value, 10))}
                    />
                    <input 
                      type="number" 
                      min="2" 
                      max="10" 
                      step="1"
                      value={selectedParams.degree} 
                    onChange={(e) => handleParamChange('degree', parseInt(e.target.value, 10))}
                    />
                  </div>
                </div>
            )}
    
            {/* coef0 参数 - 只在 Poly 和 Sigmoid 核函数时显示 */}
            {(selectedKernel === 'poly' || selectedKernel === 'sigmoid') && (
                <div className="param-group">
                  <label>系数:</label>
                  <div className="param-input">
                    <input 
                      type="range" 
                      min="-5" 
                      max="5" 
                      step="0.1"
                      value={selectedParams.coef0} 
                      onChange={(e) => handleParamChange('coef0', parseFloat(e.target.value))}
                    />
                    <input 
                      type="number" 
                      min="-5" 
                      max="5" 
                      step="0.1"
                      value={selectedParams.coef0} 
                      onChange={(e) => handleParamChange('coef0', parseFloat(e.target.value))}
                    />
                  </div>
                </div>
            )}
          </div>
        </div>
      </div>
    );
  };
  
  const renderTeachingContent = () => {
    if (!showTeaching) return null;
  
    const steps = [
      "第1步：添加数据点\n- 点击图表空白处添加新的数据点\n- 使用正例/负例按钮切换点的类别\n- 点击已有数据点可以切换其类别\n- 支持导入CSV或TXT格式的数据文件",
      "第2步：选择核函数\n- 线性核：适合线性可分的数据\n- 多项式核：可以处理非线性数据，通过调整次数和系数\n- RBF核：最常用的核函数，适合处理复杂的非线性数据\n- Sigmoid核：类似神经网络中的激活函数",
      "第3步：调整参数\n- 惩罚参数C：控制分类错误的惩罚程度，值越大越严格\n- gamma：控制RBF核的影响范围，值越大决策边界越复杂\n- 多项式次数：控制多项式核的复杂度\n- 系数：调整多项式核的偏移量",
      "第4步：训练模型\n- 点击'训练模型'按钮开始训练\n- 观察训练过程中的决策边界变化\n- 支持向量会被高亮显示\n- 可以查看训练准确率",
      "第5步：分析结果\n- 观察决策边界的位置和形状\n- 查看支持向量的分布\n- 分析分类的准确率\n- 可以调整参数重新训练",
      "第6步：动画演示\n- 使用播放按钮查看训练过程\n- 观察决策边界如何逐步优化\n- 了解支持向量是如何被选择的\n- 直观理解 SVM 的工作原理",
      "第7步：参数优化\n- 尝试不同的核函数组合\n- 调整参数观察效果变化\n- 寻找最优的参数组合\n- 理解参数对模型的影响",
      "第8步：实践应用\n- 尝试不同的数据分布\n- 观察不同核函数的适用场景\n- 理解 SVM 的优缺点\n- 掌握实际应用中的调参技巧"
    ];
  
    return (
      <div className="teaching-panel">
        <h3>SVM 教学指南</h3>
        <div className="teaching-steps">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`teaching-step ${teachingStep === index ? 'active' : ''}`}
              onClick={() => setTeachingStep(index)}
            >
              {step.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
          ))}
        </div>
        <button onClick={() => setShowTeaching(false)}>关闭指南</button>
      </div>
    );
  };
  
  const renderPlot = () => {
    const plotData = [
      // 正例点
      {
        type: 'scatter',
        mode: 'markers',
        x: data.filter((_, i) => labels[i] === 1).map(point => point[0]),
        y: data.filter((_, i) => labels[i] === 1).map(point => point[1]),
        marker: {
          color: '#FF0000',
          size: 10,
          symbol: 'circle',
          line: { color: 'white', width: 1 }
        },
        name: '正例'
      },
      // 负例点
      {
        type: 'scatter',
        mode: 'markers',
        x: data.filter((_, i) => labels[i] === 0).map(point => point[0]),
        y: data.filter((_, i) => labels[i] === 0).map(point => point[1]),
        marker: {
          color: '#0000FF',
          size: 10,
          symbol: 'circle',
          line: { color: 'white', width: 1 }
        },
        name: '负例'
      }
    ];

    return (
      <div className="plot-container" style={{
        width: '800px',
        height: '600px',
        backgroundColor: 'white',
        borderRadius: '8px',
        padding: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '20px',
        overflow: 'hidden'
      }}>
        <Plot
          data={plotData}
          layout={{
            width: 800,
            height: 600,
            title: {
              text: '数据点展示',
              font: {
                size: 20,
                color: '#333',
                family: 'Arial, sans-serif'
              },
              y: 0.95
            },
            xaxis: {
              title: {
                text: '特征1',
                font: {
                  size: 16,
                  color: '#333'
                }
              },
              range: [-100, 100],
              gridcolor: '#eee',
              zerolinecolor: '#999',
              zerolinewidth: 1,
              showgrid: true,
              dtick: 20
            },
            yaxis: {
              title: {
                text: '特征2',
                font: {
                  size: 16,
                  color: '#333'
                }
              },
              range: [-100, 100],
              gridcolor: '#eee',
              zerolinecolor: '#999',
              zerolinewidth: 1,
              showgrid: true,
              dtick: 20,
              scaleanchor: 'x',
              scaleratio: 1
            },
            showlegend: true,
            legend: {
              x: 0.9,
              y: 0.95,
              bgcolor: 'rgba(255, 255, 255, 0.9)',
              bordercolor: '#ddd',
              borderwidth: 1,
              font: {
                size: 12,
                color: '#333'
              }
            },
            dragmode: 'pan',
            margin: {
              l: 50,
              r: 50,
              t: 50,
              b: 50
            },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            hovermode: 'closest'
          }}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            scrollZoom: true,
            responsive: true
          }}
          style={{
            width: '100%',
            height: '100%'
          }}
          onClick={handlePlotClick}
        />
      </div>
    );
  };

  const renderResultPlot = () => {
    if (!visualization) return null;

    const currentState = isAnimating && visualization.intermediate_states ? 
      visualization.intermediate_states[currentAnimationFrame] : 
      {
        Z: visualization.Z,
        xx: visualization.xx,
        yy: visualization.yy,
        support_vectors: visualization.support_vectors,
        description: '最终训练结果',
        accuracy: visualization.accuracy,
        n_support_vectors: visualization.support_vectors.length
      };

    const data = [
      // 正例点
      {
        x: visualization.X.filter((_, i) => visualization.y[i] === 1).map(p => p[0]),
        y: visualization.X.filter((_, i) => visualization.y[i] === 1).map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        name: '正例',
        marker: {
          color: '#ff4d4d',
          size: 10,
          line: { color: '#fff', width: 1 }
        }
      },
      // 负例点
      {
        x: visualization.X.filter((_, i) => visualization.y[i] === 0).map(p => p[0]),
        y: visualization.X.filter((_, i) => visualization.y[i] === 0).map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        name: '负例',
        marker: {
          color: '#4d4dff',
          size: 10,
          line: { color: '#fff', width: 1 }
        }
      }
    ];

    // 添加决策边界
    if (currentState.Z && currentState.xx && currentState.yy) {
      data.push({
        x: currentState.xx[0],
        y: currentState.yy.map(row => row[0]),
        z: currentState.Z,
        type: 'contour',
        colorscale: [
          [0, 'rgba(77, 77, 255, 0.2)'],
          [0.5, 'rgba(255, 255, 255, 0)'],
          [1, 'rgba(255, 77, 77, 0.2)']
        ],
        contours: {
          start: -1,
          end: 1,
          size: 0.5,
          showlabels: true,
          labelfont: {
            size: 12,
            color: '#666'
          }
        },
        line: {
          smoothing: 1.3,
          width: 2,
          color: '#666'
        },
        showscale: false,
        name: '决策边界'
      });
    }

    // 添加支持向量标记（调整大小和样式）
    if (currentState.support_vectors) {
      data.push({
        x: currentState.support_vectors.map(p => p[0]),
        y: currentState.support_vectors.map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        name: '支持向量',
        marker: {
          symbol: 'circle-open',
          color: '#000',
          size: 20, // 增大尺寸
          line: {
            width: 2.5,  // 增加线宽
            color: '#000'
          }
        }
      });
    }

    return (
      <div className="result-container" style={{
        display: 'flex',
        gap: '20px',
        alignItems: 'flex-start',
        width: '100%',
        maxWidth: '1400px',
        margin: '0 auto',
        position: 'relative'
      }}>
        <div className="plot-wrapper" style={{
          flex: '0 0 800px',
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '15px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          marginBottom: '20px',
          position: 'relative'
        }}>
          <Plot
            data={data}
            layout={{
              width: 800,
              height: 600,
              title: {
                text: isAnimating ?
                  `训练过程 - 第${currentAnimationFrame + 1}步 / ${visualization.intermediate_states.length}步` :
                  '训练结果',
                font: {
                  size: 20,
                  color: '#333',
                  family: 'Arial, sans-serif'
                },
                y: 0.95
              },
              showlegend: true,
              legend: {
                x: 0.85,
                y: 0.95,
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                bordercolor: '#ddd',
                borderwidth: 1,
                font: {
                  size: 12,
                  color: '#333'
                }
              },
              margin: {
                l: 60,
                r: 30,
                t: 60,
                b: 60
              },
              xaxis: {
                title: {
                  text: '特征1',
                  font: {
                    size: 16,
                    color: '#333'
                  }
                },
                gridcolor: '#eee',
                zerolinecolor: '#999'
              },
              yaxis: {
                title: {
                  text: '特征2',
                  font: {
                    size: 16,
                    color: '#333'
                  }
                },
                gridcolor: '#eee',
                zerolinecolor: '#999'
              },
              paper_bgcolor: 'white',
              plot_bgcolor: 'white'
            }}
            config={{
              displayModeBar: true,
              scrollZoom: true,
              displaylogo: false
            }}
            style={{ width: '100%', height: '600px' }}
                />
              </div>
      </div>
    );
  };

  // 修改模式选择菜单组件
  const ModeSelector = () => {
    return (
      <div className="mode-selector">
        <button 
          className={`mode-button ${currentMode === 'basic' ? 'active' : ''}`}
          onClick={() => setCurrentMode('basic')}
        >
          <i className="fa fa-book"></i>
          基础知识
        </button>
        <button 
          className={`mode-button ${currentMode === 'dynamic' ? 'active' : ''}`}
          onClick={() => setCurrentMode('dynamic')}
        >
          <i className="fa fa-play"></i>
          动态演示
        </button>
        <button 
          className={`mode-button ${currentMode === 'advanced' ? 'active' : ''}`}
          onClick={() => setCurrentMode('advanced')}
        >
          <i className="fa fa-cogs"></i>
          拓展内容
        </button>
        <button 
          className={`mode-button ${currentMode === 'help' ? 'active' : ''}`}
          onClick={() => setCurrentMode('help')}
        >
          <i className="fa fa-question-circle"></i>
          帮助中心
        </button>
      </div>
    );
  };

  // 渲染基础知识内容
  const renderBasicContent = () => {
    switch (activeBasicSection) {
      case 'introduction':
        return (
          <div className="basic-section-content">
            <h3>什么是支持向量机 (SVM) ？</h3>
            <p style={{ textAlign: 'left' }}>支持向量机（Support Vector Machine，简称 SVM）是一种二分类模型，其基本思想是找到不同类别数据之间的最大间隔超平面，实现对新样本的分类。SVM 通过引入核技巧，将数据映射到更高维空间，以便于在高维空间中寻找到最优超平面。</p>
            <p style={{ textAlign: 'left' }}>SVM可以分为以下几种类型：</p>
            <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
              <li>当训练样本线性可分时，通过硬间隔最大化，学习一个<strong>线性可分支持向量机；</strong></li>
              <li>当训练样本近似线性可分时，通过软间隔最大化，学习一个<strong>线性支持向量机；</strong></li>
              <li>当训练样本线性不可分时，通过核技巧和软间隔最大化，学习一个<strong>非线性支持向量机。</strong></li>
            </ul>
            <p style={{ textAlign: 'left' }}>SVM 的主要优势：</p>
            <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
              <li>在高维空间中非常有效；</li>
              <li>对于非线性分类问题，通过核函数可以很好地处理；</li>
              <li>泛化能力强，不容易过拟合。</li>
            </ul>
          </div>
        );
      case 'core_concepts':
        return (
          <div className="basic-section-content">
            <h3 >核心概念</h3>
            <div className="concept-item">
              <h4 style={{ textAlign: 'left' }}>支持向量</h4>
              <p style={{ textAlign: 'left' }}>距离决策边界最近的数据点，它们决定了决策边界的位置。</p>
              <BlockMath math="\text{支持向量} = \{x_i | y_i(w^Tx_i + b) = 1\}" />
            </div>
            <ImageComponent 
                src="/images/support_machine.png" 
                alt="支持向量示意图" 
                className="concept-image"
                style={{ maxWidth: '400px', width: '80%', height: 'auto' }}
              />
            <div className="concept-item">
              <h4 style={{ textAlign: 'left' }}>决策边界</h4>
              <p style={{ textAlign: 'left' }}>将不同类别分开的超平面，由支持向量决定。</p>
              <BlockMath math="w^Tx + b = 0" />
            </div>
            <div className="concept-item">
              <h4 style={{ textAlign: 'left' }}>最大间隔优化问题</h4>
              <p style={{ textAlign: 'left' }}>
                寻找参数 <InlineMath math={`\\boldsymbol{w}`} /> 和 <InlineMath math={`b`} />，使得间隔最大化。
              </p>
              <BlockMath math={`
                \\begin{aligned}
                \\arg\\max_{\\boldsymbol{w}, b} \\quad & \\frac{2}{\\|\\boldsymbol{w}\\|} \\\\
                \\text{s.t.} \\quad & y_i (\\boldsymbol{w}^\\top \\boldsymbol{x}_i + b) \\geq 1, \\quad i = 1, 2, \\dots, m.
                \\end{aligned}
              `} />

              <p>优化问题转换：</p>
              <p style={{ textAlign: 'left' }}>
                为了方便计算，将最大化问题转换为最小化问题。
              </p>
              <BlockMath math={`
                \\begin{aligned}
                \\arg\\min_{\\boldsymbol{w}, b} \\quad & \\frac{1}{2} \\|\\boldsymbol{w}\\|^2 \\\\
                \\text{s.t.} \\quad & y_i (\\boldsymbol{w}^\\top \\boldsymbol{x}_i + b) \\geq 1, \\quad i = 1, 2, \\dots, m.
                \\end{aligned}
              `} />
            </div>
          </div>
        );
      case 'kernel_functions':
        return (
          <div className="basic-section-content">
            <h3>核函数详解</h3>
            <p> </p>
            <h4 style={{ textAlign: 'left' }}>核支持向量机</h4>
            <div style={{ textAlign: 'left', marginBottom: '2rem' }}>
              <p>
                设样本 <InlineMath math={`x`} /> 映射后的向量为 <InlineMath math={`\\phi(x)`} />，
                划分超平面为 <InlineMath math={`f(x) = \\boldsymbol{w}^\\top \\phi(x) + b`} />。
              </p>

              <p>原始问题：</p>
              <BlockMath math={`
                \\begin{aligned}
                \\min_{\\boldsymbol{w}, b} \\quad & \\frac{1}{2} \\|\\boldsymbol{w}\\|^2 \\\\
                \\text{s.t.} \\quad & y_i(\\boldsymbol{w}^\\top \\phi(x_i) + b) \\geq 1, \\quad i = 1, 2, \\dots, m
                \\end{aligned}
              `} />

              <p>对偶问题：</p>
              <BlockMath math={`
                \\begin{aligned}
                \\min_{\\boldsymbol{\\alpha}} \\quad & \\frac{1}{2} \\sum_{i=1}^{m} \\sum_{j=1}^{m} \\alpha_i \\alpha_j y_i y_j \\phi(x_i)^\\top \\phi(x_j) - \\sum_{i=1}^{m} \\alpha_i \\\\
                \\text{s.t.} \\quad & \\sum_{i=1}^{m} \\alpha_i y_i = 0
                \\end{aligned}
              `} />

              <p style={{ color: '#c76f00', fontWeight: 'bold' }}>（只以内积的形式出现）</p>

              <p>预测：</p>
              <BlockMath math={`
                \\begin{aligned}
                f(x) &= \\boldsymbol{w}^\\top \\phi(x) + b \\\\
                    &= \\sum_{i=1}^{m} \\alpha_i y_i \\phi(x_i)^\\top \\phi(x) + b
                \\end{aligned}
              `} />
              </div>
              <h4 style={{ textAlign: 'left' }}>核函数分类</h4>
              <table className="kernel-table">
                <thead>
                  <tr>
                    <th>输入</th>
                    <th>含义</th>
                    <th>解决问题</th>
                    <th>核函数表达式</th>
                    <th style={{ whiteSpace: 'nowrap' }}>参数<BlockMath math="\gamma"/></th>
                    <th>参数degree</th>
                    <th>参数coef0</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Linear</td>
                    <td>线性核</td>
                    <td>线性</td>
                    <td><BlockMath math="K(x,y) = x^Ty = x \cdot y" /></td>
                    <td>无</td>
                    <td>无</td>
                    <td>无</td>
                  </tr>
                  <tr>
                    <td>Poly</td>
                    <td>多项式核</td>
                    <td>偏线性</td>
                    <td><BlockMath math="K(x,y) = (\gamma(x \cdot y) + r)^d" /></td>
                    <td>有</td>
                    <td>有</td>
                    <td>有</td>
                  </tr>
                  <tr>
                    <td>Sigmoid</td>
                    <td>双曲正切核</td>
                    <td>非线性</td>
                    <td><BlockMath math="K(x,y) = \tanh(\gamma(x \cdot y) + r)" /></td>
                    <td>有</td>
                    <td>无</td>
                    <td>有</td>
                  </tr>
                  <tr>
                    <td>RBF</td>
                    <td>高斯径向基</td>
                    <td>偏非线性</td>
                    <td><BlockMath math="K(x,y) = e^{-\gamma\|x-y\|^2}, \gamma > 0" /></td>
                    <td>有</td>
                    <td>无</td>
                    <td>无</td>
                  </tr>
                </tbody>
              </table>

            <div className="kernel-explanation">
              <h4 style={{ textAlign: 'left' }}>核函数的作用</h4>
              <p style={{ textAlign: 'left' }}>核函数允许 SVM 在高维特征空间中工作，而无需显式计算该空间中的坐标。这被称为"核技巧"。</p>
              <BlockMath math="K(x,y) = \phi(x)^T\phi(y)" />
              <p style={{ textAlign: 'left' }}>其中 <InlineMath math="\phi"/> 是将输入映射到高维特征空间的函数。</p>
            </div>
          </div>
        );
        case 'parameters':
          return (
            <div className="basic-section-content">
              <h3>参数说明</h3>
        
              <div className="parameter-item">
                <h4>惩罚参数 C</h4>
                <p>控制模型对误分类点的惩罚程度。</p>
                <p>在软间隔 SVM 中，C 值越大，表示对分类错误的惩罚越严格，间隔越小；C 值越小，允许更多的分类错误，间隔越大。</p>
                <BlockMath math="\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i" />
                <p style={{ marginBottom: '20px' }}>
                  其中 <InlineMath math="\xi_i"/> 是松弛变量，允许一些点被错误分类。
                </p>
                <p>SVM 其实是一个自带 L2 正则项的分类器。SVM 防止过拟合的主要技巧就在于调整软间隔松弛变量的惩罚因子 C。C 越大表明越不能容忍错分，当无穷大时则退化为硬间隔分类器。合适的 C 大小可以照顾到整体数据而不是被一个 Outlier 带偏整个判决平面。</p>
                <p>至于 C 的具体调参，通常可以采用交叉验证来获得，每个松弛变量对应的惩罚因子可以不一样。</p>
                <p>一般情况下：</p>
                <ul>
                  <li>低偏差，高方差，即遇到过拟合时，减小C</li>
                  <li>高偏差，低方差，即遇到欠拟合时，增大C</li>
                </ul>
              </div>
        
              <div className="parameter-item">
                <h4 style={{ marginTop: '30px', marginBottom: '15px' }}>
                  <InlineMath math="\gamma"/> 参数
                </h4>
                <p style={{ marginTop: '20px' }}>
                  控制 RBF 核函数的影响范围。<InlineMath math="\gamma"/> 值越大，决策边界越复杂，可能导致过拟合。
                </p>
                <BlockMath math="K(x,y) = e^{-\gamma\|x-y\|^2}" />
              </div>
        
              <div className="parameter-item">
                <h4>Degree 参数</h4>
                <p>多项式核的度数，控制多项式的复杂度。度数越高，决策边界越复杂。</p>
                <BlockMath math="K(x,y) = (x \cdot y)^d" />
              </div>
        
              <div className="parameter-item">
                <h4>Coef0 参数</h4>
                <p>多项式核和 Sigmoid 核的常数项，影响决策边界的偏移。</p>
                <BlockMath math="K(x,y) = (\gamma(x \cdot y) + r)^d" />
              </div>
            </div>
          );
      case 'applications':
        return (
          <div className="basic-section-content">
            <h3>应用场景</h3>
            <div className="application-list">
              <div className="application-item">
                <h4>文本分类</h4>
                <p>SVM在文本分类领域有着明显的优势，尤其适合处理文本中常见的高维、稀疏的特征数据。因为文本本身的特点，大部分词语只在少数文档中出现，导致特征矩阵很多都是零，这种情况下 SVM 的效果通常比其他算法要更好。</p>
              </div>
              <div className="application-item">
                <h4>图像识别</h4>
                <p>
                  SVM 能够有效处理图像分类任务。常见的应用包括人脸识别、手写字符识别，以及物体检测等。
                </p>
              </div>

              <div className="application-item">
                <h4>生物信息学</h4>
                <p>
                  SVM 在生物信息学领域表现突出，尤其擅长解决复杂的生物数据分类问题。常用场景包括蛋白质结构和功能的分类分析，帮助科研人员理解蛋白质的特性；基因表达分析，利用基因组数据预测疾病的可能性；以及药物发现，通过识别化合物特征，加快新药的研发过程。
                </p>
              </div>

              <div className="application-item">
                <h4>金融预测</h4>
                <p>
                  在股票价格预测中，SVM 能通过历史数据准确判断趋势变化；在风险评估方面，能够帮助银行和金融机构提前识别潜在风险；在反欺诈系统中，SVM 还可以快速检测异常交易，保护用户资产安全。
                </p>
              </div>
            </div>
          </div>
        );
      case 'advantage':
        return (
          <div className="basic-section-content">
            <h3>支持向量机的优缺点分析</h3>
            
            <div className="concept-item">
              <h4>优点</h4>
              <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
                <li>在高维空间中非常有效</li>
                <li>在数据维度比样本数量大的情况下仍然有效</li>
                <li>使用核函数可以灵活地处理非线性分类问题</li>
                <li>泛化能力强，不容易过拟合</li>
                <li>理论基础扎实，基于统计学习理论</li>
                <li>可以处理小样本学习问题</li>
              </ul>
            </div>

            <div className="concept-item">
              <h4>缺点</h4>
              <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
                <li>对参数和核函数的选择比较敏感</li>
                <li>计算复杂度较高，特别是对大规模数据集</li>
                <li>对缺失数据敏感</li>
                <li>不直接提供概率估计</li>
                <li>内存消耗大，特别是使用非线性核时</li>
              </ul>
            </div>
          </div>
        );
      case 'implementation':
        return (
          <div className="basic-section-content">
            <h3>SVM 实现细节</h3>
            
            <div className="concept-item">
              <h4>优化算法</h4>
              <p>SVM 的训练过程主要涉及以下优化算法：</p>
              <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
                <li>序列最小优化算法（SMO）</li>
                <li>坐标下降法</li>
                <li>随机梯度下降</li>
                <li>内点法</li>
              </ul>
            </div>

            <div className="concept-item">
              <h4>计算复杂度</h4>
              <p>对于n个训练样本：</p>
              <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
                <li>线性 SVM：O(n)到O(n²)</li>
                <li>非线性 SVM：O(n²)到O(n³)</li>
                <li>内存复杂度：O(n²)</li>
              </ul>
            </div>

            <div className="concept-item">
              <h4>实现技巧</h4>
              <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
                <li>数据预处理和标准化</li>
                <li>核矩阵的缓存策略</li>
                <li>并行计算优化</li>
                <li>增量学习支持</li>
                <li>模型压缩和简化</li>
              </ul>
            </div>

            <div className="concept-item">
              <h4>常用工具库</h4>
              <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
                <li>LIBSVM：C++实现的经典 SVM 库</li>
                <li>scikit-learn：Python 机器学习库中的 SVM 实现</li>
                <li>liblinear：大规模线性分类库</li>
                <li>SVMLight：高效的 SVM 实现</li>
              </ul>
            </div>
          </div>
        );
        case 'formula':
          return (
            <div className="basic-section-content">
              <h3>具体公式推导</h3>         
              <div className="concept-item">
                <h4 style={{ textAlign: 'left' }}>硬间隔</h4>
                <p style={{ textAlign: 'left' }}>每个支持向量到超平面的距离可以写为：</p>
                <BlockMath math={`d = \\frac{|w^T x + b|}{\\|w\\|}`} />

                <p style={{ textAlign: 'left' }}>由上述</p>
                <BlockMath math={`y(w^T x + b) > 1 > 0`} />
                <p style={{ textAlign: 'left' }}>可以得到</p>
                <BlockMath math={`y(w^T x + b) = |w^T x + b|`} />
                <p style={{ textAlign: 'left' }}>所以我们得到：</p>
                <BlockMath math={`d = \\frac{y(w^T x + b)}{\\|w\\|}`} />

                <p style={{ textAlign: 'left' }}>最大化这个距离：</p>
                <BlockMath math={`\\max \\, 2 \\cdot \\frac{y(w^T x + b)}{\\|w\\|}`} />

                <p style={{ textAlign: 'left' }}>
                  这里乘上2倍也是为了后面推导，对目标函数没有影响。
                </p>
                <p style={{ textAlign: 'left' }}>刚刚我们得到支持向量：</p>
                <BlockMath math={`y(w^T x + b) = 1`} />
                <p style={{ textAlign: 'left' }}>所以我们得到：</p>
                <BlockMath math={`\\max \\, \\frac{2}{\\|w\\|}`} />

                <p style={{ textAlign: 'left' }}>再做一个转换：</p>
                <BlockMath math={`\\min \\frac{1}{2} \\|w\\|`} />

                <p style={{ textAlign: 'left' }}>
                  为了方便计算（去除根号），我们有：
                </p>
                <BlockMath math={`\\min \\frac{1}{2} \\|w\\|^2`} />

                <p style={{ textAlign: 'left' }}>所以最终的最优化问题是：</p>
                <BlockMath math={`\\min \\frac{1}{2} \\|w\\|^2 \\quad \\text{s.t.} \\quad y_i(w^T x_i + b) \\geq 1`} />
              </div>

              <div className="concept-item">
                <h4>软间隔</h4>
                <p>
                  在硬间隔 SVM 中，虚线内侧不应有任何样本点；
                  而在软间隔 SVM 中，由于数据不是完全线性可分，虚线内侧可能存在样本点。
                  通过为每个位于虚线内侧的样本点引入松弛变量 <InlineMath math="\xi_i" />，可以将这些样本点"移动"到支持向量所在的虚线上。
                  对于本身就在虚线外的样本点，其松弛变量可设为 0。
                  因此，我们为每个松弛变量赋予一个代价 <InlineMath math="\xi_i" />，目标函数变为：
                </p>
                <BlockMath math="\displaystyle f(w, \xi) = \frac{1}{2} \| w \|^2 + C \sum_{i=1}^N \xi_i" />
                <p>
                  其中，<InlineMath math="C > 0" /> 称为<strong>惩罚参数</strong>。
                  当 <InlineMath math="C" /> 值较大时，对误分类的惩罚增大；
                  当 <InlineMath math="C" /> 值较小时，对误分类的惩罚减小。
                  该目标函数有两层含义：
                  一是使 <InlineMath math="\frac{1}{2} \| w \|^2" /> 尽量小，即间隔尽可能大；
                  二是使误分类的数量尽量小。
                  <InlineMath math="C" /> 是调和两者的系数，是一个超参数。
                </p>
                <p>因此，软间隔 SVM 的优化问题可以描述为：</p>
                <BlockMath math="\displaystyle \min_{w, b} \frac{1}{2} \| w \|^2 + C \sum_{i=1}^N \xi_i" />
                <p>满足以下约束条件：</p>
                <BlockMath math="\displaystyle \begin{aligned}
                  & y_i (w^T x_i + b) \geq 1 - \xi_i \\
                  & \xi_i \geq 0 \\
                  & i = 1, 2, \dots, N
                \end{aligned}" />
                <p>将其表述为标准形式：</p>
                <BlockMath math="\displaystyle \min_{w, b} \frac{1}{2} \| w \|^2 + C \sum_{i=1}^N \xi_i" />
                <BlockMath math={String.raw`
                \begin{align*}
                \text{s.t.} \quad & y_i (w^T x_i + b) \geq 1 - \xi_i \\
                & \xi_i \geq 0 \\
                & i = 1, 2, \dots, N
                \end{align*}
                `} />
                
              </div>
              <ImageComponent 
                  src="/images/soft.png" 
                  alt="软间隔"
                  className="concept-image"
                  style={{ maxWidth: '400px', width: '80%', height: 'auto' }}
                />
                <p style={{ textAlign: 'center', color: '#666', fontSize: '14px', marginTop: '10px' }}>
                   软间隔示意图. 红色圈出了一些不满足约束的样本.
                 </p>
            </div>
          );
      default:
        return null;
      
    }
  };

  // 添加帮助页面内容
  const renderHelpContent = () => {
    switch (activeHelpSection) {
      case 'introduction':
        return (
          <div className="advanced-section-content">
            <h2>项目介绍</h2>

              
              <p style={{ textAlign: 'left' }}>本平台是一个基于 Web 的支持向量机（SVM）可视化教学系统，旨在通过直观的交互式演示，帮助用户深入理解 SVM 的核心概念、算法原理和应用场景。</p>
            
            <div className="concept-item">
              <h4>平台特点</h4>
              <ul>
                <li>交互式数据可视化：支持实时添加数据点，直观展示分类过程</li>
                <li>多种核函数支持：包括线性核、多项式核、RBF 核和 Sigmoid 核</li>
                <li>参数动态调整：可实时调整 SVM 参数，观察决策边界变化</li>
                <li>3D可视化：支持高维数据的 3D 展示，帮助理解核函数的作用</li>
                <li>教学资源丰富：包含基础知识、动态演示和拓展内容三个模块</li>
              </ul>
            </div>
          </div>
        );
      case 'guide':
        return (
          <div className="advanced-section-content">
            <h2>使用指南</h2>
            <div className="concept-item">
              <h4>基础知识模块</h4>
              <ul>
                <li>了解 SVM 的基本概念和原理</li>
                <li>学习核心概念和参数说明</li>
                <li>查看应用场景和优缺点分析</li>
                <li>了解实现细节和常用工具</li>
              </ul>
            </div>
            
            <div className="concept-item">
              <h4>动态演示模块</h4>
              <ul>
                <li>添加训练数据点（点击图表或上传文件）</li>
                <li>选择不同的核函数和参数</li>
                <li>观察训练过程和决策边界</li>
                <li>查看支持向量和分类结果</li>
              </ul>
            </div>
            
            <div className="concept-item">
              <h4>拓展内容模块</h4>
              <ul>
                <li>了解随机森林算法</li>
                <li>学习平分最近点法</li>
                <li>探索多分类问题</li>
                <li>理解软间隔与硬间隔</li>
                <li>学习序列最小优化算法（SMO）</li>
              </ul>
            </div>
          </div>
        );
      case 'faq':
        return (
          <div className="advanced-section-content">
            <h2>常见问题</h2>
            <div className="concept-item">
              <h4>如何添加数据点？</h4>
              <p>您可以通过以下方式添加数据点：</p>
              <ul>
                <li>点击图表空白处添加新点</li>
                <li>使用正例/负例按钮切换点的类别</li>
                <li>点击已有数据点可以切换其类别</li>
                <li>支持导入CSV或TXT格式的数据文件</li>
              </ul>
            </div>
            
            <div className="concept-item">
              <h4>如何选择合适的核函数？</h4>
              <p>核函数的选择取决于数据特点：</p>
              <ul>
                <li>线性核：适合线性可分的数据</li>
                <li>多项式核：可以处理非线性数据，通过调整次数和系数</li>
                <li>RBF核：最常用的核函数，适合处理复杂的非线性数据</li>
                <li>Sigmoid核：类似神经网络中的激活函数</li>
              </ul>
            </div>
            
            <div className="concept-item">
              <h4>如何调整参数？</h4>
              <p>主要参数说明：</p>
              <ul>
                <li>惩罚参数C：控制分类错误的惩罚程度，值越大越严格</li>
                <li>gamma：控制RBF核的影响范围，值越大决策边界越复杂</li>
                <li>多项式次数：控制多项式核的复杂度</li>
                <li>Coef0：调整多项式核的偏移量</li>
              </ul>
            </div>
          </div>
        );
      case 'support':
        return (
          <div className="advanced-section-content">
            <h2>技术支持</h2>
            <div className="concept-item">
              <h4>获取支持</h4>
              <p>如果您在使用过程中遇到任何问题，可以通过以下方式获取支持：</p>
              <ul>
                <li>访问 GitHub 项目主页</li>
                <li>提交 Issue 反馈问题</li>
                <li>联系技术支持团队</li>
              </ul>
            </div>
          </div>
        );
      default:
        return null;
    }
  };
  // 修改渲染内容函数
  const renderContent = () => {
    if (currentMode === 'welcome') {
      return <WelcomePage setCurrentMode={setCurrentMode} />;
    }
    switch (currentMode) {
      case 'basic':
        return (
          <div className="basic-mode">
            <div className="basic-sidebar">
              <h3>基础知识</h3>
              <ul className="basic-nav">
                {basicSections.map(section => (
                  <li 
                    key={section.id}
                    className={activeBasicSection === section.id ? 'active' : ''}
                    onClick={() => setActiveBasicSection(section.id)}
                  >
                    {section.title}
                  </li>
                ))}
              </ul>
            </div>
            <div className="basic-content" style={{ 
              overflow: 'hidden',
              height: 'auto',
              minHeight: 'calc(100vh - 200px)',
              display: 'flex',
              flexDirection: 'column',
              padding: '20px'
            }}>
              {renderBasicContent()}
            </div>
          </div>
        );
      case 'dynamic':
        return (
          <div className="basic-mode">
            <div className="basic-sidebar">
              <h3>动态演示</h3>
              <ul className="basic-nav">
                <li className={activeDynamicSection === 'demo' ? 'active' : ''} onClick={() => setActiveDynamicSection('demo')}>二分类演示</li>
                <li className={activeDynamicSection === 'steps' ? 'active' : ''} onClick={() => setActiveDynamicSection('steps')}>多分类演示</li>
              </ul>
            </div>
            <div className="basic-content" style={{ 
              overflow: 'hidden',
              height: 'auto',
              minHeight: 'calc(100vh - 200px)',
              display: 'flex',
              flexDirection: 'column',
              padding: '20px',
              overflowY: 'hidden'
            }}>
            <div className="basic-section-content">
              {activeDynamicSection === 'demo' && (
                <>
                  <h3>二分类演示</h3>
                  <p>点击下方按钮，开始体验 SVM 二分类问题的动态演示过程。</p>
                  <div style={{ textAlign: 'center', marginTop: '20px' }}>
                    <button 
                      onClick={() => window.open('http://localhost:8501', '_blank')}
                      style={{
                        padding: '10px 20px',
                        backgroundColor: '#4a90e2',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: 'pointer',
                        fontSize: '16px',
                        transition: 'background-color 0.3s'
                      }}
                    >
                      开始演示
                    </button>
                  </div>
                  <div className="concept-item">
                    <h4>基本概念</h4>
                    <p>二分类问题（Binary Classification）指模型需从两个互斥的类别中，为输入样本选择一个最可能的类别标签。</p>
                    <p>具体来说：</p>
                    <ul>
                      <li>输入：特征向量 <InlineMath math="x \in \mathbb{R}^d"/></li>
                      <li>输出：目标 <InlineMath math="y \in \{0, 1\}"/></li>
                      <li>模型预测：预测值为类别 1 的概率 <InlineMath math="P(y=1|x) = \hat{y}"/></li>
                    </ul>
                  </div>
                  <div className="concept-item">
                    <h4>常用方法</h4>
                    <ul>
                      <li>逻辑回归</li>
                      <li>支持向量机（SVM）</li>
                      <li>决策树</li>
                      <li>朴素贝叶斯</li>
                      <li>神经网络</li>
                    </ul>
                  </div>
                  <div className="concept-item">
                    <h4>处理思路</h4>
                    <p>对于二分类问题，SVM 通过以下方式处理：</p>
                    <ul>
                      <li>寻找最优超平面，最大化不同类别之间的间隔</li>
                      <li>使用核函数处理非线性可分的数据</li>
                      <li>通过支持向量确定决策边界</li>
                    </ul>
                  </div>
                </>
              )}
              {activeDynamicSection === 'steps' && (
                <>
                  <h3>多分类演示</h3>
                  <p>点击下方按钮，开始体验 SVM 多分类问题的动态演示过程。</p>
                  <div style={{ textAlign: 'center', marginTop: '20px' }}>
                    <button 
                      onClick={() => window.open('http://localhost:8502', '_blank')}
                      style={{
                        padding: '10px 20px',
                        backgroundColor: '#4a90e2',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: 'pointer',
                        fontSize: '16px',
                        transition: 'background-color 0.3s'
                      }}
                    >
                      开始演示
                    </button>
                  </div>
                  <div className="concept-item">
                    <h4>基本概念</h4>
                    <p>多分类问题（Multi-class Classification）指模型需从多个互斥的类别中，为输入样本选择一个最可能的类别标签。</p>
                    <p>具体来说：</p>
                    <ul>
                      <li>输入：特征向量 <InlineMath math="x \in \mathbb{R}^d"/></li>
                      <li>输出：目标 <InlineMath math="y \in \{1, 2, \dots, K\}"/></li>
                      <li>模型预测：预测每个类别的概率 <InlineMath math="P(y=k|x)"/>，所有类别概率之和为 1</li>
                    </ul>
                  </div>
                  <div className="concept-item">
                    <h4>常用方法</h4>
                    <ul>
                      <li>一对多（One-vs-Rest）</li>
                      <li>一对一（One-vs-One）</li>
                      <li>多分类 SVM</li>
                      <li>决策树和随机森林</li>
                      <li>神经网络</li>
                    </ul>
                  </div>
                  <div className="concept-item">
                    <h4>处理思路</h4>
                    <p>对于多分类问题，SVM 可以通过以下两种方式处理：</p>
                    <ul>
                      <li>
                        <strong>一对多（One-vs-Rest）</strong>：
                        <ul>
                          <li>为每个类别训练一个二分类器，将目标类别与其他所有类别区分开</li>
                          <li>训练时，将某个类别的样本标记为正类，其他所有类别标记为负类</li>
                          <li>测试时，选择分类函数值最大的类别作为最终结果</li>
                          <li>优点：只需要训练 <InlineMath math="k"/> 个分类器，分类速度较快</li>
                          <li>缺点：训练样本不平衡，负类样本远多于正类样本</li>
                        </ul>
                      </li>
                      <li>
                        <strong>一对一（One-vs-One）</strong>：
                        <ul>
                          <li>在任意两类样本之间设计一个 SVM，共需要 <InlineMath math="\frac{k(k-1)}{2}"/> 次</li>
                          <li>训练时，每次只使用两个类别的样本进行训练</li>
                          <li>测试时，采用投票机制，得票最多的类别即为最终结果</li>
                          <li>优点：每个分类器只使用两个类别的样本，训练样本更平衡</li>
                          <li>缺点：当类别很多时，需要训练的分类器数量会急剧增加</li>
                        </ul>
                      </li>
                    </ul>
                  </div>
                </>
              )}
            </div>
            </div>
          </div>
        );
      case 'advanced':
        return (
          <div className="basic-mode">
            <div className="basic-sidebar">
              <h3>拓展内容</h3>
              <ul className="basic-nav">
                <li 
                  className={activeAdvancedSection === 'random-forest' ? 'active' : ''}
                  onClick={() => setActiveAdvancedSection('random-forest')}
                >
                  随机森林
                </li>
                <li 
                  className={activeAdvancedSection === 'bisecting-nearest-point' ? 'active' : ''}
                  onClick={() => setActiveAdvancedSection('bisecting-nearest-point')}
                >
                  平分最近点法
                </li>
                <li 
                  className={activeAdvancedSection === 'soft-hard-margin' ? 'active' : ''}
                  onClick={() => setActiveAdvancedSection('soft-hard-margin')}
                >
                  软间隔与硬间隔
                </li>
                <li 
                  className={activeAdvancedSection === 'smo' ? 'active' : ''}
                  onClick={() => setActiveAdvancedSection('smo')}
                >
                  序列最小优化算法（SMO）
                </li>
              </ul>
            </div>
            <div className="basic-content" style={{ 
              overflow: 'hidden',
              height: 'auto',
              minHeight: 'calc(100vh - 200px)',
              display: 'flex',
              flexDirection: 'column',
              padding: '20px'
            }}>
              {renderAdvancedContent()}
            </div>
          </div>
        );
      case 'help':
        return (
          <div className="basic-mode">
            <div className="basic-sidebar">
              <h3>帮助中心</h3>
              <ul className="basic-nav">
                <li className={activeHelpSection === 'introduction' ? 'active' : ''} onClick={() => setActiveHelpSection('introduction')}>项目介绍</li>
                <li className={activeHelpSection === 'guide' ? 'active' : ''} onClick={() => setActiveHelpSection('guide')}>使用指南</li>
                <li className={activeHelpSection === 'faq' ? 'active' : ''} onClick={() => setActiveHelpSection('faq')}>常见问题</li>
                <li className={activeHelpSection === 'support' ? 'active' : ''} onClick={() => setActiveHelpSection('support')}>技术支持</li>
              </ul>
            </div>
            <div className="basic-content" style={{ 
              overflow: 'hidden',
              height: 'auto',
              minHeight: 'calc(100vh - 200px)',
              display: 'flex',
              flexDirection: 'column',
              padding: '20px',
              overflowY: 'hidden'
            }}>
              {renderHelpContent()}
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // 添加清除所有点的函数
  const clearAllPoints = () => {
    setData([]);
    setLabels([]);
    setVisualization({
      X: [],
      y: [],
      xx: [],
      yy: [],
      Z: [],
      support_vectors: [],
      intermediate_states: [],
      accuracy: null
    });
    setMessage('');
  };

  // 添加手动加点处理函数
  const handleManualAdd = () => {
    setShowManualAddModal(true);
  };

  const handleManualAddSubmit = () => {
    try {
      // 解析输入文本，支持多行
      const lines = manualInputText.trim().split('\n');
      const newPoints = [];
      const newLabels = [];
      
      for (const line of lines) {
        if (!line.trim()) continue;
        
        const parts = line.split(',').map(part => part.trim());
        if (parts.length !== 3) {
          throw new Error(`格式错误: ${line}，请使用"x1,x2,label"格式，label为1或-1`);
        }
        
        const x1 = parseFloat(parts[0]);
        const x2 = parseFloat(parts[1]);
        const label = parseInt(parts[2], 10);
        
        if (isNaN(x1) || isNaN(x2) || isNaN(label) || (label !== 1 && label !== -1)) {
          throw new Error(`无效的数值: ${line}，请确保x1,x2为数字，label为1或-1`);
        }
        
        newPoints.push([x1, x2]);
        // 将标签转换为0和1，其中-1转换为0
        newLabels.push(label === 1 ? 1 : 0);
      }
      
      if (newPoints.length === 0) {
        throw new Error('没有有效的点被添加');
      }
      
      // 更新数据和标签
      setData([...data, ...newPoints]);
      setLabels([...labels, ...newLabels]);
      
      showMessage(`成功添加 ${newPoints.length} 个点`, 'success');
      setShowManualAddModal(false);
      setManualInputText('');
    } catch (error) {
      showMessage(error.message, 'error');
    }
  };

  const handleManualAddCancel = () => {
    setShowManualAddModal(false);
    setManualInputText('');
  };

  // 添加滑动框加点处理函数
  const handleSliderAdd = () => {
    setShowSliderAddModal(true);
    setSliderX1(0);
    setSliderX2(0);
    setSliderLabel(1);
    setPreviewPoint([0, 0]);
  };

  const handleSliderChange = (axis, value) => {
    if (axis === 'x1') {
      setSliderX1(value);
      setPreviewPoint([value, sliderX2]);
    } else if (axis === 'x2') {
      setSliderX2(value);
      setPreviewPoint([sliderX1, value]);
    }
  };

  const handleSliderLabelChange = (label) => {
    setSliderLabel(label);
  };

  const handleSliderAddSubmit = () => {
    // 添加点到数据中
    setData([...data, [sliderX1, sliderX2]]);
    setLabels([...labels, sliderLabel]);
    
    showMessage(`成功添加点 (${sliderX1.toFixed(2)}, ${sliderX2.toFixed(2)})`, 'success');
    setShowSliderAddModal(false);
  };

  const handleSliderAddCancel = () => {
    setShowSliderAddModal(false);
  };

  // 添加滚动到顶部的函数
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // 监听activeTab变化
  useEffect(() => {
    if (activeTab === 'basics') {
      scrollToTop();
    }
  }, [activeTab]);

  const handleExtendClick = () => {
    // 跳转到8501端口
    window.open('http://localhost:8501', '_blank');
  };

  // 3D演示相关状态
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(true);
  const [showSupportVectors, setShowSupportVectors] = useState(true);
  const [cameraPosition, setCameraPosition] = useState({ x: 1.5, y: 1.5, z: 1.5 });

  // 更新相机视角
  const updateCamera = (view) => {
    const positions = {
      top: { x: 0, y: 0, z: 2 },
      front: { x: 0, y: 2, z: 0 },
      side: { x: 2, y: 0, z: 0 },
      isometric: { x: 1.5, y: 1.5, z: 1.5 }
    };
    setCameraPosition(positions[view]);
  };

  // 添加渲染拓展内容函数
  const renderAdvancedContent = () => {
    switch (activeAdvancedSection) {
      case 'random-forest':
        return (
          <div className="advanced-section-content">
            <h3>随机森林算法</h3>
            <div className="concept-item">
              <h4>基本概念</h4>
              <p>随机森林是一种集成学习方法，通过构建多个决策树并取其投票结果来进行分类或回归。</p>
            </div>
            <div className="concept-item">
              <h4>核心特点</h4>
              <ul>
                <li>使用自助采样法（bootstrap）构建多个决策树</li>
                <li>在构建每棵树时随机选择特征子集</li>
                <li>通过投票机制决定最终结果</li>
                <li>具有很好的抗过拟合能力</li>
              </ul>
            </div>
            <div className="concept-item">
              <h4>算法流程</h4>
              <ol>
                <li>从训练集中随机采样（有放回）</li>
                <li>随机选择特征子集</li>
                <li>构建决策树</li>
                <li>重复上述步骤构建多棵树</li>
                <li>通过投票或平均得到最终结果</li>
              </ol>
            </div>
            <div className="concept-item">
              <h4>随机森林和 SVM 的区别</h4>
              <p>随机森林和支持向量机都是用于解决分类和回归问题的机器学习算法，它们之间的联系在于它们都试图找到一个最佳的模型来预测输入数据的输出。随机森林通过构建多个决策树并对其进行平均来提高泛化能力，而支持向量机则通过寻找最大化边界Margin的支持向量来实现分类。</p>
              <p>主要区别：</p>
              <ul>
                <li>模型结构：随机森林是集成学习方法，由多个决策树组成；SVM 是单一模型，通过支持向量确定决策边界</li>
                <li>处理非线性：随机森林天然支持非线性分类；SVM 需要通过核函数处理非线性问题</li>
                <li>计算复杂度：随机森林训练速度快，适合大规模数据；SVM 训练时间较长，特别是使用非线性核时</li>
                <li>可解释性：随机森林提供特征重要性排序；SVM 的决策过程相对复杂，解释性较差</li>
                <li>参数敏感性：随机森林对参数不敏感，容易调参；SVM 对核函数和参数选择较为敏感</li>
              </ul>
            </div>
          </div>
        );
      case 'bisecting-nearest-point':
        return (
          <div className="advanced-section-content">
            <h3>平分最近点法</h3>
            <div className="concept-item">
              <h4>基本概念</h4>
              <p>线性分类学习机中，得到超平面有两种方法，一种是最大间隔法，另一种是平分最近点法。在动态演示中，我们采用最大间隔法实现了分类。</p>
              <p>最大间隔法（Maximum Margin Method）通过最大化不同类别之间的间隔来寻找最优分类超平面，这种方法能够获得更好的泛化性能。而平分最近点法（Bisecting Nearest Point Method）则是通过找到两个类别中最近的点，然后在这些点之间寻找一个平分超平面。</p>
              <p>两种方法的主要区别在于：</p>
              <ul>
                <li>最大间隔法追求的是最大化分类间隔，能够获得更好的泛化能力</li>
                <li>平分最近点法更注重于直接分离两个类别，计算相对简单</li>
                <li>最大间隔法对噪声和异常值更鲁棒</li>
                <li>平分最近点法在类别分布较为均匀时效果较好</li>
              </ul>
            </div>
            <div className="concept-item">
              <h4 style={{ textAlign: 'left' }}>实现原理</h4>
              <p>找到两类凸壳最近点，做垂直平分线即可获得。下例的最近点是 <InlineMath math="a, b"/> 两点，可以通过求解一个最优化问题来解决。</p>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center',
                margin: '20px 0',
                width: '100%',
                minHeight: '300px'
              }}>
                <ImageComponent 
                  src="/images/nearest.png" 
                  alt="平分最近点法示意图"
                  className="concept-image"
                  style={{ 
                    maxWidth: '400px', 
                    width: '80%', 
                    height: 'auto',
                    display: 'block',
                    margin: '0 auto'
                  }}
                />
              </div>
            </div>
          </div>
        );
      case 'soft-hard-margin':
        return (
          <div className="advanced-section-content">
            <h3>软间隔与硬间隔</h3>
            <div className="concept-item">
              <h4>基本概念</h4>
              <p>在 SVM 中，间隔（Margin）是指决策边界到最近数据点的距离。根据是否允许分类错误，可以分为硬间隔和软间隔两种方法。</p>
              <p>硬间隔（Hard Margin）要求所有样本都被正确分类，不允许任何错误。而软间隔（Soft Margin）则允许一些样本被错误分类，通过引入松弛变量来控制错误程度。</p>
            </div>
            <div className="concept-item">
              <h4>硬间隔 SVM</h4>
              <p>硬间隔 SVM 的优化目标为：</p>
              <BlockMath math="\min_{w,b} \frac{1}{2}\|w\|^2" />
              <p>约束条件：</p>
              <BlockMath math="y_i(w^Tx_i + b) \geq 1, \forall i" />
              <p>特点：</p>
              <ul>
                <li>要求数据必须线性可分</li>
                <li>对噪声和异常值非常敏感</li>
                <li>可能出现过拟合问题</li>
              </ul>
            </div>
            <div className="concept-item">
              <h4>软间隔 SVM</h4>
              <p>软间隔 SVM 的优化目标为：</p>
              <BlockMath math="\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i" />
              <p>约束条件：</p>
              <BlockMath math="y_i(w^Tx_i + b) \geq 1 - \xi_i, \xi_i \geq 0, \forall i" />
              <p>其中：</p>
              <ul>
                <li><InlineMath math="\xi_i"/> 是松弛变量，表示第i个样本的误差</li>
                <li><InlineMath math="C"/> 是惩罚参数，控制对错误分类的惩罚程度</li>
                <li><InlineMath math="C"/> 越大，对错误分类的惩罚越重，模型越接近硬间隔</li>
                <li><InlineMath math="C"/> 越小，允许更多的错误分类，模型更灵活</li>
              </ul>
            </div>
            <div className="concept-item">
              <h4>应用场景</h4>
              <ul>
                <li>硬间隔：适用于数据完全线性可分且没有噪声的情况</li>
                <li>软间隔：适用于大多数实际情况，能够处理噪声和异常值</li>
                <li>通过调整 C 参数，可以在分类准确率和模型复杂度之间取得平衡</li>
              </ul>
            </div>
          </div>
        );
      case 'smo':
        return (
          <div className="advanced-section-content">
            <h3>序列最小优化算法（SMO）</h3>
            <div className="concept-item">
              <h4>基本概念</h4>
              <p>序列最小优化算法（Sequential Minimal Optimization，SMO）是一种用于求解 SVM 对偶问题的优化算法。它通过将大规模优化问题分解为一系列小规模子问题来高效求解。</p>
              <p>SMO算法的核心思想是：每次只选择两个变量进行优化，而固定其他变量。这种方法可以保证每次优化都能得到解析解，从而大大提高计算效率。</p>
            </div>
            <div className="concept-item">
              <h4>算法步骤</h4>
              <ol>
                <li>选择两个拉格朗日乘子 <InlineMath math="\alpha_1"/> 和 <InlineMath math="\alpha_2"/> </li>
                <li>固定其他乘子，求解这两个乘子的最优值</li>
                <li>更新对应的偏置项<InlineMath math="b"/></li>
                <li>检查收敛条件，如果未收敛则重复上述步骤</li>
              </ol>
              <p>选择乘子的启发式方法：</p>
              <ul>
                <li>第一个乘子：选择违反KKT条件最严重的样本</li>
                <li>第二个乘子：选择使目标函数值变化最大的样本</li>
              </ul>
            </div>
            <div className="concept-item">
              <h4>优化策略</h4>
              <p>1. 乘子选择策略：</p>
              <ul>
                <li>外层循环：遍历所有样本，寻找违反KKT条件的样本</li>
                <li>内层循环：选择使目标函数值变化最大的第二个乘子</li>
              </ul>
              <p>2. 收敛判断：</p>
              <ul>
                <li>检查所有样本是否满足KKT条件</li>
                <li>设置最大迭代次数</li>
                <li>设置目标函数值变化阈值</li>
              </ul>
            </div>
            <div className="concept-item">
              <h4>实现细节</h4>
              <p>1. 计算核函数矩阵：</p>
              <BlockMath math="K_{ij} = K(x_i, x_j)" />
              <p>2. 更新规则：</p>
              <BlockMath math="\alpha_2^{new} = \alpha_2^{old} + \frac{y_2(E_1 - E_2)}{\eta}" />
              <BlockMath math="\alpha_1^{new} = \alpha_1^{old} + y_1y_2(\alpha_2^{old} - \alpha_2^{new})" />
              <p>其中：</p>
              <ul>
                <li><InlineMath math="E_i = f(x_i) - y_i"/> 是预测误差</li>
                <li><InlineMath math="\eta = K_{11} + K_{22} - 2K_{12}"/> 是二阶导数</li>
              </ul>
            </div>
            <div className="concept-item">
              <h4>算法优势</h4>
              <ul>
                <li>计算效率高，适合大规模数据集</li>
                <li>内存需求小，不需要存储完整的核矩阵</li>
                <li>实现简单，容易理解和调试</li>
                <li>收敛速度快，通常只需要几次迭代</li>
              </ul>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // 添加高级知识导航状态
  const [activeAdvancedSection, setActiveAdvancedSection] = useState('random-forest');

  // 添加动态演示模式状态
  const [activeDynamicSection, setActiveDynamicSection] = useState('demo');  // 修改为 'demo'，默认显示二分类演示

  // 添加帮助中心导航状态
  const [activeHelpSection, setActiveHelpSection] = useState('introduction');

  return (
    <div className="App">
      {currentMode !== 'welcome' && (
        <>
          <h1 className="main-title">DataTessellate Lab —— 基于 SVM 的仿真实验与理论教学一体化交互平台</h1>
          <ModeSelector />
        </>
      )}
      {renderContent()}
    </div>
  );
}

// 选择菜单组件
const FunctionMenu = ({ onSelect }) => {
  return (
    <div className="function-menu">
      <h2>选择功能</h2>
      <button onClick={() => onSelect('knowledge')}>基础知识</button>
      <button onClick={() => onSelect('demo')}>动态演示</button>
      <button onClick={() => onSelect('extension')}>拓展内容</button>
    </div>
  );
};

export default App;