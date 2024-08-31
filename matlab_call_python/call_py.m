close all; clear; clc;

% 确保 MATLAB 能够找到 Python 解释器
% pyenv;

% 添加 Python 文件所在的路径
pythonPath = fullfile(pwd, 'called_func.py');
if count(py.sys.path, pythonPath) == 0
    py.sys.path().append(pythonPath);
end

% 调用 Python 函数
result = py.called_func.greet('World', 3);

% 将 Python 返回值转换为 MATLAB 数据类型
resultStr = char(result);

% 处理返回值
disp(resultStr);
