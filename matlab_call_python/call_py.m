close all; clear; clc;

% ȷ�� MATLAB �ܹ��ҵ� Python ������
% pyenv;

% ��� Python �ļ����ڵ�·��
pythonPath = fullfile(pwd, 'called_func.py');
if count(py.sys.path, pythonPath) == 0
    py.sys.path().append(pythonPath);
end

% ���� Python ����
result = py.called_func.greet('World', 3);

% �� Python ����ֵת��Ϊ MATLAB ��������
resultStr = char(result);

% ������ֵ
disp(resultStr);
