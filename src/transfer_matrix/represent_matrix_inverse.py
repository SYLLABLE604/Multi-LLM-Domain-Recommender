import torch
import scipy.linalg
import numpy as np
import os

def process_matrix_inverse(input_path, output_path, matrix_key=None):
    """
    读取pth文件中的矩阵，计算逆矩阵或伪逆矩阵，并保存结果
    
    Args:
        input_path (str): 输入pth文件路径
        output_path (str): 输出pth文件路径  
        matrix_key (str, optional): 如果pth文件包含字典，指定矩阵的键名
    """
    
    try:
        # 读取pth文件
        print(f"正在读取文件: {input_path}")
        data = torch.load(input_path, map_location='cuda:4')
        
        # 提取矩阵
        if isinstance(data, dict):
            if matrix_key is None:
                # 如果没有指定键名，尝试找到第一个矩阵
                matrix = None
                for key, value in data.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                        matrix = value
                        print(f"自动选择矩阵键: {key}")
                        break
                if matrix is None:
                    raise ValueError("未找到2D矩阵，请指定matrix_key参数")
            else:
                if matrix_key not in data:
                    raise KeyError(f"键 '{matrix_key}' 不存在于文件中")
                matrix = data[matrix_key]
        elif isinstance(data, torch.Tensor):
            matrix = data
        else:
            raise ValueError("不支持的数据类型，请确保文件包含张量或字典")
        
        # 确保是2D矩阵
        if len(matrix.shape) != 2:
            raise ValueError(f"需要2D矩阵，但得到形状: {matrix.shape}")
        
        print(f"矩阵形状: {matrix.shape}")
        print(f"矩阵数据类型: {matrix.dtype}")
        
        # 转换为numpy数组用于scipy计算
        matrix_np = matrix.detach().cpu().numpy()
        
        # 尝试计算逆矩阵
        inverse_matrix = None
        method_used = ""
        
        try:
            # 检查矩阵是否为方阵
            if matrix_np.shape[0] != matrix_np.shape[1]:
                raise np.linalg.LinAlgError("矩阵不是方阵，无法计算逆矩阵")
            
            # 使用scipy计算逆矩阵
            inverse_matrix_np = scipy.linalg.inv(matrix_np)
            inverse_matrix = torch.from_numpy(inverse_matrix_np).to(matrix.dtype)
            method_used = "scipy.linalg.inv"
            
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as e:
            
            try:
                # 使用torch计算伪逆矩阵
                inverse_matrix = torch.linalg.pinv(matrix)
                method_used = "torch.linalg.pinv"
                
            except Exception as e:
                print(f"失败: {e}")
                raise RuntimeError("无法计算逆矩阵或伪逆矩阵")
        
        # 准备保存的数据
        if isinstance(data, dict):
            # 如果原始数据是字典，创建新字典保存结果
            result_data = data.copy()
            inverse_key = f"{matrix_key}_inverse" if matrix_key else "matrix_inverse"
            result_data[inverse_key] = inverse_matrix
            result_data['inverse_method'] = method_used
        else:
            # 如果原始数据是张量，直接保存逆矩阵
            result_data = inverse_matrix
        
        # 保存结果
        print(f"正在保存到: {output_path}")
        torch.save(result_data, output_path)
        
        print(f"✓ 处理完成!")
        print(f"逆矩阵形状: {inverse_matrix.shape}")
        print(f"逆矩阵数据类型: {inverse_matrix.dtype}")
        
        return inverse_matrix, method_used
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        raise

def verify_inverse(original_matrix, inverse_matrix, method_used, tolerance=1e-6):
    """
    验证逆矩阵的正确性
    """
    print("\n验证逆矩阵...")
    
    if method_used == "scipy.linalg.inv":
        # 对于真逆矩阵，检查 A * A^(-1) = I
        if original_matrix.shape[0] == original_matrix.shape[1]:
            product = torch.mm(original_matrix, inverse_matrix)
            identity = torch.eye(original_matrix.shape[0], dtype=original_matrix.dtype)
            error = torch.max(torch.abs(product - identity)).item()
            
            if error < tolerance:
                print(f"✓ 逆矩阵验证通过，最大误差: {error:.2e}")
            else:
                print(f"⚠ 逆矩阵验证警告，最大误差: {error:.2e}")
    
    elif method_used == "torch.linalg.pinv":
        # 对于伪逆矩阵，检查 A * A^+ * A = A
        product = torch.mm(torch.mm(original_matrix, inverse_matrix), original_matrix)
        error = torch.max(torch.abs(product - original_matrix)).item()
        
        if error < tolerance:
            print(f"✓ 伪逆矩阵验证通过，最大误差: {error:.2e}")
        else:
            print(f"⚠ 伪逆矩阵验证警告，最大误差: {error:.2e}")

# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    input_file = "/root/work/Crowdsourcing/MLDR/MLDR/represent_matrix/mistral/mistral.pth"
    output_file = "/root/work/Crowdsourcing/MLDR/MLDR/represent_matrix/mistral/mistral_inv.pth"
    
    # 如果输入文件不存在，创建一个示例矩阵
    if not os.path.exists(input_file):
        print("创建示例矩阵文件...")
        # 创建一个可逆的示例矩阵
        example_matrix = torch.tensor([[2.0, 1.0], 
                                     [1.0, 1.0]], dtype=torch.float32)
        torch.save(example_matrix, input_file)
        print(f"示例矩阵已保存到: {input_file}")
    
    try:
        # 处理矩阵
        original_data = torch.load(input_file, map_location='cuda:4')
        inverse_matrix, method = process_matrix_inverse(input_file, output_file)
        
        # 验证结果
        if isinstance(original_data, torch.Tensor):
            verify_inverse(original_data, inverse_matrix, method)
        
    except Exception as e:
        print(f"程序执行失败: {e}")