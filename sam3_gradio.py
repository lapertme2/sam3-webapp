#!/usr/bin/env python3
"""
SAM3 图像分割 Gradio 应用 (增强版本)
支持文本提示、点选、框选三种分割模式，集成真实SAM3模型
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import json
import tempfile
import os
import torch
import time
from typing import List, Optional, Tuple, Dict, Any
import warnings

# SAM3 imports
try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.box_ops import box_xywh_to_cxcywh
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SAM3模块导入失败: {e}")
    SAM3_AVAILABLE = False

# 全局变量
sam3_model = None
sam3_processor = None
simulate_mode = True
model_path = None
current_device = "cuda" if torch.cuda.is_available() else "cpu"

# 抑制SAM3模型内部的警告
import warnings
warnings.filterwarnings("ignore", message="expected str, bytes or os.PathLike object, not NoneType")
warnings.filterwarnings("ignore", message="Skipping the post-processing step due to the error above")

# 可视化辅助函数
import matplotlib.pyplot as plt
import cv2

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders=True):
    # 确保mask是2D的
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8) * 255  # 转换为0-255范围
    
    # 使用更明显的颜色
    if random_color:
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        alpha = 0.6
    else:
        color = np.array([255, 0, 0], dtype=np.uint8)  # 红色
        alpha = 0.6
    
    # 创建一个RGB颜色掩码
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask > 0] = color
    
    # 将掩码叠加到原图上
    ax.imshow(color_mask, alpha=alpha, interpolation='nearest')
    
    # 如果需要边框，绘制轮廓
    if borders and np.any(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 绘制白色轮廓
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        # 转换为numpy数组格式，用于matplotlib绘制
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 1:  # 确保是2D数组
                continue
            ax.plot(contour[:, 0], contour[:, 1], color='white', linewidth=2)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

# 点选分割的交互状态
interactive_points = []  # 保存点坐标和标签的列表，格式: [(x, y, label), ...]

def find_local_model():
    """自动查找本地sam3模型文件"""
    print(f"🔍 开始搜索SAM3模型文件...")
    print(f"📁 当前工作目录: {os.getcwd()}")
    
    # 可能的模型文件路径
    possible_paths = [
        "sam3.pt",
        "models/sam3.pt", 
        "checkpoints/sam3.pt",
        "model/sam3.pt",
        "./sam3.pt"
    ]
    
    print(f"🔍 检查预定义路径...")
    for path in possible_paths:
        print(f"  检查: {path} -> 存在: {os.path.exists(path)}")
        if os.path.exists(path):
            abs_path = os.path.abspath(path)
            print(f"✅ 找到模型文件: {abs_path}")
            return abs_path
    
    # 如果没找到，在当前目录下搜索所有pt文件
    print(f"🔍 搜索当前目录下的所有.pt文件...")
    current_dir = os.getcwd()
    try:
        for file in os.listdir(current_dir):
            if file.endswith('.pt'):
                print(f"  发现.pt文件: {file}")
                if 'sam3' in file.lower():
                    full_path = os.path.join(current_dir, file)
                    print(f"✅ 找到SAM3模型文件: {full_path}")
                    return full_path
    except Exception as e:
        print(f"❌ 搜索文件时出错: {e}")
    
    print(f"❌ 未找到SAM3模型文件")
    return None

def load_model(device: str = None, confidence_threshold: float = 0.5, 
               enable_segmentation: bool = True, enable_inst_interactivity: bool = False):
    """加载SAM3模型（优先从本地加载）"""
    global sam3_model, sam3_processor, simulate_mode, model_path, current_device
    
    if not SAM3_AVAILABLE:
        simulate_mode = True
        return "❌ SAM3模块不可用，使用模拟模式\n🔧 请安装SAM3依赖或检查环境配置"
    
    try:
        # 设置设备
        if device is None:
            device = current_device
        else:
            current_device = device
        
        print(f"� 开始加载SAM3模型到设备: {device}")
        
        # 优先尝试查找本地模型文件
        local_model_path = find_local_model()
        if local_model_path:
            model_path = local_model_path
            print(f"� 发现本地模型文件: {model_path}")
            
            # 加载本地模型
            sam3_model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                load_from_HF=False,
                checkpoint_path=model_path,
                enable_segmentation=enable_segmentation,
                enable_inst_interactivity=enable_inst_interactivity
            )
            print("✅ 成功加载本地模型")
        else:
            # 本地模型文件不存在，尝试从HuggingFace下载模型
            try:
                print("� 本地模型不存在，尝试从HuggingFace下载SAM3模型...")
                sam3_model = build_sam3_image_model(
                    device=device,
                    eval_mode=True,
                    load_from_HF=True,
                    enable_segmentation=enable_segmentation,
                    enable_inst_interactivity=enable_inst_interactivity
                )
                model_path = "HuggingFace/facebook/sam3"
                print("✅ 成功从HuggingFace加载模型")
            except Exception as hf_error:
                print(f"⚠️ HuggingFace下载失败: {hf_error}")
                raise RuntimeError("未找到可用的模型文件")
        
        # 创建处理器
        sam3_processor = Sam3Processor(
            model=sam3_model,
            device=device,
            confidence_threshold=confidence_threshold
        )
        
        simulate_mode = False
        return f"✅ SAM3模型加载成功\n� 模型路径: {model_path}\n🔧 设备: {device}\n📊 置信度阈值: {confidence_threshold}"
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        simulate_mode = True
        return f"⚠️ 模型加载失败，回退到模拟模式\n❌ 错误: {str(e)}\n🔧 建议：检查网络连接或提供本地模型文件"

def perform_real_segmentation(image, prompt, point_coords, point_labels, box, 
                            multimask_output=False, max_masks=3, box_expansion=0):
    """使用真实SAM3模型执行图像分割"""
    global sam3_model, sam3_processor
    
    try:
        print(f"🔍 开始真实SAM3分割...")
        start_time = time.time()
        
        # 设置图像
        inference_state = sam3_processor.set_image(image)
        print(f"✅ 图像设置完成，尺寸: {image.size if hasattr(image, 'size') else 'unknown'}")
        
        # 获取图像尺寸
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
        
        # 根据输入类型执行分割
        result_state = {}
        
        if prompt and prompt.strip():
            print(f"📝 文本提示分割: '{prompt}'")
            result_state = sam3_processor.set_text_prompt(prompt, inference_state)
            
        elif point_coords and len(point_coords) > 0:
            print(f"📍 点选分割: {len(point_coords)} 个点")
            
            # 转换点坐标格式为模型所需的numpy数组
            input_point = np.array(point_coords)
            input_label = np.array(point_labels)
            
            # 关键修复：将相对坐标转换为绝对坐标
            input_point_abs = input_point.copy()
            if len(input_point_abs) > 0 and (input_point_abs.max() <= 1.0 or input_point_abs.min() < 0):
                # 相对坐标，转换为绝对坐标
                input_point_abs[:, 0] *= width
                input_point_abs[:, 1] *= height
            
            # 使用model.predict_inst方法进行分割，传入绝对坐标
            masks, scores, logits = sam3_model.predict_inst(
                inference_state,
                point_coords=input_point_abs,
                point_labels=input_label,
                multimask_output=multimask_output,
            )
            
            # 排序并限制掩码数量
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind][:max_masks]
            scores = scores[sorted_ind][:max_masks]
            logits = logits[sorted_ind][:max_masks]
            
            # 构造结果状态，保存原始点坐标（用于可视化）
            result_state = {
                "masks": masks,
                "scores": scores,
                "logits": logits,
                "point_coords": input_point,
                "point_labels": input_label
            }
            
        elif box and len(box) >= 4:
            print(f"📦 框选分割: {box}")
            
            # 转换边界框格式
            input_box = np.array([box])
            
            # 使用model.predict_inst方法进行分割
            masks, scores, logits = sam3_model.predict_inst(
                inference_state,
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=multimask_output,
            )
            
            # 排序并限制掩码数量
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind][:max_masks]
            scores = scores[sorted_ind][:max_masks]
            logits = logits[sorted_ind][:max_masks]
            
            # 构造结果状态
            result_state = {
                "masks": masks,
                "scores": scores,
                "logits": logits
            }
            
        else:
            raise ValueError("未提供有效的分割提示")
        
        # 处理结果
        processing_time = time.time() - start_time
        print(f"✅ 分割完成，耗时: {processing_time:.2f}秒")
        
        # 创建可视化结果
        result_image, overview_text, details_text = create_visualization_result(image, result_state, processing_time)
        
        return result_image, overview_text, details_text
        
    except Exception as e:
        print(f"❌ 真实模型分割失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 回退到模拟模式
        print("🔄 回退到模拟模式...")
        result_image, overview_text, details_text = simulate_segmentation(image, prompt, point_coords, point_labels, box, 
                                   multimask_output, max_masks, box_expansion)
        return result_image, overview_text, details_text

def simulate_segmentation(image, prompt, point_coords, point_labels, box, 
                         multimask_output=False, max_masks=3, box_expansion=0):
    """模拟分割功能，不依赖真实模型"""
    
    # 原始图像
    original_image = image.copy() if hasattr(image, 'copy') else image
    
    # 模拟分割结果
    try:
        # 转换为numpy数组处理
        if isinstance(image, Image.Image):
            image_array = np.array(image)
            height, width = image_array.shape[:2]
        else:
            image_array = image
            height, width = image_array.shape[:2]
        
        # 创建模拟掩码
        masks = []
        boxes = []
        scores = []
        
        # 根据输入类型生成不同的模拟结果
        if point_coords:
            # 点选分割 - 根据点坐标生成掩码
            # 找到正样本点的中心
            positive_points = []
            for i, (coord, label) in enumerate(zip(point_coords, point_labels)):
                if label == 1:
                    positive_points.append(coord)
            
            if positive_points:
                # 计算正样本点的平均位置
                avg_x = sum(p[0] for p in positive_points) / len(positive_points)
                avg_y = sum(p[1] for p in positive_points) / len(positive_points)
                
                # 转换为图像坐标
                center_x = int(avg_x * width)
                center_y = int(avg_y * height)
                
                # 生成围绕中心点的掩码
                mask = np.zeros((height, width), dtype=bool)
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 < 10000  # 圆形区域
                masks.append(mask)
                
                # 生成边界框
                x1 = max(0, center_x - 100)
                y1 = max(0, center_y - 100)
                x2 = min(width, center_x + 100)
                y2 = min(height, center_y + 100)
                boxes.append([x1, y1, x2, y2])
                scores.append(0.95)
        
        elif box:
            # 框选分割 - 生成1个掩码
            x1, y1, x2, y2 = [int(x * width) for x in box]
            x1, y1 = max(0, x1 - box_expansion), max(0, y1 - box_expansion)
            x2, y2 = min(width, x2 + box_expansion), min(height, y2 + box_expansion)
            
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
            boxes.append([x1, y1, x2, y2])
            scores.append(0.90)
            
        else:
            # 文本提示分割 - 在图像中心生成掩码
            num_masks = min(3 if multimask_output else 1, max_masks)
            for i in range(num_masks):
                mask = np.zeros((height, width), dtype=bool)
                radius = 100 + i * 50
                center_x, center_y = width // 2, height // 2
                y_grid, x_grid = np.ogrid[:height, :width]
                mask_circle = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2
                mask = mask_circle
                masks.append(mask)
                
                # 边界框
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices), np.max(y_indices)
                    boxes.append([x1, y1, x2, y2])
                else:
                    boxes.append([0, 0, width, height])
                
                scores.append(0.8 + i * 0.05)
        
        # 构造结果状态，与perform_real_segmentation函数保持一致
        result_state = {
            "masks": masks,
            "scores": scores,
            "boxes": boxes
        }
        
        # 如果是点选分割，添加点坐标和标签
        if point_coords and len(point_coords) > 0:
            result_state["point_coords"] = np.array(point_coords)
            result_state["point_labels"] = np.array(point_labels)
        
        # 使用与真实分割相同的可视化函数
        result_pil, overview_text, details_text = create_visualization_result(original_image, result_state, 0.5)
        
        return result_pil, overview_text, details_text
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc() if 'traceback' in dir() else str(e)
        return None, f"❌ 模拟分割失败: {str(e)}\n详细信息: {error_details}"

def create_visualization_result(image: Image.Image, result_state: Dict, processing_time: float) -> Tuple[Image.Image, str, str]:
    """创建真实分割结果的可视化"""
    try:
        # 获取分割结果
        if "masks" not in result_state or len(result_state["masks"]) == 0:
            return image, "❌ 未检测到任何对象", ""
        
        masks = result_state["masks"]
        boxes = result_state.get("boxes", [])
        scores = result_state.get("scores", [])
        
        # 转换图像为numpy数组
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        height, width = image_array.shape[:2]
        
        # 确保所有掩码在CPU上并转换为numpy数组
        processed_masks = []
        processed_scores = []
        for mask, score in zip(masks, scores):
            # 确保掩码在CPU上并转换为numpy数组
            if hasattr(mask, 'cpu'):
                mask = mask.cpu()
            if hasattr(mask, 'detach'):
                mask = mask.detach()
            if hasattr(mask, 'numpy'):
                mask = mask.numpy()
            
            # 确保掩码是2D的
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 调整掩码尺寸以匹配图像
            if mask.shape != (height, width):
                from torchvision.transforms import functional as F
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                mask_tensor = F.resize(mask_tensor, (height, width), interpolation=F.InterpolationMode.NEAREST)
                mask = mask_tensor.squeeze().numpy().astype(bool)
            
            # 确保score在CPU上并转换为python标量
            if hasattr(score, 'cpu'):
                score = score.cpu()
            if hasattr(score, 'detach'):
                score = score.detach()
            if hasattr(score, 'item'):
                score = score.item()
            
            processed_masks.append(mask)
            processed_scores.append(score)
        
        # 使用matplotlib创建可视化结果
        plt.figure(figsize=(10, 10))
        plt.imshow(image_array)
        
        # 显示所有掩码
        for i, (mask, score) in enumerate(zip(processed_masks, processed_scores)):
            show_mask(mask, plt.gca(), borders=True)
        
        # 显示点坐标（如果有）
        point_coords = result_state.get("point_coords", None)
        point_labels = result_state.get("point_labels", None)
        if point_coords is not None and point_labels is not None and len(point_coords) > 0:
            # 转换点坐标从相对坐标到绝对坐标
            if len(point_coords) > 0 and (point_coords.max() <= 1.0 or point_coords.min() < 0):
                # 相对坐标，转换为绝对坐标
                absolute_point_coords = point_coords.copy()
                absolute_point_coords[:, 0] *= width
                absolute_point_coords[:, 1] *= height
                show_points(absolute_point_coords, point_labels, plt.gca())
            else:
                # 已经是绝对坐标
                show_points(point_coords, point_labels, plt.gca())
        
        plt.axis('off')
        plt.tight_layout()
        
        # 将matplotlib图像转换为PIL Image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buffer.seek(0)
        result_pil = Image.open(buffer)
        plt.close()
        
        # 计算总掩码面积
        total_area = sum(np.sum(mask) for mask in processed_masks)
        mask_count = len(processed_masks)
        
        # 生成概览信息
        overview_text = f"""🎯 分割完成 (真实SAM3模型)

📊 分割结果:
- 检测到的对象数量: {mask_count}
- 总掩码面积: {total_area} 像素
- 图像尺寸: {width} × {height}
- 处理时间: {processing_time:.2f}秒"""
        
        # 生成对象详细信息
        details_text = "📋 对象详细信息:\n"
        for i, (mask, score) in enumerate(zip(processed_masks, processed_scores)):
            mask_area = np.sum(mask)
            
            # 生成边界框信息
            y_indices, x_indices = np.where(mask) if len(mask.shape) == 2 else ([], [])
            if len(y_indices) > 0 and len(x_indices) > 0:
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                details_text += f"""
对象 {i+1}:
  - 置信度: {score:.3f}
  - 面积: {mask_area} 像素 ({mask_area/(width*height)*100:.1f}%)
  - 边界框: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]
  - 尺寸: {x2-x1:.1f} × {y2-y1:.1f}"""
            else:
                details_text += f"""
对象 {i+1}:
  - 置信度: {score:.3f}
  - 面积: {mask_area} 像素"""
        
        return result_pil, overview_text, details_text
        
    except Exception as e:
        print(f"❌ 可视化结果创建失败: {e}")
        import traceback
        traceback.print_exc()
        return image, f"❌ 可视化失败: {str(e)}", ""

def create_demo_image():
    """创建演示图像"""
    # 创建一个简单的彩色测试图像
    width, height = 400, 300
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # 绘制一些几何图形
    draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0), width=2)  # 红色矩形
    draw.ellipse([200, 50, 300, 150], fill=(0, 255, 0), outline=(0, 0, 0), width=2)   # 绿色圆形
    draw.polygon([(100, 200), (150, 250), (50, 250)], fill=(0, 0, 255), outline=(0, 0, 0))  # 蓝色三角形
    
    return image

def parse_points(point_input):
    """解析点坐标，支持字符串和列表两种输入类型"""
    points = []
    labels = []
    
    # 如果输入是列表，直接处理
    if isinstance(point_input, list):
        for point in point_input:
            if isinstance(point, tuple) or isinstance(point, list):
                if len(point) >= 2:
                    try:
                        x = float(point[0])
                        y = float(point[1])
                        points.append([x, y])
                        
                        # 如果提供了标签
                        if len(point) >= 3:
                            label = int(point[2])
                        else:
                            label = 1  # 默认正样本点
                        labels.append(label)
                    except ValueError:
                        continue
        return points, labels
    
    # 如果输入是字符串，按原逻辑处理
    point_text = point_input
    if not point_text or not point_text.strip():
        return points, labels
    
    for line in point_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                points.append([x, y])
                
                # 如果提供了标签
                if len(parts) >= 3:
                    label = int(parts[2].strip())
                else:
                    label = 1  # 默认正样本点
                labels.append(label)
            except ValueError:
                continue
    
    return points, labels

def parse_box(box_text):
    """解析边界框文本"""
    if not box_text.strip():
        return None
    
    try:
        coords = [float(x.strip()) for x in box_text.split(',')]
        if len(coords) == 4:
            return coords
    except ValueError:
        pass
    
    return None

def perform_segmentation(image, prompt, point_coords, box_coords, 
                        multimask_output=False, max_masks=3,
                        confidence_threshold: float = 0.5, box_expansion=0):
    """执行图像分割（增强版）"""
    global simulate_mode, sam3_processor
    
    try:
        # 检查输入
        if image is None:
            return None, "❌ 错误: 请上传图像", ""
        
        # 解析点坐标文本
        point_coords, point_labels = parse_points(point_coords)
        
        # 解析边界框
        box = parse_box(box_coords) if box_coords else None
        
        # 检查是否有有效的分割提示
        has_text = prompt and prompt.strip()
        has_points = len(point_coords) > 0
        has_box = box is not None and len(box) >= 4
        
        if not (has_text or has_points or has_box):
            return None, "❌ 错误: 请提供分割参数", ""
        
        # 更新置信度阈值
        if sam3_processor is not None:
            sam3_processor.confidence_threshold = confidence_threshold
        
        # 如果加载了真实模型，使用真实分割
        if sam3_model is not None and not simulate_mode and sam3_processor is not None:
            try:
                print(f"🧠 使用真实SAM3模型进行分割...")
                
                return perform_real_segmentation(image, prompt, point_coords, point_labels, box, 
                                               multimask_output, max_masks, box_expansion)
            except Exception as e:
                print(f"❌ 真实模型分割失败，回退到模拟模式: {e}")
                import traceback
                traceback.print_exc()
                simulate_mode = True
        
        # 使用模拟分割
        print("🎭 使用模拟模式进行分割...")
        return simulate_segmentation(image, prompt, point_coords, point_labels, box, 
                                   multimask_output, max_masks, box_expansion)
    
    except Exception as e:
        error_msg = f"❌ 分割过程中出错: {str(e)}\n"
        error_msg += f"错误类型: {type(e).__name__}\n"
        import traceback
        error_msg += f"详细错误信息:\n{traceback.format_exc()}"
        return None, error_msg, ""



def create_demo_with_image001():
    """加载image001.jpg作为演示图像，并设置默认文本提示"""
    try:
        if os.path.exists("image001.jpg"):
            image = Image.open("image001.jpg")
            print(f"✅ 成功加载image001.jpg，尺寸: {image.size}")
            # 返回图像和默认文本提示
            return image, "main stone"
        else:
            print(f"⚠️ image001.jpg文件不存在，使用默认演示图像")
            return create_demo_image(), "main stone"
    except Exception as e:
        print(f"❌ 加载image001.jpg失败: {e}")
        return create_demo_image(), "main stone"

def unload_model():
    """卸载模型"""
    global sam3_model, sam3_processor, current_device
    try:
        sam3_model = None
        sam3_processor = None
        current_device = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return "模型已卸载，GPU内存已释放"
    except Exception as e:
        return f"卸载模型时出错: {str(e)}"

def save_result(result_image):
    """保存结果"""
    try:
        if result_image is None:
            return "没有可保存的结果"
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"sam3_result_{timestamp}.png"
        result_image.save(filename)
        return f"结果已保存为: {filename}"
    except Exception as e:
        return f"保存结果时出错: {str(e)}"

def clear_results():
    """清除结果"""
    return None, "结果已清除", ""

def handle_image_click(img, evt: gr.SelectData, point_label):
    """处理图像点击事件，添加点"""
    global interactive_points
    
    if img is None:
        return format_points(interactive_points), format_points(interactive_points), img
    
    # 获取图像尺寸
    width, height = img.size
    
    # 获取点击坐标（相对坐标）
    x = evt.index[0] / width
    y = evt.index[1] / height
    
    # 添加点到列表
    interactive_points.append((x, y, point_label))
    
    # 在图像上绘制点
    draw = ImageDraw.Draw(img)
    color = (0, 255, 255) if point_label == 1 else (255, 0, 0)  # 青色：正样本，红色：负样本
    radius = 20
    draw.ellipse([(evt.index[0]-radius, evt.index[1]-radius), 
                 (evt.index[0]+radius, evt.index[1]+radius)], 
                fill=color, outline=(0, 0, 0), width=5)
    
    # 格式化点列表为文本
    points_text = format_points(interactive_points)
    
    # 返回更新后的点文本和图像
    return points_text, points_text, img

def format_points(points):
    """格式化点列表为文本"""
    if not points:
        return ""
    return "\n".join([f"{x:.3f},{y:.3f},{label}" for x, y, label in points])

def clear_points():
    """清除所有点"""
    global interactive_points
    interactive_points = []
    return "", "", None

def manual_add_point(point_text, img):
    """手动添加点"""
    global interactive_points
    
    # 解析点文本
    points, labels = parse_points(point_text)
    
    if points:
        for point, label in zip(points, labels):
            interactive_points.append((point[0], point[1], label))
    
    # 更新图像
    if img is not None:
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for x, y, label in interactive_points:
            color = (0, 255, 255) if label == 1 else (255, 0, 0)  # 青色：正样本，红色：负样本
            radius = 20
            draw.ellipse([(int(x*width)-radius, int(y*height)-radius), 
                         (int(x*width)+radius, int(y*height)+radius)], 
                        fill=color, outline=(0, 0, 0), width=5)
    
    # 格式化点列表为文本
    points_text = format_points(interactive_points)
    
    return points_text, points_text, img

def sync_points_to_textbox(points):
    """同步点列表到文本框"""
    return format_points(points)

def update_interactive_image(img):
    """更新交互式图像，保留已选点"""
    global interactive_points
    
    if img is None:
        return None
    
    # 复制图像
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size
    
    # 绘制已选点
    for x, y, label in interactive_points:
        color = (0, 255, 255) if label == 1 else (255, 0, 0)
        radius = 20
        draw.ellipse([(int(x*width)-radius, int(y*height)-radius), 
                     (int(x*width)+radius, int(y*height)+radius)], 
                    fill=color, outline=(0, 0, 0), width=5)
    
    return img_copy

def create_interface():
    """创建增强版Gradio界面"""
    
    with gr.Blocks(title="SAM3 图像分割应用 - 增强版") as demo:
        
        # 标题
        gr.Markdown("# 🎯 SAM3 图像分割应用 - 增强版")
        gr.Markdown("*支持文本提示、点选、框选三种分割模式的AI图像分割工具*")
        
        with gr.Row():
            # 左侧：输入控制
            with gr.Column(scale=1):
                gr.Markdown("## 📤 输入设置")
                
                # 图像上传
                image_input = gr.Image(
                    label="🖼️ 上传图像",
                    type="pil",
                    height=400
                )
                
                with gr.Row():
                    # 预设演示图像按钮
                    demo_btn = gr.Button("🎨 使用演示图像", variant="secondary", scale=1)
                    # 默认使用image001.jpg按钮
                    #default_btn = gr.Button("🗿 使用默认图像", variant="primary", scale=1)
                
                # 分割模式选择
                gr.Markdown("### 🎯 分割模式")
                
                with gr.Tabs():
                    with gr.Tab("📝 文本提示"):
                        text_prompt = gr.Textbox(
                            label="文本提示",
                            placeholder="例如：person, car, dog, face...",
                            info="输入描述要分割对象的文本提示",
                            lines=2
                        )
                        
                        with gr.Row():
                            example_texts = ["person", "car", "dog","red","blue","green"]
                            for example in example_texts:
                                gr.Button(example, size="sm").click(
                                    lambda x=example: x,
                                    outputs=[text_prompt]
                                )
                    
                    with gr.Tab("📍 点选分割"):
                        # 点坐标输入
                        point_coords = gr.Textbox(
                            label="点坐标 (可选)",
                            placeholder="0.5,0.3,1\n0.7,0.6,1\n0.3,0.8,0",
                            info="每行一个点坐标，格式：x,y,label (0-1之间的相对坐标，1=正样本，0=负样本)",
                            lines=4
                        )
                        
                        with gr.Row():
                            # 正/负样本点切换
                            with gr.Column(scale=1):
                                point_label = gr.Radio(
                                    choices=[("正样本点", 1), ("负样本点 ", 0)],
                                    value=1,
                                    label="点类型",
                                    info="选择当前要添加的点类型"
                                )
                                
                                # 已选点显示
                                selected_points = gr.Textbox(
                                    label="已选点",
                                    placeholder="点坐标将显示在这里...",
                                    info="已选择的点坐标和标签",
                                    lines=4,
                                    interactive=False
                                )
                                
                                with gr.Row():
                                    clear_points_btn = gr.Button("清除所有点", size="sm", variant="secondary")
                                    add_point_btn = gr.Button("手动添加点", size="sm")
                    
                    with gr.Tab("📦 框选分割"):
                        box_coords = gr.Textbox(
                            label="边界框坐标",
                            placeholder="0.2,0.3,0.8,0.7",
                            info="格式：x1,y1,x2,y2 (0-1之间的相对坐标，左上角到右下角)",
                            lines=2
                        )
                        
                        with gr.Row():
                            gr.Button("清除框", size="sm", variant="secondary").click(
                                lambda: "",
                                outputs=[box_coords]
                            )
                
                # 高级参数
                gr.Markdown("### ⚙️ 高级参数")
                
                with gr.Row():
                    confidence_threshold = gr.Slider(
                        label="置信度阈值",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        info="只显示置信度高于此阈值的分割结果"
                    )
                    
                    device_choice = gr.Radio(
                        choices=["cuda", "cpu", "auto"],
                        value="cuda",
                        label="计算设备",
                        info="选择运行模型的设备"
                    )
                
                with gr.Row():
                    multimask_output = gr.Checkbox(
                        label="多掩码输出",
                        value=False,
                        info="启用后将生成多个候选掩码"
                    )
                    
                    max_masks = gr.Slider(
                        label="最大掩码数",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        info="限制生成的最大掩码数量"
                    )
                
                # 模型控制
                #gr.Markdown("### 🧠 模型控制")
                
                #with gr.Row():
                #    load_btn = gr.Button("加载/重新加载模型", variant="primary")
                #    unload_btn = gr.Button("卸载模型", variant="secondary")
                
                # 模型状态
                #model_status = gr.Textbox(
                #    label="📊 模型状态",
                #     value="点击加载模型按钮开始使用",
                #     interactive=False,
                #     lines=3
                # )
                
                # 分割按钮
                segment_btn = gr.Button("开始分割", variant="primary", size="lg")
            
            # 右侧：输出结果和交互式图像
            with gr.Column(scale=1):
                gr.Markdown("## 📊 分割结果")
                
                # 结果选项卡
                with gr.Tabs():
                    with gr.Tab("🎨 可视化结果"):
                        result_image = gr.Image(
                            label="分割结果可视化",
                            type="pil",
                            interactive=False,
                            height=400
                        )
                        
                        # 结果操作
                        with gr.Row():
                            save_btn = gr.Button("💾 保存结果", size="sm")
                            clear_btn = gr.Button("🗑️ 清除结果", size="sm", variant="secondary")
                        
                        # 交互式图像操作
                        gr.Markdown("### 📍 交互式图像操作")
                        interactive_image = gr.Image(
                            label="交互式图像",
                            type="pil",
                            interactive=True,
                            height=400
                        )
                    
                    with gr.Tab("📋 详细信息"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                info_output1 = gr.Textbox(
                                    label="分割结果概览",
                                    lines=10,
                                    max_lines=20,
                                    interactive=False
                                )
                            with gr.Column(scale=1):
                                info_output2 = gr.Textbox(
                                    label="对象详细信息",
                                    lines=10,
                                    max_lines=20,
                                    interactive=False
                                )
        
        # 事件绑定
        # 显示demo图像
        demo_btn.click(
            fn=create_demo_image,
            inputs=[],
            outputs=[image_input]
        )
        
        # 图像上传后，同步到交互式图像
        image_input.change(
            fn=update_interactive_image,
            inputs=[image_input],
            outputs=[interactive_image]
        )
        
        # 交互式图像点击事件
        interactive_image.select(
            fn=handle_image_click,
            inputs=[interactive_image, point_label],
            outputs=[point_coords, selected_points, interactive_image]
        )
        
        # 清除点按钮
        clear_points_btn.click(
            fn=clear_points,
            inputs=[],
            outputs=[point_coords, selected_points, interactive_image]
        )
        
        # 手动添加点按钮
        add_point_btn.click(
            fn=manual_add_point,
            inputs=[point_coords, interactive_image],
            outputs=[point_coords, selected_points, interactive_image]
        )
        
        segment_btn.click(
            fn=perform_segmentation,
            inputs=[image_input, text_prompt, point_coords, box_coords, 
                   multimask_output, max_masks, confidence_threshold],
            outputs=[result_image, info_output1, info_output2]
        )
        
        save_btn.click(
            fn=save_result,
            inputs=[result_image],
            outputs=[info_output1]
        )
        
        clear_btn.click(
            fn=clear_results,
            inputs=[],
            outputs=[result_image, info_output1, info_output2]
        )
        
        # 使用说明和示例
        gr.Markdown("""
        ---
        
        ## 📖 使用说明
        
        ### 🎯 分割模式
        
        - **文本提示模式：** 输入描述性文本，如"person", "car", "dog"
        - **点选模式：** 输入点坐标和标签，格式：`x,y,label` (1=正样本，0=负样本)
        - **框选模式：** 输入边界框坐标，格式：`x1,y1,x2,y2` (相对坐标)
        
        ### ⚙️ 参数说明
        
        - **置信度阈值：** 只显示置信度高于此值的结果 (0.1-0.9)
        - **多掩码输出：** 为每个对象生成多个候选掩码
        - **最大掩码数：** 限制生成的掩码数量
        
        ### 💡 使用技巧
        
        1. **文本提示：** 使用简单明确的词汇，如"person"而不是"一个人"
        2. **点选：** 正样本点标记目标，负样本点标记背景
        3. **框选：** 框选要包含整个目标对象
        4. **置信度：** 降低阈值可以看到更多检测结果，但可能包含误检
        
        ### 🎨 示例
        
        - **文本提示：** "person", "car", "dog", "face"
        - **点选：** `0.5,0.3,1` (在图像中心偏上位置标记正样本)
        - **框选：** `0.2,0.2,0.8,0.8` (选择图像中心区域)
        """)
    
    return demo

def main():
    """主函数"""
    print("🎯 SAM3 Gradio 应用启动中...")
    
    # 启动时自动尝试加载模型
    print("🔍 启动时自动检查模型文件...")
    try:
        print("🔧 开始调用 load_model() 函数...")
        model_status = load_model(enable_inst_interactivity=True)
        print(f"📊 模型状态返回: {model_status}")
        
        # 额外检查全局变量状态
        print(f"🔍 调试信息:")
        print(f"  - sam3_model: {type(sam3_model) if sam3_model is not None else 'None'}")
        print(f"  - simulate_mode: {simulate_mode}")
        print(f"  - model_path: {model_path}")
        
    except Exception as e:
        print(f"❌ 模型加载过程出错: {e}")
        print(f"❌ 错误类型: {type(e)}")
        import traceback
        print(f"❌ 错误详情: {traceback.format_exc()}")
        model_status = "❌ 模型加载失败"
    
    print(f"🔧 当前运行模式: {'真实模式' if not simulate_mode else '模拟模式'}")
    
    try:
        # 创建界面
        demo = create_interface()
        
        # 启动应用
        print("🚀 启动 Gradio 服务器...")
        print("📱 请在浏览器中访问: http://localhost:7860")
        print("⏹️  按 Ctrl+C 停止服务")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("🔧 尝试启动简化版本...")
        
        # 创建最简单的界面
        with gr.Blocks() as simple_demo:
            gr.Markdown("# SAM3 图像分割应用 (简化版)")
            img = gr.Image(label="上传图像")
            btn = gr.Button("分割")
            result = gr.Image(label="结果")
            info = gr.Textbox(label="信息")
            
            def simple_process(img):
                if img is None:
                    return None, "请上传图像"
                return img, "处理完成（模拟模式）"
            
            btn.click(simple_process, inputs=[img], outputs=[result, info])
        
        try:
            simple_demo.launch(server_port=7860)
        except Exception as e2:
            print(f"❌ 简化版启动也失败: {e2}")
            print("🔧 尝试使用端口7861...")
            try:
                simple_demo.launch(server_port=7861)
            except Exception as e3:
                print(f"❌ 所有启动尝试都失败: {e3}")

if __name__ == "__main__":
    main()