# smart_car_path_planner.py
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel, Text, Scrollbar
import re

LOG_ENABLED = False

def log_print(*args, **kwargs):
    if LOG_ENABLED:
        print(*args, **kwargs)


class SmartCarPathPlanner:
    def __init__(self, master):
        self.master = master
        master.title("智能车竞赛 - 科目二绕八字路径规划软件 v6.3")
        master.geometry("1500x850")

        # 路径规划参数
        self.cone_radius = 0.15
        self.cone1_pos = (0, 1.0)
        self.cone2_pos = (0, -1.0)
        self.outer_radius = 2.0
        self.start_point = (0, 1.5)  # 规划起点
        self.min_step_size = 0.08
        self.max_step_size = 0.15
        self.max_angle_change = 30.0
        self.smooth_sigma = 1.5
        self.curvature_sensitivity = 1.0
        self.target_point_count = 0
        self.fix_endpoints = True
        self.single_step_angle_limit = 15.0
        self.optimization_passes = 3

        # 复现数据存储（独立）
        self.replay_start_point = (0, 1.5)  # 复现起点，与规划独立
        self.replay_distances = []  # 存储原始距离数据
        self.replay_angles_deg = []  # 存储原始角度数据
        self.replay_mode = ''  # 存储角度模式
        self.replay_points = []  # 存储复现路径点

        self.angle_mode = 'mode1'
        self.raw_points = []
        self.opt_points = []
        self.distance_arr = []
        self.yaw_arr = []

        self.create_widgets()
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.update_plot()

    def create_widgets(self):
        # 创建PanedWindow实现可调节分割
        main_pane = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        control_frame = ttk.Frame(main_pane, width=450)
        main_pane.add(control_frame, weight=0)

        # 右侧画布区域
        canvas_container = ttk.Frame(main_pane)
        main_pane.add(canvas_container, weight=1)

        # 创建Notebook标签页
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 标签页1: 路径规划
        planning_tab = ttk.Frame(notebook)
        notebook.add(planning_tab, text="路径规划")

        # 标签页2: 数据复现
        replay_tab = ttk.Frame(notebook)
        notebook.add(replay_tab, text="数据复现")

        # ========== 规划标签页内容 ==========
        cone_frame = ttk.LabelFrame(planning_tab, text="锥桶设置")
        cone_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(cone_frame, text="半径(m):").grid(row=0, column=0, sticky=tk.W)
        self.cone_radius_var = tk.DoubleVar(value=self.cone_radius)
        ttk.Entry(cone_frame, textvariable=self.cone_radius_var, width=8).grid(row=0, column=1)
        ttk.Button(cone_frame, text="应用", command=self.update_cone_radius).grid(row=0, column=2, padx=5)

        ttk.Label(cone_frame, text="上锥桶Y(m):").grid(row=1, column=0, sticky=tk.W)
        self.cone1_y_var = tk.DoubleVar(value=self.cone1_pos[1])
        ttk.Entry(cone_frame, textvariable=self.cone1_y_var, width=8).grid(row=1, column=1)
        ttk.Label(cone_frame, text="下锥桶Y(m):").grid(row=2, column=0, sticky=tk.W)
        self.cone2_y_var = tk.DoubleVar(value=self.cone2_pos[1])
        ttk.Entry(cone_frame, textvariable=self.cone2_y_var, width=8).grid(row=2, column=1)
        ttk.Button(cone_frame, text="更新锥桶", command=self.update_cone_pos).grid(row=1, column=2, rowspan=2, padx=5)

        # 规划起点设置（独立）
        plan_start_frame = ttk.LabelFrame(planning_tab, text="规划起点设置")
        plan_start_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(plan_start_frame, text="起点X(m):").grid(row=0, column=0, sticky=tk.W)
        self.start_x_var = tk.DoubleVar(value=self.start_point[0])
        ttk.Entry(plan_start_frame, textvariable=self.start_x_var, width=8).grid(row=0, column=1)
        ttk.Label(plan_start_frame, text="Y(m):").grid(row=0, column=2, sticky=tk.W)
        self.start_y_var = tk.DoubleVar(value=self.start_point[1])
        ttk.Entry(plan_start_frame, textvariable=self.start_y_var, width=8).grid(row=0, column=3)
        ttk.Button(plan_start_frame, text="设置规划起点", command=self.set_start_point).grid(row=0, column=4, padx=5)

        path_frame = ttk.LabelFrame(planning_tab, text="路径参数")
        path_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(path_frame, text="最小步进(m):").grid(row=0, column=0, sticky=tk.W)
        self.min_step_var = tk.DoubleVar(value=self.min_step_size)
        ttk.Entry(path_frame, textvariable=self.min_step_var, width=6).grid(row=0, column=1)
        ttk.Label(path_frame, text="最大步进(m):").grid(row=0, column=2, sticky=tk.W)
        self.max_step_var = tk.DoubleVar(value=self.max_step_size)
        ttk.Entry(path_frame, textvariable=self.max_step_var, width=6).grid(row=0, column=3)

        ttk.Label(path_frame, text="目标点数(0=步进):").grid(row=1, column=0, sticky=tk.W)
        self.target_count_var = tk.IntVar(value=self.target_point_count)
        ttk.Entry(path_frame, textvariable=self.target_count_var, width=6).grid(row=1, column=1)
        ttk.Label(path_frame, text="最大转角(°):").grid(row=1, column=2, sticky=tk.W)
        self.max_angle_var = tk.DoubleVar(value=self.max_angle_change)
        ttk.Entry(path_frame, textvariable=self.max_angle_var, width=6).grid(row=1, column=3)

        ttk.Label(path_frame, text="单次转角限制(°):").grid(row=2, column=0, sticky=tk.W)
        self.single_angle_var = tk.DoubleVar(value=self.single_step_angle_limit)
        ttk.Entry(path_frame, textvariable=self.single_angle_var, width=6).grid(row=2, column=1)
        ttk.Label(path_frame, text="优化迭代次数:").grid(row=2, column=2, sticky=tk.W)
        self.opt_passes_var = tk.IntVar(value=self.optimization_passes)
        ttk.Entry(path_frame, textvariable=self.opt_passes_var, width=6).grid(row=2, column=3)

        ttk.Label(path_frame, text="平滑系数:").grid(row=3, column=0, sticky=tk.W)
        self.smooth_sigma_var = tk.DoubleVar(value=self.smooth_sigma)
        ttk.Entry(path_frame, textvariable=self.smooth_sigma_var, width=6).grid(row=3, column=1)
        ttk.Label(path_frame, text="曲率敏感度:").grid(row=3, column=2, sticky=tk.W)
        self.curvature_sens_var = tk.DoubleVar(value=self.curvature_sensitivity)
        ttk.Entry(path_frame, textvariable=self.curvature_sens_var, width=6).grid(row=3, column=3)

        self.fix_endpoints_var = tk.BooleanVar(value=self.fix_endpoints)
        ttk.Checkbutton(path_frame, text="保持端点固定", variable=self.fix_endpoints_var).grid(row=4, column=0, columnspan=2, sticky=tk.W)

        angle_frame = ttk.LabelFrame(planning_tab, text="惯导角度表示")
        angle_frame.pack(fill=tk.X, padx=5, pady=5)
        self.angle_mode_var = tk.StringVar(value=self.angle_mode)
        ttk.Radiobutton(angle_frame, text="模式1: -180°~180° (顺正逆负)",
                        variable=self.angle_mode_var, value='mode1').pack(anchor=tk.W)
        ttk.Radiobutton(angle_frame, text="模式2: 0°~360° (顺时针递增)",
                        variable=self.angle_mode_var, value='mode2').pack(anchor=tk.W)
        ttk.Radiobutton(angle_frame, text="模式3: 累积角度 (顺→+∞, 逆→-∞)",
                        variable=self.angle_mode_var, value='mode3').pack(anchor=tk.W)
        ttk.Button(angle_frame, text="应用模式", command=self.apply_angle_mode).pack(pady=5)

        action_frame = ttk.Frame(planning_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(action_frame, text="清空手绘路径", command=self.clear_raw_path).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="手动优化(使用当前参数)", command=self.optimize_and_generate).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="显示距离/角度数据", command=self.show_data_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="打印数据到控制台", command=self.print_data_to_console).pack(fill=tk.X, pady=2)

        # ========== 复现标签页内容（完全独立） ==========
        replay_inner = ttk.Frame(replay_tab)
        replay_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(replay_inner, text="粘贴距离和角度数组 (支持C数组或逗号分隔):").pack(anchor=tk.W)

        # 距离文本框
        ttk.Label(replay_inner, text="Distance数据:").pack(anchor=tk.W, pady=(5,0))
        self.replay_distance_text = Text(replay_inner, height=6, font=("Courier", 9))
        self.replay_distance_text.pack(fill=tk.X, pady=2)

        # 角度文本框
        ttk.Label(replay_inner, text="Yaw/角度数据:").pack(anchor=tk.W, pady=(5,0))
        self.replay_yaw_text = Text(replay_inner, height=6, font=("Courier", 9))
        self.replay_yaw_text.pack(fill=tk.X, pady=2)

        # 示例按钮
        example_frame = ttk.Frame(replay_inner)
        example_frame.pack(fill=tk.X, pady=5)
        ttk.Button(example_frame, text="插入示例数据", command=self.insert_example).pack(side=tk.LEFT, padx=2)

        # 模式选择
        mode_frame = ttk.LabelFrame(replay_inner, text="数据角度模式")
        mode_frame.pack(fill=tk.X, pady=5)
        self.replay_mode_var = tk.StringVar(value='auto')
        ttk.Radiobutton(mode_frame, text="自动检测", variable=self.replay_mode_var, value='auto').pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="模式1: -180°~180°", variable=self.replay_mode_var, value='mode1').pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="模式2: 0°~360°", variable=self.replay_mode_var, value='mode2').pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="模式3: 累积角度", variable=self.replay_mode_var, value='mode3').pack(anchor=tk.W)

        # ========== 复现起点设置（完全独立，有确认按钮） ==========
        replay_start_frame = ttk.LabelFrame(replay_inner, text="复现起点设置 (独立于规划起点)")
        replay_start_frame.pack(fill=tk.X, pady=5)

        # 起点选择方式
        self.replay_start_mode = tk.StringVar(value='manual')
        ttk.Radiobutton(replay_start_frame, text="使用手动起点",
                        variable=self.replay_start_mode, value='manual').pack(anchor=tk.W, padx=10)

        manual_frame = ttk.Frame(replay_start_frame)
        manual_frame.pack(fill=tk.X, padx=30, pady=5)
        ttk.Label(manual_frame, text="X(m):").pack(side=tk.LEFT)
        self.replay_start_x = tk.DoubleVar(value=0.0)
        ttk.Entry(manual_frame, textvariable=self.replay_start_x, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(manual_frame, text="Y(m):").pack(side=tk.LEFT, padx=(10,0))
        self.replay_start_y = tk.DoubleVar(value=1.5)
        ttk.Entry(manual_frame, textvariable=self.replay_start_y, width=8).pack(side=tk.LEFT, padx=2)

        # 起点确认按钮 - 重新计算路径（使用已解析的数据）
        confirm_start_btn = ttk.Button(manual_frame, text="✓ 确认并更新起点",
                                       command=self.update_replay_start, width=15)
        confirm_start_btn.pack(side=tk.LEFT, padx=10)

        ttk.Radiobutton(replay_start_frame, text="使用数据中的起点坐标 (需要数据包含X,Y)",
                        variable=self.replay_start_mode, value='from_data').pack(anchor=tk.W, padx=10)

        # 其他选项
        opt_frame = ttk.LabelFrame(replay_inner, text="复现选项")
        opt_frame.pack(fill=tk.X, pady=5)

        self.invert_angle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="角度取反 (如果路径方向相反)",
                        variable=self.invert_angle_var, command=self.apply_angle_option).pack(anchor=tk.W, padx=10)

        btn_frame = ttk.Frame(replay_inner)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="解析数据并复现", command=self.parse_and_replay).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="保存PNG", command=self.save_replay_png).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="导入CSV", command=self.import_csv).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 复现结果显示
        self.replay_info_var = tk.StringVar(value="等待复现...")
        ttk.Label(replay_inner, textvariable=self.replay_info_var, foreground="blue").pack(pady=5)

        # ========== 右侧画布 ==========
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_container)
        toolbar.update()

        # 鼠标事件
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.drawing = False

        self.status_var = tk.StringVar(value="就绪 | 鼠标左键拖动画线，松手自动优化 (起点固定为规划起点)")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)

        self.coord_var = tk.StringVar(value="坐标: ")
        coord_label = ttk.Label(control_frame, textvariable=self.coord_var, relief=tk.FLAT)
        coord_label.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)

    def insert_example(self):
        example_dist = "double Get_distance[50] = {0.000, 0.148, 0.297, 0.446, 0.596, 0.747, 0.898, 1.049, 1.201, 1.353,\n    1.505, 1.658, 1.811, 1.964, 2.117, 2.270, 2.424, 2.578, 2.732, 2.886,\n    3.040, 3.194, 3.348, 3.502, 3.656, 3.810, 3.964, 4.118, 4.272, 4.426,\n    4.580, 4.734, 4.888, 5.042, 5.196, 5.350, 5.504, 5.658, 5.812, 5.966,\n    6.120, 6.274, 6.428, 6.582, 6.736, 6.890, 7.044, 7.198, 7.352, 7.506};"
        example_yaw = "double Get_Yaw[50] = {5.41, 5.56, 5.70, 5.84, 5.96, 6.07, 6.16, 6.24, 6.30, 6.34,\n    6.36, 6.36, 6.34, 6.30, 6.24, 6.16, 6.06, 5.94, 5.80, 5.64,\n    5.46, 5.26, 5.04, 4.80, 4.54, 4.26, 3.96, 3.64, 3.31, 2.95,\n    2.58, 2.19, 1.78, 1.36, 0.92, 0.47, 0.01, -0.46, -0.94, -1.43,\n    -1.93, -2.44, -2.96, -3.49, -4.03, -4.57, -5.12, -5.68, -6.24, -6.81};"
        self.replay_distance_text.delete("1.0", tk.END)
        self.replay_distance_text.insert("1.0", example_dist)
        self.replay_yaw_text.delete("1.0", tk.END)
        self.replay_yaw_text.insert("1.0", example_yaw)

    def parse_array_data(self, text):
        match = re.search(r'\{([\s\S]*)\}', text)
        if match:
            text = match.group(1)
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'/\*[\s\S]*?\*/', '', text)
        nums = re.findall(r'[-+]?\d*\.?\d+', text)
        return [float(x) for x in nums]

    def parse_xy_from_text(self, text):
        """尝试从文本中解析X和Y坐标数组"""
        x_match = re.search(r'(?:Get_X|x_coords?|X)\s*\[\s*\d*\s*\]\s*=\s*\{([^}]+)\}', text, re.IGNORECASE)
        y_match = re.search(r'(?:Get_Y|y_coords?|Y)\s*\[\s*\d*\s*\]\s*=\s*\{([^}]+)\}', text, re.IGNORECASE)

        xs = []
        ys = []
        if x_match:
            xs = [float(x.strip()) for x in x_match.group(1).replace('\n','').split(',') if x.strip()]
        if y_match:
            ys = [float(y.strip()) for y in y_match.group(1).replace('\n','').split(',') if y.strip()]
        return xs, ys

    def parse_and_replay(self):
        """解析数据并复现（首次解析或重新解析）"""
        try:
            dist_text = self.replay_distance_text.get("1.0", tk.END)
            yaw_text = self.replay_yaw_text.get("1.0", tk.END)

            distances = self.parse_array_data(dist_text)
            angles_raw = self.parse_array_data(yaw_text)

            # 尝试解析XY数据
            xs, ys = self.parse_xy_from_text(dist_text + yaw_text)

            if len(distances) < 2:
                messagebox.showerror("错误", "Distance数据不足")
                return
            if len(angles_raw) < 2:
                messagebox.showerror("错误", "Yaw数据不足")
                return

            use_len = min(len(distances), len(angles_raw))
            if len(distances) != len(angles_raw):
                messagebox.showwarning("提示", f"距离和角度长度不一致，自动截取前{use_len}个点")

            # 存储原始数据
            self.replay_distances = distances[:use_len]
            self.replay_angles_raw = angles_raw[:use_len]
            self.replay_xy_data = (xs, ys) if len(xs) == use_len and len(ys) == use_len else ([], [])

            # 检测角度模式
            mode = self.replay_mode_var.get()
            if mode == 'auto':
                mode = self.detect_angle_mode(self.replay_angles_raw)
            self.replay_mode = mode

            # 转换角度
            if mode == 'mode1':
                angles_deg = np.array(self.replay_angles_raw)
            elif mode == 'mode2':
                angles_deg = np.array([a if a <= 180 else a - 360 for a in self.replay_angles_raw])
            else:
                angles_deg = np.array(self.replay_angles_raw)

            if self.invert_angle_var.get():
                angles_deg = -angles_deg

            self.replay_angles_deg = angles_deg

            # 生成路径
            self._generate_replay_path()

        except Exception as e:
            messagebox.showerror("错误", f"解析失败: {str(e)}")
            self.replay_info_var.set(f"解析失败: {str(e)}")

    def _generate_replay_path(self):
        """根据当前存储的数据和起点生成路径"""
        if len(self.replay_distances) < 2:
            return

        use_len = len(self.replay_distances)
        points = []

        # 判断起点来源
        if self.replay_start_mode.get() == 'from_data':
            xs, ys = self.replay_xy_data
            if len(xs) == use_len and len(ys) == use_len:
                points = [(xs[i], ys[i]) for i in range(use_len)]
            else:
                messagebox.showwarning("警告", "数据中未找到完整的XY坐标，将使用手动起点")
                start_x = self.replay_start_x.get()
                start_y = self.replay_start_y.get()
                points.append((start_x, start_y))
                for i in range(1, use_len):
                    step = self.replay_distances[i] - self.replay_distances[i-1]
                    rad = np.deg2rad(90 - self.replay_angles_deg[i-1])
                    dx = step * np.cos(rad)
                    dy = step * np.sin(rad)
                    points.append((points[-1][0] + dx, points[-1][1] + dy))
        else:
            # 使用手动起点
            start_x = self.replay_start_x.get()
            start_y = self.replay_start_y.get()
            points.append((start_x, start_y))
            for i in range(1, use_len):
                step = self.replay_distances[i] - self.replay_distances[i-1]
                rad = np.deg2rad(90 - self.replay_angles_deg[i-1])
                dx = step * np.cos(rad)
                dy = step * np.sin(rad)
                points.append((points[-1][0] + dx, points[-1][1] + dy))

        self.replay_points = points

        # 显示在画布上
        self.display_replay_path(points, self.replay_distances, self.replay_angles_deg, self.replay_mode)

        total_dist = self.replay_distances[-1] - self.replay_distances[0]
        start_info = f"复现起点: ({points[0][0]:.2f}, {points[0][1]:.2f})"
        mode_info = f"模式: {self.replay_mode}"
        self.replay_info_var.set(f"复现成功 | 点数:{use_len} | 总距离:{total_dist:.2f}m | {mode_info} | {start_info}")

    def draw_replay_reference(self, start_x=None, start_y=None):
        """绘制复现界面的参考圆、锥桶和起点标记（无路径时）"""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # 绘制八字圆和锥桶
        circle = plt.Circle((0, 0), self.outer_radius, color='gray', fill=False, linestyle='--', alpha=0.5)
        self.ax.add_patch(circle)
        cone1 = Circle(self.cone1_pos, self.cone_radius, color='red', alpha=0.7)
        cone2 = Circle(self.cone2_pos, self.cone_radius, color='red', alpha=0.7)
        self.ax.add_patch(cone1)
        self.ax.add_patch(cone2)

        # 获取起点坐标
        if start_x is None or start_y is None:
            start_x = self.replay_start_x.get()
            start_y = self.replay_start_y.get()

        # 绘制起点标记
        self.ax.scatter(start_x, start_y, c='red', s=100, marker='o', label='复现起点', zorder=5)

        # 自动调整视图范围（包含所有参考元素）
        all_x = [self.cone1_pos[0], self.cone2_pos[0], -self.outer_radius, self.outer_radius, start_x]
        all_y = [self.cone1_pos[1], self.cone2_pos[1], -self.outer_radius, self.outer_radius, start_y]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        margin = max((x_max - x_min) * 0.2, (y_max - y_min) * 0.2, 1.0)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        half_size = max(x_max - x_min, y_max - y_min) / 2 + margin
        self.ax.set_xlim(center_x - half_size, center_x + half_size)
        self.ax.set_ylim(center_y - half_size, center_y + half_size)

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("路径复现结果 (等待数据)")
        self.ax.legend(loc='upper right')
        self.canvas.draw()

    def update_replay_start(self):
        """更新复现起点（强制使用手动起点，立即显示新起点位置）"""
        # 强制使用手动起点模式
        self.replay_start_mode.set('manual')

        # 获取当前手动起点坐标
        start_x = self.replay_start_x.get()
        start_y = self.replay_start_y.get()

        # 如果没有距离数据，只显示起点标记
        if len(self.replay_distances) < 2:
            self.draw_replay_reference(start_x, start_y)
            self.replay_info_var.set(f"起点已更新为 ({start_x:.2f}, {start_y:.2f})，请解析数据以生成完整路径。")
            return

        # 已有数据，重新生成完整路径
        self._generate_replay_path()
        self.replay_info_var.set(f"起点已更新为 ({start_x:.2f}, {start_y:.2f})，共 {len(self.replay_points)} 个路径点")

    def apply_angle_option(self):
        """应用角度取反选项"""
        if len(self.replay_distances) >= 2:
            # 重新转换角度
            if self.replay_mode == 'mode1':
                angles_deg = np.array(self.replay_angles_raw)
            elif self.replay_mode == 'mode2':
                angles_deg = np.array([a if a <= 180 else a - 360 for a in self.replay_angles_raw])
            else:
                angles_deg = np.array(self.replay_angles_raw)

            if self.invert_angle_var.get():
                angles_deg = -angles_deg

            self.replay_angles_deg = angles_deg
            self._generate_replay_path()

    def detect_angle_mode(self, angles):
        angles = np.array(angles)
        ang_min, ang_max = np.min(angles), np.max(angles)
        if ang_min >= -180 and ang_max <= 180:
            diffs = np.abs(np.diff(angles))
            if np.any(diffs > 180):
                return 'mode3'
            return 'mode1'
        elif ang_min >= 0 and ang_max <= 360:
            return 'mode2'
        else:
            return 'mode3'

    def display_replay_path(self, points, distances, angles, mode):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # ------------------- 始终绘制锥桶和八字圆参考线 -------------------
        circle = plt.Circle((0, 0), self.outer_radius, color='gray', fill=False, linestyle='--', alpha=0.5)
        self.ax.add_patch(circle)
        cone1 = Circle(self.cone1_pos, self.cone_radius, color='red', alpha=0.7)
        cone2 = Circle(self.cone2_pos, self.cone_radius, color='red', alpha=0.7)
        self.ax.add_patch(cone1)
        self.ax.add_patch(cone2)

        # 绘制复现路径
        pts = np.array(points)
        self.ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=2.5, label='复现路径')
        self.ax.scatter(pts[0, 0], pts[0, 1], c='red', s=100, marker='o', label='起点', zorder=5)
        self.ax.scatter(pts[-1, 0], pts[-1, 1], c='orange', s=100, marker='s', label='终点', zorder=5)

        # 方向箭头
        step = max(len(points) // 20, 1)
        for i in range(0, len(points) - 1, step):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            if np.hypot(dx, dy) > 0.05:
                self.ax.arrow(points[i][0], points[i][1], dx * 0.5, dy * 0.5,
                              head_width=0.05, head_length=0.08, fc='blue', ec='blue', alpha=0.6)

        # 自动调整视图范围，同时包含路径、锥桶和圆
        all_x = list(pts[:, 0]) + [self.cone1_pos[0], self.cone2_pos[0], -self.outer_radius, self.outer_radius]
        all_y = list(pts[:, 1]) + [self.cone1_pos[1], self.cone2_pos[1], -self.outer_radius, self.outer_radius]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        margin = max((x_max - x_min) * 0.2, (y_max - y_min) * 0.2, 1.0)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        half_size = max(x_max - x_min, y_max - y_min) / 2 + margin
        self.ax.set_xlim(center_x - half_size, center_x + half_size)
        self.ax.set_ylim(center_y - half_size, center_y + half_size)

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("路径复现结果")

        info = f"点数: {len(points)}\n总距离: {distances[-1] - distances[0]:.2f}m\n模式: {mode}"
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.ax.legend(loc='upper right')
        self.canvas.draw()

    def save_replay_png(self):
        if not self.replay_points:
            messagebox.showwarning("提示", "请先解析并复现路径")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")]
        )
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"图片已保存: {filename}")

    def import_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV文件", "*.csv"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if not filename:
            return
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            # 尝试解析
            distances = self.parse_array_data(content)
            if len(distances) > 0:
                self.replay_distance_text.delete("1.0", tk.END)
                self.replay_distance_text.insert("1.0", f"double distance[{len(distances)}] = {{{', '.join(f'{d:.3f}' for d in distances)}}};")
            messagebox.showinfo("提示", "CSV导入完成，请继续输入角度数据或编辑")
        except Exception as e:
            messagebox.showerror("错误", f"导入失败: {str(e)}")

    # ---------- 绘图交互 ----------
    def update_plot(self):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("八字绕行路径规划 (正北0°, 顺时针为正)")

        circle = plt.Circle((0,0), self.outer_radius, color='gray', fill=False, linestyle='--', alpha=0.5)
        self.ax.add_patch(circle)
        cone1 = Circle(self.cone1_pos, self.cone_radius, color='red', alpha=0.7, label='锥桶')
        cone2 = Circle(self.cone2_pos, self.cone_radius, color='red', alpha=0.7)
        self.ax.add_patch(cone1)
        self.ax.add_patch(cone2)

        if len(self.raw_points) > 1:
            raw_arr = np.array(self.raw_points)
            self.ax.plot(raw_arr[:,0], raw_arr[:,1], 'b-', linewidth=1.5, alpha=0.5, label='手绘原始路径')

        if len(self.opt_points) > 1:
            opt_arr = np.array(self.opt_points)
            self.ax.plot(opt_arr[:,0], opt_arr[:,1], 'g-', linewidth=2.5, label='优化后最优路径')
            self.ax.scatter(opt_arr[:,0], opt_arr[:,1], c='green', s=30, alpha=0.7, marker='s', label='优化路径点位')
            self.ax.plot(opt_arr[0,0], opt_arr[0,1], 'go', markersize=8, label='规划起点')

        self.ax.plot(self.start_point[0], self.start_point[1], 'ro', markersize=6, label='设定规划起点')
        self.ax.legend(loc='upper right')
        self.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.raw_points = []
        self.opt_points = []
        # 强制从规划起点开始绘制
        self.raw_points.append(self.start_point)
        # 如果鼠标点击位置有效且不是起点，也添加进去，形成第一条线段
        if event.xdata is not None and event.ydata is not None:
            # 避免重复添加同一个点
            if (event.xdata, event.ydata) != self.start_point:
                self.raw_points.append((event.xdata, event.ydata))
        self.update_plot()
        self.status_var.set("从规划起点开始绘制，松手自动优化")

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        if event.xdata is not None and event.ydata is not None:
            self.raw_points.append((event.xdata, event.ydata))
        if len(self.raw_points) % 5 == 0:
            self.update_plot()

    def on_release(self, event):
        if self.drawing:
            self.drawing = False
            # 如果绘制的点数太少（可能只点了起点一个点），则补充一个点或提示
            if len(self.raw_points) < 2:
                self.status_var.set("绘制路径点不足，请至少画出一条线段")
                # 可以忽略优化，保留原状态
                self.update_plot()
                return
            self.update_plot()
            self.status_var.set("手绘完成，正在自动优化...")
            self.master.update()
            self.optimize_and_generate()
            self.status_var.set(f"自动优化完成，共{len(self.opt_points)}个路径点")

    def clear_raw_path(self):
        self.raw_points = []
        self.opt_points = []
        self.distance_arr = []
        self.yaw_arr = []
        self.update_plot()
        self.status_var.set("已清除手绘路径")

    def update_cone_radius(self):
        self.cone_radius = self.cone_radius_var.get()
        self.update_plot()

    def update_cone_pos(self):
        y1 = self.cone1_y_var.get()
        y2 = self.cone2_y_var.get()
        self.cone1_pos = (0, y1)
        self.cone2_pos = (0, y2)
        self.update_plot()

    def set_start_point(self):
        """设置规划起点（仅影响规划功能）"""
        x = self.start_x_var.get()
        y = self.start_y_var.get()
        self.start_point = (x, y)
        self.update_plot()
        self.status_var.set(f"规划起点已设置为 ({x:.2f}, {y:.2f})")

    def apply_angle_mode(self):
        self.angle_mode = self.angle_mode_var.get()
        if len(self.opt_points) > 1:
            self.generate_angle_distance_from_opt()
        mode_names = {'mode1':'模式1(-180~180)', 'mode2':'模式2(0~360)', 'mode3':'模式3(累积)'}
        self.status_var.set(f"角度模式切换至 {mode_names[self.angle_mode]}")

    # ---------- 核心优化 ----------
    def optimize_and_generate(self):
        if len(self.raw_points) < 2:
            messagebox.showwarning("警告", "请先绘制路径")
            return

        max_angle_deg = self.max_angle_var.get()
        single_angle_deg = self.single_angle_var.get()
        passes = self.opt_passes_var.get()
        smooth_sigma = self.smooth_sigma_var.get()
        fix_end = self.fix_endpoints_var.get()
        min_step = self.min_step_var.get()
        max_step = self.max_step_var.get()
        sens = self.curvature_sens_var.get()
        target_cnt = self.target_count_var.get()

        pts = np.array(self.raw_points)
        if len(pts) > 10 and smooth_sigma > 0:
            try:
                tck, u = splprep([pts[:,0], pts[:,1]], s=0.5, per=False)
                u_new = np.linspace(0, 1, 500)
                x_smooth, y_smooth = splev(u_new, tck)
                smooth_pts = np.vstack([x_smooth, y_smooth]).T
            except:
                x_s = gaussian_filter1d(pts[:,0], sigma=smooth_sigma)
                y_s = gaussian_filter1d(pts[:,1], sigma=smooth_sigma)
                smooth_pts = np.vstack([x_s, y_s]).T
        else:
            smooth_pts = pts

        if target_cnt > 0:
            opt_path = self.resample_to_fixed_count(smooth_pts, target_cnt)
        else:
            curvature = self.compute_curvature(smooth_pts)
            opt_path = self.adaptive_resample(smooth_pts, curvature, min_step, max_step, sens)

        if len(opt_path) < 2:
            self.status_var.set("优化失败，重采样后路径过短")
            return

        for _ in range(passes):
            opt_path = self.limit_turning_angle(opt_path, max_angle_deg)
            opt_path = self.limit_turning_angle(opt_path, single_angle_deg)
            opt_path = self.avoid_cones(opt_path, self.cone_radius + 0.1)
            if len(opt_path) > 10 and smooth_sigma > 0:
                if fix_end:
                    x_org = opt_path[:, 0]
                    y_org = opt_path[:, 1]
                    x_sm = gaussian_filter1d(x_org, sigma=smooth_sigma * 0.5)
                    y_sm = gaussian_filter1d(y_org, sigma=smooth_sigma * 0.5)
                    x_sm[0], x_sm[-1] = x_org[0], x_org[-1]
                    y_sm[0], y_sm[-1] = y_org[0], y_org[-1]
                    opt_path = np.column_stack([x_sm, y_sm])
                else:
                    x_sm = gaussian_filter1d(opt_path[:, 0], sigma=smooth_sigma * 0.5)
                    y_sm = gaussian_filter1d(opt_path[:, 1], sigma=smooth_sigma * 0.5)
                    opt_path = np.column_stack([x_sm, y_sm])

        # 起点修正：因为 raw_points[0] 已经固定为规划起点，这里只需确保对齐（实际上应该已经对齐）
        if fix_end and len(self.raw_points) >= 2:
            # 使用手绘起点（即规划起点）平移优化路径，确保起点精确匹配
            dx_start = self.raw_points[0][0] - opt_path[0][0]
            dy_start = self.raw_points[0][1] - opt_path[0][1]
            opt_path = opt_path + np.array([dx_start, dy_start])
        else:
            if len(opt_path) > 0:
                dx = self.start_point[0] - opt_path[0][0]
                dy = self.start_point[1] - opt_path[0][1]
                opt_path = opt_path + np.array([dx, dy])

        self.opt_points = opt_path.tolist()
        self.generate_angle_distance_from_opt()
        self.update_plot()

    def resample_to_fixed_count(self, points, n):
        if len(points) < 2:
            return points
        dists = [0]
        for i in range(1, len(points)):
            dists.append(dists[-1] + np.linalg.norm(points[i]-points[i-1]))
        total = dists[-1]
        if total == 0:
            return points[:n]
        new_pts = []
        for i in range(n):
            t = i / (n-1) if n>1 else 0
            target_dist = t * total
            idx = np.searchsorted(dists, target_dist)
            if idx == 0:
                new_pts.append(points[0])
            elif idx >= len(points):
                new_pts.append(points[-1])
            else:
                d0 = dists[idx-1]
                d1 = dists[idx]
                r = (target_dist - d0) / (d1 - d0) if d1>d0 else 0
                pt = points[idx-1] + r * (points[idx] - points[idx-1])
                new_pts.append(pt)
        return np.array(new_pts)

    def compute_curvature(self, points):
        if len(points) < 3:
            return np.zeros(len(points))
        curv = np.zeros(len(points))
        for i in range(1, len(points)-1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1<1e-6 or len2<1e-6:
                continue
            ang1 = np.arctan2(v1[1], v1[0])
            ang2 = np.arctan2(v2[1], v2[0])
            delta = ang2 - ang1
            if delta > np.pi: delta -= 2*np.pi
            if delta < -np.pi: delta += 2*np.pi
            arc = (len1+len2)/2
            curv[i] = abs(delta)/arc if arc>0 else 0
        curv[0] = curv[1] if len(points)>1 else 0
        curv[-1] = curv[-2] if len(points)>1 else 0
        return gaussian_filter1d(curv, sigma=1.0)

    def adaptive_resample(self, points, curvature, min_step, max_step, sensitivity):
        if len(points) < 2:
            return points
        dists = [0]
        for i in range(1, len(points)):
            dists.append(dists[-1] + np.linalg.norm(points[i]-points[i-1]))
        total = dists[-1]
        if total == 0:
            return points
        max_curv = max(curvature) if max(curvature) > 0 else 1.0
        new_pts = [points[0]]
        cur_dist = 0
        while cur_dist < total:
            idx = self.find_index_by_distance(dists, cur_dist)
            if idx >= len(curvature):
                idx = len(curvature)-1
            norm_c = min(1.0, curvature[idx]/max_curv * sensitivity)
            step = max_step - (max_step-min_step)*norm_c
            step = max(min_step, min(max_step, step))
            next_dist = min(total, cur_dist + step)
            new_pts.append(self.interpolate_point(points, dists, next_dist))
            cur_dist = next_dist
        return np.array(new_pts)

    def find_index_by_distance(self, dists, target):
        for i, d in enumerate(dists):
            if d >= target:
                return i
        return len(dists)-1

    def interpolate_point(self, points, dists, target):
        if target <= dists[0]: return points[0]
        if target >= dists[-1]: return points[-1]
        for i in range(len(dists)-1):
            if dists[i] <= target <= dists[i+1]:
                t = (target-dists[i])/(dists[i+1]-dists[i]) if dists[i+1]>dists[i] else 0
                return points[i] + t*(points[i+1]-points[i])
        return points[-1]

    def limit_turning_angle(self, path, max_deg):
        if len(path) < 3:
            return path
        path = path.copy()
        max_rad = np.deg2rad(max_deg)
        for _ in range(3):
            new_path = path.copy()
            for i in range(1, len(path)-1):
                v1 = path[i]-path[i-1]
                v2 = path[i+1]-path[i]
                if np.linalg.norm(v1)<1e-6 or np.linalg.norm(v2)<1e-6:
                    continue
                ang1 = np.arctan2(v1[1], v1[0])
                ang2 = np.arctan2(v2[1], v2[0])
                delta = ang2-ang1
                if delta > np.pi: delta -= 2*np.pi
                if delta < -np.pi: delta += 2*np.pi
                if abs(delta) > max_rad:
                    new_ang = ang1 + np.clip(delta, -max_rad, max_rad)
                    step = np.linalg.norm(v2)
                    new_v2 = step * np.array([np.cos(new_ang), np.sin(new_ang)])
                    new_path[i+1] = path[i] + new_v2
            path = new_path
        return path

    def avoid_cones(self, path, min_dist):
        cones = [self.cone1_pos, self.cone2_pos]
        new_path = path.copy()
        for i, pt in enumerate(new_path):
            for cone in cones:
                vec = pt - np.array(cone)
                d = np.linalg.norm(vec)
                if d < min_dist and d > 0:
                    push_dir = vec / d
                    new_path[i] = pt + push_dir * (min_dist - d + 0.02)
        return new_path

    # ---------- 角度与距离生成 ----------
    def generate_angle_distance_from_opt(self):
        if len(self.opt_points) < 2:
            self.distance_arr = []
            self.yaw_arr = []
            return
        pts = np.array(self.opt_points)
        dist = [0.0]
        for i in range(1, len(pts)):
            dist.append(dist[-1] + np.linalg.norm(pts[i]-pts[i-1]))
        self.distance_arr = dist

        yaw_abs = []
        for i in range(len(pts)):
            if i == 0:
                dx = pts[1][0]-pts[0][0]
                dy = pts[1][1]-pts[0][1]
            elif i == len(pts)-1:
                dx = pts[-1][0]-pts[-2][0]
                dy = pts[-1][1]-pts[-2][1]
            else:
                dx = pts[i+1][0]-pts[i-1][0]
                dy = pts[i+1][1]-pts[i-1][1]
            ang = self.vector_to_clockwise_angle(dx, dy)
            yaw_abs.append(ang)

        if self.angle_mode == 'mode1':
            self.yaw_arr = yaw_abs
        elif self.angle_mode == 'mode2':
            self.yaw_arr = [a if a>=0 else 360+a for a in yaw_abs]
        else:
            cum = yaw_abs[0]
            cum_angles = [cum]
            for i in range(1, len(yaw_abs)):
                delta = yaw_abs[i] - yaw_abs[i-1]
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360
                cum += delta
                cum_angles.append(cum)
            self.yaw_arr = cum_angles

    def vector_to_clockwise_angle(self, dx, dy):
        ang_rad = np.arctan2(dx, dy)
        return np.rad2deg(ang_rad)

    # ---------- 数据显示 ----------
    def show_data_dialog(self):
        if len(self.distance_arr) == 0:
            messagebox.showwarning("警告", "没有数据，请先绘制并优化路径")
            return
        dialog = Toplevel(self.master)
        dialog.title("路径数据 - 距离和角度")
        dialog.geometry("700x500")
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar = Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget = Text(text_frame, wrap=tk.NONE, yscrollcommand=scrollbar.set, font=("Courier", 9))
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        content = self.generate_data_content()
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        def copy():
            self.master.clipboard_clear()
            self.master.clipboard_append(content)
            messagebox.showinfo("成功", "数据已复制到剪贴板")
        ttk.Button(button_frame, text="复制到剪贴板", command=copy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def print_data_to_console(self):
        if len(self.distance_arr) == 0:
            print("无数据，请先优化路径")
            return
        print(self.generate_data_content())

    def generate_data_content(self):
        mode_name = {'mode1':'模式1: -180°~180°', 'mode2':'模式2: 0°~360°', 'mode3':'模式3: 累积'}[self.angle_mode]
        n = len(self.distance_arr)
        content = "="*90 + "\n"
        content += f"优化后路径数据 | 角度模式: {mode_name}\n"
        content += f"路径点数: {n} | 规划起点: ({self.start_point[0]:.2f}, {self.start_point[1]:.2f})\n"
        content += "-"*90 + "\n"
        content += f"{'序号':<6} {'距离(m)':<12} {'角度(°)':<12} {'X(m)':<12} {'Y(m)':<12}\n"
        content += "-"*90 + "\n"
        for i, (d, y) in enumerate(zip(self.distance_arr, self.yaw_arr)):
            if i < len(self.opt_points):
                x, yp = self.opt_points[i]
                content += f"{i+1:<6} {d:<12.3f} {y:<12.2f} {x:<12.3f} {yp:<12.3f}\n"
        content += "-"*90 + "\n\n"
        content += f"double Get_distance[{n}] = {{\n    "
        for i, d in enumerate(self.distance_arr):
            content += f"{d:.3f}"
            if i < n-1:
                content += ", "
            if (i+1)%10==0 and i<n-1:
                content += "\n    "
        content += "\n};\n\n"
        content += f"double Get_Yaw[{n}] = {{\n    "
        for i, y in enumerate(self.yaw_arr):
            content += f"{y:.2f}"
            if i < n-1:
                content += ", "
            if (i+1)%10==0 and i<n-1:
                content += "\n    "
        content += "\n};\n"
        content += "="*90
        return content


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCarPathPlanner(root)
    root.mainloop()