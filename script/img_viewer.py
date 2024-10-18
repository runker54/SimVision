import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import pandas as pd
import os
import logging
import shutil
from threading import Thread
import aiohttp
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedMediaViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("高级媒体查看器和标注工具")
        self.root.geometry("1900x1000")

        self.df = None
        self.filtered_data = {}
        self.current_id_index = 0
        self.file_path = None
        self.zoom_level = 1.0
        self.rotation_angles = {}  # 用于存储每张图片的旋转角度
        
        # 新增：用于存储相似和不相似图片的路径
        self.similar_path = ""
        self.dissimilar_path = ""

        self.image_checkboxes = []
        self.image_data = {}  # 修改：使用 (id, index) 作为键
        self.displayed_images = {}  # 用于存储当前显示的图片引用

        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.image_cache = {}

        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Excel/CSV viewer page
        excel_frame = ttk.Frame(self.notebook)
        self.notebook.add(excel_frame, text='Excel/CSV Viewer')
        self.create_excel_viewer_widgets(excel_frame)

        # Manual URL input page
        manual_frame = ttk.Frame(self.notebook)
        self.notebook.add(manual_frame, text='Manual URL Input')
        self.create_manual_input_widgets(manual_frame)

        # 新增：标注功能的按钮
        self.create_annotation_widgets(excel_frame)

    def create_excel_viewer_widgets(self, parent):
        # File selection and filter frame
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=5)

        ttk.Button(top_frame, text="Select File", command=self.select_file).grid(row=0, column=0, padx=5)
        self.file_label = ttk.Label(top_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, padx=5)
        ttk.Button(top_frame, text="Load Data", command=self.load_data).grid(row=0, column=2, padx=5)

        ttk.Label(top_frame, text="ID Column:").grid(row=0, column=3, padx=5)
        self.id_column_var = tk.StringVar()
        self.id_column_menu = ttk.Combobox(top_frame, textvariable=self.id_column_var, state="readonly", width=15)
        self.id_column_menu.grid(row=0, column=4, padx=5)

        ttk.Label(top_frame, text="URL Columns:").grid(row=0, column=5, padx=5)
        self.url_columns_var = tk.StringVar()
        self.url_columns_listbox = tk.Listbox(top_frame, listvariable=self.url_columns_var, selectmode=tk.MULTIPLE, width=20, height=4)
        self.url_columns_listbox.grid(row=0, column=6, padx=5, rowspan=2)

        ttk.Label(top_frame, text="Filter Column:").grid(row=1, column=0, padx=5, pady=5)
        self.filter_column_var = tk.StringVar()
        self.filter_column_menu = ttk.Combobox(top_frame, textvariable=self.filter_column_var, state="readonly", width=15)
        self.filter_column_menu.grid(row=1, column=1, padx=5)
        self.filter_column_menu.bind("<<ComboboxSelected>>", self.update_filter_values)

        ttk.Label(top_frame, text="Filter Value:").grid(row=1, column=2, padx=5)
        self.filter_value_var = tk.StringVar()
        self.filter_value_menu = ttk.Combobox(top_frame, textvariable=self.filter_value_var, state="readonly", width=15)
        self.filter_value_menu.grid(row=1, column=3, padx=5)
        
        # 添加提示标签
        ttk.Label(top_frame, text="(提示: 选择 [...])").grid(row=1, column=4, padx=5, sticky='w')

        ttk.Label(top_frame, text="Specific ID:").grid(row=1, column=4, padx=5)
        self.specific_id_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.specific_id_var, width=15).grid(row=1, column=5, padx=5)

        ttk.Button(top_frame, text="Apply Filters", command=self.apply_filters).grid(row=1, column=8, padx=5)

        # Navigation frame
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=5)
        ttk.Button(nav_frame, text="Previous ID", command=self.prev_id).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next ID", command=self.next_id).pack(side=tk.LEFT, padx=5)
        self.id_label = ttk.Label(nav_frame, text="ID: ")
        self.id_label.pack(side=tk.LEFT, padx=5)

        # 创建一个框架来容纳画布和滚动条
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 创建画布和滚动条
        self.canvas = tk.Canvas(self.canvas_frame)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 添加缩放控制
        zoom_frame = ttk.Frame(parent)
        zoom_frame.pack(fill=tk.X, pady=5)
        ttk.Label(zoom_frame, text="缩放:").pack(side=tk.LEFT)
        self.zoom_scale = ttk.Scale(zoom_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, command=self.update_zoom)
        self.zoom_scale.set(1.0)
        self.zoom_scale.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def create_manual_input_widgets(self, parent):
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=10)

        self.url_entries = []
        for _ in range(2):
            self.add_url_entry(input_frame)

        ttk.Button(input_frame, text="Add URL", command=lambda: self.add_url_entry(input_frame)).pack(pady=5)
        ttk.Button(input_frame, text="Display Images", command=self.display_manual_images).pack(pady=5)

        self.manual_canvas = tk.Canvas(parent)
        self.manual_canvas.pack(fill=tk.BOTH, expand=True)

    def add_url_entry(self, parent):
        entry = ttk.Entry(parent, width=50)
        entry.pack(pady=2)
        self.url_entries.append(entry)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        if self.file_path:
            self.file_label.config(text=os.path.basename(self.file_path))

    def load_data(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file first.")
            return

        try:
            if self.file_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.file_path)
            else:
                self.df = pd.read_csv(self.file_path)
            
            self.update_column_menus()
            messagebox.showinfo("Success", f"Data loaded successfully. {len(self.df)} rows found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def update_column_menus(self):
        all_columns = list(self.df.columns)
        self.id_column_menu['values'] = all_columns
        self.url_columns_listbox.delete(0, tk.END)
        for col in all_columns:
            self.url_columns_listbox.insert(tk.END, col)
        self.filter_column_menu['values'] = all_columns

    def update_filter_values(self, event):
        column = self.filter_column_var.get()
        if column:
            unique_values = self.df[column].astype(str).unique()
            # 如果只有两个值，将它们组合成一个字符串
            if len(unique_values) == 2:
                combined_value = f"[{', '.join(unique_values)}]"
                self.filter_value_menu['values'] = [combined_value] + list(unique_values)
            else:
                self.filter_value_menu['values'] = unique_values

    def apply_filters(self):
        id_column = self.id_column_var.get()
        url_columns = [self.url_columns_listbox.get(idx) for idx in self.url_columns_listbox.curselection()]
        filter_column = self.filter_column_var.get()
        filter_value = self.filter_value_var.get()
        specific_id = self.specific_id_var.get()

        if not id_column or not url_columns:
            messagebox.showerror("Error", "Please select both ID and URL columns.")
            return

        filtered_df = self.df

        if filter_column and filter_value:
            filtered_df[filter_column] = filtered_df[filter_column].astype(str).str.strip()
            filter_value = str(filter_value).strip()
            
            # 修改：移除方括号并分割多个值
            if filter_value.startswith('[') and filter_value.endswith(']'):
                filter_values = [v.strip() for v in filter_value[1:-1].split(',')]
            else:
                filter_values = [filter_value]
            
            # 使用 isin 进行筛选
            filtered_df = filtered_df[filtered_df[filter_column].isin(filter_values)]

        self.filtered_data = {}
        for _, row in filtered_df.iterrows():
            id_value = str(row[id_column]).strip()
            urls = []
            for col in url_columns:
                if pd.notna(row[col]):
                    urls.extend(str(row[col]).split(','))  # 假设多个URL可能用逗号分隔
            urls = [url.strip() for url in urls if url.strip()]
            if urls:
                if id_value in self.filtered_data:
                    self.filtered_data[id_value].extend(urls)
                else:
                    self.filtered_data[id_value] = urls

        if specific_id:
            specific_id = str(specific_id).strip().lower()
            matching_ids = [id for id in self.filtered_data.keys() if specific_id in id.lower()]
            if matching_ids:
                self.filtered_data = {id: self.filtered_data[id] for id in matching_ids}
            else:
                messagebox.showwarning("Warning", f"ID containing '{specific_id}' not found in filtered data.")
                return

        self.current_id_index = 0
        self.update_id_display()
        messagebox.showinfo("Filter Applied", f"Filter applied. {len(self.filtered_data)} IDs found.")

    def update_id_display(self):
        if self.filtered_data:
            current_id = list(self.filtered_data.keys())[self.current_id_index]
            self.id_label.config(text=f"ID: {current_id}")
            self.clear_canvas()  # 添加这行来清除之前的图片
            self.display_current_media()
        else:
            self.id_label.config(text="No data available")
            self.clear_canvas()

    def display_current_media(self):
        if not self.filtered_data:
            return

        current_id = list(self.filtered_data.keys())[self.current_id_index]
        urls = self.filtered_data[current_id]
        self.display_images(self.canvas, urls, current_id)

    def display_manual_images(self):
        self.manual_canvas.delete("all")
        urls = [entry.get() for entry in self.url_entries if entry.get()]
        self.display_images(self.manual_canvas, urls)

    def display_images(self, canvas, urls, current_id=None):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.image_checkboxes = []
        self.displayed_images = {}
        self.rotation_angles = {}

        # 使用异步方法加载图片
        loop = asyncio.get_event_loop()
        images = loop.run_until_complete(self.load_images(urls))

        for i, image in enumerate(images):
            try:
                image_key = (current_id, i) if current_id is not None else i

                if image_key not in self.image_data:
                    self.image_data[image_key] = image
                else:
                    image = self.image_data[image_key]

                self.rotation_angles[i] = 0
                display_image = image.copy()
                display_image.thumbnail((int(400 * self.zoom_level), int(400 * self.zoom_level)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
                self.displayed_images[i] = photo

                frame = ttk.Frame(self.scrollable_frame)
                frame.grid(row=i // 4, column=i % 4, padx=5, pady=5)

                label = ttk.Label(frame, image=photo)
                label.pack()
                label.bind("<Button-1>", lambda e, img=image, idx=i: self.show_full_image(img, idx))

                ttk.Label(frame, text=f"Image {i+1}").pack()

                var = tk.BooleanVar()
                checkbox = ttk.Checkbutton(frame, variable=var)
                checkbox.pack(side=tk.LEFT)

                rotate_button = ttk.Button(frame, text="旋转", command=lambda idx=i: self.rotate_image(idx))
                rotate_button.pack(side=tk.RIGHT)

                self.image_checkboxes.append(var)

            except Exception as e:
                logging.error(f"加载图片失败，URL: {image}")
                logging.error(str(e))

    async def create_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    @lru_cache(maxsize=100)
    async def load_image(self, url):
        logging.info(f"尝试从以下地址加载图片: {url}")
        try:
            if url.startswith(('http://', 'https://')):
                await self.create_session()
                async with self.session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    content = await response.read()
                    img = await asyncio.get_event_loop().run_in_executor(
                        self.executor, lambda: Image.open(BytesIO(content))
                    )
            else:
                if not os.path.isabs(url):
                    url = os.path.abspath(url)
                logging.info(f"尝试打开本地文件: {url}")
                img = await asyncio.get_event_loop().run_in_executor(
                    self.executor, Image.open, url
                )
            return img
        except Exception as e:
            logging.error(f"加载图片时出错: {str(e)}")
            raise

    async def load_images(self, urls):
        tasks = [self.load_image(url) for url in urls]
        return await asyncio.gather(*tasks)

    def prev_id(self):
        if self.filtered_data and self.current_id_index > 0:
            self.current_id_index -= 1
            self.update_id_display()

    def next_id(self):
        if self.filtered_data and self.current_id_index < len(self.filtered_data) - 1:
            self.current_id_index += 1
            self.update_id_display()

    def clear_canvas(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.image_data.clear()  # 清除之前加载的图片数据
        self.displayed_images.clear()
        self.rotation_angles.clear()

    def create_annotation_widgets(self, parent):
        annotation_frame = ttk.Frame(parent)
        annotation_frame.pack(fill=tk.X, pady=5)

        ttk.Button(annotation_frame, text="选择相似图片路径", command=self.select_similar_path).pack(side=tk.LEFT, padx=5)
        ttk.Button(annotation_frame, text="选择不相似图片路径", command=self.select_dissimilar_path).pack(side=tk.LEFT, padx=5)
        ttk.Button(annotation_frame, text="标记为相似", command=self.mark_as_similar).pack(side=tk.LEFT, padx=5)
        ttk.Button(annotation_frame, text="标记为不相似", command=self.mark_as_dissimilar).pack(side=tk.LEFT, padx=5)

    def select_similar_path(self):
        self.similar_path = filedialog.askdirectory(title="选择存放相似图片的路径")
        if self.similar_path:
            messagebox.showinfo("成功", f"已选择相似图片路径：{self.similar_path}")

    def select_dissimilar_path(self):
        self.dissimilar_path = filedialog.askdirectory(title="选择存放不相似图片的路径")
        if self.dissimilar_path:
            messagebox.showinfo("成功", f"已选择不相似图片路径：{self.dissimilar_path}")

    def mark_as_similar(self):
        self.mark_images(self.similar_path)

    def mark_as_dissimilar(self):
        self.mark_images(self.dissimilar_path)

    def mark_images(self, target_path):
        if not target_path:
            messagebox.showerror("错误", "请先选择目标路径")
            return

        if not self.filtered_data:
            messagebox.showerror("错误", "没有可标注的图片")
            return

        current_id = list(self.filtered_data.keys())[self.current_id_index]
        urls = self.filtered_data[current_id]

        id_folder = os.path.join(target_path, current_id)
        os.makedirs(id_folder, exist_ok=True)

        selected_images = [i for i, var in enumerate(self.image_checkboxes) if var.get()]
        total_images = len(selected_images)

        progress = ttk.Progressbar(self.root, length=300, mode='determinate')
        progress.pack(pady=10)

        def save_images():
            for i, index in enumerate(selected_images):
                try:
                    image_key = (current_id, index)
                    if image_key in self.image_data:
                        image = self.image_data[image_key]
                        file_name = f"image_{index+1}.jpg"
                        file_path = os.path.join(id_folder, file_name)
                        image = image.rotate(self.rotation_angles.get(index, 0), expand=True)
                        image.save(file_path, "JPEG")
                        logging.info(f"已保存图片{file_path}")
                    else:
                        logging.warning(f"图片数据不存在：ID {current_id}, 索引 {index}")
                    
                    # 更新进度条
                    progress['value'] = (i + 1) / total_images * 100
                    self.root.update_idletasks()
                except Exception as e:
                    logging.error(f"保存图片时出错：{str(e)}")

            messagebox.showinfo("成功", f"已将选中的图片标注并保存到 {id_folder}")
            progress.destroy()

        Thread(target=save_images).start()

    def update_zoom(self, value):
        self.zoom_level = float(value)
        self.update_displayed_images()

    def update_displayed_images(self):
        for i, (frame, image) in enumerate(zip(self.scrollable_frame.winfo_children(), self.image_data.values())):
            display_image = image.copy()
            display_image = display_image.rotate(self.rotation_angles[i], expand=True)
            display_image.thumbnail((int(300 * self.zoom_level), int(300 * self.zoom_level)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            self.displayed_images[i] = photo

            label = frame.winfo_children()[0]
            if isinstance(label, ttk.Label):
                label.configure(image=photo)

        # 强制更新画布
        self.canvas.update_idletasks()

    def show_full_image(self, image, index):
        top = tk.Toplevel(self.root)
        top.title("图片查看")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # 创建一个框架来容纳画布和滚动条
        frame = ttk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True)

        # 创建画布和滚动条
        canvas = tk.Canvas(frame, width=min(image.width, screen_width-100), height=min(image.height, screen_height-100))
        h_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)

        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # 添加缩放控制
        zoom_frame = ttk.Frame(top)
        zoom_frame.pack(fill=tk.X, pady=5)
        ttk.Label(zoom_frame, text="缩放:").pack(side=tk.LEFT)
        zoom_scale = ttk.Scale(zoom_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, command=lambda v: self.update_full_image(canvas, image, index, float(v)))
        zoom_scale.set(1.0)
        zoom_scale.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 添加旋转按钮
        rotate_button = ttk.Button(zoom_frame, text="旋转", command=lambda: self.rotate_full_image(canvas, image, index))
        rotate_button.pack(side=tk.RIGHT)

        self.update_full_image(canvas, image, index, 1.0)

    def update_full_image(self, canvas, image, index, zoom):
        rotated_image = image.rotate(self.rotation_angles[index], expand=True)
        display_image = rotated_image.copy()
        display_image = display_image.resize((int(image.width * zoom), int(image.height * zoom)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # 保持对图片的引用

        canvas.configure(scrollregion=canvas.bbox("all"))

    def rotate_full_image(self, canvas, image, index):
        self.rotation_angles[index] = (self.rotation_angles[index] + 90) % 360
        self.update_full_image(canvas, image, index, float(canvas.winfo_width()) / image.width)  # 保持当前缩放比例

    def rotate_image(self, index):
        self.rotation_angles[index] = (self.rotation_angles[index] + 90) % 360
        self.update_displayed_images()

    def __del__(self):
        # 确保在应用关闭时关闭会话
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.close_session())
        self.executor.shutdown()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedMediaViewerApp(root)
    root.mainloop()
