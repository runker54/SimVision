import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class FolderImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文件夹图片查看器")
        self.root.geometry("1920x1080")

        self.image_data = {}  # 用于存储每个ID对应的图片路径
        self.current_id_index = 0
        self.current_id = None

        self.create_widgets()

    def create_widgets(self):
        # 选择目录按钮
        ttk.Button(self.root, text="选择目录", command=self.select_directory).pack(pady=10)

        # 导航按钮
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(pady=10)
        ttk.Button(nav_frame, text="上一组", command=self.prev_id).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="下一组", command=self.next_id).pack(side=tk.LEFT, padx=5)
        self.id_label = ttk.Label(nav_frame, text="ID: ")
        self.id_label.pack(side=tk.LEFT, padx=5)

        # 创建一个框架来容纳画布和滚动条
        self.canvas_frame = ttk.Frame(self.root)
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

    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.load_images(directory)

    def load_images(self, directory):
        self.image_data = {}
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                if images:
                    self.image_data[folder] = images
        
        if self.image_data:
            self.current_id_index = 0
            self.current_id = list(self.image_data.keys())[self.current_id_index]
            self.display_images()
        else:
            messagebox.showwarning("警告", "该目录下没有找到图片。")

    def display_images(self):
        # 清空之前的图片
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if self.current_id:
            self.id_label.config(text=f"ID: {self.current_id}")
            images = self.image_data[self.current_id]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # 计算每个图片的最大显示尺寸
            max_image_width = canvas_width // 5  # 每行显示5个图片
            max_image_height = int(canvas_height / (len(images) // 5 + 1))  # 根据图片数量自适应高度

            for i, image_path in enumerate(images):
                image = Image.open(image_path)
                display_image = image.copy()
                display_image.thumbnail((max_image_width, max_image_height), Image.LANCZOS)  # 自适应缩放
                photo = ImageTk.PhotoImage(display_image)
                frame = ttk.Frame(self.scrollable_frame)
                frame.grid(row=i // 5, column=i % 5, padx=5, pady=5)  # 每行显示5个图片

                label = ttk.Label(frame, image=photo)
                label.image = photo  # 保持对图片的引用
                label.pack()

    def prev_id(self):
        if self.image_data and self.current_id_index > 0:
            self.current_id_index -= 1
            self.current_id = list(self.image_data.keys())[self.current_id_index]
            self.display_images()

    def next_id(self):
        if self.image_data and self.current_id_index < len(self.image_data) - 1:
            self.current_id_index += 1
            self.current_id = list(self.image_data.keys())[self.current_id_index]
            self.display_images()

if __name__ == "__main__":
    root = tk.Tk()
    app = FolderImageViewerApp(root)
    root.mainloop()
