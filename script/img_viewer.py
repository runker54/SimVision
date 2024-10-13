import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedMediaViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Media Viewer")
        self.root.geometry("1700x900")

        self.df = None
        self.filtered_data = {}
        self.current_id_index = 0
        self.file_path = None
        self.zoom_level = 1.0

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

        # Image display
        self.canvas = tk.Canvas(parent)
        self.canvas.pack(fill=tk.BOTH, expand=True)

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
            unique_values = self.df[column].unique()
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
            filtered_df[filter_column] = filtered_df[filter_column].astype(str).str.strip().str.lower()
            filter_value = str(filter_value).strip().lower()
            filtered_df = filtered_df[filtered_df[filter_column].str.contains(filter_value, na=False)]

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
            self.display_current_media()
        else:
            self.id_label.config(text="No data available")
            self.clear_canvas()

    def display_current_media(self):
        self.clear_canvas()
        if not self.filtered_data:
            return

        current_id = list(self.filtered_data.keys())[self.current_id_index]
        urls = self.filtered_data[current_id]
        self.display_images(self.canvas, urls)

    def display_manual_images(self):
        self.manual_canvas.delete("all")
        urls = [entry.get() for entry in self.url_entries if entry.get()]
        self.display_images(self.manual_canvas, urls)

    def display_images(self, canvas, urls):
        x, y = 0, 0
        images = []  # Store image references
        for i, url in enumerate(urls):
            try:
                image = self.load_image(url)
                photo = ImageTk.PhotoImage(image)
                images.append(photo)  # Keep a reference to the image
                canvas.create_image(x, y, anchor=tk.NW, image=photo)
                canvas.create_text(x+10, y+10, anchor=tk.NW, text=f"Image {i+1}", fill="black", font=("Arial", 16))
                
                x += 500
                if (i + 1) % 3 == 0:
                    x = 0
                    y += 500
            except Exception as e:
                logging.error(f"Failed to load image from URL: {url}")
                logging.error(str(e))
        
        canvas.images = images  # Store images as an attribute of the canvas

    def load_image(self, url):
        logging.info(f"Attempting to load image from: {url}")
        try:
            if url.startswith(('http://', 'https://')):
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                if not os.path.isabs(url):
                    url = os.path.abspath(url)
                logging.info(f"Attempting to open local file: {url}")
                img = Image.open(url)
            img.thumbnail((400, 500), Image.LANCZOS)
            return img
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            raise

    def prev_id(self):
        if self.filtered_data and self.current_id_index > 0:
            self.current_id_index -= 1
            self.update_id_display()

    def next_id(self):
        if self.filtered_data and self.current_id_index < len(self.filtered_data) - 1:
            self.current_id_index += 1
            self.update_id_display()

    def clear_canvas(self):
        self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedMediaViewerApp(root)
    root.mainloop()
