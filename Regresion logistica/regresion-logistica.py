import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import warnings
warnings.filterwarnings('ignore')

class ModernRegresionLogisticaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Análisis de Regresión Logística - Machine Learning")
        self.root.geometry("1400x900")
        
        # Intentar maximizar la ventana
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                pass  # macOS o fallback
        
        # Configurar tema moderno
        self.setup_modern_theme()
        
        # Variables del modelo
        self.df = None
        self.model = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.feature_importance = None
        
        self.setup_ui()
        
    def setup_modern_theme(self):
        """Configurar tema moderno y colores"""
        # Configurar colores modernos
        self.colors = {
            'primary': '#2C3E50',      # Azul oscuro
            'secondary': '#3498DB',    # Azul claro  
            'accent': '#E74C3C',       # Rojo
            'success': '#27AE60',      # Verde
            'warning': '#F39C12',      # Naranja
            'background': '#ECF0F1',   # Gris claro
            'surface': '#FFFFFF',      # Blanco
            'text': '#2C3E50',         # Texto oscuro
            'text_light': '#7F8C8D'    # Texto claro
        }
        
        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar estilos personalizados
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 24, 'bold'),
                       foreground=self.colors['primary'],
                       background=self.colors['background'])
        
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 12),
                       foreground=self.colors['text_light'],
                       background=self.colors['background'])
        
        style.configure('Modern.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(20, 10))
        
        style.configure('Card.TFrame',
                       background=self.colors['surface'],
                       relief='flat',
                       borderwidth=1)
        
        # Configurar colores de matplotlib
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
        self.setup_matplotlib_theme()
        
        # Configurar la ventana principal
        self.root.configure(bg=self.colors['background'])
    
    def setup_matplotlib_theme(self):
        """Configuración moderna para matplotlib"""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#CCCCCC',
            'axes.linewidth': 0.8,
            'grid.color': '#E0E0E0',
            'grid.linewidth': 0.5
        })
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        # Frame principal con padding
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header moderno
        self.create_modern_header(main_container)
        
        # Panel de control con botones modernos
        self.create_control_panel(main_container)
        
        # Separador visual
        separator = tk.Frame(main_container, height=2, bg=self.colors['secondary'])
        separator.pack(fill=tk.X, pady=(10, 20))
        
        # Notebook con diseño moderno
        self.create_modern_notebook(main_container)
        
        # Footer con información
        self.create_footer(main_container)
        
        # Cargar dataset por defecto
        self.root.after(100, self.try_load_default_dataset)
    
    def create_modern_header(self, parent):
        """Crear header moderno"""
        header_frame = tk.Frame(parent, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Título principal
        title_frame = tk.Frame(header_frame, bg=self.colors['background'])
        title_frame.pack(side=tk.TOP)
        
        title_label = tk.Label(title_frame, 
                              text="🎯 ANÁLISIS DE REGRESIÓN LOGÍSTICA",
                              font=('Segoe UI', 28, 'bold'),
                              fg=self.colors['primary'],
                              bg=self.colors['background'])
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="Machine Learning • Clasificación Binaria • Análisis Predictivo",
                                 font=('Segoe UI', 12),
                                 fg=self.colors['text_light'],
                                 bg=self.colors['background'])
        subtitle_label.pack(pady=(5, 0))
    
    def create_control_panel(self, parent):
        """Crear panel de control con botones"""
        control_frame = tk.Frame(parent, bg=self.colors['background'])
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame para centrar botones
        buttons_frame = tk.Frame(control_frame, bg=self.colors['background'])
        buttons_frame.pack()
        
        # Botones modernos con iconos y colores
        self.create_modern_button(buttons_frame, "📁 CARGAR DATASET", 
                                 self.cargar_dataset, self.colors['primary'])
        
        self.create_modern_button(buttons_frame, "🚀 EJECUTAR ANÁLISIS", 
                                 self.ejecutar_analisis, self.colors['success'])
        
        self.create_modern_button(buttons_frame, "💾 GUARDAR RESULTADOS", 
                                 self.guardar_resultados, self.colors['secondary'])
        
        self.create_modern_button(buttons_frame, "🔄 LIMPIAR TODO", 
                                 self.limpiar_todo, self.colors['warning'])
    
    def create_modern_button(self, parent, text, command, color):
        """Crear botón moderno con efectos"""
        button = tk.Button(parent,
                          text=text,
                          command=command,
                          font=('Segoe UI', 10, 'bold'),
                          bg=color,
                          fg='white',
                          activebackground=self.darken_color(color),
                          activeforeground='white',
                          relief='flat',
                          cursor='hand2',
                          padx=25,
                          pady=12,
                          bd=0)
        button.pack(side=tk.LEFT, padx=8)
        
        # Efectos hover
        def on_enter(e):
            button.config(bg=self.darken_color(color))
        def on_leave(e):
            button.config(bg=color)
            
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        return button
    
    def darken_color(self, color):
        """Oscurecer un color hexadecimal"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * 0.8)) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*darkened)
    
    def create_modern_notebook(self, parent):
        """Crear notebook con estilo moderno"""
        # Crear notebook con estilo moderno
        style = ttk.Style()
        style.configure('Modern.TNotebook', background=self.colors['background'])
        style.configure('Modern.TNotebook.Tab', 
                       padding=[20, 12],
                       font=('Segoe UI', 10, 'bold'))
        
        self.notebook = ttk.Notebook(parent, style='Modern.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Crear pestañas con diseño moderno
        self.create_modern_tabs()
    
    def create_modern_tabs(self):
        """Crear todas las pestañas"""
        # 1. Exploración de Datos
        self.tab_eda = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(self.tab_eda, text="📊 EXPLORACIÓN")
        self.setup_modern_eda_tab()
        
        # 2. Preprocesamiento
        self.tab_prep = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(self.tab_prep, text="🔧 PREPROCESAMIENTO")
        self.setup_modern_prep_tab()
        
        # 3. Entrenamiento
        self.tab_train = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(self.tab_train, text="🎯 ENTRENAMIENTO")
        self.setup_modern_train_tab()
        
        # 4. Evaluación
        self.tab_eval = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(self.tab_eval, text="📈 EVALUACIÓN")
        self.setup_modern_eval_tab()
        
        # 5. Visualizaciones
        self.tab_viz = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(self.tab_viz, text="📊 VISUALIZACIONES")
        self.setup_modern_viz_tab()
        
        # 6. Interpretabilidad
        self.tab_interp = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(self.tab_interp, text="🔍 INTERPRETABILIDAD")
        self.setup_modern_interp_tab()
    
    def create_modern_card(self, parent, title, height=None):
        """Crear tarjeta moderna con sombra simulada"""
        outer_frame = tk.Frame(parent, bg='#D5D8DC', bd=0)  # Sombra
        outer_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        card_frame = tk.Frame(outer_frame, bg=self.colors['surface'], bd=0)
        card_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Header de la tarjeta
        header_frame = tk.Frame(card_frame, bg=self.colors['primary'], height=40)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame,
                              text=title,
                              font=('Segoe UI', 12, 'bold'),
                              fg='white',
                              bg=self.colors['primary'])
        title_label.pack(expand=True)
        
        # Contenido de la tarjeta
        content_frame = tk.Frame(card_frame, bg=self.colors['surface'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        return content_frame
    
    def setup_modern_eda_tab(self):
        """Configurar pestaña de exploración de datos"""
        # Panel principal con scroll
        main_panel = self.create_scrollable_panel(self.tab_eda)
        
        # Tarjeta de información del dataset
        info_content = self.create_modern_card(main_panel, "📋 INFORMACIÓN DEL DATASET")
        
        self.info_text = scrolledtext.ScrolledText(info_content, 
                                                  height=15,
                                                  font=('Consolas', 10),
                                                  bg='#F8F9FA',
                                                  fg=self.colors['text'],
                                                  relief='flat',
                                                  bd=1)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Tarjeta de visualización
        viz_content = self.create_modern_card(main_panel, "📊 DISTRIBUCIÓN DE CLASES")
        
        self.fig_dist, self.ax_dist = plt.subplots(figsize=(10, 5))
        self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, viz_content)
        self.canvas_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_modern_prep_tab(self):
        """Configurar pestaña de preprocesamiento"""
        main_panel = self.create_scrollable_panel(self.tab_prep)
        
        # Información de preprocesamiento
        prep_content = self.create_modern_card(main_panel, "⚙️ INFORMACIÓN DE PREPROCESAMIENTO")
        
        self.prep_text = scrolledtext.ScrolledText(prep_content,
                                                  height=12,
                                                  font=('Consolas', 10),
                                                  bg='#F8F9FA',
                                                  fg=self.colors['text'],
                                                  relief='flat',
                                                  bd=1)
        self.prep_text.pack(fill=tk.BOTH, expand=True)
        
        # Matriz de correlación
        corr_content = self.create_modern_card(main_panel, "🔗 MATRIZ DE CORRELACIÓN")
        
        self.fig_corr, self.ax_corr = plt.subplots(figsize=(12, 8))
        self.canvas_corr = FigureCanvasTkAgg(self.fig_corr, corr_content)
        self.canvas_corr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_modern_train_tab(self):
        """Configurar pestaña de entrenamiento"""
        train_content = self.create_modern_card(self.tab_train, "🎯 INFORMACIÓN DEL ENTRENAMIENTO")
        
        self.train_text = scrolledtext.ScrolledText(train_content,
                                                   font=('Consolas', 10),
                                                   bg='#F8F9FA',
                                                   fg=self.colors['text'],
                                                   relief='flat',
                                                   bd=1)
        self.train_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_modern_eval_tab(self):
        """Configurar pestaña de evaluación"""
        main_panel = self.create_scrollable_panel(self.tab_eval)
        
        # Métricas de evaluación
        metrics_content = self.create_modern_card(main_panel, "📊 MÉTRICAS DE EVALUACIÓN")
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_content,
                                                     height=15,
                                                     font=('Consolas', 10),
                                                     bg='#F8F9FA',
                                                     fg=self.colors['text'],
                                                     relief='flat',
                                                     bd=1)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Matriz de confusión
        cm_content = self.create_modern_card(main_panel, "🎯 MATRIZ DE CONFUSIÓN")
        
        self.fig_cm, self.ax_cm = plt.subplots(figsize=(8, 6))
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, cm_content)
        self.canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_modern_viz_tab(self):
        """Configurar pestaña de visualizaciones"""
        # Crear notebook secundario para visualizaciones
        viz_notebook = ttk.Notebook(self.tab_viz)
        viz_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Curva ROC
        roc_frame = tk.Frame(viz_notebook, bg=self.colors['surface'])
        viz_notebook.add(roc_frame, text="📈 CURVA ROC")
        
        roc_content = self.create_modern_card(roc_frame, "📈 ANÁLISIS ROC")
        
        self.fig_roc, self.ax_roc = plt.subplots(figsize=(10, 8))
        self.canvas_roc = FigureCanvasTkAgg(self.fig_roc, roc_content)
        self.canvas_roc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Distribución de probabilidades
        prob_frame = tk.Frame(viz_notebook, bg=self.colors['surface'])
        viz_notebook.add(prob_frame, text="📊 PROBABILIDADES")
        
        prob_content = self.create_modern_card(prob_frame, "📊 DISTRIBUCIÓN DE PROBABILIDADES")
        
        self.fig_prob, self.ax_prob = plt.subplots(figsize=(10, 8))
        self.canvas_prob = FigureCanvasTkAgg(self.fig_prob, prob_content)
        self.canvas_prob.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_modern_interp_tab(self):
        """Configurar pestaña de interpretabilidad"""
        main_panel = self.create_scrollable_panel(self.tab_interp)
        
        # Análisis de importancia
        importance_content = self.create_modern_card(main_panel, "🔍 ANÁLISIS DE IMPORTANCIA")
        
        self.importance_text = scrolledtext.ScrolledText(importance_content,
                                                        height=12,
                                                        font=('Consolas', 10),
                                                        bg='#F8F9FA',
                                                        fg=self.colors['text'],
                                                        relief='flat',
                                                        bd=1)
        self.importance_text.pack(fill=tk.BOTH, expand=True)
        
        # Gráfico de importancia
        viz_content = self.create_modern_card(main_panel, "📊 IMPORTANCIA DE VARIABLES")
        
        self.fig_importance, self.ax_importance = plt.subplots(figsize=(12, 8))
        self.canvas_importance = FigureCanvasTkAgg(self.fig_importance, viz_content)
        self.canvas_importance.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_scrollable_panel(self, parent):
        """Crear panel con scroll"""
        canvas = tk.Canvas(parent, bg=self.colors['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['surface'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        return scrollable_frame
    
    def create_footer(self, parent):
        """Crear footer informativo"""
        footer_frame = tk.Frame(parent, bg=self.colors['primary'], height=40)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(footer_frame,
                               text="🚀 Machine Learning Analytics • Desarrollado con Python & Scikit-learn",
                               font=('Segoe UI', 10),
                               fg='white',
                               bg=self.colors['primary'])
        footer_label.pack(expand=True)
    
    # ============ FUNCIONES DE LÓGICA DEL MODELO ============
    
    def cargar_dataset(self):
        """Cargar dataset desde archivo"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.show_modern_message(f"✅ Dataset cargado: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
                self.actualizar_eda()
            except Exception as e:
                self.show_modern_message(f"❌ Error al cargar: {str(e)}", tipo="error")
    
    def try_load_default_dataset(self):
        """Intentar cargar dataset por defecto"""
        try:
            # Intentar cargar dataset por defecto
            possible_paths = [
                "diabetes.csv",
                "./diabetes.csv",
                "../diabetes.csv",
                "./data/diabetes.csv",
                "/workspaces/Taller-clasificacion/diabetes.csv"
            ]
            
            for path in possible_paths:
                try:
                    self.df = pd.read_csv(path)
                    self.actualizar_eda()
                    self.show_modern_message("✅ Dataset de diabetes cargado automáticamente")
                    return
                except:
                    continue
                    
            # Si no encuentra ningún archivo, mostrar mensaje
            self.show_modern_message("💡 Use 'CARGAR DATASET' para seleccionar un archivo CSV", tipo="info")
        except:
            pass
    
    def ejecutar_analisis(self):
        """Ejecutar análisis completo"""
        if self.df is None:
            self.show_modern_message("❌ Por favor, carga primero un dataset.", tipo="error")
            return
        
        try:
            # Mostrar mensaje de progreso
            progress_window = self.create_progress_window()
            self.root.update()
            
            # Realizar análisis completo
            self.preprocesar_datos()
            progress_window.update_progress("Datos preprocesados... ⚙️", 25)
            
            self.entrenar_modelo()
            progress_window.update_progress("Modelo entrenado... 🎯", 50)
            
            self.evaluar_modelo()
            progress_window.update_progress("Modelo evaluado... 📊", 75)
            
            self.actualizar_todas_las_pestanas()
            progress_window.update_progress("Interfaz actualizada... ✅", 100)
            
            progress_window.close()
            self.show_modern_message("🎉 ¡Análisis completado exitosamente!")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.close()
            self.show_modern_message(f"❌ Error durante el análisis: {str(e)}", tipo="error")
    
    def create_progress_window(self):
        """Crear ventana de progreso"""
        class ProgressWindow:
            def __init__(self, parent):
                self.window = tk.Toplevel(parent.root)
                self.window.title("Procesando...")
                self.window.geometry("400x150")
                self.window.resizable(False, False)
                self.window.configure(bg=parent.colors['background'])
                
                # Centrar ventana
                self.window.transient(parent.root)
                self.window.grab_set()
                
                # Título
                title = tk.Label(self.window, 
                               text="🔄 Ejecutando Análisis",
                               font=('Segoe UI', 14, 'bold'),
                               bg=parent.colors['background'],
                               fg=parent.colors['primary'])
                title.pack(pady=20)
                
                # Barra de progreso
                self.progress_var = tk.DoubleVar()
                self.progress_bar = ttk.Progressbar(self.window,
                                                   variable=self.progress_var,
                                                   maximum=100,
                                                   length=300,
                                                   mode='determinate')
                self.progress_bar.pack(pady=10)
                
                # Etiqueta de estado
                self.status_label = tk.Label(self.window,
                                           text="Iniciando...",
                                           font=('Segoe UI', 10),
                                           bg=parent.colors['background'],
                                           fg=parent.colors['text'])
                self.status_label.pack(pady=5)
            
            def update_progress(self, message, value):
                self.progress_var.set(value)
                self.status_label.config(text=message)
                self.window.update()
                
            def close(self):
                self.window.destroy()
        
        return ProgressWindow(self)
    
    def limpiar_todo(self):
        """Limpiar todos los datos"""
        if messagebox.askyesno("Confirmar", "¿Estás seguro de que quieres limpiar todos los datos?"):
            # Resetear variables
            self.df = None
            self.model = None
            self.scaler = None
            self.X_train_scaled = None
            self.X_test_scaled = None
            self.y_train = None
            self.y_test = None
            self.y_pred = None
            self.y_pred_proba = None
            self.feature_importance = None
            
            # Limpiar todas las áreas de texto
            text_widgets = [
                self.info_text, self.prep_text, self.train_text, 
                self.metrics_text, self.importance_text
            ]
            
            for widget in text_widgets:
                widget.delete(1.0, tk.END)
            
            # Limpiar gráficos
            plots = [
                (self.fig_dist, self.ax_dist, self.canvas_dist),
                (self.fig_corr, self.ax_corr, self.canvas_corr),
                (self.fig_cm, self.ax_cm, self.canvas_cm),
                (self.fig_roc, self.ax_roc, self.canvas_roc),
                (self.fig_prob, self.ax_prob, self.canvas_prob),
                (self.fig_importance, self.ax_importance, self.canvas_importance)
            ]
            
            for fig, ax, canvas in plots:
                ax.clear()
                canvas.draw()
            
            self.show_modern_message("🧹 Todos los datos han sido limpiados")
    
    def show_modern_message(self, mensaje, tipo="info"):
        """Mostrar mensaje moderno"""
        if tipo == "error":
            messagebox.showerror("❌ Error", mensaje)
        elif tipo == "warning":
            messagebox.showwarning("⚠️ Advertencia", mensaje)
        else:
            messagebox.showinfo("✅ Información", mensaje)
    
    def preprocesar_datos(self):
        """Preprocesar los datos"""
        # Limpiar datos
        self.df = self.df.dropna()
        
        # Dividir variables
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        
        # División entrenamiento/prueba
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalización
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
    
    def entrenar_modelo(self):
        """Entrenar el modelo de regresión logística"""
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Calcular importancia de características
        X = self.df.drop("Outcome", axis=1)
        self.feature_importance = pd.DataFrame({
            'Variable': X.columns,
            'Coeficiente': self.model.coef_[0],
            'Importancia_Abs': np.abs(self.model.coef_[0])
        }).sort_values('Importancia_Abs', ascending=False)
    
    def evaluar_modelo(self):
        """Evaluar el modelo entrenado"""
        self.y_pred = self.model.predict(self.X_test_scaled)
        self.y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
    
    def actualizar_eda(self):
        """Actualizar la pestaña de exploración de datos"""
        if self.df is None:
            return
        
        # Actualizar texto de información con formato mejorado
        info_text = f"""
🎯 INFORMACIÓN DEL DATASET
{'='*60}

📊 DIMENSIONES:
    • Filas: {self.df.shape[0]:,}
    • Columnas: {self.df.shape[1]}
    • Tamaño total: {self.df.size:,} valores

📋 VARIABLES:
    {', '.join(self.df.columns.tolist())}

⚠️ VALORES NULOS:
{self.df.isnull().sum().to_string()}

📈 ESTADÍSTICAS DESCRIPTIVAS:
{self.df.describe().round(2).to_string()}

📋 PRIMERAS 5 FILAS:
{self.df.head().to_string()}

🎯 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO:
{self.df['Outcome'].value_counts().to_string()}

📊 BALANCE DE CLASES:
    • No Diabetes (0): {(self.df['Outcome'] == 0).sum()} ({(self.df['Outcome'] == 0).mean()*100:.1f}%)
    • Diabetes (1): {(self.df['Outcome'] == 1).sum()} ({(self.df['Outcome'] == 1).mean()*100:.1f}%)
"""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)
        
        # Actualizar gráfico de distribución con diseño moderno
        self.ax_dist.clear()
        
        # Crear gráfico de barras moderno
        counts = self.df['Outcome'].value_counts()
        colors = [self.colors['secondary'], self.colors['accent']]
        
        bars = self.ax_dist.bar(['No Diabetes', 'Diabetes'], counts.values, 
                               color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Añadir etiquetas en las barras
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            self.ax_dist.text(bar.get_x() + bar.get_width()/2., height + 10,
                             f'{count}\n({count/len(self.df)*100:.1f}%)',
                             ha='center', va='bottom', fontweight='bold')
        
        self.ax_dist.set_title('📊 Distribución de la Variable Objetivo', 
                              fontsize=14, fontweight='bold', pad=20)
        self.ax_dist.set_ylabel('Frecuencia', fontweight='bold')
        self.ax_dist.grid(True, alpha=0.3)
        self.ax_dist.set_ylim(0, max(counts.values) * 1.2)
        
        self.fig_dist.tight_layout()
        self.canvas_dist.draw()
    
    def actualizar_todas_las_pestanas(self):
        """Actualizar todas las pestañas con nueva información"""
        self.actualizar_prep()
        self.actualizar_train()
        self.actualizar_eval()
        self.actualizar_viz()
        self.actualizar_interp()
    
    def actualizar_prep(self):
        """Actualizar pestaña de preprocesamiento"""
        X = self.df.drop("Outcome", axis=1)
        correlations = self.df.corr()['Outcome'].sort_values(ascending=False)
        
        prep_text = f"""
🔧 INFORMACIÓN DE PREPROCESAMIENTO
{'='*60}

📊 DIMENSIONES DESPUÉS DE LIMPIEZA:
    • Dataset limpio: {self.df.shape}
    • Variables predictoras (X): {X.shape}
    • Variable objetivo (y): Outcome

🔄 DIVISIÓN DE DATOS:
    • Conjunto de entrenamiento: {self.X_train_scaled.shape}
    • Conjunto de prueba: {self.X_test_scaled.shape}
    • Proporción: 80% entrenamiento, 20% prueba

📊 DISTRIBUCIÓN POR CONJUNTO:
    • Entrenamiento: {dict(zip(*np.unique(self.y_train, return_counts=True)))}
    • Prueba: {dict(zip(*np.unique(self.y_test, return_counts=True)))}

📈 CORRELACIÓN CON VARIABLE OBJETIVO:
{correlations.round(3).to_string()}

⚙️ NORMALIZACIÓN APLICADA:
    • Método: StandardScaler (z-score)
    • Media post-normalización: ~0.00
    • Desviación estándar: ~1.00
    • Estado: ✅ Completado

🎯 VARIABLES MÁS CORRELACIONADAS:
    • Más positiva: {correlations.iloc[1]:.3f} ({correlations.index[1]})
    • Más negativa: {correlations.iloc[-1]:.3f} ({correlations.index[-1]})
"""
        
        self.prep_text.delete(1.0, tk.END)
        self.prep_text.insert(tk.END, prep_text)
        
        # Actualizar matriz de correlación con diseño moderno
        self.ax_corr.clear()
        correlation_matrix = self.df.corr()
        
        # Crear heatmap moderno
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.2f', 
                   square=True,
                   mask=mask,
                   cbar_kws={"shrink": .8},
                   ax=self.ax_corr)
        
        self.ax_corr.set_title('🔗 Matriz de Correlación', 
                              fontsize=14, fontweight='bold', pad=20)
        self.fig_corr.tight_layout()
        self.canvas_corr.draw()
    
    def actualizar_train(self):
        """Actualizar pestaña de entrenamiento"""
        # Crear información detallada del entrenamiento
        train_text = f"""
🎯 INFORMACIÓN DEL ENTRENAMIENTO
{'='*60}

🤖 MODELO SELECCIONADO:
    • Algoritmo: Regresión Logística
    • Solver: lbfgs (por defecto)
    • Regularización: L2 (por defecto)
    • Max iteraciones: 1000
    • Estado: ✅ ENTRENADO EXITOSAMENTE

📊 PARÁMETROS DEL MODELO:
    • Intercepto (b₀): {self.model.intercept_[0]:.4f}
    • Número de coeficientes: {len(self.model.coef_[0])}

🔍 COEFICIENTES DETALLADOS:
"""
        
        # Añadir información de cada coeficiente
        X = self.df.drop("Outcome", axis=1)
        for i, (var, coef) in enumerate(zip(X.columns, self.model.coef_[0])):
            effect = "📈 AUMENTA" if coef > 0 else "📉 DISMINUYE"
            train_text += f"    • {var}: {coef:8.4f} ({effect})\n"
        
        train_text += f"""
🏆 TOP 5 VARIABLES MÁS IMPORTANTES:
"""
        
        for i, (idx, row) in enumerate(self.feature_importance.head(5).iterrows(), 1):
            effect = "📈 POSITIVO" if row['Coeficiente'] > 0 else "📉 NEGATIVO"
            train_text += f"    {i}. {row['Variable']}: {effect} ({row['Coeficiente']:.4f})\n"
        
        train_text += f"""
💡 INTERPRETACIÓN:
    • Coeficientes POSITIVOS → Mayor probabilidad de diabetes
    • Coeficientes NEGATIVOS → Menor probabilidad de diabetes  
    • Magnitud → Fuerza de la relación

📈 ECUACIÓN DEL MODELO:
    logit(p) = {self.model.intercept_[0]:.3f}"""
        
        for var, coef in zip(X.columns, self.model.coef_[0]):
            sign = "+" if coef >= 0 else ""
            train_text += f" {sign}{coef:.3f}×{var}"
        
        self.train_text.delete(1.0, tk.END)
        self.train_text.insert(tk.END, train_text)
    
    def actualizar_eval(self):
        """Actualizar pestaña de evaluación"""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Calcular métricas adicionales
        baseline_accuracy = max(np.bincount(self.y_test)) / len(self.y_test)
        improvement = accuracy - baseline_accuracy
        
        # Calcular métricas detalladas desde la matriz de confusión
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        eval_text = f"""
📈 EVALUACIÓN DEL MODELO
{'='*60}

🎯 MÉTRICAS PRINCIPALES:
    • Exactitud (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)
    • AUC-ROC Score: {auc_score:.4f} ({auc_score*100:.2f}%)
    • Precisión: {precision:.4f} ({precision*100:.2f}%)
    • Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%)
    • F1-Score: {f1:.4f} ({f1*100:.2f}%)
    • Especificidad: {specificity:.4f} ({specificity*100:.2f}%)

🎯 MATRIZ DE CONFUSIÓN:
                     Predicción
    Real        No Diabetes  Diabetes
    No Diabetes      {tn:3d}      {fp:3d}
    Diabetes         {fn:3d}      {tp:3d}

📊 ANÁLISIS DETALLADO:
    • Verdaderos Negativos (TN): {tn} - Correctos No Diabetes
    • Falsos Positivos (FP): {fp} - Incorrectos como Diabetes  
    • Falsos Negativos (FN): {fn} - Incorrectos como No Diabetes
    • Verdaderos Positivos (TP): {tp} - Correctos Diabetes

🏆 COMPARACIÓN CON MODELO BASE:
    • Modelo base (clase mayoritaria): {baseline_accuracy:.4f}
    • Nuestro modelo: {accuracy:.4f}
    • Mejora absoluta: {improvement:.4f}
    • Mejora relativa: {improvement/baseline_accuracy*100:.1f}%

🎲 EJEMPLOS DE PREDICCIONES:
    • Primeras 10 predicciones: {list(self.y_pred[:10])}
    • Primeras 10 probabilidades: {[f'{p:.3f}' for p in self.y_pred_proba[:10]]}

✅ CONCLUSIÓN: {'EXCELENTE' if accuracy > 0.85 else 'BUENO' if accuracy > 0.75 else 'ACEPTABLE' if accuracy > 0.65 else 'MEJORABLE'} RENDIMIENTO
"""
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, eval_text)
        
        # Actualizar matriz de confusión con diseño moderno
        self.ax_cm.clear()
        
        # Crear heatmap moderno para la matriz de confusión
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   cbar_kws={'label': 'Número de casos'},
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'], 
                   ax=self.ax_cm,
                   square=True)
        
        # Añadir porcentajes
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                self.ax_cm.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                               ha='center', va='center', fontsize=10, color='gray')
        
        self.ax_cm.set_title('🎯 Matriz de Confusión', 
                            fontsize=14, fontweight='bold', pad=20)
        self.ax_cm.set_xlabel('Predicción', fontweight='bold')
        self.ax_cm.set_ylabel('Valor Real', fontweight='bold')
        self.fig_cm.tight_layout()
        self.canvas_cm.draw()
    
    def actualizar_viz(self):
        """Actualizar visualizaciones"""
        # Curva ROC moderna
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        
        self.ax_roc.clear()
        
        # Curva ROC con gradiente
        self.ax_roc.plot(fpr, tpr, color=self.colors['success'], lw=3, 
                        label=f'📈 Curva ROC (AUC = {auc_score:.3f})')
        self.ax_roc.fill_between(fpr, tpr, alpha=0.2, color=self.colors['success'])
        
        # Línea de referencia
        self.ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                        alpha=0.8, label='📊 Clasificador Aleatorio (AUC = 0.5)')
        
        # Punto óptimo
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        self.ax_roc.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                        label=f'🎯 Punto Óptimo (umbral={optimal_threshold:.3f})')
        
        self.ax_roc.set_xlim([0.0, 1.0])
        self.ax_roc.set_ylim([0.0, 1.05])
        self.ax_roc.set_xlabel('📊 Tasa de Falsos Positivos (1 - Especificidad)', fontweight='bold')
        self.ax_roc.set_ylabel('📈 Tasa de Verdaderos Positivos (Sensibilidad)', fontweight='bold')
        self.ax_roc.set_title('📈 Curva ROC - Análisis de Rendimiento', 
                             fontsize=14, fontweight='bold', pad=20)
        self.ax_roc.legend(loc="lower right")
        self.ax_roc.grid(True, alpha=0.3)
        
        # Añadir texto con interpretación
        interpretation = f"AUC = {auc_score:.3f} → "
        if auc_score >= 0.9:
            interpretation += "Excelente discriminación"
        elif auc_score >= 0.8:
            interpretation += "Buena discriminación"  
        elif auc_score >= 0.7:
            interpretation += "Discriminación aceptable"
        else:
            interpretation += "Discriminación pobre"
            
        self.ax_roc.text(0.6, 0.2, interpretation, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        self.fig_roc.tight_layout()
        self.canvas_roc.draw()
        
        # Distribución de probabilidades moderna
        self.ax_prob.clear()
        
        # Crear histogramas con diseño moderno
        prob_0 = self.y_pred_proba[self.y_test == 0]
        prob_1 = self.y_pred_proba[self.y_test == 1]
        
        self.ax_prob.hist(prob_0, bins=30, alpha=0.7, color=self.colors['secondary'],
                         label=f'📘 No Diabetes (n={len(prob_0)})', density=True)
        self.ax_prob.hist(prob_1, bins=30, alpha=0.7, color=self.colors['accent'],
                         label=f'📕 Diabetes (n={len(prob_1)})', density=True)
        
        # Añadir línea de decisión
        self.ax_prob.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                           label='⚖️ Umbral de Decisión (0.5)')
        
        # Añadir estadísticas
        self.ax_prob.axvline(x=np.mean(prob_0), color=self.colors['secondary'], 
                           linestyle='-', alpha=0.8, linewidth=2,
                           label=f'📊 Media No Diabetes ({np.mean(prob_0):.3f})')
        self.ax_prob.axvline(x=np.mean(prob_1), color=self.colors['accent'], 
                           linestyle='-', alpha=0.8, linewidth=2,
                           label=f'📊 Media Diabetes ({np.mean(prob_1):.3f})')
        
        self.ax_prob.set_xlabel('🎯 Probabilidad Predicha de Diabetes', fontweight='bold')
        self.ax_prob.set_ylabel('📊 Densidad', fontweight='bold')
        self.ax_prob.set_title('📊 Distribución de Probabilidades por Clase Real', 
                              fontsize=14, fontweight='bold', pad=20)
        self.ax_prob.legend()
        self.ax_prob.grid(True, alpha=0.3)
        
        self.fig_prob.tight_layout()
        self.canvas_prob.draw()
    
    def actualizar_interp(self):
        """Actualizar pestaña de interpretabilidad"""
        top_features = self.feature_importance.head(8)
        
        importance_text = f"""
🔍 ANÁLISIS DE INTERPRETABILIDAD
{'='*60}

🏆 RANKING DE IMPORTANCIA DE VARIABLES:

"""
        
        for i, (idx, row) in enumerate(top_features.iterrows(), 1):
            effect = "📈 AUMENTA" if row['Coeficiente'] > 0 else "📉 DISMINUYE"
            stars = "⭐" * min(5, int(abs(row['Coeficiente']) * 3) + 1)
            
            importance_text += f"{i:2d}. {row['Variable']:20s} {stars}\n"
            importance_text += f"    • Efecto: {effect} la probabilidad de diabetes\n" 
            importance_text += f"    • Coeficiente: {row['Coeficiente']:8.4f}\n"
            importance_text += f"    • Importancia: {row['Importancia_Abs']:8.4f}\n"
            importance_text += f"    • Interpretación: "
            
            if abs(row['Coeficiente']) > 0.5:
                importance_text += "🔥 IMPACTO ALTO\n"
            elif abs(row['Coeficiente']) > 0.2:
                importance_text += "🔶 IMPACTO MEDIO\n"
            else:
                importance_text += "🔹 IMPACTO BAJO\n"
            importance_text += "\n"
        
        importance_text += f"""
💡 GUÍA DE INTERPRETACIÓN:

🔵 COEFICIENTES POSITIVOS:
    • Por cada unidad de aumento en la variable (normalizada)
    • La probabilidad de diabetes AUMENTA
    • Factores de RIESGO

🔴 COEFICIENTES NEGATIVOS:
    • Por cada unidad de aumento en la variable (normalizada)  
    • La probabilidad de diabetes DISMINUYE
    • Factores PROTECTORES

📊 MAGNITUD:
    • |Coef| > 0.5: Impacto ALTO
    • |Coef| 0.2-0.5: Impacto MEDIO
    • |Coef| < 0.2: Impacto BAJO

🎯 TOTAL DE VARIABLES ANALIZADAS: {len(self.feature_importance)}

🏆 VARIABLE MÁS INFLUYENTE: {self.feature_importance.iloc[0]['Variable']}
    (Coeficiente: {self.feature_importance.iloc[0]['Coeficiente']:.4f})
"""
        
        self.importance_text.delete(1.0, tk.END)
        self.importance_text.insert(tk.END, importance_text)
        
        # Gráfico de importancia moderno
        self.ax_importance.clear()
        
        # Ordenar por coeficiente para mejor visualización
        feature_importance_sorted = self.feature_importance.sort_values('Coeficiente')
        
        # Crear colores dinámicos
        colors = []
        for coef in feature_importance_sorted['Coeficiente']:
            if coef > 0.3:
                colors.append('#E74C3C')  # Rojo fuerte para alto impacto positivo
            elif coef > 0:
                colors.append('#F39C12')  # Naranja para impacto positivo medio
            elif coef > -0.3:
                colors.append('#3498DB')  # Azul para impacto negativo medio
            else:
                colors.append('#2C3E50')  # Azul oscuro para alto impacto negativo
        
        # Crear gráfico de barras horizontal
        bars = self.ax_importance.barh(range(len(feature_importance_sorted)), 
                                      feature_importance_sorted['Coeficiente'], 
                                      color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Personalizar etiquetas
        self.ax_importance.set_yticks(range(len(feature_importance_sorted)))
        self.ax_importance.set_yticklabels(feature_importance_sorted['Variable'], fontweight='bold')
        self.ax_importance.set_xlabel('🔍 Coeficiente del Modelo', fontweight='bold')
        self.ax_importance.set_title('📊 Importancia de Variables en Regresión Logística', 
                                    fontsize=14, fontweight='bold', pad=20)
        
        # Añadir línea de referencia
        self.ax_importance.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Añadir valores en las barras
        for i, (bar, coef) in enumerate(zip(bars, feature_importance_sorted['Coeficiente'])):
            width = bar.get_width()
            label_x = width + (0.01 if width >= 0 else -0.01)
            ha = 'left' if width >= 0 else 'right'
            self.ax_importance.text(label_x, bar.get_y() + bar.get_height()/2, 
                                  f'{coef:.3f}', ha=ha, va='center', fontweight='bold', fontsize=9)
        
        # Añadir zonas de interpretación
        self.ax_importance.axvspan(-1, -0.3, alpha=0.1, color='blue', label='🔵 Alto Impacto Negativo')
        self.ax_importance.axvspan(-0.3, 0, alpha=0.1, color='lightblue', label='🔷 Medio Impacto Negativo')
        self.ax_importance.axvspan(0, 0.3, alpha=0.1, color='orange', label='🔶 Medio Impacto Positivo') 
        self.ax_importance.axvspan(0.3, 1, alpha=0.1, color='red', label='🔴 Alto Impacto Positivo')
        
        self.ax_importance.grid(True, alpha=0.3, axis='x')
        self.ax_importance.legend(loc='lower right', fontsize=9)
        
        self.fig_importance.tight_layout()
        self.canvas_importance.draw()
    
    def guardar_resultados(self):
        """Guardar los resultados del análisis"""
        if self.model is None:
            self.show_modern_message("❌ Primero ejecuta el análisis.", tipo="error")
            return
        
        try:
            # Crear directorio de resultados
            import os
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"resultados_regresion_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Guardar métricas
            metrics = {
                'accuracy': accuracy_score(self.y_test, self.y_pred),
                'auc_score': roc_auc_score(self.y_test, self.y_pred_proba),
                'feature_importance': self.feature_importance.to_dict('records'),
                'confusion_matrix': confusion_matrix(self.y_test, self.y_pred).tolist(),
                'model_coefficients': self.model.coef_[0].tolist(),
                'model_intercept': float(self.model.intercept_[0])
            }
            
            import json
            with open(f"{results_dir}/metricas.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Guardar gráficos
            self.fig_roc.savefig(f"{results_dir}/curva_roc.png", dpi=300, bbox_inches='tight')
            self.fig_cm.savefig(f"{results_dir}/matriz_confusion.png", dpi=300, bbox_inches='tight')
            self.fig_importance.savefig(f"{results_dir}/importancia_variables.png", dpi=300, bbox_inches='tight')
            self.fig_corr.savefig(f"{results_dir}/matriz_correlacion.png", dpi=300, bbox_inches='tight')
            
            self.show_modern_message(f"✅ Resultados guardados en: {results_dir}")
            
        except Exception as e:
            self.show_modern_message(f"❌ Error al guardar: {str(e)}", tipo="error")

def main():
    root = tk.Tk()
    app = ModernRegresionLogisticaGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()