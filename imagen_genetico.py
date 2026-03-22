import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import threading
import time
import math
from PIL import Image, ImageDraw, ImageTk

# ─────────────────────────────────────────────────────────
#  CONFIGURACION GLOBAL
# ─────────────────────────────────────────────────────────
ANCHO_IMG  = 150   # resolución interna de trabajo (más pequeño = más rápido)
ALTO_IMG   = 150

# ─────────────────────────────────────────────────────────
#  ALGORITMO GENÉTICO — GENERACIÓN DE IMAGEN CON POLÍGONOS
# ─────────────────────────────────────────────────────────

def gen_poligono(ancho, alto, n_vertices=3):
    """Crea un polígono aleatorio: vértices + color RGBA."""
    vertices = []
    for _ in range(n_vertices):
        vertices.append(random.randint(0, ancho))
        vertices.append(random.randint(0, alto))
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    a = random.randint(30, 180)  
    return vertices + [r, g, b, a]

def crear_individuo(n_poligonos, ancho, alto):
    """Un individuo = lista de n_poligonos polígonos."""
    return [gen_poligono(ancho, alto) for _ in range(n_poligonos)]

def crear_poblacion(tam, n_poligonos, ancho, alto):
    return [crear_individuo(n_poligonos, ancho, alto) for _ in range(tam)]

def renderizar(individuo, ancho, alto):
    """Dibuja los polígonos del individuo y devuelve una imagen PIL."""
    img = Image.new("RGB", (ancho, alto), (0, 0, 0))
    capa = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    draw = ImageDraw.Draw(capa)
    for poligono in individuo:
        n_coords = len(poligono) - 4
        vertices = list(zip(poligono[:n_coords:2], poligono[1:n_coords:2]))
        r, g, b, a = poligono[-4], poligono[-3], poligono[-2], poligono[-1]
        if len(vertices) >= 3:
            draw.polygon(vertices, fill=(r, g, b, a))
    img = Image.alpha_composite(img.convert("RGBA"), capa).convert("RGB")
    return img

def aptitud(individuo, objetivo_px, ancho, alto):
    """
    Compara píxel a píxel el individuo con la imagen objetivo.
    Devuelve un valor entre 0 y 1 donde 1 es perfecto.
    """
    img = renderizar(individuo, ancho, alto)
    px  = list(img.getdata())
    diferencia = 0
    max_dif = 255 * 3 * ancho * alto
    for (r1, g1, b1), (r2, g2, b2) in zip(px, objetivo_px):
        diferencia += abs(r1-r2) + abs(g1-g2) + abs(b1-b2)
    return 1 - (diferencia / max_dif)

# ── Mutación ─────────────────────────────────────────────

def mutar(individuo, tasa, ancho, alto):
    nuevo = [p[:] for p in individuo]
    for i in range(len(nuevo)):
        if random.random() < tasa:
            tipo = random.random()
            if tipo < 0.3:
                # Reemplazar polígono completo
                nuevo[i] = gen_poligono(ancho, alto)
            elif tipo < 0.6:
                # Mutar color
                nuevo[i][-4] = max(0, min(255, nuevo[i][-4] + random.randint(-30, 30)))
                nuevo[i][-3] = max(0, min(255, nuevo[i][-3] + random.randint(-30, 30)))
                nuevo[i][-2] = max(0, min(255, nuevo[i][-2] + random.randint(-30, 30)))
                nuevo[i][-1] = max(30, min(180, nuevo[i][-1] + random.randint(-20, 20)))
            else:
                # Mutar vértices ligeramente
                n_coords = len(nuevo[i]) - 4
                for j in range(n_coords):
                    lim = ancho if j % 2 == 0 else alto
                    nuevo[i][j] = max(0, min(lim, nuevo[i][j] + random.randint(-20, 20)))
    return nuevo

# ── Cruce ────────────────────────────────────────────────

def cruce(p1, p2, tasa):
    """Cruce uniforme: cada polígono viene del padre 1 o del padre 2."""
    if random.random() > tasa:
        return [p[:] for p in p1]
    hijo = []
    for pol1, pol2 in zip(p1, p2):
        hijo.append(pol1[:] if random.random() < 0.5 else pol2[:])
    return hijo

# ── Selección por torneo ─────────────────────────────────

def seleccion_torneo(poblacion, aptitudes, k=3):
    indices = random.sample(range(len(poblacion)), k)
    mejor   = max(indices, key=lambda i: aptitudes[i])
    return poblacion[mejor]

# ── Ciclo principal ──────────────────────────────────────

def algoritmo_genetico(objetivo_px, ancho, alto,
                        tam_poblacion, n_generaciones,
                        n_poligonos, tasa_cruce, tasa_mutacion,
                        elitismo, callback=None, stop_flag=None):

    poblacion  = crear_poblacion(tam_poblacion, n_poligonos, ancho, alto)
    mejor_global   = None
    mejor_aptitud  = -1
    historial      = []

    for gen in range(n_generaciones):
        if stop_flag and stop_flag():
            break

        # Evaluar toda la población
        aptitudes = [aptitud(ind, objetivo_px, ancho, alto) for ind in poblacion]

        idx_mejor = max(range(len(aptitudes)), key=lambda i: aptitudes[i])
        if aptitudes[idx_mejor] > mejor_aptitud:
            mejor_aptitud  = aptitudes[idx_mejor]
            mejor_global   = [p[:] for p in poblacion[idx_mejor]]

        historial.append(mejor_aptitud)

        if callback:
            callback(gen + 1, mejor_global, mejor_aptitud, historial)

        # Ordenar por aptitud
        orden      = sorted(range(len(poblacion)), key=lambda i: aptitudes[i], reverse=True)
        poblacion  = [poblacion[i] for i in orden]
        aptitudes  = [aptitudes[i] for i in orden]

        nueva_poblacion = []

        # Elitismo
        n_elite = max(1, int(tam_poblacion * 0.1)) if elitismo else 0
        nueva_poblacion.extend([[p[:] for p in ind] for ind in poblacion[:n_elite]])

        # Generar hijos
        while len(nueva_poblacion) < tam_poblacion:
            p1   = seleccion_torneo(poblacion, aptitudes)
            p2   = seleccion_torneo(poblacion, aptitudes)
            hijo = cruce(p1, p2, tasa_cruce)
            hijo = mutar(hijo, tasa_mutacion, ancho, alto)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    return mejor_global, mejor_aptitud, historial


# ─────────────────────────────────────────────────────────
#  COLORES Y FUENTES
# ─────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
YELLOW = "#d29922"
RED    = "#f85149"
PURPLE = "#bc8cff"
FG     = "#e6edf3"
FG2    = "#8b949e"
DARK   = "#21262d"

F_TITLE = ("Courier New", 12, "bold")
F_LABEL = ("Courier New", 10)
F_SMALL = ("Courier New", 9)
F_BTN   = ("Courier New", 10, "bold")


# ─────────────────────────────────────────────────────────
#  INTERFAZ GRÁFICA
# ─────────────────────────────────────────────────────────

class AppImagen:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genetico — Generacion de Imagen con Poligonos")
        self.root.configure(bg=BG)
        self.root.geometry("1300x820")
        self.root.minsize(1000, 700)

        self.imagen_objetivo = None   # PIL Image
        self.objetivo_px     = None   # lista de tuplas RGB
        self.img_ancho       = ANCHO_IMG
        self.img_alto        = ALTO_IMG
        self.corriendo       = False
        self._stop           = False
        self.hilo            = None
        self.t_inicio        = 0
        self.vars            = {}

        # Referencias para no perder imagenes en GC
        self._tk_objetivo  = None
        self._tk_generada  = None

        self._build()

    # ─────────────────────────────────────────────────────
    def _build(self):
        # HEADER
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text="IMAGE GENETIC OPTIMIZER",
                 font=("Courier New", 17, "bold"), bg=BG, fg=PURPLE).pack(side="left")
        tk.Label(hdr, text="   Carga una imagen, configura y presiona Ejecutar AG",
                 font=F_SMALL, bg=BG, fg=FG2).pack(side="left")

        # CUERPO
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── PANEL IZQUIERDO ──────────────────────────────
        left = tk.Frame(main, bg=PANEL,
                        highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        self._section(left, "PARAMETROS")

        self._slider(left, "Poblacion",       "pop",      20,   5,  100,  5, int)
        self._slider(left, "Generaciones",    "gen",     300,  10, 2000, 10, int)
        self._slider(left, "Poligonos",       "npol",     50,  10,  100,  5, int)
        self._slider(left, "Tasa de cruce",   "cruce",  0.85, 0.0,  1.0, 0.01, float)
        self._slider(left, "Tasa mutacion",   "mut",    0.20, 0.0,  1.0, 0.01, float)

        self._sep(left)

        self.vars["elitismo"] = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Elitismo (conservar top 10%)",
                       variable=self.vars["elitismo"],
                       font=F_SMALL, bg=PANEL, fg=FG,
                       selectcolor=DARK, activebackground=PANEL,
                       activeforeground=GREEN).pack(anchor="w", padx=14, pady=(4, 8))

        self._sep(left)

        # Boton cargar imagen
        self.btn_cargar = self._btn(left, "Cargar imagen objetivo", PURPLE, DARK, self._cargar_imagen)

        # Preview imagen objetivo
        self.lbl_preview_titulo = tk.Label(left, text="Sin imagen cargada",
                                           font=F_SMALL, bg=PANEL, fg=FG2)
        self.lbl_preview_titulo.pack(pady=(6, 2))

        self.canvas_preview = tk.Canvas(left, width=180, height=180,
                                        bg=DARK, highlightbackground=BORDER,
                                        highlightthickness=1)
        self.canvas_preview.pack(padx=14, pady=(0, 8))

        self._sep(left)

        self.btn_run  = self._btn(left, "Ejecutar AG",  BG,  GREEN, self._ejecutar)
        self.btn_stop = self._btn(left, "Detener",       BG,  RED,   self._detener)
        self.btn_stop.config(state="disabled")
        self.btn_run.config(state="disabled")

        self._sep(left)

        self._section(left, "ESTADISTICAS")
        sf = tk.Frame(left, bg=PANEL)
        sf.pack(fill="x", padx=14, pady=(0, 12))
        self.lbl_gen      = self._stat(sf, "Generacion", ACCENT)
        self.lbl_apt      = self._stat(sf, "Similitud",  GREEN)
        self.lbl_pct      = self._stat(sf, "Progreso",   YELLOW)
        self.lbl_tiempo   = self._stat(sf, "Tiempo",     FG)

        # ── PANEL DERECHO ────────────────────────────────
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)

        # Titulos de canvas
        tk.Label(right, text="IMAGEN OBJETIVO",
                 font=("Courier New", 9, "bold"), bg=BG, fg=FG2, anchor="w"
                 ).grid(row=0, column=0, sticky="w", pady=(0, 2))
        tk.Label(right, text="IMAGEN GENERADA POR EL AG",
                 font=("Courier New", 9, "bold"), bg=BG, fg=FG2, anchor="w"
                 ).grid(row=0, column=1, sticky="w", pady=(0, 2), padx=(8, 0))

        # Canvas imagen objetivo
        self.canvas_obj = tk.Canvas(right, bg=DARK,
                                    highlightbackground=BORDER, highlightthickness=1)
        self.canvas_obj.grid(row=1, column=0, sticky="nsew", padx=(0, 4))

        # Canvas imagen generada
        self.canvas_gen = tk.Canvas(right, bg=DARK,
                                    highlightbackground=BORDER, highlightthickness=1)
        self.canvas_gen.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        # Grafica evolucion
        tk.Frame(right, bg=BG, height=6).grid(row=2, columnspan=2)
        tk.Label(right, text="EVOLUCION — Similitud con la imagen objetivo (%)",
                 font=("Courier New", 9, "bold"), bg=BG, fg=FG2, anchor="w"
                 ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 2))

        self.canvas_evo = tk.Canvas(right, bg="#0a0e14", height=150,
                                    highlightbackground=BORDER, highlightthickness=1)
        self.canvas_evo.grid(row=4, column=0, columnspan=2, sticky="ew")

        # Progreso + status
        self.progress_var = tk.DoubleVar()
        style = ttk.Style()
        style.theme_use("default")
        style.configure("p.Horizontal.TProgressbar",
                        troughcolor=DARK, background=PURPLE, thickness=5)
        ttk.Progressbar(right, variable=self.progress_var,
                        style="p.Horizontal.TProgressbar",
                        maximum=100).grid(row=5, column=0, columnspan=2,
                                          sticky="ew", pady=(6, 2))

        self.lbl_status = tk.Label(right,
            text="Carga una imagen para comenzar.",
            font=F_SMALL, bg=BG, fg=FG2, anchor="w")
        self.lbl_status.grid(row=6, column=0, columnspan=2, sticky="w")

    # ── Helpers UI ───────────────────────────────────────

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=F_TITLE,
                 bg=PANEL, fg=PURPLE).pack(pady=(12, 4), padx=14, anchor="w")

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=10, pady=5)

    def _btn(self, parent, text, fg, bg, cmd):
        b = tk.Button(parent, text=text, font=F_BTN,
                      bg=bg, fg=fg, activebackground=fg, activeforeground=bg,
                      relief="flat", cursor="hand2", padx=10, pady=7, command=cmd)
        b.pack(fill="x", padx=14, pady=3)
        return b

    def _slider(self, parent, label, key, default, mn, mx, step, typ):
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="x", padx=14, pady=2)
        top = tk.Frame(frame, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=label, font=F_LABEL, bg=PANEL, fg=FG2).pack(side="left")
        fmt = "{:.2f}" if typ == float else "{}"
        var = tk.DoubleVar(value=default) if typ == float else tk.IntVar(value=default)
        self.vars[key] = var
        val_lbl = tk.Label(top, text=fmt.format(default),
                           font=F_LABEL, bg=PANEL, fg=PURPLE, width=6)
        val_lbl.pack(side="right")
        def on_change(v):
            val_lbl.config(text=fmt.format(typ(float(v))))
        tk.Scale(frame, from_=mn, to=mx, orient="horizontal",
                 variable=var, resolution=step,
                 bg=PANEL, fg=PURPLE, troughcolor=DARK,
                 highlightthickness=0, showvalue=False,
                 command=on_change).pack(fill="x")

    def _stat(self, parent, label, val_color=FG):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{label}:", font=F_SMALL,
                 bg=PANEL, fg=FG2, width=12, anchor="w").pack(side="left")
        lbl = tk.Label(row, text="—", font=F_SMALL, bg=PANEL, fg=val_color)
        lbl.pack(side="left")
        return lbl

    # ── Cargar imagen ────────────────────────────────────

    def _cargar_imagen(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Todos", "*.*")]
        )
        if not path:
            return

        img = Image.open(path).convert("RGB")

        # Detectar orientacion y ajustar resolución interna
        w, h = img.size
        if w > h:
            self.img_ancho = ANCHO_IMG
            self.img_alto  = int(ALTO_IMG * h / w)
        elif h > w:
            self.img_alto  = ALTO_IMG
            self.img_ancho = int(ANCHO_IMG * w / h)
        else:
            self.img_ancho = ANCHO_IMG
            self.img_alto  = ALTO_IMG

        self.imagen_objetivo = img.resize((self.img_ancho, self.img_alto), Image.LANCZOS)
        self.objetivo_px     = list(self.imagen_objetivo.getdata())

        # Preview en panel izquierdo
        prev = self.imagen_objetivo.resize((176, 176), Image.LANCZOS)
        self._tk_preview = ImageTk.PhotoImage(prev)
        self.canvas_preview.delete("all")
        self.canvas_preview.create_image(88, 88, image=self._tk_preview)
        self.lbl_preview_titulo.config(text=f"{w}x{h} px → {self.img_ancho}x{self.img_alto}", fg=GREEN)

        # Mostrar en canvas objetivo grande
        self._mostrar_en_canvas(self.canvas_obj, self.imagen_objetivo, "_tk_objetivo")

        self.canvas_gen.delete("all")
        self.canvas_evo.delete("all")
        self.progress_var.set(0)
        for lbl in [self.lbl_gen, self.lbl_apt, self.lbl_pct, self.lbl_tiempo]:
            lbl.config(text="—")

        self.btn_run.config(state="normal")
        self.lbl_status.config(text="Imagen cargada. Configura los parametros y presiona Ejecutar AG.")

    def _mostrar_en_canvas(self, canvas, img_pil, attr):
        canvas.update_idletasks()
        cw = canvas.winfo_width()  or 400
        ch = canvas.winfo_height() or 400
        escala = min(cw / img_pil.width, ch / img_pil.height)
        nw = max(1, int(img_pil.width  * escala))
        nh = max(1, int(img_pil.height * escala))
        img_tk = ImageTk.PhotoImage(img_pil.resize((nw, nh), Image.NEAREST))
        setattr(self, attr, img_tk)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=img_tk)

    # ── Ejecutar / Detener ───────────────────────────────

    def _ejecutar(self):
        if self.imagen_objetivo is None:
            messagebox.showwarning("Sin imagen", "Carga una imagen primero.")
            return
        if self.corriendo:
            return
        self.corriendo = True
        self._stop     = False
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_cargar.config(state="disabled")
        self.canvas_evo.delete("all")
        self.progress_var.set(0)
        self.lbl_status.config(text="Ejecutando algoritmo genetico...")
        self.t_inicio = time.time()
        self.hilo = threading.Thread(target=self._run_ag, daemon=True)
        self.hilo.start()

    def _detener(self):
        self._stop = True
        self.lbl_status.config(text="Detenido por el usuario.")

    def _run_ag(self):
        n_gen = self.vars["gen"].get()

        def cb(gen, mejor, apt, hist):
            pct     = (gen / n_gen) * 100
            elapsed = time.time() - self.t_inicio
            self.root.after(0, self._update_ui, gen, n_gen, mejor, apt, hist, pct, elapsed)

        mejor, apt, hist = algoritmo_genetico(
            objetivo_px    = self.objetivo_px,
            ancho          = self.img_ancho,
            alto           = self.img_alto,
            tam_poblacion  = self.vars["pop"].get(),
            n_generaciones = n_gen,
            n_poligonos    = self.vars["npol"].get(),
            tasa_cruce     = self.vars["cruce"].get(),
            tasa_mutacion  = self.vars["mut"].get(),
            elitismo       = self.vars["elitismo"].get(),
            callback       = cb,
            stop_flag      = lambda: self._stop
        )
        self.root.after(0, self._done, mejor, apt)

    def _update_ui(self, gen, n_gen, mejor, apt, hist, pct, elapsed):
        self.progress_var.set(pct)
        self.lbl_gen.config(text=f"{gen}/{n_gen}")
        self.lbl_apt.config(text=f"{apt*100:.2f}%")
        self.lbl_pct.config(text=f"{pct:.1f}%")
        self.lbl_tiempo.config(text=f"{elapsed:.1f} s")

        # Renderizar mejor individuo actual
        img_gen = renderizar(mejor, self.img_ancho, self.img_alto)
        self._mostrar_en_canvas(self.canvas_gen, img_gen, "_tk_generada")
        self._draw_chart(hist, n_gen)

    def _done(self, mejor, apt):
        self.corriendo = False
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_cargar.config(state="normal")
        elapsed = time.time() - self.t_inicio
        self.lbl_tiempo.config(text=f"{elapsed:.1f} s")
        self.lbl_status.config(
            text=f"Completado — Similitud final: {apt*100:.2f}%  |  {elapsed:.1f} s")

        # Mostrar imagen final
        img_gen = renderizar(mejor, self.img_ancho, self.img_alto)
        self._mostrar_en_canvas(self.canvas_gen, img_gen, "_tk_generada")

    # ── Grafica de evolucion ─────────────────────────────

    def _draw_chart(self, hist, n_gen):
        c = self.canvas_evo
        c.delete("all")
        W   = c.winfo_width()  or 800
        H   = c.winfo_height() or 150
        pad = 36

        if len(hist) < 2:
            return

        mn  = min(hist)
        mx  = max(hist)
        rng = mx - mn if mx != mn else 0.001

        def cx(i): return pad + (i / max(len(hist)-1, 1)) * (W - 2*pad)
        def cy(v): return pad + (1 - (v - mn) / rng) * (H - 2*pad)

        # Grid
        for k in range(5):
            yg = pad + k * (H - 2*pad) / 4
            c.create_line(pad, yg, W-pad, yg, fill=BORDER, dash=(2, 4))

        # Area rellena
        pts_fill = [pad, H-pad]
        pts_line = []
        for i, v in enumerate(hist):
            pts_fill += [cx(i), cy(v)]
            pts_line += [cx(i), cy(v)]
        pts_fill += [cx(len(hist)-1), H-pad]

        if len(pts_fill) >= 6:
            c.create_polygon(*pts_fill, fill="#2a1a3a", outline="")
        if len(pts_line) >= 4:
            c.create_line(*pts_line, fill=PURPLE, width=2, smooth=True)

        # Punto final
        lx, ly = cx(len(hist)-1), cy(hist[-1])
        c.create_oval(lx-4, ly-4, lx+4, ly+4, fill=YELLOW, outline="")

        # Etiquetas
        c.create_text(pad, pad-12,     text=f"{mx*100:.1f}%", fill=FG2,
                      font=("Courier New", 8), anchor="w")
        c.create_text(pad, H-pad+12,   text=f"{mn*100:.1f}%", fill=PURPLE,
                      font=("Courier New", 8), anchor="w")
        c.create_text(W-pad, H-pad+12, text=f"gen {len(hist)}", fill=FG2,
                      font=("Courier New", 8), anchor="e")


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AppImagen(root)
    root.mainloop()
