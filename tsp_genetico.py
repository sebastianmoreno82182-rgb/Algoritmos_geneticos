import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import time
import threading

#  ALGORITMO GENETICO - TSP

def calcular_distancia(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def distancia_total(ruta, ciudades):
    total = 0
    n = len(ruta)
    for i in range(n):
        total += calcular_distancia(ciudades[ruta[i]], ciudades[ruta[(i + 1) % n]])
    return total

def crear_individuo(n):
    ind = list(range(n))
    random.shuffle(ind)
    return ind

def crear_poblacion(tam_poblacion, n_ciudades):
    return [crear_individuo(n_ciudades) for _ in range(tam_poblacion)]

def aptitud(individuo, ciudades):
    dist = distancia_total(individuo, ciudades)
    return 1 / dist if dist > 0 else float('inf')

def seleccion_torneo(poblacion, ciudades, k=3):
    torneo = random.sample(poblacion, k)
    torneo.sort(key=lambda ind: aptitud(ind, ciudades), reverse=True)
    return torneo[0]

def seleccion_ruleta(poblacion, ciudades):
    aptitudes = [aptitud(ind, ciudades) for ind in poblacion]
    total = sum(aptitudes)
    r = random.uniform(0, total)
    acum = 0
    for ind, ap in zip(poblacion, aptitudes):
        acum += ap
        if acum >= r:
            return ind
    return poblacion[-1]

def cruce_ox(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    hijo = [-1] * n
    hijo[a:b+1] = p1[a:b+1]
    pos = (b + 1) % n
    for gene in p2:
        if gene not in hijo:
            hijo[pos] = gene
            pos = (pos + 1) % n
    return hijo

def mutar_swap(individuo, tasa_mutacion):
    if random.random() < tasa_mutacion:
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

def mutar_inversion(individuo, tasa_mutacion):
    if random.random() < tasa_mutacion:
        i, j = sorted(random.sample(range(len(individuo)), 2))
        individuo[i:j+1] = individuo[i:j+1][::-1]
    return individuo

def algoritmo_genetico(ciudades, tam_poblacion, n_generaciones,
                        tasa_cruce, tasa_mutacion,
                        tipo_seleccion, tipo_mutacion,
                        elitismo, callback=None, stop_flag=None):
    n = len(ciudades)
    poblacion = crear_poblacion(tam_poblacion, n)
    mejor_global = None
    mejor_distancia = float('inf')
    historial = []

    for gen in range(n_generaciones):
        if stop_flag and stop_flag():
            break

        poblacion.sort(key=lambda ind: aptitud(ind, ciudades), reverse=True)
        mejor_actual = poblacion[0]
        dist_actual = distancia_total(mejor_actual, ciudades)

        if dist_actual < mejor_distancia:
            mejor_distancia = dist_actual
            mejor_global = mejor_actual[:]

        historial.append(mejor_distancia)

        if callback:
            callback(gen + 1, mejor_global, mejor_distancia, historial)

        nueva_poblacion = []
        n_elite = max(1, int(tam_poblacion * 0.05)) if elitismo else 0
        nueva_poblacion.extend([ind[:] for ind in poblacion[:n_elite]])

        while len(nueva_poblacion) < tam_poblacion:
            if tipo_seleccion == "Torneo":
                p1 = seleccion_torneo(poblacion, ciudades)
                p2 = seleccion_torneo(poblacion, ciudades)
            else:
                p1 = seleccion_ruleta(poblacion, ciudades)
                p2 = seleccion_ruleta(poblacion, ciudades)

            hijo = cruce_ox(p1, p2) if random.random() < tasa_cruce else p1[:]

            if tipo_mutacion == "Swap":
                hijo = mutar_swap(hijo, tasa_mutacion)
            else:
                hijo = mutar_inversion(hijo, tasa_mutacion)

            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    return mejor_global, mejor_distancia, historial



#  COLORES Y FUENTES
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
YELLOW = "#d29922"
RED    = "#f85149"
FG     = "#e6edf3"
FG2    = "#8b949e"
DARK   = "#21262d"

F_TITLE = ("Courier New", 13, "bold")
F_LABEL = ("Courier New", 10)
F_SMALL = ("Courier New", 9)
F_BTN   = ("Courier New", 10, "bold")


#  INTERFAZ GRAFICA

class AppTSP:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genético - TSP Optimizador de Rutas")
        self.root.configure(bg=BG)
        self.root.geometry("1280x800")
        self.root.minsize(900, 650)

        self.ciudades   = []
        self.mejor_ruta = None
        self.corriendo  = False
        self._stop      = False
        self.hilo       = None
        self.t_inicio   = 0
        self.vars       = {}

        self._build()

    # --------------------------------------
    def _build(self):
        # ---- HEADER -----------------------
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=18, pady=(14, 4))

        tk.Label(hdr, text="TSP GENETIC OPTIMIZER",
                 font=("Courier New", 17, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr,
                 text="   Clic en el mapa para añadir lugares  -  Configura y presiona Ejecutar AG",
                 font=F_SMALL, bg=BG, fg=FG2).pack(side="left")

        # --- CONTENEDOR PRINCIPAL ------------
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # --- PANEL IZQUIERDO ---------------
        left = tk.Frame(main, bg=PANEL,
                        highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        self._section(left, "PARAMETROS")

        self._slider(left, "Poblacion",      "pop",    50,  10,  500, 10, int)
        self._slider(left, "Generaciones",   "gen",   200,  10, 2000, 10, int)
        self._slider(left, "N de ciudades",  "nciu",   15,   4,   60,  1, int)
        self._slider(left, "Tasa de cruce",    "cruce",   0.85, 0.0, 1.0, 0.01, float)
        self._slider(left, "Tasa de mutacion", "mutacion", 0.15, 0.0, 1.0, 0.01, float)

        self._sep(left)

        self._radio_group(left, "Seleccion",  "seleccion",    ["Torneo", "Ruleta"], "Torneo")
        self._radio_group(left, "Mutacion",   "mutacion_tipo", ["Swap", "Inversion"], "Swap")

        self._sep(left)

        self.vars["elitismo"] = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Elitismo (conservar top 5%)",
                       variable=self.vars["elitismo"],
                       font=F_SMALL, bg=PANEL, fg=FG,
                       selectcolor=DARK,
                       activebackground=PANEL,
                       activeforeground=GREEN
                       ).pack(anchor="w", padx=14, pady=(4, 8))

        self._sep(left)

        self.btn_gen  = self._btn(left, "Generar lugares", ACCENT, DARK, self._generar)
        self.btn_clr  = self._btn(left, "Limpiar canvas",   FG2,   DARK, self._limpiar)
        self.btn_run  = self._btn(left, "Ejecutar AG",       BG,   GREEN, self._ejecutar)
        self.btn_stop = self._btn(left, "Detener",            BG,   RED,  self._detener)
        self.btn_stop.config(state="disabled")

        self._sep(left)

        self._section(left, "ESTADISTICAS")
        sf = tk.Frame(left, bg=PANEL)
        sf.pack(fill="x", padx=14, pady=(0, 12))
        self.lbl_gen  = self._stat(sf, "Generacion", ACCENT)
        self.lbl_dist = self._stat(sf, "Mejor dist", GREEN)
        self.lbl_ciud = self._stat(sf, "Lugares",   FG)
        self.lbl_time = self._stat(sf, "Tiempo",     YELLOW)

        # --------- PANEL DERECHO -----------
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=0)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="MAPA DE RUTAS",
                 font=("Courier New", 9, "bold"),
                 bg=BG, fg=FG2, anchor="w").grid(row=0, column=0, sticky="w", pady=(0, 2))

        self.canvas = tk.Canvas(right, bg="#0a0e14",
                                highlightbackground=BORDER,
                                highlightthickness=1,
                                cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._click_mapa)

        tk.Frame(right, bg=BG, height=8).grid(row=2, column=0)

        tk.Label(right, text="EVOLUCION — Mejor distancia por generacion",
                 font=("Courier New", 9, "bold"),
                 bg=BG, fg=FG2, anchor="w").grid(row=3, column=0, sticky="w", pady=(0, 2))

        self.canvas_evo = tk.Canvas(right, bg="#0a0e14", height=170,
                                    highlightbackground=BORDER,
                                    highlightthickness=1)
        self.canvas_evo.grid(row=4, column=0, sticky="ew")

        self.progress_var = tk.DoubleVar()
        style = ttk.Style()
        style.theme_use("default")
        style.configure("g.Horizontal.TProgressbar",
                        troughcolor=DARK, background=GREEN, thickness=5)
        ttk.Progressbar(right, variable=self.progress_var,
                        style="g.Horizontal.TProgressbar",
                        maximum=100).grid(row=5, column=0, sticky="ew", pady=(6, 2))

        self.lbl_status = tk.Label(right,
            text="Listo. Añade lugares en el mapa o usa 'Generar lugares'.",
            font=F_SMALL, bg=BG, fg=FG2, anchor="w")
        self.lbl_status.grid(row=6, column=0, sticky="w")

    # ------- Helpers de construccion -----------------

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=F_TITLE,
                 bg=PANEL, fg=ACCENT).pack(pady=(12, 4), padx=14, anchor="w")

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=10, pady=5)

    def _btn(self, parent, text, fg, bg, cmd):
        b = tk.Button(parent, text=text, font=F_BTN,
                      bg=bg, fg=fg,
                      activebackground=fg, activeforeground=bg,
                      relief="flat", cursor="hand2",
                      padx=10, pady=7,
                      command=cmd)
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
                           font=F_LABEL, bg=PANEL, fg=ACCENT, width=6)
        val_lbl.pack(side="right")

        def on_change(v):
            val_lbl.config(text=fmt.format(typ(float(v))))

        tk.Scale(frame, from_=mn, to=mx, orient="horizontal",
                 variable=var, resolution=step,
                 bg=PANEL, fg=ACCENT, troughcolor=DARK,
                 highlightthickness=0, showvalue=False,
                 command=on_change).pack(fill="x")

    def _radio_group(self, parent, label, key, options, default):
        tk.Label(parent, text=label, font=F_LABEL,
                 bg=PANEL, fg=FG2).pack(anchor="w", padx=14, pady=(4, 0))
        var = tk.StringVar(value=default)
        self.vars[key] = var
        row = tk.Frame(parent, bg=PANEL)
        row.pack(anchor="w", padx=14, pady=(0, 4))
        for op in options:
            tk.Radiobutton(row, text=op, variable=var, value=op,
                           font=F_SMALL, bg=PANEL, fg=FG,
                           selectcolor=DARK,
                           activebackground=PANEL,
                           activeforeground=ACCENT).pack(side="left", padx=6)

    def _stat(self, parent, label, val_color=FG):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{label}:", font=F_SMALL,
                 bg=PANEL, fg=FG2, width=12, anchor="w").pack(side="left")
        lbl = tk.Label(row, text="—", font=F_SMALL, bg=PANEL, fg=val_color)
        lbl.pack(side="left")
        return lbl

    # ---------- Acciones ---------------------------

    def _click_mapa(self, event):
        if self.corriendo:
            return
        self.ciudades.append((event.x, event.y))
        self._draw_cities()
        self.lbl_ciud.config(text=str(len(self.ciudades)))

    def _generar(self):
        if self.corriendo:
            return
        n = self.vars["nciu"].get()
        w = max(self.canvas.winfo_width(),  500)
        h = max(self.canvas.winfo_height(), 350)
        m = 40
        self.ciudades = [(random.randint(m, w-m), random.randint(m, h-m))
                         for _ in range(n)]
        self.mejor_ruta = None
        self._draw_cities()
        self.lbl_ciud.config(text=str(n))
        self.lbl_status.config(text=f"{n} lugares generados. Presiona Ejecutar AG.")

    def _limpiar(self):
        if self.corriendo:
            return
        self.ciudades   = []
        self.mejor_ruta = None
        self.canvas.delete("all")
        self.canvas_evo.delete("all")
        self.progress_var.set(0)
        for lbl in [self.lbl_gen, self.lbl_dist, self.lbl_ciud, self.lbl_time]:
            lbl.config(text="—")
        self.lbl_status.config(text="Canvas limpio.")

    def _ejecutar(self):
        if len(self.ciudades) < 4:
            messagebox.showwarning("Pocos lugares",
                "Necesitas al menos 4 lugares.\n"
                "Agregarlos en el mapa o usa 'Generar lugares'.")
            return
        if self.corriendo:
            return
        self.corriendo = True
        self._stop     = False
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_gen.config(state="disabled")
        self.btn_clr.config(state="disabled")
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

        def cb(gen, ruta, dist, hist):
            pct     = (gen / n_gen) * 100
            elapsed = time.time() - self.t_inicio
            self.root.after(0, self._update_ui, gen, n_gen, ruta, dist, hist, pct, elapsed)

        mejor, dist, hist = algoritmo_genetico(
            ciudades        = self.ciudades,
            tam_poblacion   = self.vars["pop"].get(),
            n_generaciones  = n_gen,
            tasa_cruce      = self.vars["cruce"].get(),
            tasa_mutacion   = self.vars["mutacion"].get(),
            tipo_seleccion  = self.vars["seleccion"].get(),
            tipo_mutacion   = self.vars["mutacion_tipo"].get(),
            elitismo        = self.vars["elitismo"].get(),
            callback        = cb,
            stop_flag       = lambda: self._stop
        )
        self.mejor_ruta = mejor
        self.root.after(0, self._done, dist)

    def _update_ui(self, gen, n_gen, ruta, dist, hist, pct, elapsed):
        self.progress_var.set(pct)
        self.lbl_gen.config(text=f"{gen}/{n_gen}")
        self.lbl_dist.config(text=f"{dist:.1f} px")
        self.lbl_time.config(text=f"{elapsed:.1f} s")
        self._draw_route(ruta)
        self._draw_chart(hist, n_gen)

    def _done(self, dist):
        self.corriendo = False
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_gen.config(state="normal")
        self.btn_clr.config(state="normal")
        elapsed = time.time() - self.t_inicio
        self.lbl_time.config(text=f"{elapsed:.1f} s")
        self.lbl_status.config(
            text=f"Completado — Mejor distancia: {dist:.1f} px  |  {elapsed:.1f} s")

    # ---------- Dibujo -------------------

    def _draw_cities(self):
        self.canvas.delete("all")
        for i, (x, y) in enumerate(self.ciudades):
            r = 7
            self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                    fill=ACCENT, outline="white", width=1)
            self.canvas.create_text(x+13, y-13, text=str(i),
                                    fill=FG2, font=("Courier New", 9))

    def _draw_route(self, ruta):
        self.canvas.delete("all")
        if not ruta:
            return
        n = len(ruta)
        for i in range(n):
            a = self.ciudades[ruta[i]]
            b = self.ciudades[ruta[(i+1) % n]]
            self.canvas.create_line(a[0], a[1], b[0], b[1],
                                    fill=GREEN, width=2)
        for i, (x, y) in enumerate(self.ciudades):
            r = 7
            col = YELLOW if i == ruta[0] else ACCENT
            self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                    fill=col, outline="white", width=1)
            self.canvas.create_text(x+13, y-13, text=str(i),
                                    fill=FG2, font=("Courier New", 9))

    def _draw_chart(self, hist, n_gen):
        c = self.canvas_evo
        c.delete("all")
        W = c.winfo_width()  or 800
        H = c.winfo_height() or 170
        pad = 36

        if len(hist) < 2:
            return

        mn  = min(hist)
        mx  = max(hist)
        rng = mx - mn if mx != mn else 1

        def cx(i): return pad + (i / max(len(hist)-1, 1)) * (W - 2*pad)
        def cy(v): return pad + (1 - (v - mn) / rng) * (H - 2*pad)

        for k in range(5):
            yg = pad + k * (H - 2*pad) / 4
            c.create_line(pad, yg, W-pad, yg, fill=BORDER, dash=(2, 4))

        pts_fill = [pad, H-pad]
        pts_line = []
        for i, v in enumerate(hist):
            pts_fill += [cx(i), cy(v)]
            pts_line += [cx(i), cy(v)]
        pts_fill += [cx(len(hist)-1), H-pad]

        if len(pts_fill) >= 6:
            c.create_polygon(*pts_fill, fill="#1a3a1a", outline="")
        if len(pts_line) >= 4:
            c.create_line(*pts_line, fill=GREEN, width=2, smooth=True)

        lx, ly = cx(len(hist)-1), cy(hist[-1])
        c.create_oval(lx-4, ly-4, lx+4, ly+4, fill=YELLOW, outline="")

        c.create_text(pad, pad-12,     text=f"{mx:.0f}", fill=FG2,   font=("Courier New", 8), anchor="w")
        c.create_text(pad, H-pad+12,   text=f"{mn:.0f}", fill=GREEN, font=("Courier New", 8), anchor="w")
        c.create_text(W-pad, H-pad+12, text=f"gen {len(hist)}", fill=FG2,
                      font=("Courier New", 8), anchor="e")



#  ------------- MAIN --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AppTSP(root)
    root.mainloop()
