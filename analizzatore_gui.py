import csv
import json
import math
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator

# ============================================================
# COSTANTI DI VISUALIZZAZIONE
# ============================================================
NUM_TEST = 1000
RIPETIZIONI_PER_DOMANDA = 5


# ============================================================
# CLASSE CONTENITORE
# ============================================================
class RisultatiBenchmark:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.risultati_per_tabella = []
        self.dist_corrette = {}
        self.dist_errate = {}


# ============================================================
# LETTURA DAL CSV
# ============================================================
def carica_da_csv(filename="risultati_benchmark.csv") -> RisultatiBenchmark:
    risultati = RisultatiBenchmark()

    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                esito_corretto = str(row["corretta"]).strip().lower() == "true"
                risposta_reale = row["reale"]
                generata_dist = row["generata_dist"]

                # Ricalcolo le distribuzioni e la matrice di confusione
                if esito_corretto:
                    risultati.dist_corrette[generata_dist] = risultati.dist_corrette.get(generata_dist, 0) + 1
                    if risposta_reale == "true":
                        risultati.tp += 1
                    else:
                        risultati.tn += 1
                else:
                    risultati.dist_errate[generata_dist] = risultati.dist_errate.get(generata_dist, 0) + 1
                    if risposta_reale == "false":
                        risultati.fp += 1
                    else:
                        risultati.fn += 1

                # Ricostruzione della riga per la tabella
                dati_domanda = {
                    "id": int(row["id"]),
                    "domanda": row["domanda"],
                    "reale": risposta_reale,
                    "alternative": json.loads(row["alternative_json"]),  # Riconversione in array di dizionari
                    "prob_media_unita": row["prob_media_unita"],
                    "generata_dist": generata_dist,
                    "corretta": esito_corretto
                }
                risultati.risultati_per_tabella.append(dati_domanda)

    except FileNotFoundError:
        print(f"Errore: Il file {filename} non è stato trovato. Esegui prima il generatore.")
        exit()

    return risultati


# ============================================================
# ANALYZER: CALCOLO STATISTICHE ENTROPIA (AGGIORNATO)
# ============================================================
def analyzer(risultati: RisultatiBenchmark) -> RisultatiBenchmark:
    for riga in risultati.risultati_per_tabella:
        entropie = []
        somma_raw_t = 0.0
        somma_raw_f = 0.0

        for alt in riga["alternative"]:
            p_t = alt.get("p_true_raw", 0.0)
            p_f = alt.get("p_false_raw", 0.0)

            # 1. Calcolo entropia della singola distribuzione NON NORMALIZZATA
            ent_singola = 0.0
            if p_t > 0: ent_singola -= p_t * math.log2(p_t)
            if p_f > 0: ent_singola -= p_f * math.log2(p_f)

            entropie.append(ent_singola)

            # Accumulo le probabilità grezze per fare la media successiva
            somma_raw_t += p_t
            somma_raw_f += p_f

        k = len(riga["alternative"])
        if k > 0:
            # Salvataggio Min/Max delle entropie singole (non normalizzate)
            riga["min_ent"] = min(entropie)
            riga["max_ent"] = max(entropie)

            # 2. Calcolo della MEDIA grezza sulle k ripetizioni
            avg_raw_t = somma_raw_t / k
            avg_raw_f = somma_raw_f / k

            # 3. NORMALIZZAZIONE della MEDIA
            somma_avg = avg_raw_t + avg_raw_f
            if somma_avg > 0:
                norm_avg_t = avg_raw_t / somma_avg
                norm_avg_f = avg_raw_f / somma_avg
            else:
                norm_avg_t, norm_avg_f = 0.0, 0.0

            # 4. Calcolo entropia della media NORMALIZZATA
            ent_media = 0.0
            if norm_avg_t > 0: ent_media -= norm_avg_t * math.log2(norm_avg_t)
            if norm_avg_f > 0: ent_media -= norm_avg_f * math.log2(norm_avg_f)

            riga["avg_ent"] = ent_media
        else:
            riga["min_ent"] = 0.0
            riga["max_ent"] = 0.0
            riga["avg_ent"] = 0.0

    return risultati


# ============================================================
# INTERFACCIA GRAFICA
# ============================================================
def mostra_interfaccia_completa(res: RisultatiBenchmark):
    finestra = tk.Tk()
    finestra.title(f"Report Benchmark ({RIPETIZIONI_PER_DOMANDA} Ripetizioni su {NUM_TEST} Domande)")
    finestra.geometry("1350x650")

    notebook = ttk.Notebook(finestra)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # ── SCHEDA 1: Tabella ad Albero e Matrice ──
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Dati e Matrice")

    totale_corrette = res.tp + res.tn
    percentuale_corrette = (totale_corrette / NUM_TEST) * 100 if NUM_TEST > 0 else 0
    tk.Label(tab1, text=f"Risultati (Maggioranza): {totale_corrette}/{NUM_TEST} Corrette ({percentuale_corrette:.1f}%)",
             font=("Helvetica", 14, "bold")).pack(pady=10)

    # Matrice di confusione
    frame_matrice = tk.Frame(tab1)
    frame_matrice.pack(side=tk.BOTTOM, pady=10)
    tk.Label(frame_matrice, text="Matrice di Confusione", font=("Helvetica", 12, "bold")).grid(row=0, column=0,
                                                                                               columnspan=3, pady=5)
    tk.Label(frame_matrice, text="Modello: True").grid(row=1, column=1)
    tk.Label(frame_matrice, text="Modello: False").grid(row=1, column=2)
    tk.Label(frame_matrice, text="Realtà: True").grid(row=2, column=0)
    tk.Label(frame_matrice, text=f"TP\n{res.tp}", bg="#c6efce", width=10, height=2, relief="groove").grid(row=2,
                                                                                                          column=1)
    tk.Label(frame_matrice, text=f"FN\n{res.fn}", bg="#ffc7ce", width=10, height=2, relief="groove").grid(row=2,
                                                                                                          column=2)
    tk.Label(frame_matrice, text="Realtà: False").grid(row=3, column=0)
    tk.Label(frame_matrice, text=f"FP\n{res.fp}", bg="#ffc7ce", width=10, height=2, relief="groove").grid(row=3,
                                                                                                          column=1)
    tk.Label(frame_matrice, text=f"TN\n{res.tn}", bg="#c6efce", width=10, height=2, relief="groove").grid(row=3,
                                                                                                          column=2)

    # Tabella
    frame_tabella = tk.Frame(tab1)
    frame_tabella.pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=5)

    colonne = ("ID", "Domanda", "Probabilita", "Distribuzione", "Reale", "MinEnt", "MaxEnt", "AvgEnt")
    tabella = ttk.Treeview(frame_tabella, columns=colonne, show="tree headings")

    tabella.heading("#0", text="")
    tabella.column("#0", width=40, stretch=tk.NO, anchor="center")
    tabella.heading("ID", text="N°")
    tabella.column("ID", width=40, anchor="center")
    tabella.heading("Domanda", text="Domanda Originale / Varianti")
    tabella.column("Domanda", width=250)
    tabella.heading("Probabilita", text=f"Medie (%T / %F / %O) [{RIPETIZIONI_PER_DOMANDA} Rip]")
    tabella.column("Probabilita", width=380, anchor="center")
    tabella.heading("Distribuzione", text="Dist. [Rip.]")
    tabella.column("Distribuzione", width=70, anchor="center")
    tabella.heading("Reale", text="Reale")
    tabella.column("Reale", width=80, anchor="center")
    tabella.heading("MinEnt", text="MinEnt")
    tabella.column("MinEnt", width=60, anchor="center")
    tabella.heading("MaxEnt", text="MaxEnt")
    tabella.column("MaxEnt", width=60, anchor="center")
    tabella.heading("AvgEnt", text="AvgEnt")
    tabella.column("AvgEnt", width=60, anchor="center")

    scrollbar = ttk.Scrollbar(frame_tabella, orient=tk.VERTICAL, command=tabella.yview)
    tabella.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tabella.pack(side=tk.LEFT, fill="both", expand=True)

    tabella.tag_configure("verde", background="#c6efce", foreground="#006100")
    tabella.tag_configure("rosso", background="#ffc7ce", foreground="#9c0006")
    tabella.tag_configure("figlio", background="#f0f8ff", foreground="#333333")

    for riga in res.risultati_per_tabella:
        colore = "verde" if riga["corretta"] else "rosso"
        padre_id = tabella.insert("", tk.END, text="", values=(
            riga["id"],
            riga["domanda"][:60] + "...",
            riga["prob_media_unita"],
            riga["generata_dist"],
            riga["reale"].upper(),
            f"{riga.get('min_ent', 0):.3f}",
            f"{riga.get('max_ent', 0):.3f}",
            f"{riga.get('avg_ent', 0):.3f}"
        ), tags=(colore,))

        for i, alt in enumerate(riga["alternative"]):
            tabella.insert(padre_id, tk.END, text=f"{i + 1}.", values=(
                "",
                " ↳ " + alt["domanda_alt"][:65] + "...",
                alt['prob_unita_alt'],
                "-",
                alt["risposta_pulita"].upper(),
                "", "", ""
            ), tags=("figlio",))

    # ── SCHEDA 3 E 4 ──
    categorie_x = [f"{k}-{RIPETIZIONI_PER_DOMANDA - k}" for k in
                   range(RIPETIZIONI_PER_DOMANDA, math.ceil(RIPETIZIONI_PER_DOMANDA / 2) - 1, -1)]
    chiavi_extra = sorted(set(res.dist_corrette.keys()).union(set(res.dist_errate.keys())), reverse=True)
    for key in chiavi_extra:
        if key not in categorie_x:
            categorie_x.append(key)

    domande_giuste = [res.dist_corrette.get(cat, 0) for cat in categorie_x]
    domande_sbagliate = [res.dist_errate.get(cat, 0) for cat in categorie_x]

    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Conteggio per Distrib.")
    fig3 = Figure(figsize=(6, 4), dpi=100)
    ax3 = fig3.add_subplot(111)
    ax3.bar(categorie_x, domande_giuste, color='#4CAF50', label='Domande Corrette')
    ax3.bar(categorie_x, domande_sbagliate, bottom=domande_giuste, color='#F44336', label='Domande Errate')
    ax3.set_title("Conteggio Domande Corrette/Errate per Distribuzione")
    ax3.set_xlabel("Combinazione (Maggioranza - Minoranza)")
    ax3.set_ylabel("Numero di Domande Totali")
    ax3.legend()
    ax3.yaxis.set_major_locator(MultipleLocator(max(1, NUM_TEST // 10)))
    ax3.grid(axis='y', which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    FigureCanvasTkAgg(fig3, master=tab3).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text="Calibrazione per Distrib.")
    perc_giuste = [(g / (g + s) * 100) if (g + s) > 0 else 0 for g, s in zip(domande_giuste, domande_sbagliate)]
    perc_sbagliate = [(s / (g + s) * 100) if (g + s) > 0 else 0 for g, s in zip(domande_giuste, domande_sbagliate)]
    fig4 = Figure(figsize=(6, 4), dpi=100)
    ax4 = fig4.add_subplot(111)
    x = range(len(categorie_x))
    width = 0.35
    bars_giuste = ax4.bar([i - width / 2 for i in x], perc_giuste, width, color='#4CAF50', label='Domande Corrette (%)')
    bars_errate = ax4.bar([i + width / 2 for i in x], perc_sbagliate, width, color='#F44336',
                          label='Domande Errate (%)')

    def annota_barre(bars, color):
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9, color=color, fontweight='bold')

    annota_barre(bars_giuste, '#006100')
    annota_barre(bars_errate, '#9c0006')

    ax4.set_xticks(x)
    ax4.set_xticklabels(categorie_x)
    ax4.set_title("Diagramma di Calibrazione: Tasso Corrette/Errate per Distribuzione")
    ax4.set_xlabel("Combinazione (Maggioranza - Minoranza)")
    ax4.set_ylabel("Percentuale di Accuratezza/Errore (%)")
    ax4.set_ylim(0, 115)
    ax4.legend(loc='upper right')
    ax4.yaxis.set_major_locator(MultipleLocator(10))
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig4, master=tab4).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ── PLOT ENTROPIA ──
    max_ent_corrette = [r.get("max_ent", 0.0) for r in res.risultati_per_tabella if r["corretta"]]
    max_ent_errate = [r.get("max_ent", 0.0) for r in res.risultati_per_tabella if not r["corretta"]]
    avg_ent_corrette = [r.get("avg_ent", 0.0) for r in res.risultati_per_tabella if r["corretta"]]
    avg_ent_errate = [r.get("avg_ent", 0.0) for r in res.risultati_per_tabella if not r["corretta"]]

    # ── SCHEDA 5: BOXPLOT MAX ENTROPIA ──
    tab5 = ttk.Frame(notebook)
    notebook.add(tab5, text="Boxplot MaxEnt")
    fig5 = Figure(figsize=(6, 4), dpi=100)
    ax5 = fig5.add_subplot(111)
    dati_max = [max_ent_corrette if max_ent_corrette else [0.0], max_ent_errate if max_ent_errate else [0.0]]
    bplot1 = ax5.boxplot(dati_max, labels=['Corrette', 'Errate'], patch_artist=True)
    colors = ['#4CAF50', '#F44336']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_title("Distribuzione MaxEnt per Domande Corrette ed Errate")
    ax5.set_ylabel("Entropia Massima (MaxEnt)")
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig5, master=tab5).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ── SCHEDA 6: BOXPLOT AVG ENTROPIA ──
    tab6 = ttk.Frame(notebook)
    notebook.add(tab6, text="Boxplot AvgEnt")
    fig6 = Figure(figsize=(6, 4), dpi=100)
    ax6 = fig6.add_subplot(111)
    dati_avg = [avg_ent_corrette if avg_ent_corrette else [0.0], avg_ent_errate if avg_ent_errate else [0.0]]
    bplot2 = ax6.boxplot(dati_avg, labels=['Corrette', 'Errate'], patch_artist=True)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_title("Distribuzione AvgEnt per Domande Corrette ed Errate")
    ax6.set_ylabel("Entropia Media (AvgEnt) [0.0 - 1.0]")
    ax6.set_ylim(-0.05, 1.05)
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig6, master=tab6).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ── SCHEDA 7: ACCURATEZZA PER LIVELLI DI ENTROPIA ──
    tab7 = ttk.Frame(notebook)
    notebook.add(tab7, text="Accuratezza vs Entropia")
    fig7 = Figure(figsize=(6, 4), dpi=100)
    ax7 = fig7.add_subplot(111)
    bassa = [r["corretta"] for r in res.risultati_per_tabella if r.get("avg_ent", 0.0) <= 0.3]
    media = [r["corretta"] for r in res.risultati_per_tabella if 0.3 < r.get("avg_ent", 0.0) <= 0.7]
    alta = [r["corretta"] for r in res.risultati_per_tabella if r.get("avg_ent", 0.0) > 0.7]

    acc_bassa = (sum(bassa) / len(bassa) * 100) if len(bassa) > 0 else 0
    acc_media = (sum(media) / len(media) * 100) if len(media) > 0 else 0
    acc_alta = (sum(alta) / len(alta) * 100) if len(alta) > 0 else 0

    etichette_fasce = ['Bassa\n(0.0 - 0.3)', 'Media\n(0.3 - 0.7)', 'Alta\n(0.7 - 1.0)']
    valori_accuratezza = [acc_bassa, acc_media, acc_alta]

    barre_calibrazione = ax7.bar(etichette_fasce, valori_accuratezza, color='#2196F3', alpha=0.8)
    for bar in barre_calibrazione:
        altezza = bar.get_height()
        ax7.annotate(f'{altezza:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, altezza),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    ax7.set_title("Accuratezza del Modello per Livelli di Entropia (AvgEnt)")
    ax7.set_xlabel("Fascia di Entropia")
    ax7.set_ylabel("Accuratezza (%)")
    ax7.set_ylim(0, 115)
    ax7.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig7, master=tab7).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    finestra.mainloop()


if __name__ == "__main__":
    # 1. Carica dal file CSV
    dati_benchmark = carica_da_csv("risultati_benchmark.csv")

    # 2. Esegue i calcoli per l'entropia
    dati_benchmark = analyzer(dati_benchmark)

    # 3. Mostra la GUI
    mostra_interfaccia_completa(dati_benchmark)