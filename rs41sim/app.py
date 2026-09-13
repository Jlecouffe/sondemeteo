"""Interface graphique Tkinter : saisie des parametres et emission RS41 via Pluto."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .frame import SondeParams, build_frame
from .modulate import gfsk_modulate
from .pluto_tx import PlutoNotAvailable, PlutoTransmitter
from .trajectory import advance

APP_TITLE = "Simulateur de trame RS41 (PlutoSDR)"


class RFParams:
    def __init__(self) -> None:
        self.uri = "ip:192.168.2.1"
        self.center_freq_mhz = 405.0
        self.tx_gain_db = -30.0
        self.sample_rate_msps = 2.5
        self.deviation_hz = 2400.0
        # BT=0.5 (spec RS41 reelle) rend les transitions trop lissees pour le
        # decodeur cote radtel-tools (LocalSondeDecoder.kt), qui ne fait que
        # du comptage de passages par zero sans filtre adapte. Verifie par
        # simulation bout-en-bout AU DEBIT REEL DU PONT RT-660 (32 kHz, pas
        # 44.1 kHz) : en dessous de BT=3.0, le calage bit derive et la trame
        # se decode faux (parfois meme le marqueur GPS3 tombe juste par
        # coincidence, avec des donnees ECEF fausses derriere). BT=4.0 garde
        # de la marge, verifie correct sur plusieurs positions/indicatifs.
        self.bt = 4.0
        self.burst_period_s = 1.0
        self.file_mode = False
        self.output_dir = str(Path.home())


FIELD_LABELS = {
    "serial": "Numero de serie",
    "frame_counter": "Compteur de trame (depart)",
    "latitude_deg": "Latitude (deg)",
    "longitude_deg": "Longitude (deg)",
    "altitude_m": "Altitude (m)",
    "climb_rate_ms": "Vitesse ascensionnelle (m/s)",
    "horizontal_speed_ms": "Vitesse horizontale (m/s)",
    "heading_deg": "Cap (deg)",
    "num_sats": "Nb satellites simules",
    "pdop_x10": "PDOP x10",
    "temperature_c": "Temperature (C)",
    "humidity_pct": "Humidite (%RH)",
    "pressure_hpa": "Pression (hPa)",
    "battery_decivolts": "Batterie (x0.1 V)",
}

RF_LABELS = [
    ("uri", "URI Pluto (ex: ip:192.168.2.1 ou usb:1.4.5)"),
    ("center_freq_mhz", "Frequence centrale (MHz)"),
    ("tx_gain_db", "Gain TX (dB, ex: -30)"),
    ("sample_rate_msps", "Frequence d'echantillonnage (MSps, mini ~2.1 sur Pluto)"),
    ("deviation_hz", "Deviation FSK (Hz)"),
    ("bt", "Filtre gaussien BT (>=3.0 recommande pour rester decodable par radtel-tools a 32 kHz)"),
    ("burst_period_s", "Periode entre trames (s)"),
]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(APP_TITLE)

        self.params = SondeParams()
        self.rf = RFParams()
        self.vars: dict[str, tk.StringVar] = {}
        self.rf_vars: dict[str, tk.StringVar] = {}
        self.file_mode_var = tk.BooleanVar(value=False)

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._build_ui()
        self.root.after(150, self._drain_log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        sonde_tab = ttk.Frame(notebook)
        rf_tab = ttk.Frame(notebook)
        notebook.add(sonde_tab, text="Sonde / trajectoire / meteo")
        notebook.add(rf_tab, text="Parametres radio")

        self._build_sonde_tab(sonde_tab)
        self._build_rf_tab(rf_tab)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        self.start_btn = ttk.Button(bottom, text="Demarrer l'emission", command=self._on_start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bottom, text="Arreter", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Pret.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", padx=12)

        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill="both", expand=False, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_frame, height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _build_sonde_tab(self, parent: ttk.Frame) -> None:
        row = 0
        for f in fields(SondeParams):
            label = FIELD_LABELS.get(f.name, f.name)
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(getattr(self.params, f.name)))
            ttk.Entry(parent, textvariable=var, width=20).grid(row=row, column=1, padx=4, pady=2)
            self.vars[f.name] = var
            row += 1

    def _build_rf_tab(self, parent: ttk.Frame) -> None:
        row = 0
        for name, label in RF_LABELS:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(getattr(self.rf, name)))
            ttk.Entry(parent, textvariable=var, width=28).grid(row=row, column=1, padx=4, pady=2)
            self.rf_vars[name] = var
            row += 1

        ttk.Checkbutton(
            parent, text="Mode fichier (enregistre l'IQ, n'emet pas sur le Pluto)",
            variable=self.file_mode_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(10, 2))
        row += 1

        out_row = row
        ttk.Label(parent, text="Dossier de sortie (mode fichier)").grid(row=out_row, column=0, sticky="w", padx=4, pady=2)
        self.output_dir_var = tk.StringVar(value=self.rf.output_dir)
        ttk.Entry(parent, textvariable=self.output_dir_var, width=28).grid(row=out_row, column=1, padx=4, pady=2)
        ttk.Button(parent, text="Parcourir...", command=self._pick_output_dir).grid(row=out_row, column=2, padx=4)

    def _pick_output_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(Path.home()))
        if d:
            self.output_dir_var.set(d)

    # ------------------------------------------------------------- helpers
    def _log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _drain_log(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log)

    def _read_params(self) -> tuple[SondeParams, RFParams]:
        p = SondeParams()
        for f in fields(SondeParams):
            raw = self.vars[f.name].get().strip()
            caster = type(getattr(SondeParams(), f.name))
            try:
                setattr(p, f.name, caster(raw))
            except ValueError as exc:
                raise ValueError(f"Valeur invalide pour '{f.name}': {raw!r}") from exc

        rf = RFParams()
        rf.uri = self.rf_vars["uri"].get().strip()
        rf.center_freq_mhz = float(self.rf_vars["center_freq_mhz"].get())
        rf.tx_gain_db = float(self.rf_vars["tx_gain_db"].get())
        rf.sample_rate_msps = float(self.rf_vars["sample_rate_msps"].get())
        rf.deviation_hz = float(self.rf_vars["deviation_hz"].get())
        rf.bt = float(self.rf_vars["bt"].get())
        rf.burst_period_s = float(self.rf_vars["burst_period_s"].get())
        rf.file_mode = self.file_mode_var.get()
        rf.output_dir = self.output_dir_var.get().strip() or str(Path.home())
        return p, rf

    # --------------------------------------------------------------- actions
    def _on_start(self) -> None:
        try:
            params, rf = self._read_params()
        except ValueError as exc:
            messagebox.showerror("Parametre invalide", str(exc))
            return

        if not rf.file_mode and not messagebox.askyesno(
            "Confirmer l'emission RF",
            "Tu es sur le point d'emettre reellement sur "
            f"{rf.center_freq_mhz} MHz via le Pluto ({rf.uri}).\n\n"
            "La bande des radiosondes meteo (~400-406 MHz) est reglementee : "
            "verifie ton autorisation (banc d'essai blinde/attenue, licence "
            "radioamateur adaptee, etc.) avant de continuer.\n\n"
            "Continuer ?",
        ):
            return

        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_loop, args=(params, rf), daemon=True)
        self._worker.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Emission en cours...")

    def _on_stop(self) -> None:
        self._stop_event.set()
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Arret demande...")

    def _on_close(self) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.root.destroy()

    def _run_loop(self, params: SondeParams, rf: RFParams) -> None:
        sample_rate = rf.sample_rate_msps * 1e6
        tx = None
        try:
            if not rf.file_mode:
                tx = PlutoTransmitter(
                    uri=rf.uri,
                    center_freq_hz=rf.center_freq_mhz * 1e6,
                    sample_rate_hz=sample_rate,
                    tx_gain_db=rf.tx_gain_db,
                )
                self._log(f"Pluto connecte ({rf.uri}), TX {rf.center_freq_mhz} MHz.")
            else:
                self._log(f"Mode fichier : ecriture dans {rf.output_dir}")

            last_time = time.monotonic()
            while not self._stop_event.is_set():
                frame_bytes = build_frame(params)
                iq = gfsk_modulate(
                    frame_bytes, sample_rate,
                    deviation_hz=rf.deviation_hz, bt=rf.bt,
                )

                if rf.file_mode:
                    out_path = Path(rf.output_dir) / f"rs41_frame_{params.frame_counter:06d}.cfile"
                    iq.astype(np.complex64).tofile(out_path)
                    self._log(f"Trame #{params.frame_counter} -> {out_path.name} "
                              f"(lat={params.latitude_deg:.5f} lon={params.longitude_deg:.5f} "
                              f"alt={params.altitude_m:.1f} m)")
                else:
                    assert tx is not None
                    tx.send(iq)
                    self._log(f"Trame #{params.frame_counter} emise "
                              f"(lat={params.latitude_deg:.5f} lon={params.longitude_deg:.5f} "
                              f"alt={params.altitude_m:.1f} m)")

                params.frame_counter += 1
                now = time.monotonic()
                advance(params, now - last_time if params.frame_counter > 1 else 0.0)
                last_time = now

                self._stop_event.wait(rf.burst_period_s)

        except PlutoNotAvailable as exc:
            self._log(f"Erreur Pluto : {exc}")
            messagebox.showerror("Pluto indisponible", str(exc))
        except Exception as exc:  # pragma: no cover - remontee a l'utilisateur
            self._log(f"Erreur : {exc}")
            messagebox.showerror("Erreur", str(exc))
        finally:
            if tx is not None:
                tx.close()
            self._stop_event.set()
            self.root.after(0, self._on_stopped)

    def _on_stopped(self) -> None:
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Arrete.")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
