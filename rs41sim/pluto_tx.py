"""Pilotage de l'emission sur un ADALM-Pluto via pyadi-iio (libiio).

L'import de `adi`/`iio` est differe : l'interface graphique doit pouvoir
s'ouvrir et permettre de saisir des parametres meme si les pilotes libiio
ne sont pas installes sur la machine ; l'erreur ne doit survenir qu'au
moment ou l'utilisateur declenche reellement une emission.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


class PlutoNotAvailable(RuntimeError):
    pass


def _candidate_libiio_dirs() -> list[Path]:
    """Repertoires susceptibles de contenir libiio.dll et ses dependances.

    1) le dossier `vendor/libiio_win64` embarque avec l'application (source
       ou exe PyInstaller) ; 2) quelques emplacements courants ou libiio
       est present sur une machine de dev SDR (radioconda, PothosSDR,
       installateur officiel libiio), au cas ou le vendoring serait absent.
    """
    dirs = []

    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parent.parent
    dirs.append(base / "vendor" / "libiio_win64")

    home = Path.home()
    dirs += [
        home / "radioconda" / "Library" / "bin",
        Path("C:/Program Files/Libiio/bin"),
        Path("C:/PothosSDR/bin"),
    ]
    return dirs


def _ensure_libiio_loadable() -> None:
    """Ajoute au PATH / a la recherche de DLL le premier dossier trouve
    contenant libiio.dll, pour que `import iio` (utilise par pyadi-iio)
    puisse la localiser sur Windows."""
    if os.name != "nt":
        return
    for d in _candidate_libiio_dirs():
        if (d / "libiio.dll").is_file():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(d))
            except (AttributeError, OSError):
                pass
            return


def _set_sample_rate(sdr, rate_hz: int) -> None:
    """Regle la frequence d'echantillonnage du Pluto.

    Le setter standard de pyadi-iio (`sdr.sample_rate = ...`) reconfigure au
    passage le filtre FIR programmable de l'AD936x (ecriture de l'attribut
    `filter_fir_config`). Sur certains Pluto/versions de firmware, cette
    ecriture echoue (`OSError: [Errno 22] Invalid argument`) meme si le taux
    demande est valide. Comme ce simulateur n'a pas besoin de decimation FIR
    (4800 bauds, largement sur-echantillonne), on ecrit directement l'attribut
    IIO bas niveau `sampling_frequency` en repli si le setter standard echoue.
    """
    try:
        sdr.sample_rate = rate_hz
    except OSError:
        sdr._set_iio_attr("voltage0", "sampling_frequency", False, rate_hz)


class PlutoTransmitter:
    def __init__(self, uri: str, center_freq_hz: float, sample_rate_hz: float,
                 tx_gain_db: float, rf_bandwidth_hz: float | None = None):
        _ensure_libiio_loadable()
        try:
            import adi  # type: ignore
        except Exception as exc:  # pragma: no cover - depend de l'environnement
            raise PlutoNotAvailable(
                "Impossible d'importer pyadi-iio / libiio "
                f"(erreur d'origine : {type(exc).__name__}: {exc}). "
                "La bibliotheque native libiio.dll est introuvable sur cette "
                "machine (le pilote USB/RNDIS du Pluto seul ne suffit pas). "
                "Installe le paquet officiel 'libiio' pour Windows depuis "
                "https://github.com/analogdevicesinc/libiio/releases "
                "(installateur .exe, coche l'ajout au PATH), puis reessaie."
            ) from exc

        try:
            self._sdr = adi.Pluto(uri)
        except Exception as exc:
            raise PlutoNotAvailable(
                f"Connexion au Pluto ('{uri}') impossible : "
                f"{type(exc).__name__}: {exc}. Verifie l'URI (l'adresse par "
                "defaut du Pluto lui-meme est 'ip:192.168.2.1', 192.168.2.10 "
                "etant en general l'adresse de TON PC sur cette interface, "
                "pas celle du Pluto) et que le Pluto est bien branche."
            ) from exc
        self._sdr.tx_lo = int(center_freq_hz)
        _set_sample_rate(self._sdr, int(sample_rate_hz))
        self._sdr.tx_rf_bandwidth = int(rf_bandwidth_hz or sample_rate_hz)
        self._sdr.tx_hardwaregain_chan0 = float(tx_gain_db)
        self._sdr.tx_cyclic_buffer = False

    def send(self, iq: np.ndarray, scale: float = 2 ** 14) -> None:
        """Envoie un buffer IQ complexe (amplitude normalisee dans [-1, 1])."""
        scaled = (iq * scale).astype(np.complex64)
        self._sdr.tx(scaled)

    def close(self) -> None:
        try:
            self._sdr.tx_destroy_buffer()
        except Exception:
            pass
