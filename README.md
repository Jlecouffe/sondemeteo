# Simulateur de trame RS41 (PlutoSDR)

Application (interface graphique + `.exe`) qui construit une trame radio de
radiosonde meteo **Vaisala RS41** a partir de parametres saisis (position,
trajectoire, meteo, identifiants), la module en GFSK, et l'emet via un
**ADALM-Pluto (PlutoSDR)**. Objectif : disposer d'un signal RS41 controle
pour tester ton propre logiciel de decodage, sans dependre d'un vrai
lancement de ballon.

## ⚠️ Avertissement reglementaire

La bande utilisee par les radiosondes meteo (~400-406 MHz) est une bande
**reglementee** (meteorologie / dans certains pays partagee avec du
radioamateur sous conditions). Emettre sur cette bande sans autorisation
est illegal dans la plupart des pays.

Pour les tests :
- privilegie une liaison filaire attenuee (cable + attenuateurs) entre la
  sortie TX du Pluto et l'entree RX de ton recepteur/SDR, plutot qu'une
  emission en air libre ;
- a defaut, utilise le **mode fichier** de l'application (voir plus bas),
  qui genere les echantillons IQ dans un fichier sans jamais piloter le
  Pluto, pour rejouer le signal dans un outil comme GNU Radio/GQRX/ton
  decodeur en entree fichier ;
- verifie la reglementation locale (ANFR en France, etc.) et reste a
  puissance minimale si tu emets reellement.

## Fiabilite du format de trame

Le format RS41 est issu de retro-ingenierie communautaire (aucune
specification officielle publique). Ce module recoupe plusieurs sources
independantes (`rs1729/RS`, `projecthorus/radiosonde_auto_rx`,
`bazjo/RS41_Decoding`) pour les elements suivants, verifies par
recoupement et par test :
- mot de synchronisation et masque de brouillage (64 octets),
- CRC16 (poly `0x1021`, init `0xFFFF`) par bloc,
- parametres du code correcteur Reed-Solomon (GF(256), polynome `0x11D`,
  racine de depart `alpha^0`, 24 octets de parite, code raccourci) —
  verifie par test de syndrome (`tests/test_frame.py`),
- conversion position geodesique WGS84 -> ECEF (formules standard).

**Simplifications assumees**, a adapter si besoin dans `rs41sim/frame.py` :
- l'ordre exact des octets d'identifiant de bloc et les decalages precis
  a l'interieur de chaque bloc n'ont pas pu etre verifies bit-a-bit contre
  le firmware reel (pas d'echantillon materiel disponible pour comparer) ;
- le bloc **PTU** (temperature/humidite/pression) n'essaie pas de reproduire
  le schema reel Vaisala (mesure ADC brute + polynome de calibration
  transmis par fragments sur ~50 trames). Il encode directement les
  valeurs voulues sous forme de nombres scales (voir `_ptu_block` dans
  `rs41sim/frame.py`). Si ton decodeur a besoin du schema reel, c'est le
  point a completer.

Si tu as deja un exemple de trame reelle (capture SDR) ou le code de ton
decodeur, je peux ajuster precisement ces points pour qu'ils correspondent
exactement.

## Prerequis

- Windows 10/11 (fonctionne aussi sous Python standard sur Linux/macOS).
- Un ADALM-Pluto avec les pilotes **libiio** installes et fonctionnels
  (memes pilotes que ceux utilises par ton logiciel de decodage). Sous
  Windows, installe le paquet officiel Analog Devices "PlutoSDR/M2k USB
  Drivers" si ce n'est pas deja fait — sans lui, l'interface s'ouvre mais
  l'emission echoue avec un message explicite.
- Pour reconstruire l'exe toi-meme : Python 3.10+ et les paquets de
  `requirements.txt`.

## Utilisation (exe pre-construit)

1. Lance `dist\RS41Simulator.exe`.
2. Onglet **Sonde / trajectoire / meteo** : renseigne serie, position de
   depart, vitesse ascensionnelle/horizontale, temperature/humidite/pression.
3. Onglet **Parametres radio** : URI du Pluto (`ip:192.168.2.1` en USB-Ethernet,
   ou `usb:` en USB direct), frequence, gain TX, frequence d'echantillonnage,
   deviation, periode entre trames. Coche "Mode fichier" pour generer les
   IQ sans toucher au Pluto.
4. **Demarrer l'emission** — une trame est envoyee toutes les *N* secondes
   (parametre "Periode entre trames"), la position/altitude evolue entre
   chaque trame selon la vitesse ascensionnelle/horizontale saisie.
5. **Arreter** pour stopper proprement.

## Reconstruire depuis les sources

```powershell
python -m pip install -r requirements.txt
python main.py              # lancer directement sans construire l'exe
.\build_exe.ps1              # construire dist\RS41Simulator.exe
```

## Tests

```powershell
python -m pip install pytest
python -m pytest tests -v
```

## Structure

```
rs41sim/
  scrambling.py   en-tete, masque de brouillage
  crc.py          CRC16-CCITT
  rs_fec.py       Reed-Solomon(255,231) raccourci
  gps.py          WGS84 -> ECEF, semaine/TOW GPS
  frame.py        assemblage complet d'une trame (blocs + FEC)
  modulate.py     mise en bits, mise en forme gaussienne, GFSK -> IQ
  trajectory.py   evolution position/altitude entre deux trames
  pluto_tx.py     pilotage TX du Pluto (pyadi-iio)
  app.py          interface graphique Tkinter
tests/            tests de coherence du format de trame
main.py           point d'entree
build_exe.ps1     construction de l'exe (PyInstaller)
```
