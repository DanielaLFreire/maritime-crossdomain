"""
crossdomain — utilitários do experimento de aumento cross-domain do CITRA-3D-Real.

Submódulos:
  loaders    — leitura de anotações (YOLO/VOC/COCO/CSV/ABOShips) unificada
  profiling  — perfil estrutural e distância composta ao domínio-alvo
  prepare    — conversão YOLO + splits disjuntos por origem
"""
from . import loaders, prepare, profiling, synth  # noqa: F401

__all__ = ["loaders", "profiling", "prepare", "synth"]
__version__ = "0.1.0"
