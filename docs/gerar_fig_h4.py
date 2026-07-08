"""
Figura H4 — Δ-transferência vs distância estrutural, por via de uso.
Mostra a INTERAÇÃO: a síntese joint é robusta à distância (fica acima de 0);
o uso real degrada com a distância (despenca). Dados: extensão ABOShips +
artigo original InaTech.
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- dados (Δ mAP50 vs B2, em pontos) ---
# via síntese joint
sint_x = [0.95, 8.0]          # ABOShips, InaTech (distância; ≫ representado como 8)
sint_y = [0.69, 1.00]
# via dados reais (pré-treino direto)
real_x = [0.95, 8.0]
real_y = [-0.37, -4.15]
# ponto extra: C-joint (real co-treino) ABOShips
cjoint = (0.95, -0.05)

fig, ax = plt.subplots(figsize=(7, 4.6))

# faixas de fundo (positivo/negativo)
ax.axhspan(0, 2, color="#e8f5e9", alpha=0.6, zorder=0)
ax.axhspan(-5, 0, color="#fdecea", alpha=0.6, zorder=0)
ax.axhline(0, color="#555", lw=1, ls="--", zorder=1)

# linhas
ax.plot(sint_x, sint_y, "o-", color="#2e7d32", lw=2.5, ms=9,
        label="Via síntese joint (domain-adapted)", zorder=3)
ax.plot(real_x, real_y, "s--", color="#c62828", lw=2.5, ms=9,
        label="Via dados reais (pré-treino direto)", zorder=3)
ax.plot(*cjoint, "D", color="#ef6c00", ms=8, zorder=3,
        label="Via real (co-treino joint)")

# rótulos dos pontos
ax.annotate("ABOShips\n(+0,69)", (0.95, 0.69), textcoords="offset points",
            xytext=(8, 8), fontsize=9, color="#2e7d32")
ax.annotate("InaTech\n(+1,00)", (8.0, 1.00), textcoords="offset points",
            xytext=(-15, 10), fontsize=9, color="#2e7d32")
ax.annotate("ABOShips\n(−0,37)", (0.95, -0.37), textcoords="offset points",
            xytext=(8, -22), fontsize=9, color="#c62828")
ax.annotate("InaTech\n(−4,15)", (8.0, -4.15), textcoords="offset points",
            xytext=(-15, 8), fontsize=9, color="#c62828")

ax.set_xlabel("Distância estrutural ao domínio operacional (CITRA-3D)  →  mais longe",
              fontsize=10)
ax.set_ylabel("Δ mAP50 vs. B2 (pontos)", fontsize=10)
ax.set_title("Síntese domain-adapted neutraliza a distância;\n"
             "uso de dados reais degrada com ela", fontsize=11)
ax.text(0.5, 1.55, "transferência positiva", color="#2e7d32", fontsize=9, style="italic")
ax.text(0.5, -4.7, "transferência negativa", color="#c62828", fontsize=9, style="italic")
ax.set_xticks([0.95, 1.84, 4.08, 8.0])
ax.set_xticklabels(["0,95\n(ABOShips)", "1,84\n(SMD)", "4,08\n(SeaShips)", "≫\n(InaTech)"],
                   fontsize=8)
ax.set_ylim(-5, 2)
ax.legend(loc="center right", fontsize=8.5, framealpha=0.95)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("fig_h4_interacao.png", dpi=160, bbox_inches="tight")
plt.savefig("fig_h4_interacao.pdf", bbox_inches="tight")
print("figura salva: fig_h4_interacao.png / .pdf")
