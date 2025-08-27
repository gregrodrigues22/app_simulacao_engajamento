# ---------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------
import io
import os
import re
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# (opcional) gráficos – você pode usar mais tarde
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------
st.set_page_config(layout="wide", page_title="📊 Simulador de Engajamento")

# ---------------------------------------------------------------
# Side bar
# ---------------------------------------------------------------
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Desenvolvido com ❤️ pela P2P")
    st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Simulador de Engajamento</h1>
        <p style='color: white;'>Explore as possibilidades de engajamento da população.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Helpers
# =============================================================================
@st.cache_data(show_spinner=False)
def read_table(uploaded_file, dtype_map=None):
    """
    Lê CSV/TXT ou Parquet. Mantém tipos conforme `dtype_map`.
    """
    if uploaded_file is None:
        return None
    name = uploaded_file.name
    suffix = os.path.splitext(name)[1].lower()
    dtype_map = dtype_map or {}

    if suffix in (".csv", ".txt"):
        return pd.read_csv(uploaded_file, dtype=dtype_map)
    elif suffix in (".parquet", ".pq"):
        data = uploaded_file.read()
        return pd.read_parquet(io.BytesIO(data))
    else:
        raise ValueError("Formato não suportado. Envie CSV/TXT/Parquet.")

def schema_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "coluna": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "n_null": [df[c].isna().sum() for c in df.columns],
        "exemplo": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })

def suggest_condition_name(filename: str) -> str:
    """
    Infere nome de condição a partir do nome do arquivo.
    Ex.: 'associados_obesidade.csv' -> 'condicao_obesidade'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    base = base.lower()
    tokens = re.split(r"[^a-z0-9]+", base)
    tokens = [t for t in tokens if t and t not in {"associados", "associado", "lista", "ids", "base", "dados"}]
    return f"condicao_{tokens[-1]}" if tokens else "condicao_custom"

def to_naive_datetime(series_like):
    """
    Converte qualquer série de datas para datetime64[ns] *naive* (sem timezone).
    Aceita strings, objetos datetime com tz, etc. Retorna NaT quando não parsear.
    """
    dt = pd.to_datetime(series_like, errors="coerce", utc=True)
    # remove timezone (fica como horário UTC 'naive')
    dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt

# -----------------------------------------------------------------------------
# Plotagem Inicial
# -----------------------------------------------------------------------------

def plot_resultado_stack(resultado: pd.DataFrame,
                         colors=None,
                         show_pct_on_bars=True) -> go.Figure:
    """ 
    Converte a tabela 'resultado' (com blocos 'n' e '%') em um gráfico Plotly
    com 2 subplots (stacked bars):
      - Esquerda: ['saudaveis', 'alguma_condicao']
      - Direita : demais condições, ordenadas por 'All' (desc)

    Parâmetros
    ----------
    resultado : DataFrame
        MultiIndex columns: level0 in {'n','%'}, level1 in {False, True, 'All'}.
    colors : dict
        Mapeamento de cores, ex.: {"Sem engajamento":"lightgray", "Com engajamento":"steelblue"}.
    show_pct_on_bars : bool
        Se True, exibe “n (xx.x%)” nas barras.
    """
    # --- blocos n e % ---
    n = resultado["n"].copy()
    p = resultado["%"].copy()

    # renomeia colunas booleanas p/ rótulos.
    n = n.rename(columns={False: "Sem engajamento", True: "Com engajamento"})
    p = p.rename(columns={False: "Sem engajamento (%)", True: "Com engajamento (%)"})

    # separa linhas especiais
    idx = n.index.tolist()
    left_fixed = [c for c in ["saudaveis", "alguma_condicao"] if c in idx]
    right_rest = [c for c in idx if c not in left_fixed and c != "All"]

    # ordena detalhe pela coluna 'All' (total)
    if "All" in n.columns:
        right_rest = sorted(right_rest, key=lambda c: n.loc[c, "All"], reverse=True)

    n_left  = n.loc[left_fixed]  if left_fixed  else n.iloc[[]]
    p_left  = p.loc[left_fixed]  if left_fixed  else p.iloc[[]]
    n_right = n.loc[right_rest]  if right_rest  else n.iloc[[]]
    p_right = p.loc[right_rest]  if right_rest  else p.iloc[[]]

    if colors is None:
        colors = {"Sem engajamento": "lightgray", "Com engajamento": "steelblue"}

    cols_n = [c for c in ["Sem engajamento", "Com engajamento"] if c in n.columns]
    cols_p = [c + " (%)" for c in cols_n]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Resumo: Saudáveis vs Alguma Condição", "Detalhe por condição")
    )

    def add_panel(n_df, p_df, col_id, showlegend):
        for name_n, name_p in zip(cols_n, cols_p):
            if name_n not in n_df.columns:
                continue
            # percentuais alinhados (se existir)
            pct_vals = (
                p_df[name_p].reindex(n_df.index).values
                if (name_p in p_df.columns)
                else np.full(len(n_df), np.nan)
            )

            # texto
            texttemplate = "%{y}" if not show_pct_on_bars else "%{y} (%{customdata:.1f}%)"
            hover_tmpl   = (
                f"<b>%{{x}}</b><br>{name_n}: %{{y}}"
                + ( " (%{customdata:.2f}%)" if show_pct_on_bars else "" )
                + "<extra></extra>"
            )

            fig.add_bar(
                x=n_df.index.astype(str),
                y=n_df[name_n],
                name=name_n,
                marker_color=colors.get(name_n, None),
                textposition="inside",
                text=None if show_pct_on_bars else n_df[name_n],
                customdata=pct_vals if show_pct_on_bars else None,
                texttemplate=texttemplate if show_pct_on_bars else None,
                hovertemplate=hover_tmpl,
                showlegend=showlegend,
                row=1, col=col_id
            )

    add_panel(n_left,  p_left,  col_id=1, showlegend=True)
    add_panel(n_right, p_right, col_id=2, showlegend=False)

    def totals_max(n_df):
        if n_df.empty: return 0.0
        present = [c for c in ["Sem engajamento", "Com engajamento"] if c in n_df.columns]
        return float(n_df[present].sum(axis=1).max())

    ymax_raw = max(totals_max(n_left), totals_max(n_right))
    # respiro fixo de 10% + arredondamento para cima
    ymax = float(np.ceil(ymax_raw * 1.10)) if ymax_raw > 0 else 1.0

    def add_totals_labels(n_df, col_id):
        if n_df.empty: return
        present = [c for c in ["Sem engajamento", "Com engajamento"] if c in n_df.columns]
        totals = n_df[present].sum(axis=1)

        bump = max(1, 0.02 * ymax)   # leve afastamento do topo do stack
        fig.add_trace(
            go.Scatter(
                x=n_df.index.astype(str),
                y=totals + bump,
                text=[f"{int(v):,}".replace(",", ".") for v in totals],
                mode="text",
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
                cliponaxis=False,
            ),
            row=1, col=col_id
        )

    add_totals_labels(n_left,  col_id=1)
    add_totals_labels(n_right, col_id=2)

    # controla eixo Y nas duas subplots
    fig.update_yaxes(range=[0, ymax], row=1, col=1)
    fig.update_yaxes(range=[0, ymax], row=1, col=2)
git add .
    fig.update_layout(
        title="Gráfico de Engajamento Atual por subpopulação (n e %)",
        barmode="stack",
        xaxis_title="Condição", xaxis2_title="Condição",
        yaxis_title="Número de pessoas",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=80, b=60),
        height=520,
        uniformtext_minsize=9,
        uniformtext_mode="hide"
    )
    return fig

# -----------------------------------------------------------------------------
# Preparação Simulação
# -----------------------------------------------------------------------------

def ensure_alguma_condicao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante a coluna 'alguma_condicao' no df:
    - Se existir 'saudaveis', define como ~saudaveis;
    - Caso contrário, tenta inferir por 'condicao_*' (any True).
    Também normaliza para boolean.
    """
    if "alguma_condicao" not in df.columns:
        if "saudaveis" in df.columns:
            df["alguma_condicao"] = ~df["saudaveis"].astype(bool)
        else:
            cond_cols = [c for c in df.columns if c.startswith("condicao_")]
            if cond_cols:
                df["alguma_condicao"] = df[cond_cols].any(axis=1)
                # se quiser, derive 'saudaveis' como complemento:
                if "saudaveis" not in df.columns:
                    df["saudaveis"] = ~df["alguma_condicao"]
            else:
                # fallback: se não há nada, assume todo mundo como não-saudável (ou tudo False)
                df["alguma_condicao"] = False
    else:
        df["alguma_condicao"] = df["alguma_condicao"].astype(bool)

    # se 'saudaveis' existe, garanta bool
    if "saudaveis" in df.columns:
        df["saudaveis"] = df["saudaveis"].astype(bool)

    return df

def ensure_alguma_condicao_in_out(out: dict, col_status: str = "atendimento_ult_trim") -> dict:
    """
    Adiciona a linha 'alguma_condicao' em out['resultado'] caso não exista,
    calculando como o complemento de 'saudaveis' no df_sim.
    """
    res = out["resultado"].copy()
    df_sim = out["df_sim"]

    if "alguma_condicao" not in res.index and "saudaveis" in df_sim.columns:
        mask = (~df_sim["saudaveis"].astype(bool)) & df_sim["id_pessoa"].notna()

        counts = df_sim.loc[mask, col_status].value_counts().reindex([False, True], fill_value=0)
        n_false = int(counts.get(False, 0))
        n_true  = int(counts.get(True, 0))
        n_all   = n_false + n_true

        if n_all == 0:
            p_false = p_true = 0.0
        else:
            p_false = round(100 * n_false / n_all, 2)
            p_true  = round(100 * n_true  / n_all, 2)

        # injeta a linha com MultiIndex de colunas ('n'|'%') x (False|True|All)
        res.loc["alguma_condicao", ("n", False)] = n_false
        res.loc["alguma_condicao", ("n", True)]  = n_true
        res.loc["alguma_condicao", ("n", "All")] = n_all

        res.loc["alguma_condicao", ("%", False)] = p_false
        res.loc["alguma_condicao", ("%", True)]  = p_true
        res.loc["alguma_condicao", ("%", "All")] = 100.0

        # ordena por index, se desejar
        res = res.sort_index()
        out["resultado"] = res

    return out

# -----------------------------------------------------------------------------
# Simulação
# -----------------------------------------------------------------------------

# ---------- util: pega %True atual de uma condição, caindo para 0 se não houver ----------
def _pct_true_atual(resultado_df: pd.DataFrame, cond: str) -> float:
    try:
        v = float(resultado_df.loc[cond, ('%', True)])
        return max(0.0, min(100.0, v))  # clamp 0..100
    except Exception:
        return 0.0

def simular_ate_meta_por_subpop(
    df: pd.DataFrame,
    col_status: str = "atendimento_ult_trim",      # coluna booleana do estado (False/True)
    condition_cols: list | tuple = (
        "condicao_diabetes",
        "condicao_dislipidemia",
        "condicao_mental",
        "condicao_obesidade",
        "condicao_hipertensao",
    ),
    target_rates: dict | None = None,              # metas por condição, ex.: {"condicao_diabetes": 0.30, ...}
    healthy_col: str = "saudaveis",                # coluna booleana dos saudáveis
    healthy_target: float | None = None,           # meta dos saudáveis (0-1) — opcional
    priority_order: list | None = None,            # ordem de prioridade para resolver sobreposição
    eligible_mask: pd.Series | None = None,        # máscara extra de elegibilidade (opcional)
    deterministic: bool = False,                   # True => escolhe os primeiros; False => sorteia
    seed: int | None = None,
):
    """
    Ajusta o df simulando conversões para que cada subpopulação atinja uma
    META (% alvo) de engajamento TOTAL.

    Estratégia:
      - Para cada subpopulação (condição e/ou saudáveis):
          target_n = ceil(target_rate * tamanho_grupo)
          current_n = já True no grupo
          need = max(0, target_n - current_n)
        Seleciona `need` pessoas que estão False dentro do grupo e as converte.
      - Para grupos com sobreposição (mesma pessoa em mais de uma condição),
        usa `priority_order` para decidir quem escolhe primeiro.

    Retorna:
      {
        "df_sim": df após a simulação,
        "resultado": tabela (n e %) por condição,
        "total_true": total de True,
        "pct_true": percentual total True na base,
        "diagnostico": dataframe com quem foi convertido
      }
    """
    rng = np.random.default_rng(seed)
    df_sim = df.copy()

    # Normaliza parâmetros
    if target_rates is None:
        target_rates = {}
    # Apenas colunas que de fato existem em df
    condition_cols = [c for c in condition_cols if c in df_sim.columns]

    # prioridade padrão: condições e depois saudáveis (ou vice-versa se preferir)
    if priority_order is None:
        priority_order = list(condition_cols)
        if healthy_col in df_sim.columns and healthy_target is not None:
            priority_order.append(healthy_col)

    # Base de elegibilidade adicional (se fornecida). Por padrão, todos são elegíveis.
    if eligible_mask is None:
        eligible_mask = pd.Series(True, index=df_sim.index)

    # Vamos registrar quem foi convertido nesta rodada
    chosen_global = pd.Series(False, index=df_sim.index)

    # Helper para processar 1 grupo (condição ou saudáveis)
    def process_group(group_col: str, target_rate: float):
        nonlocal chosen_global

        # Máscara do grupo
        mask_group = df_sim[group_col].astype(bool)

        # Quem já está True dentro do grupo
        current_true = (df_sim[col_status].astype(bool) & mask_group).sum()

        # Tamanho do grupo
        size_group = int(mask_group.sum())
        if size_group == 0:
            return  # nada a fazer

        # Alvo em quantidade
        target_n = int(np.ceil(float(target_rate) * size_group))
        need = max(0, target_n - current_true)
        if need == 0:
            return

        # Candidatos: dentro do grupo, elegíveis, atualmente False e ainda não escolhidos por outro grupo
        candidates = df_sim.index[mask_group & eligible_mask & (~df_sim[col_status].astype(bool)) & (~chosen_global)]
        if len(candidates) == 0:
            return

        need = min(need, len(candidates))

        if deterministic:
            to_pick = candidates[:need]
        else:
            to_pick = rng.choice(candidates, size=need, replace=False)

        # Marca que foram escolhidos e converte no df_sim
        chosen_global.loc[to_pick] = True
        df_sim.loc[to_pick, col_status] = True

    # Executa grupos conforme prioridade
    for grp in priority_order:
        if grp == healthy_col:
            if (healthy_col in df_sim.columns) and (healthy_target is not None):
                process_group(healthy_col, healthy_target)
        else:
            # condição
            if grp in condition_cols:
                rate = float(target_rates.get(grp, 0.0))
                process_group(grp, rate)

    # ---------- Métricas de saída ----------
    # Monta long apenas com colunas existentes
    melt_vars = list(condition_cols)
    if healthy_col in df_sim.columns:
        melt_vars.append(healthy_col)

    long = df_sim.melt(
        id_vars=['id_pessoa', col_status],
        value_vars=melt_vars,
        var_name='condicao',
        value_name='tem_condicao'
    )
    long = long[long['tem_condicao']]

    contagens = pd.crosstab(long['condicao'], long[col_status])
    contagens['All'] = contagens.sum(axis=1)
    percentuais = (contagens.div(contagens['All'], axis=0) * 100).round(2)
    resultado = pd.concat({'n': contagens, '%': percentuais}, axis=1)

    total_true = int(df_sim[col_status].sum())
    total_pct_true = round(100 * total_true / len(df_sim), 2)

    diagnostico = pd.DataFrame({
        'id_pessoa': df_sim['id_pessoa'],
        'foi_convertido': chosen_global,
        'status_final': df_sim[col_status].astype(bool)
    })

    return {
        "df_sim": df_sim,
        "resultado": resultado,
        "total_true": total_true,
        "pct_true": total_pct_true,
        "diagnostico": diagnostico
    }

# -----------------------------------------------------------------------------
# Navegação (abas controladas por estado)
# -----------------------------------------------------------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📤 Upload de bases"

# Função para scrollar para o topo
#def scroll_top():
    # força o scroll no elemento pai (iframe) do app
#    components.html(
#        """
#        <script>
#            // tenta rolar a janela principal
#            if (window.parent) {
#                try { window.parent.scrollTo(0, 0); } catch (e) {}
#                // fallback: tenta rolar a seção principal do Streamlit
#                try {
#                    const sec = window.parent.document.querySelector('section.main');
#                    if (sec) { sec.scrollTo({top: 0, left: 0, behavior: "instant"}); }
#                } catch (e) {}
#            } else {
#                window.scrollTo(0, 0);
#            }
#        </script>
#        """,
#        height=0,
#    )

def scroll_top():
    st.markdown("""
    <script>
      try {
        if (window.parent) {
          window.parent.scrollTo(0, 0);
          const sec = window.parent.document.querySelector('section.main');
          if (sec) sec.scrollTo({top: 0, left: 0, behavior: "instant"});
        } else {
          window.scrollTo(0, 0);
        }
      } catch (e) {}
    </script>
    """, unsafe_allow_html=True)

# Callback do botão
def go_to_sim():
    st.session_state.active_tab = "📊 Simulação"
    scroll_top()

st.segmented_control(
    "🎲 Etapas",
    options=["📤 Upload de bases", "📊 Simulação"],
    key="active_tab"
)

show_upload = st.session_state.active_tab == "📤 Upload de bases"
show_sim    = st.session_state.active_tab == "📊 Simulação"

# ─────────────────────────────────────────────────────────────────────────────
# ABA 0 – Upload
# ─────────────────────────────────────────────────────────────────────────────
if show_upload:
    st.header("📤 Upload de bases")
    st.info(
        "Envie as 3 bases (Ativos, Interações e Condições) conforme as orientações abaixo. "
        "Cada seção tem instruções, requisitos de colunas e modelos para download. "
        "Os **dados não são persistidos**."
    )

    # ==============================
    # Templates (em memória) p/ download
    # ==============================
    _tpl_ativos = pd.DataFrame({
        "id": [1, 2],
        "id_pessoa": ["0001", "0002"],
        "id_organizacao": [10, 10],
        "etapa_funil": ["captacao", "captacao"]
    })
    _tpl_inter = pd.DataFrame({
        "id_pessoa": ["0001", "0001", "0002"],
        "data": ["2024-01-10", "2024-01-15", "2024-01-20"],
        "conteudo": ["convite", "lembrete", "convite"],
        "mes_atendimento": ["2024-01-01", "2024-02-01", "2024-01-01"],
        "tipo": ["whatsapp", "email", "whatsapp"]
    })
    _tpl_cond  = pd.DataFrame({
        "id_pessoa": ["0001", "0003"]
    })

    # ==============================
    # 1) ATIVOS
    # ==============================
    st.subheader("1) Modelo: Ativos")
    with st.expander("📎 Instruções e modelo (clique para abrir)", expanded=False):
        st.markdown(
            """
**Requisitos mínimos**  
- **Formato**: CSV, TXT ou PARQUET  
- **Colunas obrigatórias**: `id`, `id_pessoa`, `id_organizacao`, `etapa_funil`  
- `id_pessoa` será tratado como *texto* (string) para evitar perda de zeros à esquerda.
            """
        )
        st.download_button(
            "💾 Baixar modelo de Ativos (CSV)",
            data=_tpl_ativos.to_csv(index=False).encode("utf-8"),
            file_name="modelo_ativos.csv",
            mime="text/csv",
            use_container_width=True
        )

    up_ativos = st.file_uploader(
        "Envie o arquivo de **Ativos** (CSV/TXT/Parquet)",
        type=["csv", "txt", "parquet"],
        key="up_ativos_body",
        help="Tamanho máximo por arquivo: 200MB"
    )

    df_ativos = None
    if up_ativos is not None:
        try:
            df_ativos = read_table(up_ativos, dtype_map={"id_pessoa": "string"})
            req_cols = {"id", "id_pessoa", "id_organizacao", "etapa_funil"}
            miss = req_cols - set(df_ativos.columns)
            if miss:
                st.error(f"⚠️ Ativos faltando coluna(s) obrigatória(s): {', '.join(sorted(miss))}")
            else:
                st.success(f"✅ Ativos carregado: **{up_ativos.name}** — {df_ativos.shape[0]} linhas")
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**Schema (Ativos)**")
                    st.dataframe(schema_df(df_ativos), use_container_width=True)
                with c2:
                    st.markdown("**Prévia (Ativos)**")
                    st.dataframe(df_ativos.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao ler Ativos: {e}")
    else:
        st.info("Envie o arquivo **Ativos** acima.")

    # ==============================
    # 2) INTERAÇÕES
    # ==============================
    st.subheader("2) Modelo: Interações")
    with st.expander("📎 Instruções e modelo (clique para abrir)", expanded=False):
        st.markdown(
            """
**Requisitos mínimos**  
- **Formato**: CSV, TXT ou PARQUET  
- **Colunas obrigatórias**: `id_pessoa`, `data`, `conteudo`, `mes_atendimento`, `tipo`  
- `id_pessoa` será tratado como *texto* (string).
            """
        )
        st.download_button(
            "💾 Baixar modelo de Interações (CSV)",
            data=_tpl_inter.to_csv(index=False).encode("utf-8"),
            file_name="modelo_interacoes.csv",
            mime="text/csv",
            use_container_width=True
        )

    up_inter = st.file_uploader(
        "Envie o arquivo de **Interações** (CSV/TXT/Parquet)",
        type=["csv", "txt", "parquet"],
        key="up_inter_body",
        help="Tamanho máximo por arquivo: 200MB"
    )

    df_inter = None
    if up_inter is not None:
        try:
            df_inter = read_table(up_inter, dtype_map={"id_pessoa": "string"})
            req_cols = {"id_pessoa", "data", "conteudo", "mes_atendimento", "tipo"}
            miss = req_cols - set(df_inter.columns)
            if miss:
                st.error(f"⚠️ Interações faltando coluna(s) obrigatória(s): {', '.join(sorted(miss))}")
            else:
                st.success(f"✅ Interações carregado: **{up_inter.name}** — {df_inter.shape[0]} linhas")
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**Schema (Interações)**")
                    st.dataframe(schema_df(df_inter), use_container_width=True)
                with c2:
                    st.markdown("**Prévia (Interações)**")
                    st.dataframe(df_inter.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao ler Interações: {e}")
    else:
        st.info("Envie o arquivo **Interações** acima.")

    # ==============================
    # 3) CONDIÇÕES
    # ==============================
    st.subheader("3) Modelos: Condições de Saúde")
    with st.expander("📎 Instruções e modelo (clique para abrir)", expanded=False):
        st.markdown(
            """
**Requisitos mínimos**  
- **Formato**: CSV, TXT ou PARQUET  
- **Coluna obrigatória**: `id_pessoa`  
- Envie **1 ou mais arquivos**, cada um representando uma condição.  
- O nome da condição será sugerido a partir do nome do arquivo e pode ser editado.
            """
        )
        st.download_button(
            "💾 Baixar modelo de Condição (CSV)",
            data=_tpl_cond.to_csv(index=False).encode("utf-8"),
            file_name="modelo_condicao.csv",
            mime="text/csv",
            use_container_width=True
        )

    up_conds = st.file_uploader(
        "Envie **1 ou mais** arquivos de **Condições** (CSV/TXT/Parquet)",
        type=["csv", "txt", "parquet"],
        accept_multiple_files=True,
        key="up_conds_body",
        help="Tamanho máximo por arquivo: 200MB"
    )

    cond_frames = []
    cond_names  = []

    if up_conds:
        for i, up in enumerate(up_conds, start=1):
            try:
                dfc = read_table(up, dtype_map={"id_pessoa": "string"})
                if "id_pessoa" not in dfc.columns:
                    st.error(f"Arquivo **{up.name}** não tem coluna `id_pessoa`.")
                    continue

                default_cond = suggest_condition_name(up.name)
                cond_col = st.text_input(
                    f"Nome da coluna-condição para **{up.name}**",
                    value=default_cond,
                    key=f"cond_name_{i}"
                )

                st.markdown(f"**Schema — {up.name}**  → coluna sugerida: `{cond_col}`")
                st.dataframe(schema_df(dfc), use_container_width=True)
                st.markdown("**Prévia**")
                st.dataframe(dfc.head(10), use_container_width=True)

                cond_frames.append(
                    dfc[["id_pessoa"]].drop_duplicates().assign(**{cond_col: True})
                )
                cond_names.append(cond_col)

            except Exception as e:
                st.error(f"Erro ao ler {up.name}: {e}")
    else:
        st.info("Envie **1 ou mais** arquivos de **Condições** acima.")

    # ==============================
    # 4) Montar População Final
    # ==============================
    st.subheader("4) Montar População Final")
    st.caption(
        "Cruzamos Ativos com as Condições (por `id_pessoa`). Criamos `alguma_condicao`, "
        "`saudaveis` e `num_condicoes`. Também trazemos da última interação "
        "`max_mes_atendimento`, `ultimo_tipo` e `ultimo_conteudo`, e marcamos `atendimento_ult_trim` (últimos 3 meses)."
    )

    btn_ok = st.button(
        "🔧 Construir Conjunto de Dados Final",
        disabled=not (up_ativos and len(cond_frames) > 0)
    )

    if btn_ok:
        if df_ativos is None or len(cond_frames) == 0:
            st.warning("Envie Ativos e pelo menos um arquivo de Condição.")
        else:
            # 1) Base de ativos + condições
            df_final = df_ativos.copy()
            if "id_pessoa" not in df_final.columns:
                st.error("Ativos precisa ter coluna `id_pessoa`.")
                st.stop()

            df_final["id_pessoa"] = df_final["id_pessoa"].astype("string")

            # une todas as condições (cada cond_df tem id_pessoa + condicao_*)
            for cond_df in cond_frames:
                df_final = df_final.merge(cond_df, on="id_pessoa", how="left")

            # normaliza colunas condicao_* para boolean
            cond_cols = [c for c in df_final.columns if c.startswith("condicao_")]
            for c in cond_cols:
                df_final[c] = df_final[c].fillna(False).astype(bool)

            df_final["alguma_condicao"] = df_final[cond_cols].any(axis=1) if cond_cols else False
            df_final["num_condicoes"]  = df_final[cond_cols].sum(axis=1) if cond_cols else 0
            df_final["saudaveis"]      = ~df_final["alguma_condicao"]

            # 2) Traz informações da ÚLTIMA INTERAÇÃO (se houver df_inter)
            if (df_inter is not None) and (not df_inter.empty):
                inter = df_inter.copy()

                # converte 'data' para datetime naive
                if "data" in inter.columns:
                    inter["data_dt"] = to_naive_datetime(inter["data"])
                else:
                    inter["data_dt"] = pd.NaT

                # 'mes_atendimento' vira datetime naive; se não existir, usa mês de 'data'
                if "mes_atendimento" in inter.columns:
                    inter["mes_atendimento_dt"] = to_naive_datetime(inter["mes_atendimento"])
                else:
                    inter["mes_atendimento_dt"] = inter["data_dt"].dt.to_period("M").dt.to_timestamp("MS")

                # pega a última linha por id_pessoa (maior data_dt)
                inter_sorted = inter.sort_values("data_dt")
                idx_last = inter_sorted.groupby("id_pessoa")["data_dt"].idxmax()

                keep = ["id_pessoa"]
                if "tipo" in inter_sorted.columns:     keep.append("tipo")
                if "conteudo" in inter_sorted.columns: keep.append("conteudo")
                ultima = inter_sorted.loc[idx_last, keep].copy()

                rename_map = {}
                if "tipo" in ultima.columns:          rename_map["tipo"] = "ultimo_tipo"
                if "conteudo" in ultima.columns:      rename_map["conteudo"] = "ultimo_conteudo"
                ultima = ultima.rename(columns=rename_map)

                # max de mes_atendimento_dt por pessoa
                max_mes = inter.groupby("id_pessoa")["mes_atendimento_dt"].max().rename("max_mes_atendimento")

                # junta no df_final
                df_final = df_final.merge(ultima, on="id_pessoa", how="left")
                df_final = df_final.merge(max_mes, on="id_pessoa", how="left")
            else:
                for col in ["ultimo_tipo", "ultimo_conteudo", "max_mes_atendimento"]:
                    if col not in df_final.columns:
                        df_final[col] = pd.NA

            # 3) Data de corte (últimos 3 meses) e flag de engajamento trimestral
            df_final["max_mes_atendimento"] = to_naive_datetime(df_final["max_mes_atendimento"])
            hoje = pd.Timestamp.utcnow().tz_convert("UTC").tz_localize(None)
            tres_meses_atras = hoje - pd.DateOffset(months=3)
            # between requer datetime64[ns] vs Timestamp; agora ambos estão 'naive'
            df_final["atendimento_ult_trim"] = df_final["max_mes_atendimento"].between(
                tres_meses_atras, hoje, inclusive="both"
            )

            # (opcional) remover colunas operacionais
            for col_drop in ["id_organizacao", "etapa_funil", "id"]:
                if col_drop in df_final.columns:
                    df_final.drop(columns=[col_drop], inplace=True)

            # feedback
            st.success("✅ Conjunto de Dados Final construído.")
            st.markdown("**Colunas criadas:** " + ", ".join(
                ["alguma_condicao", "num_condicoes", "saudaveis",
                 "max_mes_atendimento", "ultimo_tipo", "ultimo_conteudo", "atendimento_ult_trim"]
            ))
            st.markdown("**Condições identificadas:** " + (", ".join(cond_cols) if cond_cols else "—"))

            st.subheader("Prévia do Conjunto de Dados Final")
            st.dataframe(df_final.head(20), use_container_width=True)

            # 🔑 Guarda no estado da sessão para usar na aba de simulação
            st.session_state["df_final"] = df_final

            st.download_button(
                "💾 Baixar Conjunto de Dados Final (CSV)",
                data=df_final.to_csv(index=False).encode("utf-8"),
                file_name="df_final.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("**Modelo mínimo recomendado para simulação**")
            st.code(
                "id_pessoa, condicao_*, alguma_condicao, saudaveis,\n"
                "max_mes_atendimento, ultimo_tipo, ultimo_conteudo, atendimento_ult_trim",
                language="text",
            )

            # Botão que muda para a seção de Simulação
            st.button("▶️ Ir para Simulação", on_click=go_to_sim, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# ABA 1 – Simulação
# ─────────────────────────────────────────────────────────────────────────────
if show_sim:

    if st.session_state.get("active_tab") == "📊 Simulação":
            scroll_top()

            st.header("📊 Simulação")
            
            # 1) Carrega do estado
            df_final_loaded = st.session_state.get("df_final")  # pega o df salvo

            # 2) Garante que existe antes de seguir
            if df_final_loaded is None or len(df_final_loaded) == 0:
                st.warning("Nenhum `Conjunto de Dados Final` disponível. Gere na aba anterior.")
                st.stop()

            # 3) Base original para simulações
            if "df_base" not in st.session_state:
                # primeira vez: salva uma cópia limpa
                st.session_state.df_base = st.session_state["df_final"].copy()

            # 4) Sempre trabalhe a partir de df_base
            df_base = st.session_state.df_base  # <-- SEMPRE use esta como referência

            st.subheader("1) Estado atual")

            # — Métricas rápidas
            cols = st.columns(4)
            with cols[0]:
                st.metric("Pessoas", f"{len(df_final_loaded):,}".replace(",", "."))
            with cols[1]:
                if "atendimento_ult_trim" in df_final_loaded.columns:
                    st.metric("Pessoas Engajadas", f"{df_final_loaded['atendimento_ult_trim'].sum():,}".replace(",", "."))
            with cols[2]:
                if "alguma_condicao" in df_final_loaded.columns:
                    TAU = df_final_loaded['atendimento_ult_trim'].sum() / len(df_final_loaded) * 100
                    TAU = round(TAU, 2)
                    TAU = f"{TAU}%"
                    st.metric("TAU", TAU)

            col_condicoes = [c for c in df_final_loaded.columns if "condicao_" in c]
            col_condicoes_saudaveis = col_condicoes + ["saudaveis"] + ["alguma_condicao"]

            #if "atendimento_ult_trim" in df_final_loaded.columns:  
            #    options = st.multiselect("Quais condições quer analisar",
            #    col_condicoes_saudaveis,
            #    default=col_condicoes_saudaveis)

            # ajuste nomes das colunas de condição
            cols = col_condicoes_saudaveis.copy()

            df = df_final_loaded.copy()
            df[cols] = df[cols].fillna(False).astype(bool)
            df['atendimento_ult_trim'] = df['atendimento_ult_trim'].fillna(False).astype(bool)

            # Formato longo: uma linha por pessoa-condição
            long = df.melt(
                id_vars=['id_pessoa','atendimento_ult_trim'],
                value_vars=cols,
                var_name='condicao',
                value_name='tem_condicao'
            )

            # considerar apenas quem TEM a condição
            long = long[long['tem_condicao']]

            # Contagem
            contagens = pd.crosstab(long['condicao'], long['atendimento_ult_trim'])

            # Percentual linha a linha
            percentuais = (contagens.div(contagens.sum(axis=1), axis=0) * 100).round(2)

            # Adicionar coluna All nas contagens
            contagens['All'] = contagens.sum(axis=1)
            percentuais['All'] = 100.0  # ou NaN se preferir não mostrar

            # Adicionar linha total (All)
            contagens.loc['All'] = contagens.sum()
            percentuais.loc['All'] = (contagens.loc['All'] / contagens.loc['All', 'All'] * 100).round(2)

            # Combinar contagens e %
            resultado = pd.concat({'n': contagens, '%': percentuais}, axis=1)

            st.markdown("**Tabela de Engajamento Atual por subpopulação (n e %)**")
            st.dataframe(resultado.loc[cols], use_container_width=True)

            fig = plot_resultado_stack(
                resultado,
                colors={"Sem engajamento":"lightgray", "Com engajamento":"steelblue"},
                show_pct_on_bars=True
                )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("2) Estado Simulado")

            # sempre parte de uma cópia limpa
            df_for_sim = df_base.copy()

            st.info(
            "Selecione abaixo o percentual de engajamento estimado para cada condição. ")

            st.caption(
            "O valor que você coloca no slider será o total esperado da subpopulação no cenário simulado. Apenas **aumentos** serão simulados "
            )

            # ---------- detectar condições ----------
            col_condicoes = [c for c in df_final_loaded.columns if c.startswith("condicao_")]
            col_condicoes = sorted(col_condicoes)  # opcional

            # ---------- valores default dos sliders com base no "resultado" atual ----------
            def_saudaveis       = _pct_true_atual(resultado, 'saudaveis')
            def_alguma_condicao = _pct_true_atual(resultado, 'alguma_condicao')
            def_por_cond = {c: _pct_true_atual(resultado, c) for c in col_condicoes}

            # Usar um formulário para os sliders (envio único)
            with st.form("frm_engajamento"):
                cL, cR = st.columns(2)

                with cL:
                    st.caption("Engajamento — Saudáveis / Alguma Condição (%)")
                    meta_saudaveis = st.slider(
                        "Engajamento — **Saudáveis** (%)",
                        min_value=0, max_value=100, value=int(round(def_saudaveis))
                    )

                with cR:
                    st.caption("Engajamento por condição (%)")
                    # Sliders dinâmicos para cada condicao_*
                    expectativa_por_cond = {}
                    for c in col_condicoes:
                        rotulo = c.replace("condicao_", "").replace("_", " ").title()
                        expectativa_por_cond[c] = st.slider(
                            f"{rotulo}", 0, 100, value=int(round(def_por_cond[c]))
                        )

                submitted = st.form_submit_button("▶️ Rodar simulação", type="primary")

            
            # ---------- ao enviar, montar o dicionário e simular ----------
            if submitted:
                # dicionário 0–1 para a função
                # metas vindas dos sliders (em %), converta para [0..1]
                healthy_target = meta_saudaveis / 100.0

                target_rates = {}
                for c in col_condicoes:           # ex.: ["condicao_diabetes", "condicao_dislipidemia", ...]
                    alvo_pct = st.session_state.get(f"slider_meta_{c}", 0)  # ou a variável correspondente ao slider
                    target_rates[c] = float(alvo_pct) / 100.0

                # Ordem de prioridade (ex.: condições antes dos saudáveis; ajuste como preferir)
                priority = [*col_condicoes, "saudaveis"]

                df_for_sim = ensure_alguma_condicao(df_for_sim)

                # Rodar a simulação
                out = simular_ate_meta_por_subpop(
                    df_for_sim,
                    col_status="atendimento_ult_trim",
                    condition_cols=col_condicoes,
                    target_rates=target_rates,
                    healthy_col="saudaveis",
                    healthy_target=healthy_target,
                    priority_order=priority,       # ou deixe None para padrão
                    deterministic=True,           # True se quiser comportamento reprodutível sem aleatoriedade
                    seed=42
                )

                st.session_state["last_sim_out"] = out

                # garante 'alguma_condicao' no agregado
                out = ensure_alguma_condicao_in_out(out, col_status="atendimento_ult_trim")

                # métricas e tabela
                st.markdown("### 3) Resultado do cenário")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Pessoas", f"{len(df_final_loaded):,}".replace(",", "."))
                with m2:
                    st.metric("Pessoas engajadas (pós-simulação)", f"{out['total_true']:,}".replace(",", "."))
                with m3:
                    st.metric("TAU (pós-simulação)", f"{out['pct_true']:.2f}%")

                st.markdown("**Tabela de Conversão por subpopulação (n e %)**")
                st.dataframe(out["resultado"], use_container_width=True)

                fig = plot_resultado_stack(
                out["resultado"],
                colors={"Sem engajamento":"lightgray", "Com engajamento":"steelblue"},
                show_pct_on_bars=True
                )
                st.plotly_chart(fig, use_container_width=True)

                cols_btn = st.columns([1, 1, 4])

                with cols_btn[0]:
                    if st.button("↺ Resetar base"):
                        # volta a base de simulação ao estado limpo
                        st.session_state.df_base = st.session_state["df_final"].copy()
                        st.success("Base de simulação resetada para o estado original.")
