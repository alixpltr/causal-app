import streamlit as st
import pandas as pd
import os, sys, tempfile, subprocess, time
from pathlib import Path
import shutil
from dataclasses import dataclass

def clear_session_dir():
    """Delete all files created for this user session and reset state."""
    session_dir = Path(st.session_state.get("session_dir", ""))
    if session_dir and session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    st.session_state.pop("session_dir", None)
    st.success("All uploaded files and outputs were deleted from this device.")
    st.rerun()

# where your scripts live
SCRIPTS_DIR = Path.cwd() / "methods"
SCRIPTS_DIR.mkdir(exist_ok=True)

METHODS = {
    "PC-FISHERZ (bootstrap)": "pc.py",
    "PC-KCI (bootstrap)": "pc_boot_kci.py",
    "FGES (bootstrap)": "fges_boot.py",
    "MissDAG (bootstrap)": "missdag_boot.py",
    "DAG-GNN (bootstrap)": "daggnn_boot.py",
    "ADMG DISCOVERY (bootstrap)": "admg_boot.py",
    "DirectLiNGAM (bootstrap)": "lingam_boot.py",
    "CAM-UV (bootstrap)": "camuv_boot.py",
    "DirectLiNGAM (bootstrap)": "lingam_boot.py",
    "NOTEARS (bootstrap)": "notears_boot.py",
    "NOTEARS-MLP (bootstrap)": "notears_mlp_boot.py",
    "NOTEARS-SOB (bootstrap)": "notears_sob_boot.py",
    "FCI-FISHERZ (bootstrap)": "fci_boot.py",
    "FCI-KCI (bootstrap)": "fci_kci_boot.py",
    "GFCI (bootstrap)": "gfci_boot.py",
    "CORL (bootstrap)": "corl_boot.py",
    "GRANDAG (bootstrap)": "grandag_boot.py",
    "iMIIC (R)": None,
}

@dataclass
class Answers:
    data_type: str
    latents: str
    relation: str
    noise: str
    missing: str
    n: int
    p: int
    priority: str

# ---- Method capability profiles ----
METHOD_CAPS = {
    "PC‑FISHERZ (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": False,
        "nonlinear_ok": False,
        "scale_fast": True,
        "good_high_dim": True,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "PC‑KCI (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": False,
        "nonlinear_ok": True,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "Shapley‑PC (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": False,
        "nonlinear_ok": False,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "GES (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": True,
        "nonlinear_ok": False,
        "scale_fast": True,
        "good_high_dim": True,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "FGES (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": True,    
        "supports_binary": True,
        "supports_mixed": True,      
        "linear_pref": True,
        "nonlinear_ok": False,
        "scale_fast": True,
        "good_high_dim": True,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "DirectLiNGAM (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": True,
        "nonlinear_ok": False,
        "scale_fast": True,
        "good_high_dim": True,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "CAM‑UV (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": False,
        "nonlinear_ok": True,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "NOTEARS (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": True,
        "nonlinear_ok": False,
        "scale_fast": True,
        "good_high_dim": True,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "DAG‑GNN (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": False,
        "nonlinear_ok": True,
        "scale_fast": False,
        "good_high_dim": True,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "GFCI (bootstrap)": {
        "assumes_no_latents": False,  
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": True,
        "nonlinear_ok": False,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": True,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "FCI‑FISHERZ (bootstrap)": {
        "assumes_no_latents": False,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": True,
        "nonlinear_ok": False,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": True,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "MissDAG (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": True,
        "nonlinear_ok": False,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": True,       
    },
    "CORL (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": False,
        "nonlinear_ok": True,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "GRANDAG (bootstrap)": {
        "assumes_no_latents": True,
        "supports_discrete": False,
        "supports_binary": False,
        "supports_mixed": False,
        "linear_pref": False,
        "nonlinear_ok": True,
        "scale_fast": False,
        "good_high_dim": True,
        "outputs_pag": False,
        "outputs_admg": False,
        "handles_missing": False,
    },
    "iMIIC (R)": {
        "assumes_no_latents": False,
        "supports_discrete": True,
        "supports_binary": True,
        "supports_mixed": True,
        "linear_pref": False,
        "nonlinear_ok": False,
        "scale_fast": False,
        "good_high_dim": False,
        "outputs_pag": False,
        "outputs_admg": True,
        "handles_missing": False,
    },
}


def rank_methods(ans: Answers, methods_dict: dict[str, str]):
    def matches(name: str):
        c = METHOD_CAPS.get(name, {})
        m, why = 0, []
        if ans.data_type == "Mixed" and c.get("supports_mixed", False):
            m += 1; why.append("mixed supported")
        if ans.data_type == "Discrete" and c.get("supports_discrete", False):
            m += 1; why.append("discrete supported")
        if ans.data_type == "Continuous + Binary" and c.get("supports_binary", False):
            m += 1; why.append("binary supported")
        if ans.data_type == "Continuous" and not c.get("supports_mixed", False):
            m += 1; why.append("continuous-friendly")
        if ans.data_type == "I'm not sure":
            if c.get("supports_mixed", False) or c.get("supports_binary", False) or c.get("supports_discrete", False):
                m += 1; why.append("flexible types")
        if ans.latents == "Yes (latents possible)":
            if not c.get("assumes_no_latents", True):
                m += 1; why.append("handles latents (PAG/ADMG)")
        elif ans.latents == "No (no latents)":
            if c.get("assumes_no_latents", True):
                m += 1; why.append("assumes sufficiency")
        if ans.relation == "Linear" and c.get("linear_pref", False):
            m += 1; why.append("linear SEM")
        if ans.relation == "Nonlinear (general)" and c.get("nonlinear_ok", False):
            m += 1; why.append("nonlinear ok")
        if ans.relation == "Nonlinear (additive)" and ("CAM-UV" in name or c.get("nonlinear_ok", False)):
            m += 1; why.append("additive/nonlinear ok")
        if ans.noise == "Gaussian-ish" and name in ("NOTEARS (bootstrap)", "GES (bootstrap)", "FGES (bootstrap)", "PC-FISHERZ (bootstrap)"):
            m += 1; why.append("Gaussian-ish ok")
        if ans.noise == "Non-Gaussian/Heavy-tailed" and name.startswith("DirectLiNGAM"):
            m += 1; why.append("non-Gaussian linear")
        if ans.missing in ("MCAR/MAR", "MNAR/Unknown") and c.get("handles_missing", False):
            m += 1; why.append("handles missingness")
        high_dim = (ans.p >= 50) or (ans.p > ans.n)
        if high_dim and (c.get("good_high_dim", False) or c.get("scale_fast", False)):
            m += 1; why.append("scales/high-dim ok")
        return m, why

    rows = []
    for name in methods_dict.keys():
        m, why = matches(name)
        rows.append({"name": name, "matches": m, "why": why})

    def tie_priority(r):
        score = 0
        nm = r["name"].upper()
        c = METHOD_CAPS.get(r["name"], {})
        if ans.data_type == "Mixed" and "KCI" in nm: score += 3
        if ans.latents == "Yes (latents possible)" and (c.get("outputs_pag") or c.get("outputs_admg")): score += 3
        if ans.priority == "Speed/Scale" and (c.get("scale_fast") or c.get("good_high_dim")): score += 1
        if ans.data_type == "Mixed" and "FISHERZ" in nm: score -= 2
        return score

    rows.sort(key=lambda r: (r["matches"], tie_priority(r)), reverse=True)

    # special: Mixed -> promote iMIIC first
    if ans.data_type == "Mixed" and "iMIIC (R)" in methods_dict:
        def find(name):
            for r in rows:
                if r["name"] == name: return r
            return None
        imi = find("iMIIC (R)")
        rest = [r for r in rows if r["name"] != "iMIIC (R)"]
        top_rest = rest[:2]
        top3 = [{"rank":1,"name":"iMIIC (R)","score":int(imi["matches"]),"why":", ".join(dict.fromkeys(imi["why"])) or "best suited for mixed data"}]
        for i, r in enumerate(top_rest, start=2):
            top3.append({"rank":i,"name":r["name"],"score":int(r["matches"]),"why":", ".join(dict.fromkeys(r["why"]))})
        return top3, "Best for Mixed"

    top3 = []
    for i, r in enumerate(rows[:3], start=1):
        top3.append({"rank": i, "name": r["name"], "score": int(r["matches"]), "why": ", ".join(dict.fromkeys(r["why"]))})
    if not top3:
        return [], "No clear fit"
    top = top3[0]["score"]
    second = top3[1]["score"] if len(top3) > 1 else -999
    margin = top - second
    if top >= 4 and margin >= 1: label = "Recommended"
    elif top >= 3: label = "Tentative"
    else: label = "No clear fit"
    return top3, label

st.set_page_config(page_title="Causal Discovery (PC + GES bootstrap)", layout="centered")
st.title("🧭 Causal Discovery")

# Delete button + small privacy note
if st.session_state.get("session_dir"):
    if st.button("Delete my data", help="Remove the uploaded CSV and all outputs for this session"):
        clear_session_dir()
    st.caption("(When run locally, all processing and files stay on your device — nothing is uploaded.)")

st.markdown("**Step 1 — Upload your dataset (CSV)**")
uploaded = st.file_uploader("Choose a CSV", type=["csv"])

if uploaded:
    # ===== Save the CSV for this session =====
    if "session_dir" not in st.session_state:
        st.session_state.session_dir = Path(tempfile.mkdtemp(prefix="causal_session_"))
    session_dir = Path(st.session_state.session_dir)
    data_path = session_dir / "uploaded.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded.read())

    # ===== Quick preview =====
    try:
        st.write("Preview:", pd.read_csv(data_path, nrows=5))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # ===================== Step 2 — Quick questions (recommendation) =====================
    st.markdown("---")
    st.markdown("**Step 2 — Quick questions (to recommend a method)**")

    st.radio(
        "What kind of data do you have?",
        ["Continuous", "Continuous + Binary", "Discrete", "Mixed", "I'm not sure"],
        horizontal=True, key="data_type", index=4
    )
    st.radio(
        "Are latent confounders possible?",
        ["Yes (latents possible)", "No (no latents)", "I'm not sure"],
        horizontal=True, key="latents", index=2
    )


    with st.expander("Advanced assumptions (optional)", expanded=False):
        st.radio(
            "Rough functional form between variables?",
            ["Linear", "Nonlinear (general)", "Nonlinear (additive)", "I'm not sure"],
            key="relation", horizontal=True, index=3
        )
        st.radio(
            "Noise character?",
            ["Gaussian-ish", "Non-Gaussian/Heavy-tailed", "I'm not sure"],
            key="noise", horizontal=True, index=2
        )
        st.radio(
            "Missingness?",
            ["None", "MCAR/MAR", "MNAR/Unknown", "I'm not sure"],
            key="missing", horizontal=True, index=3
        )
        st.radio(
            "What do you prioritize?", ["Accuracy", "Speed/Scale", "Either"],
            key="priority", horizontal=True, index=2
        )

    # infer n, p from uploaded CSV
    try:
        _df_all = pd.read_csv(data_path)
        n_rows, n_cols = _df_all.shape
    except Exception:
        n_rows, n_cols = 0, 0

    ans = Answers(
        data_type=st.session_state.get("data_type", "I'm not sure"),
        latents=st.session_state.get("latents", "I'm not sure"),
        relation=st.session_state.get("relation", "I'm not sure"),
        noise=st.session_state.get("noise", "I'm not sure"),
        missing=st.session_state.get("missing", "I'm not sure"),
        n=int(n_rows), p=int(n_cols),
        priority=st.session_state.get("priority", "Either"),
    )
        # ==== If user is unsure, default to FGES ====
    # ==== Only default to FGES if ALL answers are 'I'm not sure' ====
    default_to_fges = all([
        ans.data_type == "I'm not sure",
        ans.latents == "I'm not sure",
        ans.relation == "I'm not sure",
        ans.noise == "I'm not sure",
        ans.missing == "I'm not sure",
        ans.priority == "Either"
    ])


    if default_to_fges:
        st.markdown("---")
        st.info("All options were left as 'I'm not sure' — defaulting to **FGES (bootstrap)** as a safe starting point.")
        top3 = [{"rank": 1, "name": "FGES (bootstrap)", "score": 999, "why": "robust default (or PC also)"}]
        conf_label = "Recommended"

    else:
        # Normal method ranking
        top3, conf_label = rank_methods(ans, METHODS)



    st.markdown("---")
    st.subheader("Step 3 — Configure & run")

    # Show Top-3 + confidence label
    if top3:
        st.markdown("**Top 3 matches**")
        tag_map = {"Recommended": "✅ Recommended","Tentative": "⚠️ Tentative","No clear fit": "ℹ️ No clear fit","Best for Mixed": "🌈 Best for Mixed"}
        for t in top3:
            line = f"#{t['rank']} — {t['name']} · matches {t['score']}"
            if t["rank"] == 1:
                tag = tag_map.get(conf_label, conf_label)
                line += f"  ·  {tag}"
                st.success(line + (f"  ·  {t['why']}" if t["why"] else ""))
            else:
                st.info(line + (f"  ·  {t['why']}" if t["why"] else ""))
    else:
        st.caption("No ranked suggestions available with the current answers.")

    with st.form("run_form", clear_on_submit=False):
        method_names = list(METHODS.keys())
        default_index = 0
        if top3 and top3[0]["name"] in METHODS:
            default_index = method_names.index(top3[0]["name"])
        choice = st.selectbox("Method", method_names, index=default_index, key="method_choice")

        if choice == "iMIIC (R)":
            df_preview = pd.read_csv(data_path, nrows=200)
            # --- automatic detection ---
            auto_cats = []
            for col in df_preview.columns:
                nunique = df_preview[col].nunique(dropna=True)
                if df_preview[col].dtype == "object" or nunique <= 12:
                    auto_cats.append(col)

            st.caption(f"By default, these variables are treated as categorical: {', '.join(auto_cats) if auto_cats else 'none'}")

            # --- allow user to adjust ---
            st.session_state["categorical"] = st.multiselect(
                "Categorical (factors) — others will be treated as continuous:",
                df_preview.columns.tolist(),
                default=auto_cats,
                key="cats"
            )

            # ---------- ONLY TWO FILTERS ----------
            st.slider(
                "CMI′ filter (keep edges with info_shifted ≥ this value)",
                min_value=0.0, max_value=450.0,
                value=float(st.session_state.get("cmi_filter", 0.0)),
                step=0.5, key="cmi_filter",
                help="Threshold on shifted conditional mutual information (info_shifted)."
            )
            st.slider(
                "Confidence ratio filter (conf_threshold)",
                min_value=0.0, max_value=0.01,
                value=float(st.session_state.get("conf_threshold", 0.01)),
                step=0.0005, key="conf_threshold",
                help="MIIC confidence ratio cutoff; requires shufflings > 0. Default 0.01."
            )

            # Display the fixed defaults (non-editable)
            st.caption(
                "Fixed iMIIC settings here: Orientation=yes · Propagation=yes · Latent=yes · "
                "Allow Negative Info=yes · Complexity=NML · n_eff=-1 · Seed=0 · Consistent=skeleton · "
                "Max iters=95 · Consensus=0.79 · KL-distance=yes · Shufflings=99 · "
                "Orientation threshold (ratio)=1."
            )

        elif "bootstrap" in choice.lower():
            st.number_input(
                "Bootstraps (n_boot)",
                min_value=1, max_value=500,
                value=int(st.session_state.get("n_boot", 5)),
                step=1,
                key="n_boot"
            )

        submitted = st.form_submit_button("Run")

    if submitted:
        st.info(f"Running **{choice}** …")

        if choice == "iMIIC (R)":
            # --- Setup output directory ---
            run_dir = session_dir / f"run_{int(time.time())}_iMIIC"
            run_dir.mkdir(parents=True, exist_ok=True)

            # --- Collect UI values ---
            categorical = st.session_state.get("categorical", [])
            cats_r = ", ".join([f'\"{c}\"' for c in categorical])

            conf_threshold = float(st.session_state.get("conf_threshold", 0.01))
            cmi_filter = float(st.session_state.get("cmi_filter", 0.0))

            # --- Make paths safe for R (slashes) ---
            safe_data_path = str(data_path).replace("\\", "/")
            safe_run_dir   = str(run_dir).replace("\\", "/")

            # --- Build R script with fixed defaults + 2 user filters ---
            r_code = f"""
    library(miic); library(igraph)
    set.seed(0)

    # ===== Read & type variables =====
    df <- read.csv("{safe_data_path}", check.names=TRUE, na.strings=c("","NA","NaN"))
    cats <- c({cats_r})
    for (nm in intersect(make.names(cats, unique=TRUE), names(df))) {{
      df[[nm]] <- as.factor(df[[nm]])
    }}
    for (nm in setdiff(names(df), make.names(cats, unique=TRUE))) {{
      x <- df[[nm]]
      if (is.factor(x)) x <- as.character(x)
      suppressWarnings(df[[nm]] <- as.numeric(x))
    }}

    # ===== Run iMIIC with fixed defaults you specified =====
    miic_obj <- miic(
      input_data          = df,
      cplx                = "nml",
      orientation         = TRUE,
      ort_proba_ratio     = 1,
      propagation         = TRUE,
      latent              = "yes",
      n_eff               = -1,
      n_shuffles          = 99,
      conf_threshold      = {conf_threshold},
      test_mar            = TRUE,            # KL-distance
      consistent          = "skeleton",
      max_iteration       = 95,
      consensus_threshold = 0.79,
      negative_info       = TRUE
    )

    # ===== Save full artifacts =====
    write.csv(miic_obj$summary, file.path("{safe_run_dir}", "imiic_summary.csv"), row.names=FALSE)
    if (!is.null(miic_obj$proba_adj_matrix)) {{
      write.csv(miic_obj$proba_adj_matrix, file.path("{safe_run_dir}", "imiic_proba_adj_matrix.csv"), row.names=TRUE)
    }}
    if (!is.null(miic_obj$adj_matrix)) {{
      write.csv(miic_obj$adj_matrix, file.path("{safe_run_dir}", "imiic_adj_matrix.csv"), row.names=TRUE)
    }}

    # ===== Filter by CMI′ (info_shifted) =====
    sum_df <- miic_obj$summary
    if (!is.null(sum_df) && nrow(sum_df) > 0 && "info_shifted" %in% names(sum_df)) {{
      sum_filt <- subset(sum_df, info_shifted >= {cmi_filter})
      write.csv(sum_filt, file.path("{safe_run_dir}", "imiic_summary_filtered_by_cmi.csv"), row.names=FALSE)
    }}

    # ===== Export & plot FILTERED graph =====
    g <- export(miic_obj, "igraph")
    g_filt <- g
    if ("info_shifted" %in% igraph::edge_attr_names(g)) {{
      keep_idx <- which(E(g)$info_shifted >= {cmi_filter})
      g_filt <- igraph::subgraph.edges(g, E(g)[keep_idx], delete.vertices = FALSE)
    }}
    png(file.path("{safe_run_dir}", "imiic_graph_filtered.png"), width=1600, height=1200)
    plot(g_filt)
    dev.off()
    """

            # Save & run R script
            script_path = run_dir / "imiic_run.R"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(r_code)

            proc = subprocess.run(
                ["Rscript", str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if proc.returncode != 0:
                st.error("iMIIC failed")
                st.code(proc.stderr)
                st.stop()

            st.success("iMIIC run complete!")

            # ===== Show summaries =====
            edges = pd.read_csv(run_dir / "imiic_summary.csv")
            st.subheader("Full summary (unfiltered)")
            st.dataframe(edges.head(30))

            filt_path = run_dir / "imiic_summary_filtered_by_cmi.csv"
            if filt_path.exists():
                st.subheader("Filtered summary by CMI′ (info_shifted)")
                st.dataframe(pd.read_csv(filt_path).head(30))

            # ===== Show graphs =====
            img = run_dir / "imiic_graph_filtered.png"
            if img.exists():
                st.image(str(img), caption=img.name, use_container_width=True)

            # ===== Optional matrices =====
            proba_path = run_dir / "imiic_proba_adj_matrix.csv"
            if proba_path.exists():
                proba_df = pd.read_csv(proba_path, index_col=0)
                st.subheader("Edge orientation probabilities (proba_adj_matrix)")
                st.dataframe(proba_df)

            adj_path = run_dir / "imiic_adj_matrix.csv"
            if adj_path.exists():
                adj_df = pd.read_csv(adj_path, index_col=0)
                st.subheader("Adjacency matrix (consensus/raw)")
                st.dataframe(adj_df)

            # ===== Downloads =====
            st.markdown("### Downloads from this run")
            exts = {".csv", ".npy", ".txt", ".pdf", ".png"}
            files = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            for f in files:
                with open(f, "rb") as fh:
                    st.download_button(f"Download {f.name}", fh.read(), file_name=f.name)

            st.stop()  # prevent downstream Python runner

        else:
            # per-run output folder
            run_dir = session_dir / f"run_{int(time.time())}_{choice.replace(' ','_').replace('(','').replace(')','')}"
            run_dir.mkdir(parents=True, exist_ok=True)

            script_path = SCRIPTS_DIR / METHODS[choice]
            if not script_path.exists():
                st.error(f"Script not found: {script_path}")
                st.stop()

            # Base command
            cmd = [sys.executable, str(script_path), "--data", str(data_path), "--outdir", str(run_dir)]

            # Add n_boot only for bootstrap methods
            if "bootstrap" in choice.lower():
                cmd += ["--n_boot", str(int(st.session_state.get("n_boot", 10)))]

            st.caption(f"Command: {' '.join(cmd)}")

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(SCRIPTS_DIR),
            )

        # show logs
        if proc.stdout:
            with st.expander("stdout"):
                st.code(proc.stdout)
        if proc.stderr:
            with st.expander("stderr"):
                st.code(proc.stderr)
        if proc.returncode != 0:
            st.error(f"Exited with code {proc.returncode}. See errors above.")
            st.stop()

        # ==== Display images ====
        if "bootstrap" in choice.lower():
            consensus = run_dir / "ges_boot_consensus.png"
            annotated = run_dir / "ges_boot_consensus_annotated.png"
            shown_any = False
            if consensus.exists():
                st.success("Consensus (merged) graph:")
                st.image(str(consensus), caption=consensus.name, use_container_width=True)
                shown_any = True
            if annotated.exists():
                st.success("Consensus with annotations (skeleton%, same-direction%):")
                st.image(str(annotated), caption=annotated.name, use_container_width=True)
                shown_any = True
            if not shown_any:
                pngs = sorted(run_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime)
                if pngs:
                    st.success("Graphs produced:")
                    for p in pngs:
                        st.image(str(p), caption=p.name, use_container_width=True)
                else:
                    st.warning(f"No PNG found in {run_dir}.")
        else:
            pngs = sorted(run_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pngs:
                st.success("Done! Showing your graph:")
                st.image(str(pngs[0]), caption=pngs[0].name, use_container_width=True)
            else:
                st.warning(f"No PNG found in {run_dir}. Make sure your script saves one to --outdir.")

        # ==== Downloads ====
        st.markdown("### Downloads from this run")
        exts = {".csv", ".npy", ".txt", ".pdf", ".png"}
        files = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        if not files:
            st.write("No artifacts were saved.")
        else:
            for f in files:
                with open(f, "rb") as fh:
                    st.download_button(f"Download {f.name}", fh.read(), file_name=f.name)