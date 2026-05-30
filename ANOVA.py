import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import studentized_range
from matplotlib import font_manager
import io
import itertools
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

try:
    from help_text import render_help_section
    HELP_AVAILABLE = True
except ImportError:
    HELP_AVAILABLE = False

# ==========================================
# 共通設定・関数
# ==========================================
st.set_page_config(page_title="栽培試験データ統合プラットフォーム", page_icon="🌱", layout="wide")

def set_japanese_font():
    try:
        import japanize_matplotlib
        japanize_matplotlib.japanize()
        return
    except ImportError:
        pass
    candidates = ['IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP',
                  'Hiragino Sans', 'Hiragino Maru Gothic Pro', 'MS Gothic',
                  'Yu Gothic', 'Meiryo']
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            return

set_japanese_font()

def get_cld_letters(groups_sorted, tukey_df):
    ns_pairs = set()
    for _, row in tukey_df.iterrows():
        g1, g2 = str(row['group1']), str(row['group2'])
        if not row['reject']:
            ns_pairs.add((g1, g2))
            ns_pairs.add((g2, g1))
    for g in groups_sorted:
        ns_pairs.add((g, g))

    letters = {g: [] for g in groups_sorted}
    letter_groups = []
    current_idx = 0

    for gi in groups_sorted:
        absorb = [gj for gj in groups_sorted if (gi, gj) in ns_pairs]
        matched = any(set(lg) == set(absorb) for lg in letter_groups)
        if not matched:
            new_letter = chr(ord('a') + current_idx)
            current_idx += 1
            letter_groups.append(absorb)
            for g in absorb:
                letters[g].append(new_letter)

    return {g: ''.join(sorted(set(v))) for g, v in letters.items()}

def tukey_from_anova_model(model, factor, df, target):
    groups = sorted(df[factor].astype(str).unique())
    n_groups = len(groups)
    mse = model.mse_resid
    df_resid = model.df_resid

    group_stats = df.groupby(factor)[target].agg(['mean', 'count'])

    results = []
    for g1, g2 in itertools.combinations(groups, 2):
        m1 = group_stats.loc[g1, 'mean']
        m2 = group_stats.loc[g2, 'mean']
        n1 = group_stats.loc[g1, 'count']
        n2 = group_stats.loc[g2, 'count']
        se = np.sqrt(mse * (1/n1 + 1/n2) / 2)
        q_stat = abs(m1 - m2) / se
        p_val = float(1 - studentized_range.cdf(q_stat, n_groups, df_resid))
        results.append({
            'group1': g1, 'group2': g2, 'meandiff': m1 - m2,
            'q_stat': q_stat, 'p-adj': p_val, 'reject': p_val < 0.05
        })
    return pd.DataFrame(results)

def build_multi_excel_report(results_dict, ss_type):
    wb = openpyxl.Workbook()
    ws_dummy = wb.active 

    header_fill = PatternFill("solid", fgColor="2E7D32")
    sig_fill    = PatternFill("solid", fgColor="FFCCCC")
    header_font = Font(bold=True, color="FFFFFF")
    bold_font   = Font(bold=True)
    center      = Alignment(horizontal="center")
    thin        = Side(style="thin")
    border      = Border(left=thin, right=thin, top=thin, bottom=thin)

    def apply_formatting(ws, is_anova=False):
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center
            cell.border = border
        for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
            for i, cell in enumerate(row):
                cell.border = border
                cell.alignment = center
                if is_anova and i == 0:
                    cell.font = bold_font
                if is_anova and ws.cell(row=1, column=i+1).value == '判定':
                    if cell.value in ['*', '**']:
                        cell.fill = sig_fill
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max(max_len + 2, 10)

    for t_col, res in results_dict.items():
        if not res['anova_success']: continue
        
        formula = res['formula']
        report_anova = res['report_anova']
        tukey_results = res['tukey_results']
        eval_col = res['eval_col']

        ws_anova = wb.create_sheet(title=f"ANOVA_{t_col}"[:31])
        ws_anova.append(["要因"] + list(report_anova.columns))
        
        for idx, row in report_anova.iterrows():
            clean_idx = str(idx).replace('C(Q("', '').replace('"))', '').replace(':', ' × ')
            ws_anova.append([clean_idx] + [row[c] for c in report_anova.columns])
        
        apply_formatting(ws_anova, is_anova=True)
        ws_anova.insert_rows(1, 4)
        ws_anova.cell(row=1, column=1, value=f"目的変数(元): {t_col}")
        ws_anova.cell(row=2, column=1, value=f"解析対象(変換後): {eval_col}  |  変換方法: {res['trans_type']}")
        ws_anova.cell(row=3, column=1, value=f"モデル: {formula}  |  SS タイプ: Type {ss_type}")

        for factor, data_dict in tukey_results.items():
            ws_tukey = wb.create_sheet(title=f"Tukey_{t_col[:10]}_{factor[:10]}"[:31])
            df_rep = data_dict['report']
            ws_tukey.append(list(df_rep.columns))
            for _, row in df_rep.iterrows():
                ws_tukey.append(list(row))
            apply_formatting(ws_tukey)
            ws_tukey.insert_rows(1, 4)
            ws_tukey.cell(row=1, column=1, value=f"目的変数: {eval_col}  |  要因: {factor}")
            ws_tukey.cell(row=2, column=1, value=f"変換方法: {res['trans_type']}")
            ws_tukey.cell(row=3, column=1, value="※ 同じ文字を持つ水準間に有意差なし（Tukey法, α=0.05）")

    wb.remove(ws_dummy)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()

# ==========================================
# UI および モード分岐
# ==========================================
st.sidebar.title("🌱 メニュー")
app_mode = st.sidebar.radio("機能を選択", ["📝 1. 調査様式作成", "📊 2. データ解析"])
if HELP_AVAILABLE:
    with st.sidebar: render_help_section()

if app_mode == "📝 1. 調査様式作成":
    st.title("📝 PlotBuilder - 調査データ様式ジェネレーター")
    # 様式作成ブロック（前回のコードと同じため省略・実装済みのものをご利用ください）
    st.info("データ解析モードを選択してください。")

elif app_mode == "📊 2. データ解析":
    st.title("📊 栽培試験データ 多要因分散分析＆成績書作成ツール")

    if 'analyzed' not in st.session_state: st.session_state.analyzed = False

    data_source = st.radio("データの入力方法：", ["📋 Excelからデータを貼り付ける", "🥔 サンプルデータで試す"], horizontal=True)
    df_real = None

    if data_source == "🥔 サンプルデータで試す":
        varieties = ['シマアカリ', 'ニシユタカ', 'アイユタカ']
        fertilizers = ['少肥', '標準', '多肥']
        data = []
        for rep, v, f in itertools.product([1,2,3], varieties, fertilizers):
            base_y = 300
            if v == 'シマアカリ': base_y += 40
            if f == '多肥': base_y += 60
            if v == 'シマアカリ' and f == '多肥': base_y += 50
            # 個数（ポアソン分布）
            base_t = 10
            if v == 'シマアカリ': base_t += 1
            tubers = np.random.poisson(lam=max(1, base_t))
            # 割合（二項分布的に。多肥だと規格内率低下とする）
            rate = 85
            if f == '多肥': rate -= 15
            market_rate = np.clip(np.random.normal(rate, 5), 0, 100)
            data.append([rep, v, f, np.random.normal(base_y, 25), tubers, market_rate])
        df_real = pd.DataFrame(data, columns=['反復', '品種', '施肥量', '総収量_kg', '上いも数_個', '規格内率_パーセント'])
        st.success("✅ サンプルデータを読み込みました。（個数・パーセントデータを含みます）")
    else:
        pasted_data = st.text_area("Excelデータを貼り付け (Ctrl+V)", height=150)
        if st.button("📥 読み込む"): pass
        if pasted_data:
            try:
                df_real = pd.read_csv(io.StringIO(pasted_data), sep=None, engine='python')
                st.success("✅ データを読み込みました。")
            except Exception as e:
                st.error(f"エラー: {e}")

    if df_real is not None:
        st.subheader("⚙️ 解析設定")
        cols = df_real.columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
            default_targets = [c for c in cols if df_real[c].dtype in [np.float64, np.int64] and c not in ["反復", "rep"]]
            target_cols = st.multiselect("目的変数（複数可）", cols, default=default_targets)
        with col2:
            available_factors = [c for c in cols if c not in target_cols and c != "備考"]
            factor_cols = st.multiselect("主効果とする要因", available_factors, default=available_factors)

        # ---------------------------------------------
        # 🌟 データ変換 UI追加
        # ---------------------------------------------
        st.markdown("#### 📐 データ変換設定 (オプション)")
        st.info("カウントデータ(個数など)や割合データ(％)を正規分布に近づけるための変数変換を指定します。")
        transformation_dict = {}
        for t_col in target_cols:
            trans_type = st.selectbox(
                f"【{t_col}】の変換方法", 
                [
                    "変換なし", 
                    "対数変換 ( log10(x + 0.5) ) ※カウントデータ向け", 
                    "アークサイン変換 ( arcsin(√(x/100)) ) ※%データ向け"
                ],
                key=f"trans_{t_col}"
            )
            transformation_dict[t_col] = trans_type

        ss_type = st.radio("平方和のタイプ", [2, 3], format_func=lambda x: f"Type {x}", horizontal=True)

        selected_interactions = []
        if len(factor_cols) >= 2:
            interaction_labels = [f"{c[0]} × {c[1]}" for c in itertools.combinations(factor_cols, 2)]
            selected_int = st.multiselect("考慮する交互作用", interaction_labels, default=[])
            selected_interactions = [tuple(x.split(" × ")) for x in selected_int]

        rep_candidates = [c for c in factor_cols if '反復' in c or 'rep' in c.lower() or 'Rep' in c]

        if st.button("🚀 選択した全変数の解析を実行する", type="primary") and factor_cols and target_cols:
            st.session_state.analyzed = True
            st.session_state.target_cols = target_cols
            st.session_state.factor_cols = factor_cols
            st.session_state.selected_interactions = selected_interactions
            st.session_state.ss_type = ss_type

            results_dict = {}
            for t_col in target_cols:
                trans_type = transformation_dict[t_col]
                eval_col = t_col
                
                df_eval = df_real.copy()
                df_eval[t_col] = pd.to_numeric(df_eval[t_col], errors='coerce')
                df_eval = df_eval.dropna(subset=[t_col] + factor_cols)

                # ---------------------------------------------
                # 🌟 変換ロジックの適用
                # ---------------------------------------------
                if "対数変換" in trans_type:
                    eval_col = f"{t_col}_Log10"
                    df_eval[eval_col] = np.log10(df_eval[t_col] + 0.5)
                elif "アークサイン変換" in trans_type:
                    eval_col = f"{t_col}_Arcsin"
                    # 0-100の%データを想定。0-1に収めてから変換、出力は角度(°)
                    val_prop = np.clip(df_eval[t_col] / 100.0, 0.0, 1.0)
                    df_eval[eval_col] = np.degrees(np.arcsin(np.sqrt(val_prop)))

                for f in factor_cols:
                    df_eval[f] = df_eval[f].astype(str)

                formula_terms = [f'C(Q("{f}"))' for f in factor_cols]
                for f1, f2 in selected_interactions:
                    formula_terms.append(f'C(Q("{f1}")):C(Q("{f2}"))')
                formula = f'Q("{eval_col}") ~ ' + ' + '.join(formula_terms)

                anova_success = False
                report_anova = None
                tukey_results = {}

                try:
                    model = ols(formula, data=df_eval).fit()
                    anova_res = anova_lm(model, typ=ss_type)
                    anova_res['mean_sq'] = anova_res['sum_sq'] / anova_res['df']
                    anova_res['寄与率(%)'] = (anova_res['sum_sq'] / anova_res['sum_sq'].sum()) * 100

                    def get_sig(p):
                        if pd.isna(p): return ""
                        if p < 0.01: return "**"
                        if p < 0.05: return "*"
                        return "ns"

                    anova_res['判定'] = anova_res['PR(>F)'].apply(get_sig)
                    report_anova = anova_res[['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)', '判定', '寄与率(%)']].copy()
                    report_anova.columns = ['自由度', '平方和(SS)', '平均平方(MS)', 'F値', 'p値', '判定', '寄与率(%)']
                    anova_success = True
                except Exception as e:
                    st.error(f"【{t_col}】 分散分析エラー: {e}")

                tukey_targets = [f for f in factor_cols if f not in rep_candidates]
                for factor in tukey_targets:
                    summary_stats = df_eval.groupby(factor)[eval_col].agg(['count', 'mean', 'std']).reset_index()
                    summary_stats.rename(columns={'count': 'N', 'mean': '平均値(変換後)', 'std': '標準偏差(変換後)'}, inplace=True)
                    
                    # 参考値として変換前のオリジナル平均も結合
                    if eval_col != t_col:
                        orig_mean = df_eval.groupby(factor)[t_col].mean().reset_index()
                        orig_mean.rename(columns={t_col: '平均値(オリジナル)'}, inplace=True)
                        summary_stats = pd.merge(summary_stats, orig_mean, on=factor)

                    try:
                        if anova_success:
                            tukey_df = tukey_from_anova_model(model, factor, df_eval, eval_col)
                        else:
                            tukey_obj = pairwise_tukeyhsd(endog=df_eval[eval_col], groups=df_eval[factor], alpha=0.05)
                            tukey_df = pd.DataFrame(data=tukey_obj._results_table.data[1:], columns=tukey_obj._results_table.data[0])

                        means_sorted = df_eval.groupby(factor)[eval_col].mean().sort_values(ascending=False)
                        groups_sorted = means_sorted.index.astype(str).tolist()
                        cld_map = get_cld_letters(groups_sorted, tukey_df)
                        letters_df = pd.DataFrame({'_g': list(cld_map.keys()), '有意差': list(cld_map.values())})
                        letters_df.rename(columns={'_g': factor}, inplace=True)

                        final_report = pd.merge(summary_stats, letters_df, on=factor)
                        final_report = final_report.sort_values('平均値(変換後)', ascending=False).reset_index(drop=True)
                        tukey_results[factor] = {'report': final_report, 'order': groups_sorted}
                    except Exception:
                        pass 

                results_dict[t_col] = {
                    'eval_col': eval_col,
                    'trans_type': trans_type,
                    'df_eval': df_eval,
                    'formula': formula,
                    'anova_success': anova_success,
                    'report_anova': report_anova,
                    'tukey_results': tukey_results
                }
            st.session_state.results_dict = results_dict

        # ==========================================
        # 解析結果の表示
        # ==========================================
        if st.session_state.analyzed:
            st.divider()
            
            view_target = st.selectbox("📊 結果を表示する目的変数を切り替え", st.session_state.target_cols)
            res_current = st.session_state.results_dict[view_target]
            df_eval = res_current['df_eval']
            t_col = view_target
            eval_col = res_current['eval_col']
            trans_type = res_current['trans_type']
            f_cols = st.session_state.factor_cols
            s_ints = st.session_state.selected_interactions
            ss_t = st.session_state.ss_type
            anova_success = res_current['anova_success']
            report_anova = res_current['report_anova']
            tukey_results = res_current['tukey_results']
            formula = res_current['formula']

            sns.set_theme(style="whitegrid")
            set_japanese_font()

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                f"📋 1. 分散分析表", f"📊 2. 主効果と多重比較", f"✖️ 3. 交互作用", "📥 4. Excel出力", "🤖 5. AIプロンプト"
            ])

            with tab1:
                if anova_success:
                    st.header(f"1. 分散分析表 - 【{eval_col}】")
                    if "変換なし" not in trans_type:
                        st.warning(f"💡 適用したデータ変換: {trans_type}")
                    st.dataframe(
                        report_anova.style.format({
                            '自由度': '{:.0f}', '平方和(SS)': '{:.1f}', '平均平方(MS)': '{:.1f}',
                            'F値': '{:.3f}', 'p値': '{:.4f}', '寄与率(%)': '{:.2f}'
                        }).apply(lambda x: ['background-color: #ffcccc' if v in ['*', '**'] else '' for v in x], subset=['判定']),
                        use_container_width=True
                    )

            with tab2:
                st.header(f"2. 主効果の集計表 - 【{eval_col}】")
                for factor, data_dict in tukey_results.items():
                    st.subheader(f"▶ 要因: {factor}")
                    col_t1, col_t2 = st.columns([2, 3])
                    with col_t1:
                        st.dataframe(data_dict['report'].style.format(precision=1), use_container_width=True)
                    with col_t2:
                        fig_box, ax_box = plt.subplots(figsize=(8, 4))
                        sns.boxplot(x=factor, y=eval_col, data=df_eval, order=data_dict['order'], ax=ax_box, color='#f0f0f0', showfliers=False)
                        sns.stripplot(x=factor, y=eval_col, data=df_eval, order=data_dict['order'], ax=ax_box, color='black', alpha=0.5)
                        for gname in data_dict['order']:
                            letter = data_dict['report'].loc[data_dict['report'][factor] == gname, '有意差'].values[0]
                            xpos   = data_dict['order'].index(gname)
                            ymax   = df_eval[df_eval[factor] == gname][eval_col].max()
                            ax_box.text(xpos, ymax + abs(ymax) * 0.02, letter, ha='center', va='bottom', fontweight='bold', color='#d62728', fontsize=12)
                        ax_box.set_ylabel(eval_col)
                        st.pyplot(fig_box)
                        plt.close(fig_box)

            with tab3:
                if s_ints and anova_success:
                    st.header(f"3. 交互作用図 - 【{eval_col}】")
                    int_options = [f"{c[0]} × {c[1]}" for c in s_ints]
                    selected_int_plot = st.selectbox("グラフ化する交互作用", int_options, key=f"int_sel_{t_col}")
                    plot_x, plot_hue = selected_int_plot.split(" × ")
                    if st.toggle("X軸と色分けを入れ替え", key=f"tog_{t_col}"): plot_x, plot_hue = plot_hue, plot_x

                    fig_int, ax_int = plt.subplots(figsize=(10, 6))
                    sns.pointplot(x=plot_x, y=eval_col, hue=plot_hue, data=df_eval, ax=ax_int, dodge=True, capsize=.1, markers=['o', 's', '^', 'D', 'v', '<', '>'], err_kws={'linewidth': 1})
                    ax_int.set_title(f"【交互作用図】 {plot_x} × {plot_hue} (縦軸: {eval_col})", fontsize=16)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=plot_hue)
                    plt.tight_layout()
                    st.pyplot(fig_int)
                    plt.close(fig_int)

            with tab4:
                st.header("📥 4. 全変数の解析結果をExcelで一括ダウンロード")
                excel_bytes = build_multi_excel_report(st.session_state.results_dict, ss_t)
                st.download_button("📥 Excelでダウンロード", data=excel_bytes, file_name="anova_multitrait_transformed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

            with tab5:
                st.header(f"🤖 5. AI解説用プロンプト")
                if anova_success:
                    lines = [
                        "あなたは優秀な農業データアナリストです。",
                        f"以下の栽培試験の統計解析結果（解析対象: {eval_col}）の考察を作成してください。",
                        "", "### 【解析の前提】",
                        f"- 元データ: {t_col}",
                        f"- データ変換: {trans_type}",
                        f"- 要因: {', '.join(f_cols)}"
                    ]
                    lines += ["", "### 【分散分析結果】", "※ **(1%有意), *(5%有意), ns(有意差なし)"]
                    for idx, row in report_anova.iterrows():
                        if idx == "Residual": continue
                        clean = str(idx).replace('C(Q("', '').replace('"))', '').replace(':', ' × ')
                        lines.append(f"- {clean}: p={row['p値']:.4f} [{row['判定'] or 'ns'}], 寄与率={row['寄与率(%)']:.1f}%")
                    lines += ["", "### 【多重比較結果（Tukey法）】"]
                    for factor, dd in tukey_results.items():
                        lines.append(f"\n▼ {factor}:")
                        for _, r in dd['report'].iterrows():
                            lines.append(f"  - {r[factor]}: {r['平均値(変換後)']:.1f} [{r['有意差']}] (元データ平均: {r.get('平均値(オリジナル)', r['平均値(変換後)']):.1f})")
                    st.code("\n".join(lines), language="markdown")
