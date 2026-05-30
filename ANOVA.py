import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib import font_manager
import io
import itertools

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

def get_cld_letters(df, target, group, tukey_summary):
    means = df.groupby(group)[target].mean().sort_values(ascending=False)
    groups_list = means.index.astype(str).tolist()
    
    cld = {g: [] for g in groups_list}
    current_letter = ord('a')
    
    for i in range(len(groups_list)):
        g1 = groups_list[i]
        non_diff = [g1]
        for j in range(i + 1, len(groups_list)):
            g2 = groups_list[j]
            mask = ((tukey_summary['group1'].astype(str) == g1) & (tukey_summary['group2'].astype(str) == g2)) | \
                   ((tukey_summary['group1'].astype(str) == g2) & (tukey_summary['group2'].astype(str) == g1))
            reject = tukey_summary.loc[mask, 'reject'].values
            if len(reject) > 0 and not reject[0]: 
                non_diff.append(g2)
        
        shared_letters = set(cld[non_diff[0]])
        for g in non_diff[1:]:
            shared_letters = shared_letters.intersection(cld[g])
        
        if not shared_letters:
            for g in non_diff:
                cld[g].append(chr(current_letter))
            current_letter += 1
    
    return pd.DataFrame({
        group: list(cld.keys()),
        '有意差': ["".join(l) for l in cld.values()]
    }), groups_list


# ==========================================
# サイドバーナビゲーション
# ==========================================
st.sidebar.title("🌱 メニュー")
app_mode = st.sidebar.radio(
    "機能を選択してください",
    ["📝 1. 調査様式作成 (PlotBuilder)", "📊 2. データ解析 (ANOVA)"]
)
st.sidebar.divider()
st.sidebar.info("💡 **使い方**\n\n1. 「調査様式作成」で整然データのテンプレートを作り、データを入力します。\n2. 「データ解析」にそのデータを貼り付けることで、すぐに解析が可能です。")

# ==========================================
# モード1: PlotBuilder (様式作成)
# ==========================================
if app_mode == "📝 1. 調査様式作成 (PlotBuilder)":
    st.title("📝 PlotBuilder - 調査データ様式ジェネレーター")
    st.markdown("試験の要因数と水準を設定すると、統計解析にそのまま使える「整然データ（Tidy Data）」形式の入力様式を自動生成します。")

    with st.expander("💡 新人さんへ：データ入力 4つの鉄則（必ず読んでね！）", expanded=True):
        st.markdown("""
        統計解析（RやPython、またはこのツールの「データ解析」機能）をスムーズに行うため、以下のルールを守って入力してください。
        
        1. **1行1データの原則**: 「横」にデータを伸ばさず「縦」に追加する。
        2. **セル結合は絶対禁止**: 結合するとプログラムで読み込めなくなります。
        3. **「数値」と「文字」を混ぜない**: 例）「45(病害)」はNG。特記事項は必ず「備考」列へ。
        4. **欠損値の扱いを統一**: 枯死などでデータが取れなかった場合は「空欄」にする（「-」や「欠」は入力しない）。
        """)

    st.divider()

    st.subheader("1. 試験区の設計")
    num_factors = st.number_input("設定する要因の数", min_value=1, max_value=5, value=2, step=1)

    factor_names = []
    factor_levels_list = []

    st.markdown("#### 要因と水準の入力")
    for i in range(num_factors):
        col1, col2 = st.columns([1, 2])
        with col1:
            default_name = ["品種", "施肥量", "栽植密度"][i] if i < 3 else f"要因{i+1}"
            f_name = st.text_input(f"要因 {i+1} の名前", value=default_name, key=f"name_{i}")
            factor_names.append(f_name)
        with col2:
            default_levels = ["シマアカリ, ニシユタカ, アイユタカ", "少肥, 標準, 多肥", "疎, 密"][i] if i < 3 else "水準A, 水準B"
            f_levels = st.text_input(f"要因 {i+1} の水準（カンマ区切り）", value=default_levels, key=f"levels_{i}")
            cleaned_levels = [x.strip() for x in f_levels.split(",") if x.strip() != ""]
            factor_levels_list.append(cleaned_levels)

    st.markdown("#### 反復と調査項目の設定")
    col3, col4 = st.columns(2)
    with col3:
        reps = st.number_input("反復（ブロック）数", min_value=1, max_value=20, value=3)
    with col4:
        target_var = st.text_input("目的変数（調査項目）", "収量_kg")

    st.divider()

    if st.button("📝 この設計で入力様式を作成する", type="primary"):
        rep_list = list(range(1, reps + 1))
        combinations = list(itertools.product(rep_list, *factor_levels_list))
        columns = ["反復"] + factor_names
        df_template = pd.DataFrame(combinations, columns=columns)
        df_template = df_template.sort_values(by=["反復"] + factor_names).reset_index(drop=True)
        
        df_template[target_var] = None
        df_template["備考"] = None
        
        st.subheader("2. 生成された入力様式")
        total_plots = reps
        for levels in factor_levels_list:
            total_plots *= len(levels)
            
        st.success(f"全 {total_plots} 区画の入力様式が作成されました！")
        st.info("💡 ダウンロードしたCSVに調査データを入力したら、左のメニューから**「データ解析 (ANOVA)」**を開き、データを貼り付けるだけですぐに解析結果が得られます。")
        
        st.dataframe(df_template, use_container_width=True)
        
        csv = df_template.to_csv(index=False).encode('utf-8-sig') 
        st.download_button(
            label="📥 この様式をダウンロード (CSV)",
            data=csv,
            file_name="field_survey_template.csv",
            mime="text/csv",
        )

# ==========================================
# モード2: 多要因分散分析＆成績書作成
# ==========================================
elif app_mode == "📊 2. データ解析 (ANOVA)":
    st.title("📊 栽培試験データ 多要因分散分析＆成績書作成ツール")

    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    data_source = st.radio(
        "データの入力方法：", 
        ["📋 Excelからデータを貼り付ける（推奨）", "🥔 サンプルデータ（明確な交互作用あり）で試す"],
        horizontal=True
    )

    df_real = None

    if data_source == "🥔 サンプルデータ（明確な交互作用あり）で試す":
        varieties = ['シマアカリ', 'ニシユタカ', 'デジマ', 'アイユタカ']
        locations = ['西之表', '中種子', '南種子', '熊毛']
        fertilizers = ['無施肥', '少肥', '標準', '多肥']
        
        data = []
        for v, l, f in itertools.product(varieties, locations, fertilizers):
            for _ in range(3):
                base = 1000
                if v == 'シマアカリ': base += 200
                elif v == 'ニシユタカ': base += 100
                
                if l == '西之表': base += 100
                elif l == '熊毛': base += 80
                
                if f == '少肥': base += 50
                elif f == '標準': base += 150
                elif f == '多肥': base += 200
                
                # 交互作用のシミュレーション
                if v == 'シマアカリ' and f == '多肥': 
                    base += 300 
                
                yield_val = np.random.normal(base, 100)
                data.append([v, l, f, yield_val])
                
        df_real = pd.DataFrame(data, columns=['品種', '地域', '施肥量', '収量'])
        st.success(f"✅ サンプルデータを読み込みました（全{len(df_real)}件）。")

    else:
        st.info("💡 **データの貼り付け手順:**\n\n"
                "左のメニュー「**調査様式作成 (PlotBuilder)**」等で作成したデータファイルの表をコピーして貼り付けます。\n\n"
                "1. Excelから、解析したいデータ（1行目は列名）をコピーします。\n"
                "2. 下の黒枠内をクリックして、**貼り付け（Ctrl + V）** を行います。\n"
                "3. 貼り付け後、キーボードの **`Ctrl` + `Enter`** を押すか、すぐ下の **「📥 データを確定して読み込む」** ボタンを押してください。")
        
        with st.expander("👀 正しいデータ形式（整然データ）の例を見る", expanded=False):
            st.markdown("""
            **【OKな例】要因ごとに列が分かれており、縦にデータが並んでいる状態**
            | 反復 | 品種 | 施肥量 | 収量 |
            | :--- | :--- | :--- | :--- |
            | 1 | ニシユタカ | 少肥 | 1050 |
            | 1 | シマアカリ | 多肥 | 1300 |
            | 2 | ニシユタカ | 少肥 | 1020 |
            """)
        
        pasted_data = st.text_area("ここにExcelデータを貼り付けてください (貼り付け後、Ctrl+Enter で確定)", height=150)
        
        if st.button("📥 データを確定して読み込む"):
            pass 
        
        if pasted_data:
            try:
                df_real = pd.read_csv(io.StringIO(pasted_data), sep=None, engine='python')
                st.success("✅ データを正常に読み込みました。下の「解析設定」に進んでください。")
            except Exception as e:
                st.error(f"データの読み込みに失敗しました。コピーした範囲が正しいか確認してください。エラー詳細: {e}")

    # 解析設定と実行
    if df_real is not None:
        with st.expander("🔍 読み込んだデータのプレビュー", expanded=False):
            st.dataframe(df_real.head(10))

        st.subheader("⚙️ モデルの設定と解析")
        cols = df_real.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("目的変数（数値データ）", cols, index=len(cols)-1)
        with col2:
            # ▼▼▼ バグ修正箇所 ▼▼▼
            available_factors = [c for c in cols if c != target_col]
            factor_cols = st.multiselect(
                "主効果とする要因（複数選択可）",
                available_factors,
                default=available_factors  # options と同じリストを使う
            )
            # ▲▲▲ バグ修正箇所 ▲▲▲

        possible_interactions = []
        if len(factor_cols) >= 2:
            possible_interactions = list(itertools.combinations(factor_cols, 2))
            interaction_labels = [f"{c[0]} × {c[1]}" for c in possible_interactions]
            
            selected_interaction_labels = st.multiselect(
                "考慮する交互作用（オプション・必要なものだけ選択）", 
                interaction_labels, 
                default=[]
            )
            selected_interactions = [possible_interactions[interaction_labels.index(label)] for label in selected_interaction_labels]
        else:
            selected_interactions = []

        if st.button("🚀 解析を実行する", type="primary") and factor_cols:
            df_clean = df_real.copy()
            df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[target_col] + factor_cols)
            
            for f in factor_cols:
                df_clean[f] = df_clean[f].astype(str)

            st.session_state.analyzed = True
            st.session_state.df_clean = df_clean
            st.session_state.target_col = target_col
            st.session_state.factor_cols = factor_cols
            st.session_state.selected_interactions = selected_interactions

        if st.session_state.analyzed:
            st.divider()
            st.info("💡 設定を変更した場合は、再度「🚀 解析を実行する」ボタンを押して結果を更新してください。")
            
            df_eval = st.session_state.df_clean
            t_col = st.session_state.target_col
            f_cols = st.session_state.factor_cols
            s_ints = st.session_state.selected_interactions

            sns.set_theme(style="whitegrid")
            set_japanese_font()

            # ANOVA計算
            formula_terms = [f'C(Q("{f}"))' for f in f_cols]
            for f1, f2 in s_ints:
                formula_terms.append(f'C(Q("{f1}")):C(Q("{f2}"))')
            formula = f'Q("{t_col}") ~ ' + ' + '.join(formula_terms)
            
            try:
                model = ols(formula, data=df_eval).fit()
                anova_res = anova_lm(model, typ=2)
                anova_res['mean_sq'] = anova_res['sum_sq'] / anova_res['df']
                anova_res['寄与率(%)'] = (anova_res['sum_sq'] / anova_res['sum_sq'].sum()) * 100
                
                def get_significance_mark(p):
                    if pd.isna(p): return ""
                    if p < 0.01: return "**"
                    elif p < 0.05: return "*"
                    else: return "ns"
                    
                anova_res['判定'] = anova_res['PR(>F)'].apply(get_significance_mark)
                report_anova = anova_res[['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)', '判定', '寄与率(%)']].copy()
                report_anova.columns = ['自由度', '平方和(SS)', '平均平方(MS)', 'F値', 'p値', '判定', '寄与率(%)']
                anova_success = True
            except Exception as e:
                st.error(f"分散分析の計算中にエラーが発生しました: {e}")
                anova_success = False

            # Tukey計算
            tukey_results = {}
            for factor in f_cols:
                summary_stats = df_eval.groupby(factor)[t_col].agg(['count', 'mean', 'std']).reset_index()
                summary_stats.rename(columns={'count': 'N', 'mean': '平均値', 'std': '標準偏差'}, inplace=True)
                try:
                    tukey = pairwise_tukeyhsd(endog=df_eval[t_col], groups=df_eval[factor], alpha=0.05)
                    tukey_summary = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                    letters_df, groups_order = get_cld_letters(df_eval, t_col, factor, tukey_summary)
                    final_report = pd.merge(summary_stats, letters_df, on=factor)
                    final_report = final_report.sort_values('平均値', ascending=False).reset_index(drop=True)
                    tukey_results[factor] = {'report': final_report, 'order': groups_order}
                except Exception:
                    pass 

            tab1, tab2, tab3, tab4 = st.tabs(["📋 1. 分散分析表", "📊 2. 主効果と多重比較", "✖️ 3. 交互作用の解析", "🤖 4. AI解説用プロンプト出力"])

            with tab1:
                if anova_success:
                    st.header("1. 分散分析表 (成績書用フォーマット)")
                    st.caption(f"実行されたモデル: `{formula}`")
                    st.dataframe(
                        report_anova.style.format({
                            '自由度': '{:.0f}', '平方和(SS)': '{:.1f}', '平均平方(MS)': '{:.1f}', 
                            'F値': '{:.3f}', 'p値': '{:.4f}', '寄与率(%)': '{:.2f}'
                        }).apply(lambda x: ['background-color: #ffcccc' if (v in ['*', '**']) else '' for v in x], subset=['判定']),
                        use_container_width=True
                    )

            with tab2:
                st.header("2. 主効果の集計表 (Tukeyの多重比較付き)")
                for factor, data_dict in tukey_results.items():
                    st.subheader(f"▶ 要因: {factor}")
                    col_t1, col_t2 = st.columns([2, 3])
                    
                    with col_t1:
                        st.dataframe(data_dict['report'].style.format({'平均値': '{:.1f}', '標準偏差': '{:.1f}'}), use_container_width=True)
                    
                    with col_t2:
                        fig_box, ax_box = plt.subplots(figsize=(8, 4))
                        sns.boxplot(x=factor, y=t_col, data=df_eval, order=data_dict['order'], ax=ax_box, color='#f0f0f0', showfliers=False)
                        sns.stripplot(x=factor, y=t_col, data=df_eval, order=data_dict['order'], ax=ax_box, color='black', alpha=0.5)
                        
                        for group_name in data_dict['order']:
                            letter = data_dict['report'].loc[data_dict['report'][factor] == group_name, '有意差'].values[0]
                            x_pos = data_dict['order'].index(group_name)
                            y_max = df_eval[df_eval[factor] == group_name][t_col].max()
                            ax_box.text(x_pos, y_max + (y_max*0.02), letter, ha='center', va='bottom', fontweight='bold', color='#d62728', fontsize=12)
                        
                        ax_box.set_ylabel(t_col)
                        st.pyplot(fig_box)

            with tab3:
                if s_ints:
                    st.header("3. 交互作用図（要因間の相乗効果の確認）")
                    st.info("💡 タブ1の分散分析表で**有意（* または **）**となった交互作用がある場合、グラフにして比較してみましょう。")
                    
                    int_options = [f"{c[0]} × {c[1]}" for c in s_ints]
                    selected_int_plot = st.selectbox("グラフ化する交互作用を選択してください", int_options)
                    plot_x, plot_hue = s_ints[int_options.index(selected_int_plot)]
                    
                    if st.toggle("X軸と色分け(凡例)を入れ替える"):
                        plot_x, plot_hue = plot_hue, plot_x

                    fig_int, ax_int = plt.subplots(figsize=(10, 6))
                    sns.pointplot(x=plot_x, y=t_col, hue=plot_hue, data=df_eval, ax=ax_int, 
                                  dodge=True, capsize=.1, markers=['o', 's', '^', 'D', 'v', '<', '>'], err_kws={'linewidth': 1})
                    ax_int.set_title(f"【交互作用図】 {plot_x} × {plot_hue}", fontsize=16)
                    ax_int.set_xlabel(plot_x, fontsize=12)
                    ax_int.set_ylabel(f"平均 {t_col}", fontsize=12)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=plot_hue)
                    plt.tight_layout()
                    st.pyplot(fig_int)
                else:
                    st.info("💡 解析設定で「交互作用」が選択されていないため、グラフは表示されません。")

            with tab4:
                st.header("🤖 4. AI解説用プロンプト作成")
                st.markdown("以下のテキストを右上のコピーボタン（📋）でコピーし、ChatGPTやClaude、Geminiなどに貼り付けてください。")
                
                if anova_success:
                    prompt_text = "あなたは優秀な農業データアナリストです。\n以下の栽培試験の統計解析結果（分散分析および多重比較）に基づいて、結果の概要と実践的な考察の文章を作成してください。専門用語は適度に噛み砕き、結論が明確に伝わるようにしてください。\n\n"
                    
                    prompt_text += "### 【解析の前提】\n"
                    prompt_text += f"- 目的変数（調べた値）: {t_col}\n"
                    prompt_text += f"- 考慮した要因: {', '.join(f_cols)}\n"
                    if s_ints:
                        prompt_text += f"- 考慮した交互作用: {', '.join([f'{c[0]}×{c[1]}' for c in s_ints])}\n"
                    prompt_text += "\n"

                    prompt_text += "### 【1. 分散分析 (ANOVA) の結果】\n"
                    prompt_text += "※判定マーク: **(1%有意), *(5%有意), ns(有意差なし)\n"
                    for index, row in report_anova.iterrows():
                        if index == "Residual": continue
                        mark = row['判定'] if row['判定'] != "" else "ns"
                        clean_index = str(index).replace('C(Q("', '').replace('"))', '').replace(':', ' × ')
                        prompt_text += f"- {clean_index}: p値 = {row['p値']:.4f} [{mark}], 寄与率 = {row['寄与率(%)']:.1f}%\n"
                    prompt_text += "\n"

                    prompt_text += "### 【2. 主効果と多重比較 (Tukey法) の結果】\n"
                    prompt_text += "※同じアルファベット（有意差列）を持つ水準間には統計的な有意差がありません。\n"
                    for factor, data_dict in tukey_results.items():
                        prompt_text += f"\n▼ {factor} ごとの平均値（高い順）:\n"
                        for _, row in data_dict['report'].iterrows():
                            prompt_text += f"  - {row[factor]}: {row['平均値']:.1f} (有意差: {row['有意差']})\n"
                    
                    prompt_text += "\n### 【AIへの指示（出力フォーマット）】\n"
                    prompt_text += "以下の構成で出力してください。\n"
                    prompt_text += "1. **結論の要約**: どの要因が最も結果に影響を与えているか（寄与率を参考に）。\n"
                    prompt_text += "2. **グループ間の具体的な比較**: 有意差のアルファベットを基に「AはBより統計的に有意に優れている」などを記述。\n"
                    if s_ints:
                        prompt_text += "3. **交互作用の解釈**: もし交互作用が有意であれば、特定の組み合わせでどのような相乗効果（または悪化）が起きているか。\n"
                    prompt_text += "4. **現場へのフィードバック案**: この結果を踏まえ、次回の試験や実際の栽培に向けた具体的なアドバイス。\n"

                    st.code(prompt_text, language="markdown")
                else:
                    st.warning("分散分析が正常に完了していないため、プロンプトを生成できません。")
