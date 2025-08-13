#Cox GUI v1.1.0 by Mehrdad S. Beni -- August 2025
#easy to use Cox regression based on lifelines lib 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
from lifelines import CoxPHFitter  #using lifelines Cox fitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt  #for plots
import numpy as np
import traceback  #error reporting
import os
import datetime


class CoxRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cox Regression GUI Analyzer")
        self.root.geometry("900x760")

        self.df = None
        self.fitted_model = None
        self.last_run_covariates = []
        self.last_run_data_subset = None
        self.last_model_type = None  #unadjusted (single/batch) or adjusted
        self.last_plot_fig = None    #keep a handle to the last partial-effects figure
        self.last_summary_df = None  #last model's summary table (if any)

        #GUI setup
        style = ttk.Style()
        style.theme_use('alt')

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        file_frame = ttk.LabelFrame(main_frame, text="1. Load Data", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        self.load_button = ttk.Button(file_frame, text="Load CSV File", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded.")
        self.file_label.pack(side=tk.LEFT, padx=5)

        columns_frame = ttk.LabelFrame(main_frame, text="2. Select Columns", padding="10")
        columns_frame.pack(fill=tk.BOTH, pady=5)

        ttk.Label(columns_frame, text="Duration (Time-to-Event):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.duration_var = tk.StringVar()
        self.duration_combo = ttk.Combobox(columns_frame, textvariable=self.duration_var, state="readonly", width=25)
        self.duration_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(columns_frame, text="Event (1=Event, 0=Censored):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.event_var = tk.StringVar()
        self.event_combo = ttk.Combobox(columns_frame, textvariable=self.event_var, state="readonly", width=25)
        self.event_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(columns_frame, text="Variables of Interest (Unadjusted, batch):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.voi_listbox_frame = ttk.Frame(columns_frame)
        self.voi_listbox_frame.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        self.voi_listbox = tk.Listbox(self.voi_listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, height=6, width=28)
        self.voi_listbox_scrollbar = ttk.Scrollbar(self.voi_listbox_frame, orient=tk.VERTICAL, command=self.voi_listbox.yview)
        self.voi_listbox.configure(yscrollcommand=self.voi_listbox_scrollbar.set)
        self.voi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.voi_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(columns_frame, text="Covariates (for Adjusted Model):").grid(row=0, column=2, padx=15, pady=2, sticky=tk.W)
        self.covariates_listbox_frame = ttk.Frame(columns_frame)
        self.covariates_listbox_frame.grid(row=0, column=3, rowspan=3, padx=5, pady=2, sticky=tk.NSEW)
        self.covariates_listbox = tk.Listbox(self.covariates_listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, height=8, width=36)
        self.covariates_listbox_scrollbar = ttk.Scrollbar(self.covariates_listbox_frame, orient=tk.VERTICAL, command=self.covariates_listbox.yview)
        self.covariates_listbox.configure(yscrollcommand=self.covariates_listbox_scrollbar.set)
        self.covariates_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.covariates_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        columns_frame.grid_columnconfigure(3, weight=1)

        analysis_frame = ttk.LabelFrame(main_frame, text="3. Run Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=5)

        self.run_unadjusted_button = ttk.Button(analysis_frame, text="Run Unadjusted Cox (Batch)", command=self.run_unadjusted_batch)
        self.run_unadjusted_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.run_adjusted_button = ttk.Button(analysis_frame, text="Run Adjusted Cox Regression", command=self.run_adjusted_cox)
        self.run_adjusted_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        self.check_prophaz_button = ttk.Button(analysis_frame, text="Check Proportional Hazards", command=self.check_proportional_hazards)
        self.check_prophaz_button.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        plot_effects_frame = ttk.Frame(analysis_frame)
        plot_effects_frame.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        self.plot_effects_button = ttk.Button(plot_effects_frame, text="Plot Partial Effects for:", command=self.plot_partial_effects)
        self.plot_effects_button.pack(side=tk.LEFT, padx=(0, 5))
        self.plot_covariate_var = tk.StringVar()
        self.plot_covariate_combo = ttk.Combobox(plot_effects_frame, textvariable=self.plot_covariate_var, state="readonly", width=20)
        self.plot_covariate_combo.pack(side=tk.LEFT)

        results_frame = ttk.LabelFrame(main_frame, text="4. Results", padding="10")
        results_frame.pack(expand=True, fill=tk.BOTH, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=18)
        self.results_text.pack(expand=True, fill=tk.BOTH)

        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill=tk.X, pady=(0, 8))
        self.save_button = ttk.Button(save_frame, text="Save Report...", command=self.save_report)
        self.save_button.pack(side=tk.RIGHT)

        self.results_text.configure(state='disabled')


    def load_csv(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.df = pd.read_csv(file_path)
            self.file_label.config(text=os.path.basename(file_path))
            self.update_column_selectors()
            self.results_text_clear()
            self.results_text_append("CSV file loaded successfully.\n")
            self.results_text_append(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns.\n")
            self.results_text_append("First 5 rows:\n")
            self.results_text_append(str(self.df.head()) + "\n")
            self._reset_last_run()
        except Exception as e:
            messagebox.showerror("Error Loading CSV", f"Failed to load or parse CSV file.\nError: {e}")
            self.df = None
            self.file_label.config(text="No file loaded.")
            self.update_column_selectors()

    def update_column_selectors(self):
        columns = list(self.df.columns) if self.df is not None else []
        self.duration_combo['values'] = columns
        self.event_combo['values'] = columns

        self.voi_listbox.delete(0, tk.END)
        self.covariates_listbox.delete(0, tk.END)
        for col in columns:
            self.voi_listbox.insert(tk.END, col)
            self.covariates_listbox.insert(tk.END, col)

        if not columns:
            self.duration_var.set('')
            self.event_var.set('')

    def results_text_append(self, text_to_append):
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, text_to_append)
        self.results_text.configure(state='disabled')
        self.results_text.see(tk.END)

    def results_text_clear(self):
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.configure(state='disabled')

    def _reset_last_run(self):
        self.fitted_model = None
        self.last_run_covariates = []
        self.last_run_data_subset = None
        self.last_model_type = None
        self.plot_covariate_combo['values'] = []
        self.plot_covariate_var.set('')
        self.last_plot_fig = None
        self.last_summary_df = None


    def _validate_core_cols(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return None, None, None
        duration_col = self.duration_var.get()
        event_col = self.event_var.get()
        if not duration_col or not event_col:
            messagebox.showerror("Error", "Please select Duration and Event columns.")
            return None, None, None
        return self.df.copy(), duration_col, event_col

    def _coerce_and_warn(self, df, duration_col, event_col):
        #duration to numeric
        try:
            df[duration_col] = pd.to_numeric(df[duration_col])
            if df[duration_col].isnull().any():
                self.results_text_append(
                    f"Warning: Duration column '{duration_col}' has missing values after numeric conversion. "
                    "Rows with these NaNs will be dropped per-model.\n"
                )
        except Exception:
            messagebox.showerror("Error", f"Duration column '{duration_col}' must be convertible to a numeric type.")
            return None

        #event to numeric (0/1)
        try:
            df[event_col] = pd.to_numeric(df[event_col])
            if df[event_col].isnull().any():
                self.results_text_append(
                    f"Warning: Event column '{event_col}' has missing values after numeric conversion. "
                    "Rows with these NaNs will be dropped per-model.\n"
                )
            if not df[event_col].dropna().isin([0, 1]).all():
                messagebox.showwarning(
                    "Warning",
                    f"Event column '{event_col}' contains values other than 0 and 1 (and NaNs). Ensure 1=event, 0=censored."
                )
        except Exception:
            messagebox.showerror("Error", f"Event column '{event_col}' must be convertible to a numeric type.")
            return None

        return df

    def _prepare_data_for_single_covariate(self, raw_df, duration_col, event_col, cov_name):
        if cov_name == duration_col or cov_name == event_col:
            self.results_text_append(f"Skipping '{cov_name}' (cannot be Duration or Event).\n")
            return None

        df = raw_df.copy()

        #coerce covariate to numeric, but allow categorical by mapping categories to codes for plotting;
        #lifelines CoxPH requires numeric covariates when using the formula string, so we coerce.
        series = pd.to_numeric(df[cov_name], errors='coerce')
        if series.isnull().all():
            self.results_text_append(f"Skipping '{cov_name}': no valid numeric data after conversion.\n")
            return None
        df[cov_name] = series

        #drop only rows needed for this model
        before = len(df)
        df = df[[duration_col, event_col, cov_name]].dropna()
        dropped = before - len(df)
        if dropped > 0:
            self.results_text_append(f"Note: dropped {dropped} rows due to NA in [{duration_col}, {event_col}, {cov_name}].\n")

        #require variability
        if df[cov_name].nunique(dropna=True) <= 1:
            self.results_text_append(f"Skipping '{cov_name}': no variance in the filtered data.\n")
            return None

        return df

    def _prepare_data_for_covariates(self, raw_df, duration_col, event_col, covariates):
        df = raw_df.copy()
        cols_used = [duration_col, event_col]
        valid_covs = []

        for name in covariates:
            if name == duration_col or name == event_col:
                messagebox.showerror("Error", f"Covariate '{name}' cannot be Duration or Event.")
                return None, None
            numeric = pd.to_numeric(df[name], errors='coerce')
            if numeric.isnull().all():
                messagebox.showwarning("Data Warning", f"Covariate '{name}' had no numeric data and will be excluded.")
                continue
            df[name] = numeric
            cols_used.append(name)
            valid_covs.append(name)

        if not valid_covs:
            return None, None

        before = len(df)
        df = df[cols_used].dropna()
        dropped = before - len(df)
        if dropped > 0:
            self.results_text_append(f"Note: dropped {dropped} rows due to NA in selected columns for adjusted model.\n")

        #remove constants
        final_covs = []
        for c in valid_covs:
            if df[c].nunique(dropna=True) <= 1:
                messagebox.showwarning("Data Warning", f"Covariate '{c}' had no variance and was excluded.")
            else:
                final_covs.append(c)

        if not final_covs:
            return None, None

        return df, final_covs


    def run_cox_model(self, data_subset, duration_col, event_col, covariate_cols_for_formula):
        cph = CoxPHFitter()
        self.last_run_data_subset = None
        self.last_summary_df = None
        try:
            columns_for_fit_df = [duration_col, event_col]
            formula_str = None

            if covariate_cols_for_formula:
                formula_str = " + ".join(covariate_cols_for_formula)
                self.results_text_append(f"Fitting model with formula: '{formula_str}'\n")
                columns_for_fit_df.extend(covariate_cols_for_formula)
                columns_for_fit_df = sorted(set(columns_for_fit_df))
                fit_df = data_subset[columns_for_fit_df].copy()
                cph.fit(fit_df, duration_col=duration_col, event_col=event_col, formula=formula_str)
            else:
                self.results_text_append("Warning: No covariates provided. Fitting baseline Cox model.\n")
                fit_df = data_subset[columns_for_fit_df].copy()
                cph.fit(fit_df, duration_col=duration_col, event_col=event_col)

            self.results_text_append("\n--- Cox Model Summary ---\n")
            summary_df = cph.summary
            #self.results_text_append(str(summary_df) + "\n")
            self.results_text_append(summary_df.to_string(max_cols=None, line_width=None) + "\n")
            self.last_summary_df = summary_df.copy() if summary_df is not None else None

            if covariate_cols_for_formula and (summary_df is not None) and not summary_df.empty:
                try:
                    preds = cph.predict_partial_hazard(fit_df[covariate_cols_for_formula])
                    c_index = concordance_index(fit_df[duration_col], -preds, fit_df[event_col])
                    self.results_text_append(f"\nConcordance Index (C-statistic): {c_index:.4f}\n")
                except Exception as ci_e:
                    self.results_text_append(f"\nCould not calculate Concordance Index: {ci_e}\n")
            else:
                self.results_text_append("\nConcordance Index not applicable (no covariates in the final model).\n")

            self.fitted_model = cph
            self.last_run_covariates = covariate_cols_for_formula
            self.last_run_data_subset = fit_df.copy()

            self.plot_covariate_combo['values'] = self.last_run_covariates
            if self.last_run_covariates:
                self.plot_covariate_var.set(self.last_run_covariates[0])
            else:
                self.plot_covariate_var.set('')

        except Exception as e:
            messagebox.showerror("Cox Regression Error", f"An error occurred during model fitting: {e}")
            self.results_text_append(f"\nError during Cox regression: {e}\n")
            self.results_text_append(f"Traceback:\n{traceback.format_exc()}\n")
            self._reset_last_run()

    def run_unadjusted_batch(self):
        self.results_text_clear()
        self.results_text_append("Running Unadjusted Cox Regression (Batch)...\n")

        raw_df, duration_col, event_col = self._validate_core_cols()
        if raw_df is None:
            self.results_text_append("Failed to start (see error popups).\n")
            return

        raw_df = self._coerce_and_warn(raw_df, duration_col, event_col)
        if raw_df is None:
            self.results_text_append("Failed to prepare data (see error popups).\n")
            return

        selected_indices = self.voi_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one Variable of Interest (VOI) from the list.")
            return
        vois = [self.voi_listbox.get(i) for i in selected_indices]

        all_results = []
        kept_any = False

        #loop over selected covars
        for voi in vois:
            self.results_text_append(f"\n--- Univariate model for '{voi}' ---\n")
            df_voi = self._prepare_data_for_single_covariate(raw_df, duration_col, event_col, voi)
            if df_voi is None or len(df_voi) == 0:
                self.results_text_append(f"Skipping '{voi}' due to data issues.\n")
                continue

            #fit single covar 
            self.run_cox_model(df_voi, duration_col, event_col, [voi])
            kept_any = True

            #collect a compact row for batch table if summary exists
            if self.last_summary_df is not None and not self.last_summary_df.empty:
                #lifelines summary index is covariate name
                row = self.last_summary_df.loc[voi]
                all_results.append({
                    "variable": voi,
                    "coef": row["coef"],
                    "exp(coef)": row["exp(coef)"],
                    "se(coef)": row["se(coef)"],
                    "z": row["z"],
                    "p": row["p"],
                    "CI 95% lower": row["coef lower 95%"],
                    "CI 95% upper": row["coef upper 95%"],
                    "n": len(df_voi)
                })

        if kept_any and all_results:
            batch_df = pd.DataFrame(all_results).set_index("variable")
            self.results_text_append("\n=== Batch Univariate Summary (most recent per-model estimates) ===\n")
            #self.results_text_append(str(batch_df) + "\n")
            self.results_text_append(batch_df.to_string(max_cols=None, line_width=None) + "\n")

            self.last_model_type = "unadjusted"
        else:
            self.results_text_append("\nNo univariate models were fit successfully.\n")

    def run_adjusted_cox(self):
        self.results_text_clear()
        self.results_text_append("Running Adjusted Cox Regression...\n")

        raw_df, duration_col, event_col = self._validate_core_cols()
        if raw_df is None:
            self.results_text_append("Failed to start (see error popups).\n")
            return

        raw_df = self._coerce_and_warn(raw_df, duration_col, event_col)
        if raw_df is None:
            self.results_text_append("Failed to prepare data (see error popups).\n")
            return

        selected_indices = self.covariates_listbox.curselection()
        if not selected_indices:
            self.results_text_append("Error: No covariates selected for adjusted model.\n")
            messagebox.showerror("Model Error", "Please select at least one covariate for the adjusted model.")
            return
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]

        data_subset, covariate_cols_for_formula = self._prepare_data_for_covariates(raw_df, duration_col, event_col, covariates)
        if data_subset is None:
            self.results_text_append("Failed to prepare data for adjusted model.\n")
            messagebox.showerror("Model Error", "No valid covariates remain after processing.")
            return

        self.results_text_append(f"Covariates for formula: {', '.join(covariate_cols_for_formula)}\n")
        self.results_text_append(f"Data subset for model has {len(data_subset)} rows.\n")

        cols_to_show = [duration_col, event_col] + covariate_cols_for_formula
        self.results_text_append(f"Head of relevant columns in data_subset:\n{str(data_subset[cols_to_show].head())}\n")

        self.run_cox_model(data_subset, duration_col, event_col, covariate_cols_for_formula)
        self.last_model_type = "adjusted"


    def check_proportional_hazards(self):
        if self.fitted_model is None or self.last_run_data_subset is None:
            messagebox.showerror("Error", "Please run a Cox model first and ensure data was available for it.")
            return
        if not self.last_run_covariates:
            messagebox.showwarning("Info", "No covariates were in the last successfully fitted model.")
            return

        self.results_text_append("\n--- Proportional Hazards Assumption Check ---\n")
        try:
            df_for_ph_check = self.last_run_data_subset
            self.results_text_append(f"Checking PH on data with {len(df_for_ph_check)} rows. Using model's internal formula.\n")

            #lifelines prints to stdout by default; we capture return and show what we can.
            results_ph = self.fitted_model.check_assumptions(df_for_ph_check, p_value_threshold=0.05, show_plots=False)

            self.results_text_append(f"Result type: {type(results_ph)}\n")
            if isinstance(results_ph, pd.DataFrame):
                if not results_ph.empty:
                    self.results_text_append(results_ph.to_string() + "\n")
                else:
                    self.results_text_append("Proportional hazards test returned an empty DataFrame (no violations detected or not applicable).\n")
            elif isinstance(results_ph, list):
                if not results_ph:
                    self.results_text_append("PH test returned an empty list (no tabular output for this configuration).\n")
                else:
                    self.results_text_append("PH test returned a list of results. See console output for details.\n")
            elif results_ph is None:
                self.results_text_append("PH test returned None (messages may have been printed to console).\n")
            else:
                self.results_text_append(f"PH test returned an unexpected type: {type(results_ph)}\n")

            self.results_text_append("Note: Low p-values (e.g., < 0.05) suggest a potential violation of the PH assumption.\n")

        except Exception as e:
            if any(msg in str(e) for msg in ["must have at least one column", "Found no covariates", "less than 2 levels"]):
                messagebox.showwarning("Assumption Check Info",
                                       f"PH test could not be performed due to covariate/data conditions. Details: {e}")
                self.results_text_append(f"\nPH test not applicable or failed. Reason: {e}\n")
            else:
                messagebox.showerror("Assumption Check Error", f"Error during proportional hazards check: {e}")
                self.results_text_append(f"\nError during assumption check: {e}\n")
                self.results_text_append(f"Traceback:\n{traceback.format_exc()}\n")

    def choose_plot_values(self, series):
        if pd.api.types.is_numeric_dtype(series):
            uniq = np.unique(series.dropna().values)
            if len(uniq) <= 1:
                return None  #nothing to vary
            q = np.unique(np.quantile(uniq, [0.1, 0.25, 0.5, 0.75, 0.9]))
            q = q.tolist()
            if len(q) >= 2:
                return q[:5]
            vals = sorted(uniq.tolist())[:5]
            return vals if len(vals) >= 2 else None
        else:
            cats = series.dropna().astype(str)
            if cats.nunique() <= 1:
                return None
            top = cats.value_counts().index.tolist()[:5]
            return top if len(top) >= 2 else None

    def plot_partial_effects(self):
        if self.fitted_model is None or self.last_run_data_subset is None:
            messagebox.showerror("Error", "Please run a Cox model first and ensure data was available for it.")
            return

        covariate_to_plot = self.plot_covariate_var.get()
        if not covariate_to_plot:
            messagebox.showerror("Error", "Please select a covariate to plot from the dropdown.")
            return

        if covariate_to_plot not in self.last_run_covariates:
            messagebox.showerror("Error", f"Covariate '{covariate_to_plot}' was not in the last model run.")
            return

        self.results_text_append(f"\n--- Plotting Partial Effects for '{covariate_to_plot}' ---\n")

        try:
            df = self.last_run_data_subset
            if covariate_to_plot not in df.columns:
                messagebox.showerror("Plotting Error", f"Covariate '{covariate_to_plot}' not found in last run data.")
                return

            series = df[covariate_to_plot]
            values = self.choose_plot_values(series)

            if values is None:
                self.results_text_append(
                    f"'{covariate_to_plot}' lacks sufficient variation for plotting partial effects.\n"
                )
                messagebox.showwarning("Plotting Info",
                                       f"Not enough distinct values in '{covariate_to_plot}' to draw partial effects.")
                return

            self.results_text_append(f"Using values for '{covariate_to_plot}': {values}\n")

            fig, ax = plt.subplots()
            self.fitted_model.plot_partial_effects_on_outcome(
                covariates=[covariate_to_plot],
                values=values,
                plot_baseline=True,
                ax=ax
            )
            ax.set_title(f"Partial Effects on Survival: {covariate_to_plot}")
            plt.tight_layout()
            self.last_plot_fig = fig  #save handle for Save Report
            plt.show()

            self.results_text_append(f"Plot for '{covariate_to_plot}' displayed.\n")

        except Exception as e:
            messagebox.showerror("Plotting Error", f"Error plotting partial effects for {covariate_to_plot}: {e}")
            self.results_text_append(f"\nError plotting partial effects for {covariate_to_plot}: {e}\n")
            self.results_text_append(f"Traceback:\n{traceback.format_exc()}\n")

    #save report function
    def save_report(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data and run an analysis before saving.")
            return

        default_name = f"cox_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("txt file", "*.txt"), ("All files", "*.*")]
        )
        if not save_path:
            return

        try:
            #writing all
            header_lines = []
            header_lines.append("Cox Regression GUI Analyzer — Report")
            header_lines.append(f"Saved: {datetime.datetime.now().isoformat(timespec='seconds')}")
            header_lines.append(f"Data shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            if self.last_model_type:
                header_lines.append(f"Last model type: {self.last_model_type}")
            if self.last_run_covariates:
                header_lines.append(f"Last model covariates: {', '.join(self.last_run_covariates)}")
            header = "\n".join(header_lines) + "\n\n"

            contents = self.results_text.get("1.0", tk.END)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(contents)


            msg = f"Report saved to:\n{save_path}"

            messagebox.showinfo("Report Saved", msg)

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report.\nError: {e}")
            self.results_text_append(f"\nError saving report: {e}\n")
            self.results_text_append(f"Traceback:\n{traceback.format_exc()}\n")


#main
if __name__ == "__main__":
    root = tk.Tk()
    app = CoxRegressionApp(root)
    root.mainloop()
