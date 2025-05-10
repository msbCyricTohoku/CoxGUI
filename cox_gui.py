#Cox GUI v1.0 by Mehrdad S. Beni -- May 2025
#easy to use Cox regression based on lifelines lib 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
from lifelines import CoxPHFitter #using lifeline cox fitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt #for plots
import traceback #error reporting

class CoxRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cox Regression Analyzer")
        self.root.geometry("800x700")

        self.df = None
        self.fitted_model = None
        self.last_run_covariates = []
        self.last_run_data_subset = None


        #GUI setup
        style = ttk.Style()
        style.theme_use('clam') 

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        file_frame = ttk.LabelFrame(main_frame, text="1. Load Data", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        self.load_button = ttk.Button(file_frame, text="Load CSV File", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded.")
        self.file_label.pack(side=tk.LEFT, padx=5)

        columns_frame = ttk.LabelFrame(main_frame, text="2. Select Columns", padding="10")
        columns_frame.pack(fill=tk.X, pady=5)

        ttk.Label(columns_frame, text="Duration (Time-to-Event):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.duration_var = tk.StringVar()
        self.duration_combo = ttk.Combobox(columns_frame, textvariable=self.duration_var, state="readonly", width=25)
        self.duration_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(columns_frame, text="Event (1=Event, 0=Censored):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.event_var = tk.StringVar()
        self.event_combo = ttk.Combobox(columns_frame, textvariable=self.event_var, state="readonly", width=25)
        self.event_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(columns_frame, text="Variable of Interest:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.voi_var = tk.StringVar()
        self.voi_combo = ttk.Combobox(columns_frame, textvariable=self.voi_var, state="readonly", width=25)
        self.voi_combo.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(columns_frame, text="Covariates (for Adjusted Model):").grid(row=0, column=2, padx=15, pady=2, sticky=tk.W)
        self.covariates_listbox_frame = ttk.Frame(columns_frame)
        self.covariates_listbox_frame.grid(row=0, column=3, rowspan=3, padx=5, pady=2, sticky=tk.NSEW)
        self.covariates_listbox = tk.Listbox(self.covariates_listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, height=5, width=30)
        self.covariates_listbox_scrollbar = ttk.Scrollbar(self.covariates_listbox_frame, orient=tk.VERTICAL, command=self.covariates_listbox.yview)
        self.covariates_listbox.configure(yscrollcommand=self.covariates_listbox_scrollbar.set)
        self.covariates_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.covariates_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        analysis_frame = ttk.LabelFrame(main_frame, text="3. Run Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=5)

        self.run_unadjusted_button = ttk.Button(analysis_frame, text="Run Unadjusted Cox Regression", command=self.run_unadjusted_cox)
        self.run_unadjusted_button.grid(row=0, column=0, padx=5, pady=5)

        self.run_adjusted_button = ttk.Button(analysis_frame, text="Run Adjusted Cox Regression", command=self.run_adjusted_cox)
        self.run_adjusted_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.check_prophaz_button = ttk.Button(analysis_frame, text="Check Proportional Hazards", command=self.check_proportional_hazards)
        self.check_prophaz_button.grid(row=1, column=0, padx=5, pady=5)
        
        plot_effects_frame = ttk.Frame(analysis_frame)
        plot_effects_frame.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        self.plot_effects_button = ttk.Button(plot_effects_frame, text="Plot Partial Effects for:", command=self.plot_partial_effects)
        self.plot_effects_button.pack(side=tk.LEFT, padx=(0,5))
        self.plot_covariate_var = tk.StringVar()
        self.plot_covariate_combo = ttk.Combobox(plot_effects_frame, textvariable=self.plot_covariate_var, state="readonly", width=20)
        self.plot_covariate_combo.pack(side=tk.LEFT)

        results_frame = ttk.LabelFrame(main_frame, text="4. Results", padding="10")
        results_frame.pack(expand=True, fill=tk.BOTH, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=15, width=80)
        self.results_text.pack(expand=True, fill=tk.BOTH)
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
            self.file_label.config(text=file_path.split('/')[-1])
            self.update_column_selectors()
            self.results_text_clear()
            self.results_text_append("CSV file loaded successfully.\n")
            self.results_text_append(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns.\n")
            self.results_text_append("First 5 rows:\n")
            self.results_text_append(str(self.df.head()) + "\n")
            self.fitted_model = None
            self.last_run_covariates = []
            self.last_run_data_subset = None 
            self.plot_covariate_combo['values'] = []
            self.plot_covariate_var.set('')
        except Exception as e:
            messagebox.showerror("Error Loading CSV", f"Failed to load or parse CSV file.\nError: {e}")
            self.df = None
            self.file_label.config(text="No file loaded.")
            self.update_column_selectors()

    def update_column_selectors(self):
        columns = list(self.df.columns) if self.df is not None else []
        self.duration_combo['values'] = columns
        self.event_combo['values'] = columns
        self.voi_combo['values'] = columns
        self.covariates_listbox.delete(0, tk.END)
        for col in columns:
            self.covariates_listbox.insert(tk.END, col)
        if not columns:
            self.duration_var.set('')
            self.event_var.set('')
            self.voi_var.set('')

    def results_text_append(self, text_to_append):
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, text_to_append)
        self.results_text.configure(state='disabled')
        self.results_text.see(tk.END)

    def results_text_clear(self):
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.configure(state='disabled')

    def _get_selected_columns(self, include_covariates_for_adjusted=False, include_voi=False):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return None

        duration_col = self.duration_var.get()
        event_col = self.event_var.get()

        if not duration_col or not event_col:
            messagebox.showerror("Error", "Please select Duration and Event columns.")
            return None

        selected_data = self.df.copy() 

        try:
            selected_data[duration_col] = pd.to_numeric(selected_data[duration_col])
            if selected_data[duration_col].isnull().any(): 
                self.results_text_append(f"Warning: Duration column '{duration_col}' has missing values after numeric conversion. Rows with these NaNs will be dropped.\n")
        except ValueError: 
            messagebox.showerror("Error", f"Duration column '{duration_col}' must be convertible to a numeric type.")
            return None
        
        try:
            selected_data[event_col] = pd.to_numeric(selected_data[event_col])
            if selected_data[event_col].isnull().any():
                 self.results_text_append(f"Warning: Event column '{event_col}' has missing values after numeric conversion. Rows with these NaNs will be dropped.\n")
            if not selected_data[event_col].dropna().isin([0, 1]).all(): 
                messagebox.showwarning("Warning", f"Event column '{event_col}' contains values other than 0 and 1 (and NaNs). Ensure 1 means event and 0 means censored.")
        except ValueError:
            messagebox.showerror("Error", f"Event column '{event_col}' must be convertible to a numeric type.")
            return None

        cols_to_use_for_dropna = [duration_col, event_col]
        raw_covariate_col_names = [] 
        
        if include_covariates_for_adjusted:
            selected_indices = self.covariates_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select at least one covariate for the adjusted model.")
                return None
            raw_covariate_col_names = [self.covariates_listbox.get(i) for i in selected_indices]
        elif include_voi:
            voi_col_name = self.voi_var.get()
            if not voi_col_name:
                messagebox.showerror("Error", "Please select a 'Variable of Interest'.")
                return None
            raw_covariate_col_names = [voi_col_name]

        processed_covariate_cols_for_formula = []
        for col_name in raw_covariate_col_names:
            if col_name == duration_col or col_name == event_col:
                messagebox.showerror("Error", f"Covariate/VOI '{col_name}' cannot be the same as Duration or Event column.")
                return None
            try:
                #forcing to numeric, coercing errors. If a column is truly categorical and
                #needs to be treated as such by lifelines (e.g. with C() syntax),
                #this simple conversion might not be ideal. But for basic numeric covariates, it's fine.
                numeric_col_series = pd.to_numeric(selected_data[col_name], errors='coerce')
                
                if numeric_col_series.isnull().all():
                    msg = f"Covariate/VOI '{col_name}' contains no valid numeric data after conversion and will be excluded."
                    if include_voi: messagebox.showerror("Data Error", msg); return None
                    else: messagebox.showwarning("Data Warning", msg); continue 
                
                selected_data[col_name] = numeric_col_series 
                cols_to_use_for_dropna.append(col_name)
                processed_covariate_cols_for_formula.append(col_name) 

            except Exception as e: 
                msg = f"Could not process covariate/VOI '{col_name}' as numeric (Error: {e}). It will be skipped."
                if include_voi: messagebox.showerror("Data Error", msg); return None
                else: messagebox.showwarning("Conversion Warning", msg); continue
        
        if include_voi and not processed_covariate_cols_for_formula:
             messagebox.showerror("Error", "The selected Variable of Interest could not be processed or was invalid.")
             return None
        if include_covariates_for_adjusted and not processed_covariate_cols_for_formula:
            messagebox.showerror("Error", "No valid covariates selected or remaining after initial processing for the adjusted model.")
            return None

        original_rows = len(selected_data)
        selected_data.dropna(subset=cols_to_use_for_dropna, inplace=True)
        
        if len(selected_data) < original_rows:
            self.results_text_append(f"Note: {original_rows - len(selected_data)} rows with missing values in selected columns (Duration, Event, and processed Covariates/VOI) were dropped.\n")
        
        if len(selected_data) == 0:
            messagebox.showerror("Error", "No data remains after dropping missing values from selected columns.")
            return None

        final_valid_formula_cols = []
        if processed_covariate_cols_for_formula: 
            for col_name in processed_covariate_cols_for_formula:
                if selected_data[col_name].nunique(dropna=True) <= 1: 
                    msg = f"Covariate/VOI '{col_name}' has no variance (all values are the same) in the filtered data and will be excluded."
                    if include_voi and len(processed_covariate_cols_for_formula) == 1: 
                        messagebox.showerror("Data Error", msg)
                        return None
                    else: 
                        messagebox.showwarning("Data Warning", msg)
                        continue 
                final_valid_formula_cols.append(col_name)
        
        if (include_voi or include_covariates_for_adjusted) and not final_valid_formula_cols:
            messagebox.showerror("Error", "No usable covariates/VOI with variance remain after data cleaning. Cannot fit model.")
            return None
            
        return selected_data, duration_col, event_col, final_valid_formula_cols


    def run_cox_model(self, data_subset, duration_col, event_col, covariate_cols_for_formula):
        cph = CoxPHFitter()
        self.last_run_data_subset = None 
        try:
            current_formula_str = None
            columns_for_fit_df = [duration_col, event_col] 

            if covariate_cols_for_formula:
                #this is the formula section for adjusted, the cox function needs specific formula style
                current_formula_str = " + ".join(covariate_cols_for_formula) 
                self.results_text_append(f"Fitting model with formula: '{current_formula_str}'\n")
                columns_for_fit_df.extend(covariate_cols_for_formula)
                #ensure no duplicate columns
                columns_for_fit_df = sorted(list(set(columns_for_fit_df))) 
                fit_df = data_subset[columns_for_fit_df].copy()
                cph.fit(fit_df, duration_col=duration_col, event_col=event_col, formula=current_formula_str)
            else:
                self.results_text_append("Warning: No valid covariates provided. Fitting a baseline Cox model (intercept only).\n")
                fit_df = data_subset[columns_for_fit_df].copy()
                cph.fit(fit_df, duration_col=duration_col, event_col=event_col)

            self.results_text_append("\n--- Cox Model Summary ---\n")
            summary_df = cph.summary
            self.results_text_append(str(summary_df) + "\n")

            if covariate_cols_for_formula and not summary_df.empty : 
                try:
                    #here we predict on the same data used for fit, using only the covariate columns
                    predictions = cph.predict_partial_hazard(fit_df[covariate_cols_for_formula]) 
                    c_index = concordance_index(fit_df[duration_col], -predictions, fit_df[event_col])
                    self.results_text_append(f"\nConcordance Index (C-statistic): {c_index:.4f}\n")
                except Exception as ci_e:
                    self.results_text_append(f"\nCould not calculate Concordance Index: {ci_e}\n")
            else:
                 self.results_text_append("\nConcordance Index not applicable (no covariates in the final model).\n")
            
            self.fitted_model = cph 
            self.last_run_covariates = covariate_cols_for_formula 
            self.last_run_data_subset = fit_df.copy() #store the exact df used for fitting

            self.plot_covariate_combo['values'] = self.last_run_covariates
            if self.last_run_covariates:
                self.plot_covariate_var.set(self.last_run_covariates[0])
            else:
                self.plot_covariate_var.set('')

        except Exception as e:
            messagebox.showerror("Cox Regression Error", f"An error occurred during model fitting: {e}")
            self.results_text_append(f"\nError during Cox regression: {e}\n")
            self.results_text_append(f"Traceback:\n{traceback.format_exc()}\n")
            self.fitted_model = None
            self.last_run_covariates = []
            self.last_run_data_subset = None
            self.plot_covariate_combo['values'] = []
            self.plot_covariate_var.set('')


    def run_unadjusted_cox(self):
        self.results_text_clear()
        self.results_text_append("Running Unadjusted Cox Regression...\n")
        
        prepared_data = self._get_selected_columns(include_voi=True)
        if prepared_data is None:
            self.results_text_append("Failed to prepare data for unadjusted model (see error popups for details).\n")
            return
        
        data_subset, duration_col, event_col, voi_col_list_for_formula = prepared_data
        
        if not voi_col_list_for_formula: 
            self.results_text_append("Error: Variable of Interest list is empty after data preparation, cannot run model.\n")
            messagebox.showerror("Model Error", "No valid Variable of Interest to use for the model.")
            return

        self.results_text_append(f"Duration column: {duration_col}\n")
        self.results_text_append(f"Event column: {event_col}\n")
        self.results_text_append(f"Variable of Interest (for formula): {voi_col_list_for_formula[0]}\n")
        self.results_text_append(f"Data subset for model has {len(data_subset)} rows.\n")
        
        cols_to_show = [duration_col, event_col] + voi_col_list_for_formula
        self.results_text_append(f"Head of relevant columns in data_subset:\n{str(data_subset[cols_to_show].head())}\n")
        self.results_text_append(f"Info for VOI '{voi_col_list_for_formula[0]}' in data_subset:\n{str(data_subset[voi_col_list_for_formula[0]].describe())}\n")

        self.run_cox_model(data_subset, duration_col, event_col, voi_col_list_for_formula)

    def run_adjusted_cox(self):
        self.results_text_clear()
        self.results_text_append("Running Adjusted Cox Regression...\n")

        prepared_data = self._get_selected_columns(include_covariates_for_adjusted=True)
        if prepared_data is None:
            self.results_text_append("Failed to prepare data for adjusted model (see error popups for details).\n")
            return

        data_subset, duration_col, event_col, covariate_cols_for_formula = prepared_data
        
        if not covariate_cols_for_formula:
            self.results_text_append("Error: No valid covariates remaining after data preparation, cannot run adjusted model.\n")
            messagebox.showerror("Model Error", "No valid covariates to use for the adjusted model.")
            return
            
        self.results_text_append(f"Covariates for formula: {', '.join(covariate_cols_for_formula)}\n")
        self.results_text_append(f"Data subset for model has {len(data_subset)} rows.\n")
        cols_to_show = [duration_col, event_col] + covariate_cols_for_formula
        self.results_text_append(f"Head of relevant columns in data_subset:\n{str(data_subset[cols_to_show].head())}\n")

        self.run_cox_model(data_subset, duration_col, event_col, covariate_cols_for_formula)

    def check_proportional_hazards(self):
        if self.fitted_model is None or self.last_run_data_subset is None:
            messagebox.showerror("Error", "Please run a Cox model first and ensure data was available for it.")
            return
        
        if not self.last_run_covariates: 
             messagebox.showwarning("Info", "No covariates were in the last successfully fitted model. Proportional hazards check is for covariates.")
             return

        self.results_text_append("\n--- Proportional Hazards Assumption Check ---\n")
        try:
            df_for_ph_check = self.last_run_data_subset 
            
            self.results_text_append(f"Checking PH on data with {len(df_for_ph_check)} rows. Model's internal formula will be used.\n")
            
            results_ph = self.fitted_model.check_assumptions(df_for_ph_check, p_value_threshold=0.05, show_plots=False)
            
            self.results_text_append(f"Type of result from check_assumptions: {type(results_ph)}\n")
            if isinstance(results_ph, (list, pd.DataFrame)):
                 self.results_text_append(f"Content of result: {str(results_ph)}\n")

            self.results_text_append("Test Results (p-values for deviation from proportionality):\n")
            
            if isinstance(results_ph, pd.DataFrame):
                if not results_ph.empty:
                    self.results_text_append(results_ph.to_string() + "\n")
                else:
                    self.results_text_append("Proportional hazards test returned an empty DataFrame.\n"
                                             "This may occur if no violations are detected or if the test is not applicable to the specific covariates/data.\n")
            elif isinstance(results_ph, list):
                if not results_ph: 
                    self.results_text_append("Proportional hazards test returned an empty list.\n"
                                             "This can happen with single continuous covariates or specific data conditions where the standard p-value table is not generated by the `lifelines` library.\n"
                                             "It means the automated test didn't produce a summary p-value for the covariate(s) in this specific scenario.\n")
                                             
                else: 
                    self.results_text_append("Proportional hazards test returned a list of results (which is unexpected when `show_plots=False` is used with `lifelines`):\n")
                    for i, item in enumerate(results_ph):
                        self.results_text_append(f"Item {i}: {str(item)}\n")
            elif results_ph is None:
                 self.results_text_append("Proportional hazards test did not return any results (None).\n")
            else: 
                 self.results_text_append(f"Proportional hazards test returned an unexpected type: {type(results_ph)}. Content: {str(results_ph)}\n")

            self.results_text_append("A low p-value (e.g., < 0.05) in the test results would suggest that the proportional hazard assumption may be violated for that covariate.\n")
            
        except Exception as e:
            if "must have at least one column" in str(e) or "Found no covariates" in str(e) or "less than 2 levels" in str(e):
                 messagebox.showwarning("Assumption Check Info", f"The proportional hazards test could not be performed. This might be due to the nature of the covariates (e.g., all constant after some filtering by the test) or specific data conditions. Details: {e}")
                 self.results_text_append(f"\nProportional hazards test not applicable or failed. Reason: {e}\n")
            else:
                messagebox.showerror("Assumption Check Error", f"Error during proportional hazards check: {e}")
                self.results_text_append(f"\nError during assumption check: {e}\n")
                self.results_text_append(f"Traceback:\n{traceback.format_exc()}\n")


    def plot_partial_effects(self):
        if self.fitted_model is None or self.last_run_data_subset is None:
            messagebox.showerror("Error", "Please run a Cox model first and ensure data was available for it.")
            return

        covariate_to_plot = self.plot_covariate_var.get()
        if not covariate_to_plot:
            messagebox.showerror("Error", "Please select a covariate to plot from the dropdown.")
            return

        if covariate_to_plot not in self.last_run_covariates: 
            messagebox.showerror("Error", f"Covariate '{covariate_to_plot}' was not in the last model run or is not available for plotting.")
            return
        
        self.results_text_append(f"\n--- Plotting Partial Effects for '{covariate_to_plot}' ---\n")
        
        try:
            fig, ax = plt.subplots() 
            
            plot_values = None 
            if covariate_to_plot in self.last_run_data_subset.columns:
                cov_series = self.last_run_data_subset[covariate_to_plot].dropna()
                if pd.api.types.is_numeric_dtype(cov_series) and cov_series.nunique() > 1:
                    quantiles = cov_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unique()
                    quantiles.sort()
                    if len(quantiles) >= 2 : 
                        plot_values = quantiles.tolist()
                    elif cov_series.nunique() > 0 : 
                        plot_values = sorted(list(cov_series.unique()))[:5] 
                        if len(plot_values) < 2: plot_values = None 
            
            self.results_text_append(f"Attempting to plot '{covariate_to_plot}' with values: {plot_values}\n")

            self.fitted_model.plot_partial_effects_on_outcome(
                covariates=[covariate_to_plot], 
                values=plot_values, 
                plot_baseline=True, 
                ax=ax 
            )
            plt.tight_layout() 
            plt.show() 
            
            self.results_text_append(f"Plot for '{covariate_to_plot}' displayed in a new window.\n")

        except Exception as e:
            messagebox.showerror("Plotting Error", f"Error plotting partial effects for {covariate_to_plot}: {e}")
            self.results_text_append(f"\nError plotting partial effects for {covariate_to_plot}: {e}\n")
            self.results_text_append(f"Traceback:\n{traceback.format_exc()}\n")



#main function call here
if __name__ == "__main__":
    root = tk.Tk()
    app = CoxRegressionApp(root)
    root.mainloop()
