import numpy as np
import pandas as pd


def bold_extreme_values(data, best=-1, second_best=-1, decimal_places:int  = 2):
    if data == best:
        return "\\textbf{%s}" % format_number(data, decimal_places=decimal_places)

    if data == second_best:
        return "\\underline{%s}" % format_number(data, decimal_places=decimal_places)

    return format_number(data, decimal_places=decimal_places)

def format_number(data, decimal_places: int = 2, maximum_length_before_comma: int = 6):
    format_string = ("{:." + str(decimal_places) + "f}")
    format_string_scientific_notation = ("{:." + str(decimal_places) + "e}")
    formated_string = format_string.format(data)
    if len(formated_string) > maximum_length_before_comma  + decimal_places:
        formated_string = format_string_scientific_notation.format(data)
    return formated_string


def generate_result_table(df, stddev_df, stddev: bool = False, decimal_places: int = 3,
                          show_avg_and_median: bool = True):

    df_no_format = df.copy()
    df_decimal_format = df.copy()[["GP Baseline"]]
    df = df.drop(columns=["GP Baseline", "Model", "Hyperparameters", "Dataset"])

    if show_avg_and_median:
        df_avg = df.mean()
        df_median = df.median()

    for k in range(len(df.index)):
        df.iloc[k] = df.iloc[k].apply(
            lambda data: bold_extreme_values(data, best=df.iloc[k].min(), second_best=np.partition(df.iloc[k].array.to_numpy(), 1)[1], decimal_places=decimal_places))

    for k in range(len(df_decimal_format.index)):
        df_decimal_format.iloc[k] = df_decimal_format.iloc[k].apply(lambda data: format_number(data, decimal_places=decimal_places))

    df["GP Baseline"] = df_decimal_format["GP Baseline"]

    if stddev:
        for k in range(len(stddev_df.index)):
            stddev_df.iloc[k] = stddev_df.iloc[k].apply(
                lambda data: format_number(data, decimal_places=decimal_places))

        df = df.astype(str) + " $\\pm$ " + stddev_df.astype(str)

    if show_avg_and_median:
        df.loc['avg'] = df_avg
        df.loc['median'] = df_median

    df.insert(0, "Model", df_no_format["Model"])
    df.insert(1, "Hyperparameters", df_no_format["Hyperparameters"])
    df.insert(2, "Dataset", df_no_format["Dataset"])
    df["Model"] = df["Model"].str.replace('XGBoost', 'XGB')
    df["Model"] = df["Hyperparameters"].str.replace('_', '\_')
    df["Dataset"] = df["Dataset"].str.replace('blood-transfusion-service-center', 'blood-transfusion')

    # Set column header to bold title case
    df.columns = (df.columns.to_series()
                  .apply(lambda r:
                         '\\multicolumn{1}{c}{{' + r.replace('_', '\_') + '}}'))

    return df.to_latex(index=False, escape=False)