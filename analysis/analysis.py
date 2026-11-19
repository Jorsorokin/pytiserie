from pytiserie.spring import compute_spring_metrics
from pathlib import Path
import pandas as pd


DATA_PATH = Path(__file__).parent.parent / "data" / "2026" / "croissant_data.csv"


def get_spring_measurements(data: pd.DataFrame) -> pd.DataFrame:
    df = data.loc[data.metric.str.startswith("height_")]
    df['t'] = df.metric.str.split("_").str[-1].str[-1].astype(int)  # starting with t=0
    df = df.sort_values(by='t', ascending=True)
    return df


def main():
    data = pd.read_csv(DATA_PATH)
    all_results = []

    for (bakery, df) in data.groupby('bakery'):
        print(f"Bakery: {bakery}")
        spring_data = get_spring_measurements(df)

        for typ in ("single", "double"):
            double = typ == "double"
            spring_metrics = compute_spring_metrics(spring_data['value'].values, double=double)
            spring_metrics['bakery'] = bakery
            spring_metrics['fit_type'] = typ
            all_results.append(spring_metrics)

    results_df = pd.DataFrame(all_results).set_index("bakery")
    print(results_df)


if __name__ == "__main__":
    main()