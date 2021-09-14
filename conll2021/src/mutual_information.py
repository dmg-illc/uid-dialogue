import argparse
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path_nocontext", type=str, required=True,
        help="The path to the input pandas dataframe containing decontextualised entropy measurements."
    )
    parser.add_argument(
        "--data_path_context1", type=str, required=True,
        help="The path to the input pandas dataframe containing contextualised entropy measurements."
    )
    parser.add_argument(
        "--data_path_context2", type=str,
        help="The path to the input pandas dataframe containing contextualised entropy measurements."
    )
    parser.add_argument(
        "--context1", type=str, required=True,
        help="The name of the first contextual unit."
    )
    parser.add_argument(
        "--context2", type=str,
        help="The name of the second contextual unit."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output directory path for a .csv file containing the output dataframe."
    )
    parser.add_argument(
        "--max_position", default=-1, type=int,
        help="The maximum sentence position."
    )
    args = parser.parse_args()

    df_d = pd.read_csv(args.data_path_nocontext)
    df_c1 = pd.read_csv(args.data_path_context1)

    if args.max_position > 0:
        try:
            df_d = df_d[df_d['position'] <= args.max_position]
            df_c1 = df_c1[df_c1['position'] <= args.max_position]
        except KeyError:
            try:
                df_d = df_d[df_d['position_in_doc'] <= args.max_position]
                df_c1 = df_c1[df_c1['position_in_doc'] <= args.max_position]
            except KeyError:
                df_d = df_d[df_d['position_in_dialogue'] <= args.max_position]
                df_c1 = df_c1[df_c1['position_in_dialogue'] <= args.max_position]

    if len(df_d) - 1 == len(df_c1):
        df_d = df_d[:-1]

    if args.data_path_context2:
        df_c2 = pd.read_csv(args.data_path_context2)
        df_c_list = [(df_c1, args.context1), (df_c2, args.context2)]
    else:
        df_c_list = [(df_c1, args.context1)]

    for df_c, c_unit in df_c_list:
        assert len(df_d) == len(df_c), \
            '{}: Dataframes must have the same number of data points.: {} {}'.format(c_unit, len(df_d), len(df_c))

    df_new = df_d.copy()

    print(df_c['normalised_h'].isnull().sum())

    for df_c, c_unit in df_c_list:
        print(c_unit)
        mi = []

        for (index, row_d), (_, row_c) in zip(df_d.iterrows(), df_c.iterrows()):
            try:
                _mi = row_d['normalised_h'] - row_c['normalised_h']
                mi.append(_mi)
            except KeyError:
                # print(row_d['normalised_h'], row_c['normalised_h'])
                mi.append(np.nan)

        mi_field = 'mi_{}'.format(c_unit)
        xu_h_c_field = 'xu_h_{}'.format(c_unit)
        h_c_field = 'normalised_h_{}'.format(c_unit)

        df_new.loc[:, mi_field] = mi
        df_new.loc[:, xu_h_c_field] = df_c['xu_h'].values
        df_new.loc[:, h_c_field] = df_c['normalised_h'].values


        # mi_bar = df_new.groupby('length').agg({mi_field: 'mean'})
        # xu_mi = []
        # for index, row in df_new.iterrows():
        #     try:
        #         xu_mi.append(row[mi_field] / mi_bar.loc[row['length'], mi_field])
        #     except KeyError:
        #         xu_mi.append(np.nan)
        # df_new.loc[:, 'xu_{}'.format(mi_field)] = xu_mi

    df_new.to_csv(
        args.out_path,
        index=False
    )

    print('Saved dataframe to {}'.format(args.out_path))