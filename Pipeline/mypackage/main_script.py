# mypackage/main_script.py
import argparse
from data_processing import load_data, classify_route, get_denoised_signal, create_binary_df, df_to_geojson_anomalies, signal_anomaly_neighborhood, df_to_geojson_neighborhood, predict_signal_h2, predict_signal_ch4, plot_geojson

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Processing Pipeline")
    parser.add_argument("mission_name",help="Name of the mission to analize (D+dd/mm/yy)")
    parser.add_argument("data_file", help="Path to the input data file")
    parser.add_argument("output_path", help="Path to save the processed results")
    parser.add_argument("smoothing",help="Want to smooth the signal? (yes or no)")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Step 1: Load Data
    data = load_data(args.data_file,args.mission_name)


    # Step 2: Process Data
    df_1=data
    label=classify_route(df_1)
    gases=['CO', 'NO2', 'C3H8', 'C4H10', 'CH4', 'H2', 'C2H5OH']
    #Smooth the signal (yes or no)
    if args.smoothing=='yes':
        #Denoise the signal
        df_denoised=get_denoised_signal(df_1,0.1,gases)
        #Update gas measurements with the denoised signal
        df_1.loc[:, gases] = df_denoised[gases].values
    
    #Create a binary df with the anomalies
    df_1=create_binary_df(df_1,20,gases)
    gpd_anomal=df_to_geojson_anomalies(df_1,args.mission_name,args.output_path,df_1.columns)
    #Create df with neighborhoods
    signal_neigh=signal_anomaly_neighborhood(df_1,gases,3)
    gdp_neigh=df_to_geojson_neighborhood(signal_neigh,gases,args.output_path)
    #Prediction 
    predict_signal_h2(df_1,'Pipeline/models/linear_reg_mult_input_H2.sav',['CH4','CO','A_1','C3H8','C4H10','T_3'])
    predict_signal_ch4(df_1,'Pipeline/models/linear_reg_mult_input_CH4.sav',['A_1', 'CO', 'C3H8', 'T_3', 'C2H5OH', 'C4H10'])

    #Plotting all the signals
    plot_geojson(gpd_anomal,gdp_neigh,gases)


if __name__ == "__main__":
    main()
