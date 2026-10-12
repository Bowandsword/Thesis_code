import numpy as np
import pandas as pd
def setupdataframes(keepdata,imagenumber_start) :

    if keepdata == 0 :
        # force 255 rows its just a lot and therfore plenty 
        numbers_list = np.arange(0, 255) * 0 
        numbers_series = pd.Series(numbers_list)
        # make df
        df_volume   = pd.DataFrame()
        df_xpos     = pd.DataFrame()
        df_xrange   = pd.DataFrame()
        df_ypos     = pd.DataFrame()
        df_yrange   = pd.DataFrame()
        df_ID       = pd.DataFrame()
        df_meanradi = pd.DataFrame()
        # force it to have 255 rows 
        df_volume['Have 255 you have no choice'] = numbers_series
        df_xpos['Have 255 you have no choice'] = numbers_series
        df_ypos['Have 255 you have no choice'] = numbers_series
        df_ID['Have 255 you have no choice'] = numbers_series
        df_meanradi['Have 255 you have no choice'] = numbers_series
        df_xrange['Have 255 you have no choice'] = numbers_series
        df_yrange['Have 255 you have no choice'] = numbers_series
        imagenumber = imagenumber_start 
    if keepdata == 1 :
        # load me the dataframes 
        file_path = 'df_volume.csv'
        df_volume = pd.read_csv(file_path)

        file_path = 'df_xpos.csv'
        df_xpos = pd.read_csv(file_path)

        file_path = 'df_ypos.csv'
        df_ypos = pd.read_csv(file_path)

        file_path = 'df_ID.csv'
        df_ID = pd.read_csv(file_path)
        
        file_path = 'df_meanradi.csv'
        df_meanradi = pd.read_csv(file_path)        

        last_column_name = df_volume.columns[-1]
        cut_column = last_column_name.replace("time = ","")
        imagenumber = int(cut_column) + 1

    return df_volume , df_xpos , df_ypos , df_ID, df_meanradi , imagenumber , df_yrange , df_xrange

