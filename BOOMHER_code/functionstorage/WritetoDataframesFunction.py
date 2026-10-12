import pandas as pd

def  Writetodataframes(imagenumber,volumedata,X_data,Y_data,df_volume,df_xpos,df_ypos,df_ID,df_meanradi,meanradi,df_xrange , df_yrange,radiset1,radiset2):
    
    columnname = "time = "+str(imagenumber)
    #columnname_time = "timedata = " +str(imagenumber)
        # volume
    volume_series = pd.Series(volumedata)
    df_volume[columnname] = volume_series * 1000 # *1000 to get in to nL from µL 
        # postions
    X_series = pd.Series(X_data)
    df_xpos[columnname] = X_series    

    Y_series = pd.Series(Y_data)
    df_ypos[columnname] = Y_series 
    
    # ranges 
    
    rangeX_series = pd.Series(radiset1)
    df_xrange[columnname] = rangeX_series 
    
    rangeY_series = pd.Series(radiset2)
    df_yrange[columnname] = rangeY_series 

        # ID
    #volume_series  = pd.Series(volumedata)
    #df_ID[columnname] = volume_series * 0 # makes the ID system a bunch of 0 and NaN 

        #meanradi (we average the x and y radi hence the mean)
    meanradi_series = pd.Series(meanradi)
    df_meanradi[columnname] = meanradi_series
    
    return df_volume , df_xpos, df_ypos , df_ID ,df_meanradi , df_xrange , df_yrange
    